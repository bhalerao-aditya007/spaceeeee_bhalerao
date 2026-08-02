#!/usr/bin/env python3
"""
Spacecraft Autonomy — Integrated Web Dashboard
Connects all 5 agents via Redis pub/sub and exposes them through
a unified web interface with real-time WebSocket updates.

Run from project root:
    python interface/app.py
    # or
    uvicorn interface.app:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
import os
import sys
import redis_shim
import redis
redis.Redis = redis_shim.Redis  
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import asyncio
import json
import time
import queue
import base64
import threading
import traceback
from io import BytesIO
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any

import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Dependency checks — graceful fallback when modules are unavailable
# ---------------------------------------------------------------------------
try:
    import redis as redis_lib
    _REDIS_LIB = True
except ImportError:
    _REDIS_LIB = False

try:
    from orchestrator.orchestrator import Orchestrator
    from orchestrator.message_schemas import (
        HumanOverrideMessage, OverrideLevel, ActionType,
        PoseEstimateMessage, SituationVectorMessage,
        ActionRecommendationMessage, ConsensusActionMessage,
    )
    _ORCH = True
except Exception:
    _ORCH = False

try:
    from simulation.scenario_engine import ScenarioEngine
    from simulation.scenarios.scenario_library import (
        nominal_docking, thermal_anomaly, perception_challenge, perfect_storm,
    )
    _SIM = True
except Exception:
    _SIM = False

try:
    from cognition.cognition_agent import HyperdimensionalCognitionLayer
    _COG = True
except Exception:
    _COG = False

try:
    from action.counterfactual import CounterfactualEngine
    from action.physics import default_spacecraft_config
    _ACT = True
except Exception:
    _ACT = False

try:
    from perception.perception_agent import PerceptionAgent
    _PERC = True
except Exception:
    _PERC = False

# ---------------------------------------------------------------------------
# Real model loading
# ---------------------------------------------------------------------------
_perception_agent: Optional[Any] = None
_MODEL_LOADED = False
_MODEL_INFO = {}

def _download_model_if_needed():
    """
    Fetch the ONNX checkpoint from Google Drive at container startup if it's
    not already present locally (or is a stale Git-LFS pointer file). This
    sidesteps needing `git lfs pull` to run during Render's build.
    """
    model_path = os.path.join(PROJECT_ROOT, "perception", "checkpoints", "best.pt")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Already present and looks like a real model file, not an LFS pointer
    if os.path.exists(model_path) and os.path.getsize(model_path) > 1_000_000:
        print(f"  [Model] Checkpoint already present locally "
              f"({os.path.getsize(model_path) / 1e6:.1f}MB) — skipping download")
        return

    file_id = os.environ.get("1h-pSmUkO2N0JiYAtnNdABFOjgAlOMaQ5")
    if not file_id:
        print("  [Model] GDRIVE_ONNX_FILE_ID env var not set — skipping download")
        return

    try:
        import gdown
    except ImportError:
        print("  [Model] gdown not installed — check requirements_web.txt")
        return

    print(f"  [Model] Downloading checkpoint from Google Drive (id={file_id})...")
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)
        if os.path.exists(model_path) and os.path.getsize(model_path) > 1_000_000:
            print(f"  [Model] Download complete: {os.path.getsize(model_path) / 1e6:.1f}MB")
        else:
            print("  [Model] Download finished but file is too small — "
                  "check Drive sharing permissions (must be 'Anyone with the link')")
    except Exception as exc:
        print(f"  [Model] Download failed: {exc}")
        traceback.print_exc()

def _load_perception_model():
    global _perception_agent, _MODEL_LOADED, _MODEL_INFO
    if not _PERC:
        print("  [Model] PerceptionAgent module not available")
        return
    model_path = os.path.join(PROJECT_ROOT, "perception", "checkpoints", "best.pt")
    if not os.path.exists(model_path):
        print(f"  [Model] Checkpoint not found: {model_path}")
        return
    fsize = os.path.getsize(model_path)
    if fsize < 1000:
        print(f"  [Model] Checkpoint too small ({fsize}B) - may be Git LFS pointer")
        print(f"  [Model] Run: git lfs pull")
        return
    try:
        _perception_agent = PerceptionAgent(
            model_path=model_path,
            n_elevation=32,
            n_inplane=8,
            n_jensen_rotations=8,
            run_jensen_gain=True,
        )
        _MODEL_LOADED = True
        import torch
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        _MODEL_INFO = {
            "backbone": "resnet50" if ckpt.get("model_class") == "PoseNet_ResNet50" else str(ckpt.get("cfg", {}).get("backbone", "unknown")),
            "epoch": int(ckpt.get("epoch", 0)),
            "rot_err_deg": float(round(ckpt.get("rot_err_deg", 0), 2)),
            "trans_err_m": float(round(ckpt.get("trans_err_m", 0), 4)),
            "img_size": int(ckpt.get("cfg", {}).get("img_size", 224)),
            "params": int(len(ckpt.get("state_dict", {}))),
            "file_size_mb": float(round(fsize / 1024 / 1024, 1)),
        }
        print(f"  [Model] LOADED: {_MODEL_INFO['backbone']}, "
              f"epoch {_MODEL_INFO['epoch']}, "
              f"{_MODEL_INFO['file_size_mb']}MB")
    except Exception as exc:
        print(f"  [Model] Failed to load: {exc}")
        traceback.print_exc()

_download_model_if_needed()
_load_perception_model()

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
STATE = {
    "redis_connected": False,
    "orchestrator_running": False,
    "scenario_running": False,
    "current_scenario": None,
    "model_loaded": _MODEL_LOADED,
    "modules": {
        "redis": _REDIS_LIB,
        "orchestrator": _ORCH,
        "simulation": _SIM,
        "perception": _PERC,
        "cognition": _COG,
        "action": _ACT,
    },
    "latest": {
        "perception": None,
        "cognition": None,
        "action": None,
        "consensus": None,
        "escalation": None,
        "status": None,
    },
    "event_log": [],
    "decision_history": [],
}

_orchestrator: Optional[Orchestrator] = None
_scenario_engine_stop = threading.Event()
_redis_running = threading.Event()

# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.connections: List[WebSocket] = []
        self._lock = threading.Lock()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        with self._lock:
            self.connections.append(ws)

    def disconnect(self, ws: WebSocket):
        with self._lock:
            if ws in self.connections:
                self.connections.remove(ws)

    async def broadcast(self, message: dict):
        with self._lock:
            targets = list(self.connections)
        dead = []
        for ws in targets:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        if dead:
            with self._lock:
                for ws in dead:
                    if ws in self.connections:
                        self.connections.remove(ws)


manager = ConnectionManager()
_msg_q: queue.Queue = queue.Queue(maxsize=2000)

# ---------------------------------------------------------------------------
# Redis subscriber (runs in background thread)
# ---------------------------------------------------------------------------
CHANNELS = [
    "perception.out", "cognition.out", "action.out",
    "orchestrator.consensus", "orchestrator.escalation",
    "orchestrator.status", "human.in",
]


def _summarize(channel: str, data: dict) -> str:
    """One-line summary of a Redis message for the event log."""
    try:
        if channel == "perception.out":
            return (f"JG={data.get('jensen_gain','?')}° "
                    f"conf={data.get('confidence_level','?')} "
                    f"trust={'✓' if data.get('is_trustworthy') else '✗'}")
        if channel == "cognition.out":
            tag = "ANOMALY" if data.get("anomaly_detected") else "nominal"
            return f"{tag} → {data.get('recommended_action','?')}"
        if channel == "action.out":
            return (f"{data.get('primary_action','?')} "
                    f"score={data.get('primary_score','?')} "
                    f"coll={data.get('collision_prob','?')}")
        if channel == "orchestrator.consensus":
            c = "✓" if data.get("consensus_reached") else "✗"
            return f"→ {data.get('final_action','?')} consensus={c}"
        if channel == "orchestrator.escalation":
            return f"ESCALATION: {str(data.get('reason',''))[:60]}"
        if channel == "orchestrator.status":
            return f"status={data.get('overall_status','?')}"
        if channel == "human.in":
            return f"OVERRIDE L{data.get('override_level','?')} → {data.get('selected_action','?')}"
    except Exception:
        pass
    return json.dumps(data, default=str)[:80]


def _redis_subscriber():
    """Blocking loop — runs in its own thread. Auto-reconnects on errors."""
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

    while True:
        try:
            r = redis_lib.Redis.from_url(redis_url, decode_responses=False,
                                         socket_timeout=None,
                                         socket_keepalive=True)
            r.ping()
            STATE["redis_connected"] = True
            _msg_q.put({"type": "system_event", "event": "redis_connected"})
        except Exception as exc:
            STATE["redis_connected"] = False
            _msg_q.put({"type": "system_event", "event": "redis_error",
                         "message": str(exc)})
            time.sleep(2)
            continue

        ps = r.pubsub()
        ps.subscribe(*CHANNELS)
        _redis_running.set()

        try:
            for raw in ps.listen():
                if not _redis_running.is_set():
                    ps.close()
                    return
                if raw["type"] != "message":
                    continue

                channel = raw["channel"].decode() if isinstance(raw["channel"], bytes) else raw["channel"]
                try:
                    payload = json.loads(raw["data"])
                except Exception:
                    payload = {"raw": raw["data"].decode() if isinstance(raw["data"], bytes) else str(raw["data"])}

                # Update latest state
                _map = {
                    "perception.out": "perception",
                    "cognition.out": "cognition",
                    "action.out": "action",
                    "orchestrator.consensus": "consensus",
                    "orchestrator.escalation": "escalation",
                    "orchestrator.status": "status",
                }
                key = _map.get(channel)
                if key:
                    STATE["latest"][key] = payload
                if channel == "orchestrator.consensus":
                    STATE["decision_history"].append(payload)
                    if len(STATE["decision_history"]) > 100:
                        STATE["decision_history"] = STATE["decision_history"][-100:]

                ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
                entry = {"time": ts, "channel": channel,
                         "summary": _summarize(channel, payload)}
                STATE["event_log"].append(entry)
                if len(STATE["event_log"]) > 300:
                    STATE["event_log"] = STATE["event_log"][-300:]

                _msg_q.put({"type": "redis_message", "channel": channel,
                             "data": payload, "timestamp": time.time()})
        except Exception as exc:
            print(f"  [Redis] Subscriber error: {exc}, reconnecting in 1s...")
            STATE["redis_connected"] = False
            try:
                ps.close()
            except Exception:
                pass
            time.sleep(1)
            continue


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(application):
    # Startup
    if _REDIS_LIB:
        threading.Thread(target=_redis_subscriber, daemon=True).start()
    asyncio.create_task(_broadcast_loop())
    yield
    # Shutdown
    _redis_running.clear()

app = FastAPI(title="Spacecraft Autonomy Dashboard", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _broadcast_loop():
    """Drain _msg_q and push to all WebSocket clients."""
    while True:
        batch = []
        try:
            while True:
                batch.append(_msg_q.get_nowait())
        except queue.Empty:
            pass
        for msg in batch:
            await manager.broadcast(msg)
        await asyncio.sleep(0.05)


# ── HTML frontend ──────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


# ── Status ─────────────────────────────────────────────────────────────────
@app.get("/api/status")
async def api_status():
    return {
        "redis_connected": STATE["redis_connected"],
        "orchestrator_running": STATE["orchestrator_running"],
        "scenario_running": STATE["scenario_running"],
        "current_scenario": STATE["current_scenario"],
        "model_loaded": _MODEL_LOADED,
        "modules": STATE["modules"],
        "has_data": {k: v is not None for k, v in STATE["latest"].items()},
        "event_count": len(STATE["event_log"]),
        "decision_count": len(STATE["decision_history"]),
    }


@app.get("/api/model/status")
async def model_status():
    return {
        "loaded": _MODEL_LOADED,
        "info": _MODEL_INFO if _MODEL_LOADED else None,
        "perception_available": _PERC,
    }


@app.get("/api/latest")
async def api_latest():
    return STATE["latest"]


@app.get("/api/events")
async def api_events():
    return STATE["event_log"][-100:]


@app.get("/api/decisions")
async def api_decisions():
    return STATE["decision_history"][-50:]


# ── Orchestrator control ───────────────────────────────────────────────────
@app.post("/api/orchestrator/start")
async def start_orch():
    global _orchestrator
    if not _ORCH:
        return JSONResponse({"error": "Orchestrator module not available"}, 500)
    if STATE["orchestrator_running"]:
        return {"status": "already_running"}
    if not STATE["redis_connected"]:
        return JSONResponse({"error": "Redis not connected"}, 503)
    try:
        _orchestrator = Orchestrator()
        _orchestrator.start()
        STATE["orchestrator_running"] = True
        _msg_q.put({"type": "system_event", "event": "orchestrator_started"})
        return {"status": "started"}
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, 500)


@app.post("/api/orchestrator/stop")
async def stop_orch():
    global _orchestrator
    if _orchestrator and STATE["orchestrator_running"]:
        _orchestrator.stop()
        STATE["orchestrator_running"] = False
        _orchestrator = None
        _msg_q.put({"type": "system_event", "event": "orchestrator_stopped"})
        return {"status": "stopped"}
    return {"status": "not_running"}


# ── Scenarios ──────────────────────────────────────────────────────────────
_SCENARIOS = {}
if _SIM:
    _SCENARIOS = {
        "nominal": nominal_docking,
        "thermal": thermal_anomaly,
        "perception": perception_challenge,
        "perfect_storm": perfect_storm,
    }


@app.get("/api/scenarios")
async def list_scenarios():
    return {"available": list(_SCENARIOS.keys()),
            "running": STATE["scenario_running"],
            "current": STATE["current_scenario"]}


@app.post("/api/scenario/{name}")
async def run_scenario(name: str, speed: float = 5.0):
    if not _SIM:
        return JSONResponse({"error": "Simulation module not available"}, 500)
    if not STATE["redis_connected"]:
        return JSONResponse({"error": "Redis not connected"}, 503)
    if STATE["scenario_running"]:
        return JSONResponse({"error": "A scenario is already running"}, 409)
    if name not in _SCENARIOS:
        return JSONResponse({"error": f"Unknown scenario '{name}'. "
                             f"Available: {list(_SCENARIOS.keys())}"}, 404)

    _scenario_engine_stop.clear()

    def _run():
        STATE["scenario_running"] = True
        STATE["current_scenario"] = name
        _msg_q.put({"type": "system_event", "event": "scenario_started",
                     "scenario": name})
        try:
            eng = ScenarioEngine()
            eng.run_scenario(_SCENARIOS[name](), speed=speed)
        except Exception as exc:
            _msg_q.put({"type": "system_event", "event": "scenario_error",
                         "error": str(exc)})
        finally:
            STATE["scenario_running"] = False
            STATE["current_scenario"] = None
            _msg_q.put({"type": "system_event", "event": "scenario_complete",
                         "scenario": name})

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started", "scenario": name, "speed": speed}


# ── Override (Armstrong Protocol) ──────────────────────────────────────────
class OverrideRequest(BaseModel):
    level: str = "acknowledge"
    action: str = "hold_position"
    rationale: str = ""
    operator: str = "commander"


@app.post("/api/override")
async def send_override(req: OverrideRequest):
    if not STATE["redis_connected"]:
        return JSONResponse({"error": "Redis not connected"}, 503)
    try:
        r = redis_lib.Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
        msg = {
            "agent_id": "human",
            "message_type": "human_override",
            "timestamp": time.time(),
            "message_id": str(time.time_ns()),
            "override_level": req.level,
            "selected_action": req.action,
            "rationale": req.rationale,
            "modified_params": {},
            "operator_id": req.operator,
        }
        r.publish("human.in", json.dumps(msg))
        _msg_q.put({"type": "system_event", "event": "override_sent",
                     "level": req.level, "action": req.action})
        return {"status": "sent", "level": req.level, "action": req.action}
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, 500)


# ── Test injection ─────────────────────────────────────────────────────────
class InjectPerceptionRequest(BaseModel):
    jensen_gain: float = 2.5
    confidence: str = "moderate"
    distance: float = 10.0


@app.post("/api/inject/perception")
async def inject_perception(req: InjectPerceptionRequest):
    if not STATE["redis_connected"]:
        return JSONResponse({"error": "Redis not connected"}, 503)
    try:
        r = redis_lib.Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
        msg = {
            "agent_id": "perception",
            "message_type": "pose_estimate",
            "timestamp": time.time(),
            "message_id": str(time.time_ns()),
            "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "t": [req.distance, 0.0, 0.0],
            "quaternion": [1.0, 0.0, 0.0, 0.0],
            "jensen_gain": req.jensen_gain,
            "confidence_level": req.confidence,
            "confidence_label": f"{req.confidence.upper()} CONFIDENCE",
            "sigma_R_deg": round(req.jensen_gain * 0.6, 2),
            "sigma_t_m": round(0.05 * req.distance, 2),
            "nearest_anchor_idx": 0,
            "anchor_distance_deg": round(req.jensen_gain * 0.4, 2),
            "is_trustworthy": req.jensen_gain < 15.0,
            "processing_time_ms": 33.0,
            "image_shape": [224, 224, 3],
        }
        r.publish("perception.out", json.dumps(msg))
        _msg_q.put({"type": "system_event", "event": "perception_injected"})
        return {"status": "injected", "jensen_gain": req.jensen_gain}
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, 500)


class InjectCognitionRequest(BaseModel):
    anomaly: bool = False
    anomaly_type: str = "none"
    severity: str = "nominal"
    novelty: float = 0.1
    recommended_action: str = "proceed_slow"


@app.post("/api/inject/cognition")
async def inject_cognition(req: InjectCognitionRequest):
    if not STATE["redis_connected"]:
        return JSONResponse({"error": "Redis not connected"}, 503)
    try:
        r = redis_lib.Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
        msg = {
            "agent_id": "cognition",
            "message_type": "situation_vector",
            "timestamp": time.time(),
            "message_id": str(time.time_ns()),
            "situation_id": f"manual_{int(time.time())}",
            "anomaly_detected": req.anomaly,
            "anomaly_type": req.anomaly_type,
            "anomaly_severity": req.severity,
            "novelty_score": req.novelty,
            "similar_case_id": "" if req.anomaly else "case_2847",
            "similar_case_outcome": "" if req.anomaly else "success",
            "recommended_action": req.recommended_action,
            "action_confidence": 0.6 if req.anomaly else 0.91,
            "explanation": f"Manual injection: {req.anomaly_type}" if req.anomaly else "Manual injection: nominal",
        }
        r.publish("cognition.out", json.dumps(msg))
        _msg_q.put({"type": "system_event", "event": "cognition_injected"})
        return {"status": "injected"}
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, 500)


# ── Camera frame processing (REAL MODEL) ──────────────────────────────────
class FrameRequest(BaseModel):
    image: str  # base64-encoded JPEG/PNG


@app.post("/api/perception/frame")
async def process_camera_frame(req: FrameRequest):
    """Process a camera frame through the real EfficientNet-B3 model."""
    t_start = time.time()

    # Decode base64 image
    try:
        img_data = req.image
        if "," in img_data:
            img_data = img_data.split(",", 1)[1]
        img_bytes = base64.b64decode(img_data)
        from PIL import Image
        img_pil = Image.open(BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img_pil)
        
        # Save last uploaded image for offline debugging
        try:
            os.makedirs(os.path.join(PROJECT_ROOT, "perception", "outputs"), exist_ok=True)
            img_pil.save(os.path.join(PROJECT_ROOT, "perception", "outputs", "last_uploaded.png"))
            print("  [Debug] Saved last uploaded image to perception/outputs/last_uploaded.png")
        except Exception as e:
            print(f"  [Debug] Failed to save last uploaded image: {e}")
    except Exception as exc:
        return JSONResponse({"error": f"Failed to decode image: {exc}"}, 400)

    if _MODEL_LOADED and _perception_agent is not None:
        # ── Real model inference ──
        try:
            output = _perception_agent.predict(img_np)
            result = {
                "agent_id": "perception",
                "message_type": "pose_estimate",
                "source": "real_model",
                "timestamp": time.time(),
                "message_id": str(time.time_ns()),
                "R": output.pose.R,
                "t": output.pose.t,
                "quaternion": output.pose.quaternion,
                "jensen_gain": output.uncertainty.jensen_gain,
                "confidence_level": output.uncertainty.confidence_level,
                "confidence_label": output.uncertainty.confidence_label,
                "sigma_R_deg": output.uncertainty.sigma_R_deg,
                "sigma_t_m": output.uncertainty.sigma_t_m,
                "nearest_anchor_idx": output.uncertainty.nearest_anchor_idx,
                "anchor_distance_deg": output.uncertainty.anchor_distance_deg,
                "is_trustworthy": output.is_trustworthy,
                "processing_time_ms": output.metadata["processing_time_ms"],
                "image_shape": list(img_np.shape),
            }

            # Publish to Redis
            if STATE["redis_connected"]:
                try:
                    r = redis_lib.Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
                    r.publish("perception.out", json.dumps(result, default=str))
                except Exception:
                    pass

            # Also push directly to WebSocket
            _msg_q.put({"type": "redis_message", "channel": "perception.out",
                         "data": result, "timestamp": time.time()})
            STATE["latest"]["perception"] = result

            # Print raw model outputs to terminal
            print(f"\n>>> [MODEL INFERENCE OUTPUT] <<<")
            print(f"  Translation (t): {[round(x, 4) for x in output.pose.t]}")
            print(f"  Quaternion (q):  {[round(x, 4) for x in output.pose.quaternion]}")
            print(f"  Jensen Gain:     {output.uncertainty.jensen_gain:.2f}°")
            print(f"  Confidence:      {output.uncertainty.confidence_label}")
            print(f"  Processing Time: {output.metadata['processing_time_ms']:.1f}ms")
            print(f"=================================\n")

            total_ms = round((time.time() - t_start) * 1000, 1)
            return {
                "status": "processed",
                "model": "real",
                "backbone": _MODEL_INFO.get("backbone", "efficientnet_b3"),
                "jensen_gain": output.uncertainty.jensen_gain,
                "confidence": output.uncertainty.confidence_level,
                "trustworthy": output.is_trustworthy,
                "pose": {"t": output.pose.t, "quaternion": output.pose.quaternion},
                "total_ms": total_ms,
                "inference_ms": output.metadata["processing_time_ms"],
            }
        except Exception as exc:
            return JSONResponse({"error": f"Model inference failed: {exc}"}, 500)
    else:
        # ── Synthetic fallback ──
        h, w = img_np.shape[:2]
        brightness = float(np.mean(img_np)) / 255.0
        jg = round(2.0 + (1.0 - brightness) * 10.0, 2)
        conf = "high" if jg < 5 else "moderate" if jg < 15 else "low"
        result = {
            "agent_id": "perception",
            "message_type": "pose_estimate",
            "source": "synthetic",
            "timestamp": time.time(),
            "message_id": str(time.time_ns()),
            "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "t": [10.0, 0.0, 0.0],
            "quaternion": [1.0, 0.0, 0.0, 0.0],
            "jensen_gain": jg,
            "confidence_level": conf,
            "confidence_label": f"{conf.upper()} CONFIDENCE",
            "sigma_R_deg": round(jg * 0.6, 2),
            "sigma_t_m": 0.5,
            "nearest_anchor_idx": 0,
            "anchor_distance_deg": round(jg * 0.4, 2),
            "is_trustworthy": jg < 15.0,
            "processing_time_ms": round((time.time() - t_start) * 1000, 1),
            "image_shape": [h, w, 3],
        }
        if STATE["redis_connected"]:
            try:
                r = redis_lib.Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
                r.publish("perception.out", json.dumps(result))
            except Exception:
                pass
        _msg_q.put({"type": "redis_message", "channel": "perception.out",
                     "data": result, "timestamp": time.time()})
        STATE["latest"]["perception"] = result
        total_ms = round((time.time() - t_start) * 1000, 1)
        return {
            "status": "processed",
            "model": "synthetic (model not loaded)",
            "jensen_gain": jg,
            "confidence": conf,
            "total_ms": total_ms,
        }


# ── Chat ───────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    text: str


@app.post("/api/chat")
async def chat(req: ChatRequest):
    text = req.text.lower().strip()
    p = STATE["latest"]["perception"]
    c = STATE["latest"]["cognition"]
    a = STATE["latest"]["action"]
    o = STATE["latest"]["consensus"]

    if any(k in text for k in ("status", "report", "state")):
        jg = p.get("jensen_gain", "N/A") if p else "N/A"
        act = o.get("final_action", "N/A") if o else "N/A"
        orch = "RUNNING" if STATE["orchestrator_running"] else "STOPPED"
        redis_s = "Connected" if STATE["redis_connected"] else "Disconnected"
        return {"response": (f"System: Orchestrator {orch}, Redis {redis_s}. "
                             f"Jensen Gain: {jg}°. Current action: {act}."),
                "route": "deterministic"}

    if any(k in text for k in ("explain", "why", "reason", "justify")):
        if o:
            return {"response": (f"Decision: {o.get('final_action','?')}. "
                                 f"Reasoning: {o.get('reasoning','None available')}."),
                    "route": "deterministic"}
        return {"response": "No decisions made yet.", "route": "deterministic"}

    if any(k in text for k in ("option", "alternative", "action", "what can")):
        if a:
            lines = [f"Primary: {a.get('primary_action','?')} "
                     f"(score={a.get('primary_score','?')}, "
                     f"collision={a.get('collision_prob','?')})"]
            for alt in a.get("alternatives", []):
                lines.append(f"  Alt: {alt.get('action','?')} "
                             f"(score={alt.get('score','?')})")
            return {"response": "\n".join(lines), "route": "deterministic"}
        return {"response": "No action data yet. Start a scenario first.",
                "route": "deterministic"}

    if "override" in text:
        return {"response": ("Use the Override panel below. Levels: "
                             "1-Acknowledge, 2-Modify, 3-Replace, 4-Reject (full manual)."),
                "route": "deterministic"}

    if any(k in text for k in ("perception", "pose", "jensen")):
        if p:
            return {"response": (f"Perception: JG={p.get('jensen_gain','?')}° "
                                 f"conf={p.get('confidence_level','?')} "
                                 f"trustworthy={'Yes' if p.get('is_trustworthy') else 'No'} "
                                 f"position={p.get('t','?')}"),
                    "route": "deterministic"}
        return {"response": "No perception data yet.", "route": "deterministic"}

    if any(k in text for k in ("cognition", "anomaly", "hdc", "situation")):
        if c:
            return {"response": (f"Cognition: anomaly={c.get('anomaly_detected','?')} "
                                 f"type={c.get('anomaly_type','?')} "
                                 f"novelty={c.get('novelty_score','?')} "
                                 f"recommendation={c.get('recommended_action','?')}"),
                    "route": "deterministic"}
        return {"response": "No cognition data yet.", "route": "deterministic"}

    if any(k in text for k in ("ps", "problem", "statement", "background", "about")):
        return {"response": "The problem statement is: Synchronous Multi-modal Belief Integration with Orbital Self-Interpretability for Spacecraft. In deep-space proximity operations and habitat management, AI systems perform perceiving the environment (pose estimation) and responding to anomalies (autonomous control) — but often operate as disconnected black boxes. This system bridges 'what the AI sees' and 'why the AI acts' to prevent fatal delays during communication blackouts.", "route": "knowledge_base"}

    if any(k in text for k in ("space", "orbit", "iss", "deep", "mars", "moon")):
        return {"response": "Operating in deep space presents unique challenges: zero gravity, extreme thermal shifts, and high radiation. Proximity operations require millimeter-precision pose estimation. Our orbital self-interpretability framework is designed for exactly these harsh, high-stakes environments where communication latency with Earth (up to 20+ minutes for Mars) necessitates autonomous, explainable AI.", "route": "knowledge_base"}

    if any(k in text for k in ("armstrong", "protocol", "override")):
        return {"response": "The Armstrong Protocol defines 4 human override levels: 1) Acknowledge, 2) Modify Constraints, 3) Replace Action, 4) Full Manual Override. It acts as the ultimate safety net.", "route": "knowledge_base"}

    if any(k in text for k in ("jensen", "gain", "hdc", "hyperdimensional")):
        return {"response": "Jensen Gain (JG) is an uncertainty metric derived from Jensen-Shannon divergence in our Perception Agent. Hyperdimensional Cognition (HDC) uses robust vector-symbolic architectures to detect novel anomalies in spacecraft telemetry.", "route": "knowledge_base"}

    return {"response": ("Commands: 'status report', 'explain', "
                         "'what are my options', 'perception', "
                         "'cognition', 'override', 'problem statement', 'space'"),
            "route": "help"}


# ── WebSocket ──────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        await ws.send_json({
            "type": "initial_state",
            "status": {
                "redis_connected": STATE["redis_connected"],
                "orchestrator_running": STATE["orchestrator_running"],
                "scenario_running": STATE["scenario_running"],
                "model_loaded": _MODEL_LOADED,
                "modules": STATE["modules"],
            },
            "latest": _safe_latest(),
            "event_log": STATE["event_log"][-50:],
        })
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(ws)


def _safe_latest() -> dict:
    """Return STATE['latest'] with numpy arrays converted."""
    import copy
    out = {}
    for k, v in STATE["latest"].items():
        if v is None:
            out[k] = None
        else:
            out[k] = json.loads(json.dumps(v, default=str))
    return out


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))

    print("=" * 55)
    print("  SPACECRAFT AUTONOMY -- Web Dashboard")
    print("=" * 55)
    print("  Modules available:")
    for mod, ok in STATE["modules"].items():
        tag = "[OK]" if ok else "[--]"
        print(f"    {mod:15s} {tag}")
    print(f"  Model: {'LOADED (' + _MODEL_INFO.get('backbone','') + ')' if _MODEL_LOADED else 'NOT LOADED'}")
    print(f"\n  Starting server at http://localhost:{port}")
    print("=" * 55)

    uvicorn.run(app, host="0.0.0.0", port=port)
