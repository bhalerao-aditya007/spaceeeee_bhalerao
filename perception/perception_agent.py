import numpy as np
from scipy.spatial.transform import Rotation
from datetime import datetime, timezone
from typing import Callable, Optional
from dataclasses import dataclass, asdict
import json

from perception.models.hopf_grid import HopfFibrationGrid
from perception.models.jensen_gain import JensenGainMonitor


@dataclass
class PoseEstimate:
    R: list
    t: list
    quaternion: list


@dataclass
class UncertaintyEstimate:
    jensen_gain: float
    confidence_level: str
    confidence_label: str
    sigma_R_deg: float
    sigma_t_m: float
    nearest_anchor_idx: int
    anchor_distance_deg: float


@dataclass
class PerceptionOutput:
    pose: PoseEstimate
    uncertainty: UncertaintyEstimate
    metadata: dict

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @property
    def is_trustworthy(self) -> bool:
        return self.uncertainty.confidence_level in ("high", "moderate")

    @property
    def R_numpy(self) -> np.ndarray:
        return np.array(self.pose.R)

    @property
    def t_numpy(self) -> np.ndarray:
        return np.array(self.pose.t)


class PerceptionAgent:

    VERSION = "0.1.0"

    def __init__(self,
                 model_path: Optional[str] = None,
                 pose_fn: Optional[Callable] = None,
                 n_elevation: int = 64,
                 n_inplane: int = 16,
                 n_jensen_rotations: int = 16,
                 run_jensen_gain: bool = True):

        if model_path is None and pose_fn is None:
            raise ValueError("Provide either model_path or pose_fn")

        self.run_jensen_gain = run_jensen_gain
        self._pose_fn = pose_fn
        self._model = None

        self.grid = HopfFibrationGrid(
            n_elevation=n_elevation,
            n_inplane=n_inplane
        )

        self.jg_monitor = JensenGainMonitor(n_rotations=n_jensen_rotations)

        if model_path is not None:
            self._load_model(model_path)

        print(f"PerceptionAgent v{self.VERSION} ready")
        print(f"  Grid: {self.grid.total_anchors} anchors")
        print(f"  Jensen Gain: {'enabled' if run_jensen_gain else 'disabled'}")

    def _load_model(self, model_path: str):
        import onnxruntime as ort
        self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._img_size = 224
        self._norm_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._norm_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self._pose_fn = None
        print(f"ONNX model loaded: {model_path}")
    def _pose_fn_wrapper(self, image: np.ndarray):
        if hasattr(self, "_session"):
            from PIL import Image
            if image.dtype != np.uint8:
                image_u8 = (image * 255).clip(0, 255).astype(np.uint8)
            else:
                image_u8 = image
            pil = Image.fromarray(image_u8).resize((self._img_size, self._img_size))
            arr = np.asarray(pil, dtype=np.float32) / 255.0
            arr = (arr - self._norm_mean) / self._norm_std
            arr = arr.transpose(2, 0, 1)[None, ...].astype(np.float32)  # (1,3,H,W)

            quat, trans = self._session.run(None, {"image": arr})
            q = quat[0]                       # [w,x,y,z]
            q_scipy = np.array([q[1], q[2], q[3], q[0]])
            R = Rotation.from_quat(q_scipy).as_matrix()
            return R, trans[0]

        elif self._pose_fn is not None:
            result = self._pose_fn(image)
            R, t = result if isinstance(result, tuple) else (result, np.zeros(3))
            return np.array(R), np.array(t)
        raise RuntimeError("No model loaded")
        
    def _R_to_quaternion(self, R: np.ndarray) -> list:
        rot = Rotation.from_matrix(R)
        q = rot.as_quat()
        return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]

    def _estimate_sigma_R(self, jensen_gain: float) -> float:
        return 0.6 * jensen_gain

    def _estimate_sigma_t(self, jensen_gain: float,
                          t_magnitude: float) -> float:
        return 0.05 * t_magnitude * (1 + jensen_gain / 10.0)

    def predict(self, image: np.ndarray) -> PerceptionOutput:
        t_start = datetime.now(timezone.utc)

        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        R, t = self._pose_fn_wrapper(image)

        anchor_idx, anchor_dist, R_anchor = self.grid.find_nearest_anchor(R)

        if self.run_jensen_gain:
            def _pose_only(img):
                R_pred, _ = self._pose_fn_wrapper(img)
                return R_pred

            jg_result = self.jg_monitor.compute(
                pose_fn=_pose_only,
                image=image,
                compensate_inplane=True
            )
            jensen_gain = jg_result["jensen_gain"]
            confidence_level = jg_result["confidence_level"]
            confidence_label = jg_result["confidence_label"]
        else:
            jensen_gain = 0.0
            confidence_level = "high"
            confidence_label = "HIGH CONFIDENCE (Jensen Gain skipped)"

        t_end = datetime.now(timezone.utc)
        processing_ms = (t_end - t_start).total_seconds() * 1000
        t_magnitude = float(np.linalg.norm(t))

        output = PerceptionOutput(
            pose=PoseEstimate(
                R=R.tolist(),
                t=t.tolist(),
                quaternion=self._R_to_quaternion(R)
            ),
            uncertainty=UncertaintyEstimate(
                jensen_gain=float(jensen_gain),
                confidence_level=confidence_level,
                confidence_label=confidence_label,
                sigma_R_deg=self._estimate_sigma_R(jensen_gain),
                sigma_t_m=self._estimate_sigma_t(jensen_gain, t_magnitude),
                nearest_anchor_idx=int(anchor_idx),
                anchor_distance_deg=float(np.degrees(anchor_dist))
            ),
            metadata={
                "timestamp": t_start.isoformat(),
                "model_version": self.VERSION,
                "processing_time_ms": round(processing_ms, 2),
                "image_shape": list(image.shape),
                "grid_anchors": self.grid.total_anchors,
                "jensen_gain_enabled": self.run_jensen_gain
            }
        )

        return output
