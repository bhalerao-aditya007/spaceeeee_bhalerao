import fakeredis

_SERVER = fakeredis.FakeServer()

def Redis(*args, **kwargs):
    kwargs.pop("host", None)
    kwargs.pop("port", None)
    kwargs.pop("db", None)
    return fakeredis.FakeStrictRedis(server=_SERVER, decode_responses=kwargs.get("decode_responses", False))
