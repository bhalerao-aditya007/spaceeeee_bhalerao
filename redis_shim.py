import fakeredis

_SERVER = fakeredis.FakeServer()  # shared in-process fake redis instance


class Redis:
    """
    Drop-in shim for redis.Redis when no real Redis server exists
    (e.g. Render free tier). All Redis(...) / Redis.from_url(...) calls
    across the whole app share the same in-memory FakeServer, so
    pub/sub still works between threads in this one process.
    """

    def __new__(cls, *args, **kwargs):
        return cls._make(**kwargs)

    @classmethod
    def from_url(cls, url=None, *args, **kwargs):
        return cls._make(**kwargs)

    @staticmethod
    def _make(**kwargs):
        return fakeredis.FakeStrictRedis(
            server=_SERVER,
            decode_responses=kwargs.get("decode_responses", False),
        )
