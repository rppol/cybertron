import logging
import time

from prometheus_client import Counter, Histogram

PATHS = {"/v1/classify", "/v1/embed", "/health", "/health/deep", "/metrics"}

log = logging.getLogger(__name__)


class Timing:
    def __init__(self, app, registry):
        self.app = app
        self.requests = Counter("cybertron_requests", "HTTP requests", ["path", "code"], registry=registry)
        self.seconds = Histogram("cybertron_request_seconds", "HTTP request latency", ["path"], registry=registry)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        start, code = time.perf_counter(), 500

        async def send_status(message):
            nonlocal code
            if message["type"] == "http.response.start":
                code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_status)
        finally:
            seconds = time.perf_counter() - start
            path = scope["path"] if scope["path"] in PATHS else "other"
            self.requests.labels(path, str(code)).inc()
            self.seconds.labels(path).observe(seconds)
            log.debug("%s %s %d %.1f ms", scope["method"], path, code, 1000 * seconds)
