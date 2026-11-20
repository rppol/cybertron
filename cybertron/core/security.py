import secrets
import time

from fastapi import HTTPException, Request


def guard(request: Request):
    cfg, rate = request.app.state.cfg, request.app.state.rate
    if cfg.api_key:
        given = request.headers.get("x-api-key", "").encode()
        if not secrets.compare_digest(given, cfg.api_key.get_secret_value().encode()):
            raise HTTPException(401, "missing or wrong X-API-Key")
    if cfg.rate_limit and not rate.allow(request.client.host if request.client else ""):
        raise HTTPException(429, "rate limit reached", headers={"Retry-After": str(rate.retry_after())})


class Window:
    """Fixed-window request count per client host; the table clears each window."""

    def __init__(self, cfg):
        self.limit, self.length = cfg.rate_limit, cfg.rate_window
        self.start, self.hits = time.monotonic(), {}

    def allow(self, host):
        now = time.monotonic()
        if now - self.start >= self.length:
            self.start, self.hits = now, {}
        self.hits[host] = self.hits.get(host, 0) + 1
        return self.hits[host] <= self.limit

    def retry_after(self):
        return max(1, round(self.start + self.length - time.monotonic()))
