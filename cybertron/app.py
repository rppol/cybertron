"""HTTP service. Run with `uvicorn cybertron.app:create_app --factory` and one worker."""

import asyncio
import ctypes
import gc
import queue
import secrets
import sys
import time
from contextlib import asynccontextmanager, suppress
from typing import Literal

import anyio.to_thread
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from .config import Settings
from .engine import Engine

PATHS = {"/v1/classify", "/v1/embed", "/health", "/health/deep", "/metrics"}


class ClassifyIn(BaseModel):
    texts: list[str]
    top_k: int = Field(1, ge=1)
    meta: list[dict] | None = None


class EmbedIn(BaseModel):
    texts: list[str]
    pooling: Literal["mean", "cls"] = "mean"
    normalize: bool = True
    meta: list[dict] | None = None


def create_app(cfg: Settings | None = None) -> FastAPI:
    cfg = cfg or Settings()
    registry = CollectorRegistry()
    texts = Counter("cybertron_texts", "Texts accepted for inference", ["path"], registry=registry)
    free = Gauge("cybertron_free_slots", "Idle InferRequests in the pool", registry=registry)

    @asynccontextmanager
    async def lifespan(app):
        engine = app.state.engine = Engine(cfg)
        free.set_function(engine.pool.qsize)
        anyio.to_thread.current_default_thread_limiter().total_tokens = engine.info["slots"] + 8
        gc.collect()
        gc.freeze()
        old = gc.get_threshold()
        gc.set_threshold(*cfg.gc_thresholds)
        task = asyncio.create_task(_collect(cfg.gc_interval)) if cfg.gc_interval else None
        try:
            yield
        finally:
            if task:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            engine.close()
            gc.unfreeze()
            gc.set_threshold(*old)

    app = FastAPI(title="cybertron", lifespan=lifespan)
    app.add_middleware(Timing, registry=registry)

    rate = _Window(cfg)

    def guard(request: Request):
        if cfg.api_key:
            given = request.headers.get("x-api-key", "").encode()
            if not secrets.compare_digest(given, cfg.api_key.get_secret_value().encode()):
                raise HTTPException(401, "missing or wrong X-API-Key")
        if cfg.rate_limit and not rate.allow(request.client.host if request.client else ""):
            raise HTTPException(429, "rate limit reached", headers={"Retry-After": str(rate.retry_after())})

    def check(request, body):
        if len(body.texts) > cfg.max_texts or any(len(t) > cfg.max_chars for t in body.texts):
            raise HTTPException(413, f"at most {cfg.max_texts} texts of {cfg.max_chars} characters")
        if body.meta is not None and len(body.meta) != len(body.texts):
            raise HTTPException(422, "meta needs one entry per text")
        texts.labels(request.url.path).inc(len(body.texts))

    @app.post("/v1/classify", dependencies=[Depends(guard)])
    async def classify(body: ClassifyIn, request: Request):
        engine = request.app.state.engine
        if not engine.labels:
            raise HTTPException(404, "model has no classification head")
        check(request, body)
        return {"results": await admit(engine, engine.classify, body.texts, body.top_k, body.meta)}

    @app.post("/v1/embed", dependencies=[Depends(guard)])
    async def embed(body: EmbedIn, request: Request):
        engine = request.app.state.engine
        check(request, body)
        vecs = await admit(engine, engine.embed, body.texts, body.pooling, body.normalize, body.meta)
        return {"embeddings": vecs.tolist()}

    @app.get("/health")
    async def health(request: Request):
        return {"status": "ok", "info": request.app.state.engine.info}

    @app.get("/health/deep")
    async def deep(request: Request):
        seconds = await run_in_threadpool(request.app.state.engine.probe)
        if seconds is None:
            raise HTTPException(429, "a probe is already running", headers={"Retry-After": "1"})
        return {"status": "ok", "seconds": seconds}

    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return app


async def admit(engine, fn, *args):
    """Run fn(req, *args) in a worker thread on a free InferRequest, or answer 429 at once."""
    try:
        req = engine.pool.get_nowait()
    except queue.Empty:
        raise HTTPException(429, "all inference slots are busy", headers={"Retry-After": "1"}) from None
    try:
        return await run_in_threadpool(fn, req, *args)
    finally:
        engine.pool.put_nowait(req)


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
            path = scope["path"] if scope["path"] in PATHS else "other"
            self.requests.labels(path, str(code)).inc()
            self.seconds.labels(path).observe(time.perf_counter() - start)


class _Window:
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


async def _collect(interval):
    while True:
        await asyncio.sleep(interval)
        gc.collect()
        if sys.platform == "linux":
            with suppress(OSError, AttributeError):
                ctypes.CDLL("libc.so.6").malloc_trim(0)
