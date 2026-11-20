"""HTTP service. Run with `uvicorn cybertron.main:create_app --factory` and one worker."""

from contextlib import asynccontextmanager

import anyio.to_thread
from fastapi import FastAPI
from prometheus_client import CollectorRegistry, Counter, Gauge

from .core.config import Settings
from .core.logging_setup import setup
from .core.memory_utils import tuned_gc
from .core.middlewares import Timing
from .core.security import Window
from .routes import health, inference, metrics
from .services.engine import Engine


def create_app(cfg: Settings | None = None) -> FastAPI:
    cfg = cfg or Settings()
    setup(cfg.log_level)
    registry = CollectorRegistry()
    texts = Counter("cybertron_texts", "Texts accepted for inference", ["path"], registry=registry)
    free = Gauge("cybertron_free_slots", "Idle InferRequests in the pool", registry=registry)

    @asynccontextmanager
    async def lifespan(app):
        engine = app.state.engine = Engine(cfg)
        free.set_function(engine.pool.qsize)
        anyio.to_thread.current_default_thread_limiter().total_tokens = engine.info["slots"] + 8
        async with tuned_gc(cfg.gc_interval, cfg.gc_thresholds):
            try:
                yield
            finally:
                engine.close()

    app = FastAPI(title="cybertron", lifespan=lifespan)
    app.state.cfg, app.state.registry, app.state.texts, app.state.rate = cfg, registry, texts, Window(cfg)
    app.add_middleware(Timing, registry=registry)
    for module in (inference, health, metrics):
        app.include_router(module.router)
    return app
