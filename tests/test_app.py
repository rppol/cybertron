import gc
import queue
import threading
from types import SimpleNamespace

import anyio
import pytest
from fastapi.testclient import TestClient
from tiny import build

from cybertron.app import admit, create_app
from cybertron.config import Settings


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    path = tmp_path_factory.mktemp("bert")
    build(path, "bert")
    return path


@pytest.fixture
def anyio_backend():
    return "asyncio"


def client(model, **kw):
    return TestClient(create_app(Settings(model=model, slots=2, gc_interval=0, **kw)), raise_server_exceptions=False)


def drain(app):
    pool = app.state.engine.pool
    return [pool.get_nowait() for _ in range(pool.qsize())]


def test_routes(model):
    with client(model) as c:
        assert c.get("/health").json()["info"]["tasks"] == ["classify", "embed"]
        assert c.get("/health/deep").json()["seconds"] > 0
        r = c.post("/v1/classify", json={"texts": ["hello", "bad"], "top_k": 2}).json()["results"]
        assert len(r) == 2 and len(r[0]) == 2 and {"label", "score"} <= r[0][0].keys()
        e = c.post("/v1/embed", json={"texts": ["hello"], "pooling": "cls", "normalize": False}).json()
        assert len(e["embeddings"][0]) == 32
        assert c.post("/v1/embed", json={"texts": ["a"], "pooling": "max"}).status_code == 422
        metrics = c.get("/metrics").text
        for name in ["cybertron_requests_total", "cybertron_request_seconds", "cybertron_texts_total", "cybertron_free_slots"]:
            assert name in metrics
        assert 'path="/v1/classify"' in metrics


def test_busy_returns_429_and_deep_health_answers(model):
    with client(model) as c:
        reqs = drain(c.app)
        r = c.post("/v1/classify", json={"texts": ["a"]}, headers={"X-Health-Probe": "1"})
        assert r.status_code == 429 and r.headers["Retry-After"] == "1"
        assert c.get("/health/deep").status_code == 200
        for req in reqs:
            c.app.state.engine.pool.put(req)
        assert c.post("/v1/classify", json={"texts": ["a"]}).status_code == 200


def test_overlapping_probe_gets_429(model):
    with client(model) as c:
        with c.app.state.engine._probe_lock:
            assert c.get("/health/deep").status_code == 429


def test_hook_error_returns_slot(model, monkeypatch):
    with client(model) as c:
        engine = c.app.state.engine

        def boom(texts, meta):
            raise RuntimeError("hook failed")

        monkeypatch.setattr(engine, "_pre", boom)
        assert c.post("/v1/classify", json={"texts": ["a"]}).status_code == 500
        assert engine.pool.qsize() == 2


def one_slot():
    fake = SimpleNamespace(pool=queue.Queue())
    fake.pool.put("req")
    return fake


@pytest.mark.anyio
async def test_cancel_during_inference_returns_slot():
    fake, started, release = one_slot(), threading.Event(), threading.Event()

    def work(req):
        started.set()
        release.wait()

    async with anyio.create_task_group() as tg:
        tg.start_soon(admit, fake, work)
        await anyio.to_thread.run_sync(started.wait)
        tg.cancel_scope.cancel()
        assert fake.pool.qsize() == 0
        release.set()
    assert fake.pool.qsize() == 1


@pytest.mark.anyio
async def test_cancel_before_thread_starts_returns_slot():
    fake, ran = one_slot(), []
    limiter = anyio.to_thread.current_default_thread_limiter()
    total, limiter.total_tokens = limiter.total_tokens, 1
    try:
        await limiter.acquire()
        async with anyio.create_task_group() as tg:
            tg.start_soon(admit, fake, ran.append)
            await anyio.wait_all_tasks_blocked()
            assert fake.pool.qsize() == 0
            tg.cancel_scope.cancel()
        limiter.release()
    finally:
        limiter.total_tokens = total
    assert fake.pool.qsize() == 1 and not ran


def test_api_key(model):
    with client(model, api_key="s3cret") as c:
        assert c.post("/v1/classify", json={"texts": ["a"]}).status_code == 401
        assert c.post("/v1/classify", json={"texts": ["a"]}, headers={"X-API-Key": "nope"}).status_code == 401
        assert c.post("/v1/classify", json={"texts": ["a"] * 999}).status_code == 401
        assert c.post("/v1/embed", json={"texts": ["a"]}, headers={"X-API-Key": "s3cret"}).status_code == 200
        for path in ["/health", "/health/deep", "/metrics"]:
            assert c.get(path).status_code == 200


def test_rate_limit_ignores_forwarded_for(model):
    with client(model, rate_limit=2) as c:
        codes = [
            c.post("/v1/classify", json={"texts": ["a"]}, headers={"X-Forwarded-For": f"10.0.0.{i}"}).status_code
            for i in range(4)
        ]
        assert codes == [200, 200, 429, 429]
        assert c.get("/health").status_code == 200


def test_limits(model):
    with client(model, max_texts=2, max_chars=5) as c:
        assert c.post("/v1/classify", json={"texts": ["a"] * 3}).status_code == 413
        assert c.post("/v1/embed", json={"texts": ["abcdef"]}).status_code == 413
        assert c.post("/v1/classify", json={"texts": ["a"], "meta": [{}, {}]}).status_code == 422
        assert c.post("/v1/classify", json={"texts": ["a"], "top_k": 0}).status_code == 422
        assert c.post("/v1/classify", json={"texts": []}).json() == {"results": []}


def test_headless_has_no_classify(tmp_path):
    build(tmp_path, "bert", "none")
    with client(tmp_path) as c:
        assert c.post("/v1/classify", json={"texts": ["a"]}).status_code == 404
        assert c.post("/v1/embed", json={"texts": ["a"]}).status_code == 200


def test_gc_frozen_then_released(model):
    with client(model, gc_thresholds=(1000, 5, 5)):
        assert gc.get_freeze_count() > 0 and gc.get_threshold() == (1000, 5, 5)
    assert gc.get_freeze_count() == 0


def test_two_lifespans(model):
    for _ in range(2):
        with TestClient(create_app(Settings(model=model, slots=1, gc_interval=0.01))) as c:
            assert c.post("/v1/classify", json={"texts": ["a"]}).status_code == 200
            engine = c.app.state.engine
        assert engine.compiled is None and engine.pool.qsize() == 0
