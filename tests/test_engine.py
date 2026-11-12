import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch
from tiny import build

from cybertron.config import Settings
from cybertron.engine import Engine, compile, convert, properties
from cybertron.model import load

TEXTS = ["hello world", "the card number is 1234 5678 9012", "a"]


def compiled_for(path, **kw):
    cfg = Settings(model=path, **kw)
    loaded = load(cfg)
    return loaded, *compile(convert(loaded, cfg), cfg)


def feed(loaded, texts):
    enc = loaded.tok(texts, padding=True, truncation=True, max_length=loaded.max_len, return_tensors="np")
    return {n: enc[n] for n in loaded.net.inputs}


@pytest.mark.parametrize(("precision", "tol"), [("FP32", 1e-4), ("FP16", 1e-2), ("INT8", 1e-1)])
def test_precision_matches_torch(tmp_path, precision, tol):
    build(tmp_path, "bert")
    loaded, compiled, info = compiled_for(tmp_path, precision=precision)
    inputs = feed(loaded, TEXTS)
    with torch.no_grad():
        want = loaded.net(*[torch.from_numpy(inputs[n]) for n in loaded.net.inputs])
    got = compiled(inputs)
    for i, w in enumerate(want):
        np.testing.assert_allclose(got[i], w.numpy(), rtol=tol, atol=tol)
    assert info["precision"] == precision
    if precision == "FP32":
        assert info["inference_precision"] == "f32"


def test_int8_has_int_weights(tmp_path):
    build(tmp_path, "bert")
    cfg = Settings(model=tmp_path, precision="INT8")
    model = convert(load(cfg), cfg)
    types = {op.get_output_element_type(0).to_string() for op in model.get_ops() if op.get_type_name() == "Constant"}
    assert types & {"i8", "u8"}


def test_two_shapes_and_bounds(tmp_path):
    build(tmp_path, "bert")
    loaded, compiled, info = compiled_for(tmp_path, batch_size=4, max_len=16)
    assert info["inputs"] == ["input_ids", "attention_mask", "token_type_ids"]
    assert str(compiled.inputs[0].get_partial_shape()) == "[1..4,1..16]"
    assert compiled(feed(loaded, TEXTS))[0].shape == (3, 3)
    assert compiled(feed(loaded, ["a"]))[0].shape == (1, 3)


def test_no_token_type_ids(tmp_path):
    build(tmp_path, "distilbert")
    _, compiled, info = compiled_for(tmp_path)
    assert info["inputs"] == ["input_ids", "attention_mask"]


def test_cache_blob_written(tmp_path):
    build(tmp_path / "m", "bert")
    compiled_for(tmp_path / "m", cache_dir=tmp_path / "cache")
    assert list((tmp_path / "cache").glob("*.blob"))


def test_typed_properties(tmp_path):
    build(tmp_path, "bert")
    _, _, info = compiled_for(tmp_path, hint="LATENCY", streams=2, threads=2, ov_config={"EXECUTION_MODE_HINT": "ACCURACY"})
    assert (info["hint"], info["streams"], info["threads"]) == ("LATENCY", 2, 2)


def test_unsupported_default_warns(caplog):
    with caplog.at_level(logging.WARNING):
        props = properties(Settings(model="/m", threads=3), supported=set())
    assert "INFERENCE_PRECISION_HINT" in caplog.text
    assert "INFERENCE_PRECISION_HINT" not in props and props["INFERENCE_NUM_THREADS"] == 3


def test_unsupported_user_property_raises(tmp_path):
    build(tmp_path, "bert")
    with pytest.raises(RuntimeError, match="BOGUS"):
        compiled_for(tmp_path, ov_config={"BOGUS": "1"})


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    path = tmp_path_factory.mktemp("bert")
    build(path, "bert")
    e = Engine(Settings(model=path, batch_size=4, max_len=32, buckets=[8, 16], slots=8))
    yield e
    e.close()


def take(engine):
    return engine.pool.get_nowait()


class Spy:
    def __init__(self, req):
        self.req, self.shapes = req, []

    def infer(self, feed):
        self.shapes += [v.shape for v in feed.values()]
        return self.req.infer(feed)

    def get_output_tensor(self, i):
        return self.req.get_output_tensor(i)


LONG = " ".join(["hello"] * 40)
MIXED = ["a", "hello world", "the card is good", LONG, "bad", "the the the the the the the the", "x"]


def test_only_buckets_fed(engine):
    spy = Spy(take(engine))
    engine.classify(spy, MIXED)
    engine.pool.put(spy.req)
    assert {s[1] for s in spy.shapes} <= {8, 16, 32}
    assert max(s[0] for s in spy.shapes) <= 4
    assert engine.info["buckets"] == [8, 16, 32]


def test_batched_equals_single(engine):
    req = take(engine)
    batched = engine.classify(req, MIXED, top_k=3)
    single = [engine.classify(req, [t], top_k=3)[0] for t in MIXED]
    emb = engine.embed(req, MIXED)
    emb1 = np.vstack([engine.embed(req, [t]) for t in MIXED])
    engine.pool.put(req)
    for b, s in zip(batched, single):
        assert [x["label"] for x in b] == [x["label"] for x in s]
        np.testing.assert_allclose([x["score"] for x in b], [x["score"] for x in s], atol=1e-5)
    np.testing.assert_allclose(emb, emb1, atol=1e-5)


def test_top_k_sorted_and_capped(engine):
    req = take(engine)
    out = engine.classify(req, ["hello", "bad"], top_k=10)
    engine.pool.put(req)
    for row in out:
        scores = [x["score"] for x in row]
        assert len(row) == 3 and scores == sorted(scores, reverse=True)
        assert sum(scores) == pytest.approx(1, abs=1e-5)


def test_embed_options(engine):
    req = take(engine)
    mean = engine.embed(req, MIXED, pooling="mean", normalize=False)
    cls = engine.embed(req, MIXED, pooling="cls", normalize=False)
    unit = engine.embed(req, MIXED, pooling="mean", normalize=True)
    engine.pool.put(req)
    assert mean.shape == cls.shape == (len(MIXED), 32)
    assert not np.allclose(mean, cls)
    np.testing.assert_allclose(np.linalg.norm(unit, axis=1), 1, atol=1e-5)
    np.testing.assert_allclose(unit, mean / np.linalg.norm(mean, axis=1, keepdims=True), atol=1e-5)


def test_normalize_zero_vector(engine, monkeypatch):
    monkeypatch.setattr(engine, "_run", lambda req, texts, out: np.zeros((len(texts), 4), np.float32))
    assert not np.isnan(engine.embed(None, ["a"])).any()


def test_empty_input(engine):
    assert engine.classify(None, []) == [] and engine.embed(None, []).shape[0] == 0


def test_threads_equal_serial(engine):
    texts = [f"hello {i} " * (i % 9 + 1) for i in range(64)]
    req = take(engine)
    serial = engine.classify(req, texts, top_k=3)
    engine.pool.put(req)

    def work(chunk):
        r = engine.pool.get()
        try:
            return engine.classify(r, chunk, top_k=3)
        finally:
            engine.pool.put(r)

    with ThreadPoolExecutor(8) as ex:
        parts = list(ex.map(work, [texts[i::8] for i in range(8)]))
    for i in range(8):
        for got, want in zip(parts[i], serial[i::8]):
            assert [x["label"] for x in got] == [x["label"] for x in want]
            np.testing.assert_allclose([x["score"] for x in got], [x["score"] for x in want], atol=1e-5)
    assert engine.pool.qsize() == 8


def test_probe_ignores_pool(engine):
    reqs = [take(engine) for _ in range(8)]
    assert engine.probe() > 0
    for r in reqs:
        engine.pool.put(r)


def test_probe_busy_returns_none(engine):
    with engine._probe_lock:
        assert engine.probe() is None


def test_bad_post_hook(engine, monkeypatch):
    req = take(engine)
    try:
        monkeypatch.setattr(engine, "_post", lambda logits, meta, labels: logits * np.nan)
        with pytest.raises(ValueError, match="NaN"):
            engine.classify(req, ["a"])
        monkeypatch.setattr(engine, "_post", lambda logits, meta, labels: logits + np.inf)
        with pytest.raises(ValueError, match="inf"):
            engine.classify(req, ["a"])
        monkeypatch.setattr(engine, "_post", lambda logits, meta, labels: logits - np.inf)
        with pytest.raises(ValueError, match="finite"):
            engine.classify(req, ["a"])
        monkeypatch.setattr(engine, "_pre", lambda texts, meta: texts[:-1])
        with pytest.raises(ValueError, match="pre"):
            engine.classify(req, ["a", "b"])
    finally:
        engine.pool.put(req)


def test_example_hooks(tmp_path):
    build(tmp_path, "bert")
    e = Engine(Settings(model=tmp_path, hooks="examples.hooks", slots=1))
    req = take(e)
    out = e.classify(req, ["hello", "hello"], top_k=3, meta=[{}, {"deny": ["red", "blue"]}])
    assert out[1][0]["label"] == "green" and out[1][0]["score"] == pytest.approx(1)
    assert {x["label"]: x["score"] for x in out[1]}["red"] == 0
    np.testing.assert_allclose(e.embed(req, ["id 1234567"]), e.embed(req, ["id #"]))
    e.pool.put(req)
    e.close()
    assert e.pool.qsize() == 0


def test_multi_label_sigmoid(tmp_path):
    build(tmp_path, "bert", problem_type="multi_label_classification")
    e = Engine(Settings(model=tmp_path, slots=1))
    req = take(e)
    scores = [x["score"] for x in e.classify(req, ["hello"], top_k=3)[0]]
    e.pool.put(req)
    assert sum(scores) != pytest.approx(1) and all(0 < s < 1 for s in scores)


@pytest.mark.parametrize(
    ("arch", "kw"),
    [("bert", {"model_max_length": int(1e30)}), ("roberta", {"max_position_embeddings": 20})],
)
def test_long_text_is_cut(tmp_path, arch, kw):
    build(tmp_path, arch, **kw)
    e = Engine(Settings(model=tmp_path, slots=1))
    req = take(e)
    assert len(e.classify(req, [LONG * 5, "a"])) == 2
    e.pool.put(req)


def test_headless_has_no_classify(tmp_path):
    build(tmp_path, "bert", "none")
    e = Engine(Settings(model=tmp_path, slots=1))
    assert e.info["tasks"] == ["embed"]
    with pytest.raises(ValueError, match="head"):
        e.classify(None, ["a"])
