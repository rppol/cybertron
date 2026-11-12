import logging

import numpy as np
import pytest
import torch
from tiny import build

from cybertron.config import Settings
from cybertron.engine import compile, convert, properties
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
