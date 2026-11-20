from pathlib import Path

import pytest
from pydantic import ValidationError

from cybertron.config import Settings


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("CYBERTRON_MODEL", "/m")
    monkeypatch.setenv("CYBERTRON_PRECISION", "INT8")
    monkeypatch.setenv("CYBERTRON_BUCKETS", "[8, 24]")
    monkeypatch.setenv("CYBERTRON_OV_CONFIG", '{"EXECUTION_MODE_HINT": "ACCURACY"}')
    monkeypatch.setenv("CYBERTRON_API_KEY", "s3cret")
    cfg = Settings()
    assert cfg.precision == "INT8"
    assert cfg.buckets == [8, 24, 128]
    assert cfg.ov_config == {"EXECUTION_MODE_HINT": "ACCURACY"}
    assert cfg.api_key.get_secret_value() == "s3cret"
    assert "s3cret" not in repr(cfg)


def test_defaults():
    cfg = Settings(model="/m")
    assert (cfg.precision, cfg.hint, cfg.slots, cfg.batch_size, cfg.max_len) == ("FP32", "THROUGHPUT", 0, 32, 128)
    assert cfg.buckets == [16, 32, 64, 128]
    assert cfg.api_key is None


@pytest.mark.parametrize(
    "bad",
    [
        {"precision": "INT4"},
        {"hint": "FAST"},
        {"batch_size": 0},
        {"buckets": [0, 16]},
        {"max_len": 0},
        {"slots": -1},
        {"rate_window": 0},
        {"gc_thresholds": [1, 2]},
    ],
)
def test_invalid(bad):
    with pytest.raises(ValidationError):
        Settings(model="/m", **bad)


def test_model_required(monkeypatch):
    monkeypatch.delenv("CYBERTRON_MODEL", raising=False)
    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.parametrize("key", ["PERFORMANCE_HINT", "NUM_STREAMS", "INFERENCE_NUM_THREADS", "CACHE_DIR"])
def test_ov_config_rejects_owned_keys(key):
    with pytest.raises(ValidationError, match=key):
        Settings(model="/m", ov_config={key: "1"})


@pytest.mark.parametrize(
    ("max_len", "buckets", "want"),
    [(128, [16, 32], [16, 32, 128]), (64, [16, 128, 32, 32], [16, 32, 64]), (16, [16], [16])],
)
def test_top_bucket_is_max_len(max_len, buckets, want):
    assert Settings(model="/m", max_len=max_len, buckets=buckets).buckets == want


def test_every_knob_documented():
    doc = (Path(__file__).parent.parent / "docs" / "tuning.md").read_text()
    assert [f for f in Settings.model_fields if f"CYBERTRON_{f.upper()}" not in doc] == []
