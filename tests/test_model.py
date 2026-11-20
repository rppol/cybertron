import json

import pytest
import torch
from safetensors.torch import save_file
from tiny import LABELS, build
from transformers import GPT2Config

from cybertron.core.config import Settings
from cybertron.services.model import load

TEXTS = ["hello world", "the card number is 1234 5678", "a"]


def run(loaded, texts=TEXTS):
    enc = loaded.tok(texts, padding=True, return_tensors="pt")
    with torch.no_grad():
        return enc, loaded.net(*[enc[n] for n in loaded.net.inputs])


@pytest.mark.parametrize("fmt", ["safetensors", "pt"])
@pytest.mark.parametrize("arch", ["bert", "distilbert", "roberta", "minilm"])
def test_native_head(tmp_path, arch, fmt):
    ref = build(tmp_path, arch, "native", fmt)
    loaded = load(Settings(model=tmp_path))
    assert loaded.labels == LABELS and not loaded.multi
    enc, (logits, mean, first) = run(loaded)
    with torch.no_grad():
        want = ref(**{n: enc[n] for n in loaded.net.inputs}).logits
    torch.testing.assert_close(logits, want, rtol=0, atol=1e-6)
    assert mean.shape == first.shape == (3, ref.config.hidden_size)


@pytest.mark.parametrize("arch", ["bert", "distilbert", "roberta"])
def test_inputs_follow_model(tmp_path, arch):
    build(tmp_path, arch)
    want = ["input_ids", "attention_mask"] + (["token_type_ids"] if arch == "bert" else [])
    assert load(Settings(model=tmp_path)).net.inputs == want


def test_headless(tmp_path):
    ref = build(tmp_path, "bert", "none")
    loaded = load(Settings(model=tmp_path))
    assert loaded.labels == []
    enc, (mean, first) = run(loaded)
    with torch.no_grad():
        hidden = ref(**enc).last_hidden_state
    torch.testing.assert_close(first, hidden[:, 0], rtol=0, atol=1e-6)
    mask = enc["attention_mask"].unsqueeze(-1).float()
    torch.testing.assert_close(mean, (hidden * mask).sum(1) / mask.sum(1), rtol=0, atol=1e-6)


@pytest.mark.parametrize("fmt", ["safetensors", "pt"])
@pytest.mark.parametrize("acts", [("relu",), ("gelu", "tanh")])
def test_custom_head(tmp_path, fmt, acts):
    ref = build(tmp_path, "distilbert", "custom", fmt, activations=acts)
    loaded = load(Settings(model=tmp_path))
    assert loaded.labels == LABELS
    enc, (logits, _, _) = run(loaded)
    with torch.no_grad():
        want = ref(**{n: enc[n] for n in loaded.net.inputs})
    torch.testing.assert_close(logits, want, rtol=0, atol=1e-6)


def test_unknown_layout(tmp_path):
    ref = build(tmp_path, "bert", "custom")
    state = {k.replace("classifier.", "head."): v for k, v in ref.state_dict().items()}
    save_file(state, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="unrecognised head layout"):
        load(Settings(model=tmp_path))


def test_head_without_weights(tmp_path):
    build(tmp_path, "bert", "none")
    config = json.loads((tmp_path / "config.json").read_text())
    config["architectures"] = ["BertForSequenceClassification"]
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="missing weights"):
        load(Settings(model=tmp_path))


def test_vocab_mismatch(tmp_path):
    build(tmp_path, "bert", vocab_size=20)
    with pytest.raises(ValueError, match="tokenizer"):
        load(Settings(model=tmp_path))


@pytest.mark.parametrize("overrides", [{"id2label": {0: "score"}, "label2id": {"score": 0}}, {"problem_type": "regression"}])
def test_regression_rejected(tmp_path, overrides):
    build(tmp_path, "bert", **overrides)
    with pytest.raises(ValueError, match="regression"):
        load(Settings(model=tmp_path))


def test_decoder_rejected(tmp_path):
    GPT2Config().save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="encoder"):
        load(Settings(model=tmp_path))


def test_labels_override(tmp_path):
    build(tmp_path, "bert")
    assert load(Settings(model=tmp_path, labels=["x", "y", "z"])).labels == ["x", "y", "z"]
    with pytest.raises(ValueError, match="labels"):
        load(Settings(model=tmp_path, labels=["x", "y"]))


def test_multi_label(tmp_path):
    build(tmp_path, "bert", problem_type="multi_label_classification")
    assert load(Settings(model=tmp_path)).multi


def test_no_pickle(tmp_path):
    build(tmp_path, "bert", fmt="pt")
    torch.save({"x": object()}, tmp_path / "model.pt")
    with pytest.raises(Exception, match="(?i)weights_only|unsupported"):
        load(Settings(model=tmp_path))


@pytest.mark.parametrize(
    ("arch", "overrides", "tok_max", "want"),
    [
        ("bert", {}, 512, 128),
        ("bert", {}, int(1e30), 128),
        ("bert", {"max_position_embeddings": 40}, 512, 40),
        ("roberta", {"max_position_embeddings": 34}, 512, 32),
        ("bert", {}, 24, 24),
    ],
)
def test_max_len(tmp_path, arch, overrides, tok_max, want):
    build(tmp_path, arch, model_max_length=tok_max, **overrides)
    assert load(Settings(model=tmp_path)).max_len == want
