import inspect
from pathlib import Path
from typing import NamedTuple

import torch
from safetensors.torch import load_file
from torch import nn
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from transformers.models.auto.modeling_auto import MODEL_FOR_MASKED_LM_MAPPING_NAMES

ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "silu": nn.SiLU}


class Loaded(NamedTuple):
    net: "Net"
    tok: object
    labels: list[str]
    multi: bool
    max_len: int


class Net(nn.Module):
    """Encoder plus optional head. Returns (logits, mean, first) with a head, else (mean, first)."""

    def __init__(self, encoder, head, inputs):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.inputs = inputs

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        extra = {} if token_type_ids is None else {"token_type_ids": token_type_ids}
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, **extra)
        hidden = out.hidden_states[-1]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = ((hidden * mask).sum(1) / mask.sum(1).clamp(min=1), hidden[:, 0])
        logits = self.head(hidden[:, 0]) if self.head is not None else getattr(out, "logits", None)
        return pooled if logits is None else (logits, *pooled)


def load(cfg) -> Loaded:
    path = Path(cfg.model)
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    if config.is_encoder_decoder or config.is_decoder or config.model_type not in MODEL_FOR_MASKED_LM_MAPPING_NAMES:
        raise ValueError(f"{config.model_type} is not an encoder model")
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if len(tok) > config.vocab_size:
        raise ValueError(f"tokenizer has {len(tok)} tokens but the embedding table has {config.vocab_size}")

    state = _weights(path)
    if any(k.startswith("backbone.") for k in state):
        encoder, head = _hf(AutoModel, config, _strip(state, "backbone."), strict=True), _custom(config, state)
        width = head[-1].out_features
    elif any(a.endswith("ForSequenceClassification") for a in config.architectures or []):
        encoder, head = _hf(AutoModelForSequenceClassification, config, state, strict=True), None
        width = config.num_labels
    else:
        encoder, head = _hf(AutoModel, config, state, strict=False), None
        width = 0

    labels = []
    if width:
        if width == 1 or config.problem_type == "regression":
            raise ValueError("regression heads are not supported")
        labels = cfg.labels or [config.id2label[i] for i in range(width)]
        if len(labels) != width:
            raise ValueError(f"head has {width} outputs but there are {len(labels)} labels")

    inputs = ["input_ids", "attention_mask"]
    if "token_type_ids" in tok.model_input_names and "token_type_ids" in inspect.signature(encoder.forward).parameters:
        inputs.append("token_type_ids")
    net = Net(encoder, head, inputs).eval()

    offset = getattr(encoder.base_model.embeddings, "padding_idx", -1) + 1
    max_len = min(cfg.max_len, tok.model_max_length, config.max_position_embeddings - offset)
    return Loaded(net, tok, labels, config.problem_type == "multi_label_classification", max_len)


def _weights(path):
    for f in [path / "model.safetensors", path / "pytorch_model.bin", *sorted(path.glob("*.pt"))]:
        if f.is_file():
            return load_file(f) if f.suffix == ".safetensors" else torch.load(f, map_location="cpu", weights_only=True)
    raise ValueError(f"no model.safetensors, pytorch_model.bin or *.pt in {path}")


def _hf(cls, config, state, strict):
    model, info = cls.from_pretrained(None, config=config, state_dict=state, output_loading_info=True)
    if info["missing_keys"] or info["mismatched_keys"]:
        raise ValueError(f"missing weights: {info['missing_keys'] or info['mismatched_keys']}")
    if strict and info["unexpected_keys"]:
        raise ValueError(f"unrecognised head layout: {info['unexpected_keys']}")
    return model


def _custom(config, state):
    """Rebuild `classifier.N` from key indices and shapes; activations come from config.engine_head."""
    head = _strip(state, "classifier.")
    other = [k for k in state if not k.startswith(("backbone.", "classifier."))]
    if other or not head:
        raise ValueError(f"unrecognised head layout: {other[:5] or 'no classifier.* weights'}")
    try:
        idx = sorted({int(k.split(".")[0]) for k in head})
    except ValueError:
        raise ValueError(f"unrecognised head layout: {sorted(head)}") from None
    acts = getattr(config, "engine_head", {}).get("activations", ["relu"] * (len(idx) - 1))
    if len(acts) != len(idx) - 1 or set(acts) - ACTIVATIONS.keys() or any(b == a + 1 for a, b in zip(idx, idx[1:])):
        raise ValueError(f"unrecognised head layout: linears at {idx}, activations {acts}")

    layers = [nn.Identity() for _ in range(idx[-1] + 1)]
    for n, i in enumerate(idx):
        out, inp = head[f"{i}.weight"].shape
        layers[i] = nn.Linear(inp, out, bias=f"{i}.bias" in head)
        if n < len(acts):
            layers[i + 1] = ACTIVATIONS[acts[n]]()
    seq = nn.Sequential(*layers)
    seq.load_state_dict(head)
    return seq


def _strip(state, prefix):
    return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
