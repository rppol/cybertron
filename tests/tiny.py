"""Build tiny random encoder checkpoints on disk, with no network access."""

import json
import string
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    BertConfig,
    BertTokenizerFast,
    DistilBertConfig,
    DistilBertTokenizerFast,
    RobertaConfig,
    RobertaTokenizerFast,
)
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

LABELS = ["red", "green", "blue"]
WORDS = ["the", "is", "good", "bad", "card", "number", "hello", "world"]
ACTS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}


def build(out, arch="bert", head="native", fmt="safetensors", activations=("relu",), model_max_length=512, **overrides):
    """Write a checkpoint to `out` and return the torch module it was saved from.

    head: "native" (AutoModelForSequenceClassification), "none" (AutoModel),
    or "custom" (`backbone.*` plus `classifier.N.*`).
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    tok = _tokenizer(arch, out, model_max_length)
    tok.save_pretrained(out)
    config = _config(arch, len(tok), **overrides)

    if head == "native":
        net = AutoModelForSequenceClassification.from_config(config)
    elif head == "none":
        net = AutoModel.from_config(config)
    else:
        net = Custom(AutoModel.from_config(config), list(activations), config.num_labels)
        config.engine_head = {"activations": list(activations)}
    net.eval()
    config.architectures = [type(net).__name__] if head != "custom" else [type(net.backbone).__name__]
    config.save_pretrained(out)

    state = {k: v.contiguous() for k, v in net.state_dict().items()}
    if fmt == "pt":
        torch.save(state, out / "model.pt")
    else:
        save_file(state, out / "model.safetensors")
    return net


class Custom(nn.Module):
    def __init__(self, backbone, activations, num_labels):
        super().__init__()
        self.backbone = backbone
        width = backbone.config.hidden_size
        layers = []
        for i, act in enumerate(activations):
            nxt = 16 >> i
            layers += [nn.Linear(width, nxt), ACTS[act](), nn.Dropout(0.1)]
            width = nxt
        self.classifier = nn.Sequential(*layers, nn.Linear(width, num_labels))

    def forward(self, **inputs):
        return self.classifier(self.backbone(**inputs).last_hidden_state[:, 0])


def _config(arch, vocab, **overrides):
    labels = dict(id2label=dict(enumerate(LABELS)), label2id={n: i for i, n in enumerate(LABELS)})
    if arch == "distilbert":
        kw = dict(dim=32, n_layers=2, n_heads=2, hidden_dim=64, max_position_embeddings=128)
        return DistilBertConfig(vocab_size=vocab, **kw, **labels, **overrides)
    if arch == "roberta":
        kw = dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64)
        kw.update(max_position_embeddings=130, type_vocab_size=1, pad_token_id=1, bos_token_id=0, eos_token_id=2)
        return RobertaConfig(vocab_size=vocab, **{**kw, **labels, **overrides})
    if arch == "minilm":
        kw = dict(hidden_size=384, num_hidden_layers=6, num_attention_heads=12, intermediate_size=1536)
    else:
        kw = dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64)
    return BertConfig(vocab_size=vocab, max_position_embeddings=128, **{**kw, **labels, **overrides})


def _tokenizer(arch, out, model_max_length):
    if arch == "roberta":
        chars = list(bytes_to_unicode().values())
        vocab = ["<s>", "<pad>", "</s>", "<unk>"] + chars + ["th", "the", "<mask>"]
        (out / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(vocab)}))
        (out / "merges.txt").write_text("#version: 0.2\nt h\nth e\n")
        return RobertaTokenizerFast(str(out / "vocab.json"), str(out / "merges.txt"), model_max_length=model_max_length)
    chars = list(string.ascii_lowercase + string.digits + string.punctuation)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + WORDS + chars + ["##" + c for c in chars]
    (out / "vocab.txt").write_text("\n".join(vocab) + "\n")
    cls = DistilBertTokenizerFast if arch == "distilbert" else BertTokenizerFast
    return cls(str(out / "vocab.txt"), model_max_length=model_max_length)
