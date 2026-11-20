import importlib
import logging
import queue
import threading
import time

import nncf
import numpy as np
import openvino as ov
import torch
from openvino._offline_transformations import compress_model_transformation

from .model import load

log = logging.getLogger(__name__)


class Engine:
    def __init__(self, cfg):
        hooks = importlib.import_module(cfg.hooks) if cfg.hooks else None
        self._pre, self._post = getattr(hooks, "pre", None), getattr(hooks, "post", None)
        loaded = load(cfg)
        self.tok, self.labels, self.multi = loaded.tok, loaded.labels, loaded.multi
        self.batch_size = cfg.batch_size
        self.buckets = [b for b in cfg.buckets if b < loaded.max_len] + [loaded.max_len]
        self.compiled, self.info = compile(convert(loaded, cfg), cfg)
        slots = cfg.slots or self.info["optimal_requests"]
        self.pool = queue.Queue()
        for _ in range(slots):
            self.pool.put(self.compiled.create_infer_request())
        self._probe_req = self.compiled.create_infer_request()
        self._probe_lock = threading.Lock()
        self.info |= {"slots": slots, "buckets": self.buckets, "tasks": ["classify", "embed"] if self.labels else ["embed"]}
        log.info("engine ready: %s", self.info)

    def classify(self, req, texts, top_k=1, meta=None):
        if not self.labels:
            raise ValueError("model has no classification head")
        if not texts:
            return []
        logits = self._run(req, self._prep(texts, meta), 0)
        if self._post:
            logits = np.asarray(self._post(logits, meta, self.labels), dtype=np.float32)
        _check(logits, (len(texts), len(self.labels)))
        if self.multi:
            probs = 0.5 * (1 + np.tanh(logits / 2))
        else:
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = e / e.sum(axis=1, keepdims=True)
        top = np.argsort(-probs, axis=1, kind="stable")[:, :top_k]
        return [[{"label": self.labels[j], "score": float(p[j])} for j in row] for p, row in zip(probs, top)]

    def embed(self, req, texts, pooling="mean", normalize=True, meta=None):
        if not texts:
            return np.empty((0, 0), np.float32)
        vecs = self._run(req, self._prep(texts, meta), len(self.compiled.outputs) - (1 if pooling == "cls" else 2))
        if normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs /= np.where(norms == 0, 1, norms)
        return vecs

    def probe(self):
        """Seconds for one inference on a reserved request, or None while another probe runs."""
        if not self._probe_lock.acquire(blocking=False):
            return None
        try:
            start = time.perf_counter()
            if not np.isfinite(self._run(self._probe_req, ["health check"], 0)).all():
                raise RuntimeError("probe output is not finite")
            return time.perf_counter() - start
        finally:
            self._probe_lock.release()

    def close(self):
        while not self.pool.empty():
            self.pool.get_nowait()
        self.compiled = self._probe_req = None

    def _prep(self, texts, meta):
        if self._pre is None:
            return texts
        out = list(self._pre(list(texts), meta))
        if len(out) != len(texts) or not all(isinstance(t, str) for t in out):
            raise ValueError("pre hook must return one string per text")
        return out

    def _run(self, req, texts, output):
        enc = self.tok(texts, truncation=True, max_length=self.buckets[-1])
        lengths = [len(ids) for ids in enc["input_ids"]]
        order = sorted(range(len(texts)), key=lengths.__getitem__)
        pad = {"input_ids": self.tok.pad_token_id or 0}
        result = None
        for start in range(0, len(order), self.batch_size):
            rows = order[start : start + self.batch_size]
            width = next(b for b in self.buckets if b >= lengths[rows[-1]])
            feed = {}
            for name in self.info["inputs"]:
                feed[name] = np.full((len(rows), width), pad.get(name, 0), np.int64)
                for r, i in enumerate(rows):
                    feed[name][r, : lengths[i]] = enc[name][i]
            req.infer(feed)
            values = req.get_output_tensor(output).data
            if result is None:
                result = np.empty((len(texts), values.shape[1]), np.float32)
            result[rows] = values
        return result


def convert(loaded, cfg) -> ov.Model:
    net = loaded.net
    example = loaded.tok(["an example input"] * 2, truncation=True, max_length=loaded.max_len, return_tensors="pt")
    # Some transformers versions drop an all-ones mask while tracing; a padded row keeps it in the graph.
    example["attention_mask"][1, -1] = 0
    with torch.no_grad():
        model = ov.convert_model(net, example_input={n: example[n] for n in net.inputs})
    dims = ov.PartialShape([ov.Dimension(1, cfg.batch_size), ov.Dimension(1, loaded.max_len)])
    for port, name in zip(model.inputs, net.inputs):
        port.get_tensor().set_names({name})
    model.reshape({port: dims for port in model.inputs})
    if cfg.precision == "FP16":
        compress_model_transformation(model)
    elif cfg.precision == "INT8":
        model = nncf.compress_weights(model, mode=nncf.CompressWeightsMode.INT8_ASYM)
    return model


def compile(model, cfg):
    core = ov.Core()
    if cfg.cache_dir:
        core.set_property({"CACHE_DIR": str(cfg.cache_dir)})
    supported = {str(p) for p in core.get_property("CPU", "SUPPORTED_PROPERTIES")}
    compiled = core.compile_model(model, "CPU", properties(cfg, supported))
    info = {
        "precision": cfg.precision,
        "inference_precision": compiled.get_property("INFERENCE_PRECISION_HINT").to_string(),
        "hint": str(compiled.get_property("PERFORMANCE_HINT")),
        "streams": int(compiled.get_property("NUM_STREAMS")),
        "threads": int(compiled.get_property("INFERENCE_NUM_THREADS")),
        "optimal_requests": int(compiled.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")),
        "inputs": [port.get_any_name() for port in compiled.inputs],
    }
    return compiled, info


def properties(cfg, supported):
    # INT8 weights with f16 execution gave NaN logits under concurrent load on arm64 (OpenVINO 2025.4.1).
    defaults = {"INFERENCE_PRECISION_HINT": "f32"} if cfg.precision != "FP16" else {}
    for key in defaults.keys() - supported:
        log.warning("CPU plugin does not support %s, leaving it unset", key)
    typed = {
        "PERFORMANCE_HINT": cfg.hint,
        "NUM_STREAMS": cfg.streams,
        "INFERENCE_NUM_THREADS": cfg.threads,
        "ENABLE_HYPER_THREADING": cfg.hyper_threading,
        "ENABLE_CPU_PINNING": cfg.cpu_pinning,
    }
    return (
        {k: v for k, v in defaults.items() if k in supported}
        | {k: v for k, v in typed.items() if v is not None}
        | cfg.ov_config
    )


def _check(logits, shape):
    if logits.shape != shape:
        raise ValueError(f"post hook returned shape {logits.shape}, want {shape}")
    if np.isnan(logits).any():
        raise ValueError("logits contain NaN")
    if np.isposinf(logits).any():
        raise ValueError("logits contain +inf")
    if not np.isfinite(logits).any(axis=1).all():
        raise ValueError("every row needs at least one finite logit")
