import importlib
import logging
import queue
import threading
import time

import numpy as np

from . import postprocessing, preprocessing
from .convert import compile, convert
from .model import load

log = logging.getLogger(__name__)


class Engine:
    def __init__(self, cfg):
        hooks = importlib.import_module(cfg.hooks) if cfg.hooks else None
        self._pre = getattr(hooks, "pre", None) or preprocessing.pre
        self._post = getattr(hooks, "post", None) or postprocessing.post
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
        logits = self._run(req, preprocessing.apply(self._pre, texts, meta), 0)
        logits = postprocessing.apply(self._post, logits, meta, self.labels)
        return postprocessing.top(logits, self.labels, self.multi, top_k)

    def embed(self, req, texts, pooling="mean", normalize=True, meta=None):
        if not texts:
            return np.empty((0, 0), np.float32)
        vecs = self._run(req, preprocessing.apply(self._pre, texts, meta), len(self.compiled.outputs) - (1 if pooling == "cls" else 2))
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
