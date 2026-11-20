import numpy as np


def apply(post, logits, meta, labels):
    out = np.asarray(post(logits, meta, labels), dtype=np.float32)
    _check(out, (len(logits), len(labels)))
    return out


def top(logits, labels, multi, k):
    if multi:
        probs = 0.5 * (1 + np.tanh(logits / 2))
    else:
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
    best = np.argsort(-probs, axis=1, kind="stable")[:, :k]
    return [[{"label": labels[j], "score": float(p[j])} for j in row] for p, row in zip(probs, best)]


def post(logits, meta, labels):
    return logits


def _check(logits, shape):
    if logits.shape != shape:
        raise ValueError(f"post hook returned shape {logits.shape}, want {shape}")
    if np.isnan(logits).any():
        raise ValueError("logits contain NaN")
    if np.isposinf(logits).any():
        raise ValueError("logits contain +inf")
    if not np.isfinite(logits).any(axis=1).all():
        raise ValueError("every row needs at least one finite logit")
