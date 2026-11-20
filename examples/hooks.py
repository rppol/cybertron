"""Example hooks: mask long digit runs before tokenizing, and drop labels that a row's metadata denies.

Enable with CYBERTRON_HOOKS=examples.hooks.
"""

import re

import numpy as np

DIGITS = re.compile(r"\d{6,}")


def pre(texts, meta):
    return [DIGITS.sub("#", t) for t in texts]


def post(logits, meta, labels):
    for row, m in enumerate(meta or []):
        deny = [labels.index(name) for name in (m or {}).get("deny", []) if name in labels]
        logits[row, deny] = -np.inf
    return logits
