def apply(pre, texts, meta):
    out = list(pre(list(texts), meta))
    if len(out) != len(texts) or not all(isinstance(t, str) for t in out):
        raise ValueError("pre hook must return one string per text")
    return out


def pre(texts, meta):
    return texts
