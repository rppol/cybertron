import gc
import random
from concurrent.futures import ThreadPoolExecutor

import psutil
import pytest
from tiny import WORDS, build

from cybertron.config import Settings
from cybertron.engine import Engine

pytestmark = pytest.mark.slow


def test_rss_flat_after_warmup(tmp_path):
    build(tmp_path, "minilm")
    engine = Engine(Settings(model=tmp_path, batch_size=16, max_len=128))
    rng = random.Random(0)

    def text(words):
        return " ".join(rng.choices(WORDS, k=words))

    reqs = [engine.pool.get() for _ in range(engine.info["slots"])]
    for req in reqs:
        for bucket in engine.buckets:
            for size in range(1, 17):
                engine.classify(req, [text(bucket - 2)] * size)
    for req in reqs:
        engine.pool.put(req)

    def work(batch):
        req = engine.pool.get()
        try:
            engine.classify(req, batch)
        finally:
            engine.pool.put(req)

    proc = psutil.Process()
    gc.collect()
    before = proc.memory_info().rss
    batches = [[text(rng.randint(1, 126)) for _ in range(rng.randint(1, 16))] for _ in range(500)]
    with ThreadPoolExecutor(len(reqs)) as ex:
        list(ex.map(work, batches))
    gc.collect()
    after = proc.memory_info().rss
    print(f"RSS {before / 2**20:.0f} MB -> {after / 2**20:.0f} MB")
    assert (after - before) / before < 0.10
