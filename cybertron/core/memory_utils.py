import asyncio
import ctypes
import gc
import sys
from contextlib import asynccontextmanager, suppress


@asynccontextmanager
async def tuned_gc(interval, thresholds):
    gc.collect()
    gc.freeze()
    old = gc.get_threshold()
    gc.set_threshold(*thresholds)
    task = asyncio.create_task(_collect(interval)) if interval else None
    try:
        yield
    finally:
        if task:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        gc.unfreeze()
        gc.set_threshold(*old)


async def _collect(interval):
    while True:
        await asyncio.sleep(interval)
        gc.collect()
        if sys.platform == "linux":
            with suppress(OSError, AttributeError):
                ctypes.CDLL("libc.so.6").malloc_trim(0)
