import queue

from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool


async def admit(engine, fn, *args):
    """Run fn(req, *args) in a worker thread on a free InferRequest, or answer 429 at once."""
    try:
        req = engine.pool.get_nowait()
    except queue.Empty:
        raise HTTPException(429, "all inference slots are busy", headers={"Retry-After": "1"}) from None
    try:
        return await run_in_threadpool(fn, req, *args)
    finally:
        engine.pool.put_nowait(req)
