from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from ..schemas.responses import DeepOut, HealthOut

router = APIRouter()


@router.get("/health", response_model=HealthOut)
async def health(request: Request):
    return {"status": "ok", "info": request.app.state.engine.info}


@router.get("/health/deep", response_model=DeepOut)
async def deep(request: Request):
    seconds = await run_in_threadpool(request.app.state.engine.probe)
    if seconds is None:
        raise HTTPException(429, "a probe is already running", headers={"Retry-After": "1"})
    return {"status": "ok", "seconds": seconds}
