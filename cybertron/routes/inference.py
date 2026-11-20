from fastapi import APIRouter, Depends, HTTPException, Request

from ..core.concurrency import admit
from ..core.security import guard
from ..schemas.requests import ClassifyIn, EmbedIn, TextsIn
from ..schemas.responses import ClassifyOut, EmbedOut

router = APIRouter(prefix="/v1", dependencies=[Depends(guard)])


@router.post("/classify", response_model=ClassifyOut)
async def classify(body: ClassifyIn, request: Request):
    engine = request.app.state.engine
    if not engine.labels:
        raise HTTPException(404, "model has no classification head")
    check(request, body)
    return {"results": await admit(engine, engine.classify, body.texts, body.top_k, body.meta)}


@router.post("/embed", response_model=EmbedOut)
async def embed(body: EmbedIn, request: Request):
    engine = request.app.state.engine
    check(request, body)
    vecs = await admit(engine, engine.embed, body.texts, body.pooling, body.normalize, body.meta)
    return {"embeddings": vecs.tolist()}


def check(request: Request, body: TextsIn):
    cfg = request.app.state.cfg
    if len(body.texts) > cfg.max_texts or any(len(t) > cfg.max_chars for t in body.texts):
        raise HTTPException(413, f"at most {cfg.max_texts} texts of {cfg.max_chars} characters")
    if body.meta is not None and len(body.meta) != len(body.texts):
        raise HTTPException(422, "meta needs one entry per text")
    request.app.state.texts.labels(request.url.path).inc(len(body.texts))
