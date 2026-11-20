from fastapi import APIRouter, Request, Response
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

router = APIRouter()


@router.get("/metrics", response_class=PlainTextResponse)
async def metrics(request: Request):
    return Response(generate_latest(request.app.state.registry), media_type=CONTENT_TYPE_LATEST)
