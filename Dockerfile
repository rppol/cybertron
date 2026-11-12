FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.14 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY cybertron ./cybertron
COPY examples ./examples
RUN uv sync --frozen --no-dev --no-editable

RUN useradd --system --uid 10001 app && mkdir /cache && chown app /cache
USER app

ENV PATH=/app/.venv/bin:$PATH \
    CYBERTRON_MODEL=/model \
    CYBERTRON_CACHE_DIR=/cache \
    HF_HUB_OFFLINE=1 \
    MALLOC_ARENA_MAX=2 \
    MALLOC_TRIM_THRESHOLD_=131072 \
    MALLOC_MMAP_THRESHOLD_=131072
VOLUME /cache
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["uvicorn", "cybertron.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
