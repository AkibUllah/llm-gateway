# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install ".[postgres,redis]"

COPY config ./config
COPY scripts ./scripts
RUN mkdir -p /app/data && chown app:app /app/data

ENV GATEWAY_CONFIG=/app/config/gateway.yaml \
    PRICING_CONFIG=/app/config/pricing.yaml \
    DATABASE_URL=sqlite+aiosqlite:////app/data/gateway.db \
    PORT=8080

USER app
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen(f\"http://127.0.0.1:{os.environ.get('PORT','8080')}/health\").status == 200 else 1)"

CMD ["llm-gateway", "serve"]
