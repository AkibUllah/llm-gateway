"""OpenAI-compatible endpoints: /v1/chat/completions, /v1/embeddings, /v1/models."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from llm_gateway.api.deps import get_state, require_key, wants_cache
from llm_gateway.keys import KeyContext
from llm_gateway.logs import request_id_var
from llm_gateway.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ModelList,
)

router = APIRouter(prefix="/v1", tags=["OpenAI-compatible"])

STREAM_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


@router.post(
    "/chat/completions",
    response_model=ChatCompletionResponse,
    summary="Create a chat completion (set stream=true for server-sent events)",
)
async def chat_completions(
    body: ChatCompletionRequest, request: Request, key: KeyContext = Depends(require_key)
):
    gateway = get_state(request).gateway
    request_id = request_id_var.get() or ""
    if body.stream:
        stream, headers = await gateway.chat_stream(
            body, key, request_id=request_id, use_cache=wants_cache(request)
        )
        return StreamingResponse(
            stream, media_type="text/event-stream", headers={**headers, **STREAM_HEADERS}
        )
    response, headers = await gateway.chat(
        body, key, request_id=request_id, use_cache=wants_cache(request)
    )
    return JSONResponse(response.model_dump(exclude_none=True), headers=headers)


@router.post("/embeddings", response_model=EmbeddingResponse, summary="Create embeddings")
async def embeddings(
    body: EmbeddingRequest, request: Request, key: KeyContext = Depends(require_key)
):
    gateway = get_state(request).gateway
    response, headers = await gateway.embeddings(body, key, request_id=request_id_var.get() or "")
    return JSONResponse(response.model_dump(), headers=headers)


@router.get("/models", response_model=ModelList, summary="List model aliases this key can use")
async def list_models(request: Request, key: KeyContext = Depends(require_key)) -> ModelList:
    return get_state(request).gateway.models(key)
