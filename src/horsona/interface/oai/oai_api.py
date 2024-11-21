from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from horsona.llm.chat_engine import AsyncChatEngine

from .oai_models import (
    ChatCompletionChoice,
    ChatCompletionChunkChoice,
    ChatCompletionChunkResponse,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    DeltaMessage,
)

llm_engines: dict[str, AsyncChatEngine] = {}
router = APIRouter(prefix="/api")


async def _get_streaming_response(engine: AsyncChatEngine, request_dict: dict):
    response: AsyncGenerator[str, None] = engine.query_stream(**request_dict)

    async def stream_response():
        # First chunk should include role
        i = 0
        async for chunk in response:
            # Add data: prefix and double newline suffix for SSE format
            if i == 0:
                yield (
                    "data: "
                    + ChatCompletionChunkResponse(
                        id="chatcmpl-" + str(i),
                        model=request_dict["model"],
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta=DeltaMessage(role="assistant", content=chunk),
                                finish_reason=None,
                            )
                        ],
                    ).model_dump_json()
                    + "\n\n"
                )
                i += 1
            else:
                yield (
                    "data: "
                    + ChatCompletionChunkResponse(
                        id="chatcmpl-" + str(i),
                        model=request_dict["model"],
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta=DeltaMessage(content=chunk),
                                finish_reason=None,
                            )
                        ],
                    ).model_dump_json()
                    + "\n\n"
                )
                i += 1

        # Send final chunk with finish_reason
        yield (
            "data: "
            + ChatCompletionChunkResponse(
                id="chatcmpl-" + str(i + 1),
                model=request_dict["model"],
                choices=[
                    ChatCompletionChunkChoice(
                        index=0, delta=DeltaMessage(), finish_reason="stop"
                    )
                ],
            ).model_dump_json()
            + "\n\n"
        )

        # Send final [DONE] message
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


async def _get_nonstreaming_response(engine: AsyncChatEngine, request_dict: dict):
    try:
        response = await engine.query(**request_dict)

        return ChatCompletionResponse(
            model=request_dict["model"],
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=response,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=ChatCompletionUsage(),
        )
    except Exception as e:
        raise e


@router.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
) -> ChatCompletionResponse:
    if request.n != 1 and request.n is not None:
        raise HTTPException(
            status_code=400, detail="Multiple completions not supported"
        )

    model_name = request.model
    engine = llm_engines[model_name]
    request_dict = request.model_dump(exclude_unset=True)

    try:
        if request.stream:
            return await _get_streaming_response(engine, request_dict)
        else:
            return await _get_nonstreaming_response(engine, request_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def add_llm_engine(engine: AsyncChatEngine, name: str | None = None):
    llm_engines[name or engine.name] = engine


class ModelObject(BaseModel):
    id: str
    created: int
    owned_by: str
    object: str = "model"


class ModelsResponse(BaseModel):
    data: list[ModelObject]
    object: str = "list"


@router.get("/v1/models")
async def list_models() -> ModelsResponse:
    """List available models."""
    return ModelsResponse(
        data=[
            ModelObject(id=model_name, created=0, owned_by="horsona", object="model")
            for model_name in llm_engines.keys()
        ],
        object="list",
    )
