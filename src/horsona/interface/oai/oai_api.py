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

    print(request_dict)

    try:
        if request.stream:
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
                                model=request.model,
                                choices=[
                                    ChatCompletionChunkChoice(
                                        index=0,
                                        delta=DeltaMessage(
                                            role="assistant", content=chunk
                                        ),
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
                                model=request.model,
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
                        model=request.model,
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

        else:
            response = await engine.query(**request_dict)

            return ChatCompletionResponse(
                model=request.model,
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
        raise HTTPException(status_code=500, detail=str(e))


def add_llm_engine(engine: AsyncChatEngine):
    llm_engines[engine.name] = engine


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
