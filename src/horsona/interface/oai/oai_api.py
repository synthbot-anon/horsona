from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field

from horsona.llm.base_engine import AsyncLLMEngine
from horsona.llm.chat_engine import AsyncChatEngine
from horsona.llm.engine_utils import compile_user_prompt

from .oai_models import *

llm_engines: dict[str, AsyncChatEngine] = {}
router = APIRouter(prefix="/api")


@router.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
) -> ChatCompletionResponse:
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming not supported")
    if request.n != 1 and request.n is not None:
        raise HTTPException(
            status_code=400, detail="Multiple completions not supported"
        )

    model_name = request.model
    engine = llm_engines[model_name]

    try:
        response = await engine.query(
            messages=request.messages,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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


def add_llm_engine(engine: AsyncChatEngine):
    llm_engines[engine.name] = engine
