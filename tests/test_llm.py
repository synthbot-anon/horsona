from typing import Generator

import pytest
from horsona.llm.base_engine import AsyncLLMEngine
from horsona.llm.cerebras_engine import AsyncCerebrasEngine
from pydantic import BaseModel


@pytest.mark.asyncio
async def test_chat_engine(reasoning_llm: AsyncLLMEngine):
    class Response(BaseModel):
        name: str

    response = await reasoning_llm.query_object(
        Response,
        NAME="Celestia",
        TASK="Respond with the NAME.",
    )

    assert response.name == "Celestia"
