from typing import Generator

import pytest
from horsona.llm.base_engine import AsyncLLMEngine
from horsona.llm.cerebras_engine import AsyncCerebrasEngine
from pydantic import BaseModel


@pytest.fixture(scope="module")
def llm() -> Generator[AsyncLLMEngine, None, None]:
    yield AsyncCerebrasEngine(model="llama3.1-70b")


@pytest.mark.asyncio
async def test_chat_engine(llm: AsyncLLMEngine):
    class Response(BaseModel):
        name: str

    response = await llm.query_object(
        Response,
        NAME="Celestia",
        TASK="Respond with the NAME.",
    )

    assert response.name == "Celestia"
