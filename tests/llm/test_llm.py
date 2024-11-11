import pytest
from pydantic import BaseModel

from horsona.llm.base_engine import AsyncLLMEngine


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


@pytest.mark.asyncio
async def test_load(reasoning_llm: AsyncLLMEngine):
    state_dict = reasoning_llm.state_dict()
    restored = AsyncLLMEngine.load_state_dict(state_dict)

    assert isinstance(restored, type(reasoning_llm))
