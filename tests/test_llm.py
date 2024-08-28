import asyncio

import pytest
from pydantic import BaseModel

from horsona.llm.cerebras_engine import AsyncCerebrasEngine


@pytest.mark.asyncio
async def test_chat_engine():
    llm = AsyncCerebrasEngine(model="llama3.1-70b")

    class Response(BaseModel):
        name: str
    
    response = await llm.query_object(
        Response,
        NAME="Celestia",
        TASK="Respond with the NAME.",
    )

    assert response.name == "Celestia"
