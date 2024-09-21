from typing import Generator

import pytest
from horsona.llm.base_engine import AsyncLLMEngine
from horsona.llm.cerebras_engine import AsyncCerebrasEngine
from horsona.llm.multi_engine import create_multi_engine
from pydantic import BaseModel


@pytest.mark.asyncio
async def test_multi_engine(
    cerebras_llama31_70b: AsyncLLMEngine, firework_llama31_70b: AsyncLLMEngine
):
    multi = create_multi_engine(cerebras_llama31_70b, firework_llama31_70b)
    response = await multi.query_block(str, TASK="say hello world")

    print(response)
