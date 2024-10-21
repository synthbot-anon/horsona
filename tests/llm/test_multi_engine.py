import pytest
from horsona.llm.base_engine import AsyncLLMEngine
from horsona.llm.multi_engine import create_multi_engine


@pytest.mark.asyncio
async def test_multi_engine(
    fireworks_llama31_70b: AsyncLLMEngine, openai_gpt4o_mini: AsyncLLMEngine
):
    multi = create_multi_engine(fireworks_llama31_70b, openai_gpt4o_mini)
    response = await multi.query_block("text", TASK="say hello world")

    print(response)
