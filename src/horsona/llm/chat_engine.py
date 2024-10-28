from abc import ABC
from typing import Type, TypeVar, Union

from pydantic import BaseModel

from .base_engine import AsyncLLMEngine
from .engine_utils import (
    compile_user_prompt,
    generate_obj_query_messages,
    parse_block_response,
    parse_obj_response,
)

__all__ = ["AsyncChatEngine"]

T = TypeVar("T", bound=BaseModel)
S = TypeVar("S", bound=Union[str, T])


class AsyncChatEngine(AsyncLLMEngine, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def query_object(self, response_model: Type[T], **kwargs) -> T:
        prompt_args = {k: v for k, v in kwargs.items() if k == k.upper()}
        api_args = {k: v for k, v in kwargs.items() if k != k.upper()}

        response = await self.query(
            messages=await generate_obj_query_messages(response_model, prompt_args),
            **api_args,
        )

        return parse_obj_response(response_model, response)

    async def query_block(self, block_type: str, **kwargs) -> str:
        prompt_args = {k: v for k, v in kwargs.items() if k == k.upper()}
        api_args = {k: v for k, v in kwargs.items() if k != k.upper()}

        response = await self.query(
            messages=await _generate_block_query_messages(block_type, prompt_args),
            **api_args,
        )

        return parse_block_response(block_type, response)

    async def query_continuation(self, prompt: str, **kwargs) -> str:
        prompt_args = {k: v for k, v in kwargs.items() if k == k.upper()}
        api_args = {k: v for k, v in kwargs.items() if k != k.upper()}

        prompt = await compile_user_prompt(**prompt_args)

        response = await self.query(
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": prompt},
                {
                    "role": "user",
                    "content": "Please continue. Just the continuation, nothing else.",
                },
            ],
            **api_args,
        )

        return response


async def _generate_block_query_messages(block_type: str, prompt_args):
    """
    Generate messages for a block query.

    This function creates a system message and a user message for querying
    an LLM to generate a response in a specific block format.

    Args:
        block_type (str): The type of block to generate (e.g., "python", "sql").
        prompt_args: Arguments to include in the user prompt.

    Returns:
        list: A list of message dictionaries for the LLM query.
    """
    prompt = await compile_user_prompt(**prompt_args)
    system_prompt = (
        "Respond with a single fenced code block and nothing else. Provide "
        f"the response within: ```{block_type}\ncontent\n```.\n\n"
        f"The content should be {block_type}-formatted."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
