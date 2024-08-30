from abc import ABC, abstractmethod
from typing import TypeVar, Union

from pydantic import BaseModel

from .base_engine import AsyncLLMEngine
from .engine_utils import (compile_user_prompt, generate_obj_query_messages,
                           parse_block_response, parse_obj_response)

__all__ = ["AsyncChatEngine", "ChatEngine"]

T = TypeVar("T", bound=BaseModel)
S = TypeVar("S", bound=Union[str, T])


def _retry(n=3):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            for i in range(n):
                try:
                    return await func(*args, **kwargs)
                except Exception:
                    if i == n - 1:
                        raise

        return wrapper

    return decorator


class AsyncChatEngine(AsyncLLMEngine, ABC):
    def __init__(self, fallback: "AsyncLLMEngine" = None):
        self.fallback = fallback

    @abstractmethod
    async def query(self, **kwargs) -> str:
        """
        Send a query to the LLM.

        This is an abstract method that should be implemented by subclasses to interact
        with specific LLM APIs.

        Args:
            **kwargs: Arbitrary keyword arguments for the query. Example: max_tokens.
        """
        pass

    @_retry(5)
    async def query_object(self, response_model: type[T], **kwargs) -> T:
        prompt_args = {k: v for k, v in kwargs.items() if k == k.upper()}
        api_args = {k: v for k, v in kwargs.items() if k != k.upper()}

        try:
            response = await self.query(
                messages=await generate_obj_query_messages(response_model, prompt_args),
                **api_args,
            )
            return parse_obj_response(response_model, response)
        except Exception:
            if self.fallback:
                return await self.fallback.query_object(response_model, **kwargs)
            else:
                raise

    @_retry(5)
    async def query_block(self, block_type: str, **kwargs) -> T:
        prompt_args = {k: v for k, v in kwargs.items() if k == k.upper()}
        api_args = {k: v for k, v in kwargs.items() if k != k.upper()}

        try:
            response = await self.query(
                messages=_generate_block_query_messages(block_type, prompt_args),
                **api_args,
            )
            return parse_block_response(block_type, response)
        except Exception:
            if self.fallback:
                return await self.fallback.query_block(block_type, **kwargs)
            else:
                raise

    def query_continuation(self, prompt: str, **kwargs) -> str:
        try:
            response = self.query(
                messages=[
                    {"role": "assistant", "content": prompt},
                    {
                        "role": "user",
                        "content": "Please continue. Just the continuation, nothing else.",
                    },
                ],
                **kwargs,
            )
        except Exception:
            if self.fallback:
                return self.fallback.query_continuation(prompt, **kwargs)
            else:
                raise

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
