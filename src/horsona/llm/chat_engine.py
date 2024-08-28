from abc import ABC, abstractmethod
from typing import TypeVar, Union

from pydantic import BaseModel

from .base_engine import AsyncLLMEngine, LLMEngine
from .engine_utils import (compile_user_prompt, generate_obj_query_messages,
                           parse_block_response, parse_obj_response)

__all__ = ["AsyncChatEngine", "ChatEngine"]

T = TypeVar("T", bound=BaseModel)
S = TypeVar("S", bound=Union[str, T])


class ChatEngine(LLMEngine, ABC):
    def __init__(self, fallback: "LLMEngine" = None):
        """
        Initialize the LLMEngine.

        Args:
            fallback (LLMEngine, optional): Another LLMEngine instance to use as a fallback
                                            if this engine's queries fail. Defaults to None.
        """
        self.fallback = fallback

    @abstractmethod
    def query(self, **kwargs) -> str:
        """
        Send a query to the LLM.

        This is an abstract method that should be implemented by subclasses to interact
        with specific LLM APIs.

        Args:
            **kwargs: Arbitrary keyword arguments for the query. Example: max_tokens.
        """
        pass

    def query_object(self, response_model: type[T], **kwargs) -> T:
        prompt_args = {k: v for k, v in kwargs.items() if k == k.upper()}
        api_args = {k: v for k, v in kwargs.items() if k != k.upper()}

        try:
            response = self.query(
                messages=generate_obj_query_messages(response_model, prompt_args),
                **api_args,
            )
        except Exception:
            if self.fallback:
                return self.fallback.query_object(response_model, **kwargs)
            else:
                raise

        return parse_obj_response(response_model, response)

    def query_block(self, block_type: str, **kwargs) -> str:
        prompt_args = {k: v for k, v in kwargs.items() if k == k.upper()}
        api_args = {k: v for k, v in kwargs.items() if k != k.upper()}

        try:
            response = self.query(
                messages=_generate_block_query_messages(block_type, prompt_args),
                **api_args,
            )
        except Exception:
            if self.fallback:
                return self.fallback.query_block(block_type, **kwargs)
            else:
                raise

        return parse_block_response(block_type, response)


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

    async def query_object(self, response_model: type[T], **kwargs) -> T:
        prompt_args = {k: v for k, v in kwargs.items() if k == k.upper()}
        api_args = {k: v for k, v in kwargs.items() if k != k.upper()}

        try:
            response = await self.query(
                messages=generate_obj_query_messages(response_model, prompt_args),
                **api_args,
            )
        except Exception:
            if self.fallback:
                return await self.fallback.query_object(response_model, **kwargs)
            else:
                raise

        return parse_obj_response(response_model, response)

    async def query_block(self, block_type: str, **kwargs) -> T:
        prompt_args = {k: v for k, v in kwargs.items() if k == k.upper()}
        api_args = {k: v for k, v in kwargs.items() if k != k.upper()}

        try:
            response = await self.query(
                messages=_generate_block_query_messages(block_type, prompt_args),
                **api_args,
            )
        except Exception:
            if self.fallback:
                return await self.fallback.query_block(block_type, **kwargs)
            else:
                raise

        return parse_block_response(block_type, response)


def _generate_block_query_messages(block_type: str, prompt_args):
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
    prompt = compile_user_prompt(**prompt_args)
    system_prompt = (
        "Respond with a single fenced code block and nothing else. Provide "
        f"the response within: ```{block_type}\ncontent\n```.\n\n"
        f"The content should be {block_type}-formatted."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
