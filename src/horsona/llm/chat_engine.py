from abc import ABC, abstractmethod
from typing import AsyncGenerator, Type, TypeVar, Union

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
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @abstractmethod
    async def query(self, **kwargs) -> AsyncGenerator[str, None]:
        """
        Send a raw query to the chat LLM API.

        Args:
            **kwargs: API-specific arguments including:
                messages: List of chat messages
                stream: Whether to stream the response
                etc.

        Yields:
            str: Response text chunks from the LLM
        """
        ...

    async def query_response(self, **kwargs) -> tuple[str, int]:
        prompt_args = {k: v for k, v in kwargs.items() if k == k.upper()}
        api_args = {k: v for k, v in kwargs.items() if k != k.upper()}

        if prompt_args:
            api_args.setdefault("messages", []).extend(
                {"role": "user", "content": await compile_user_prompt(**prompt_args)}
            )

        if "stream" in api_args:
            assert api_args["stream"] == False
        else:
            api_args["stream"] = False

        result = []
        async for chunk in self.query(**api_args):
            result.append(chunk)

        return "".join(result)

    async def query_stream(self, **kwargs) -> AsyncGenerator[str, None]:
        prompt_args = {k: v for k, v in kwargs.items() if k == k.upper()}
        api_args = {k: v for k, v in kwargs.items() if k != k.upper()}

        if prompt_args:
            api_args.setdefault("messages", []).extend(
                {"role": "user", "content": await compile_user_prompt(**prompt_args)}
            )

        if "stream" in api_args:
            assert api_args["stream"] == True
        else:
            api_args["stream"] = True

        async for chunk in self.query(**api_args):
            yield chunk

    async def query_object(self, response_model: Type[T], **kwargs) -> T:
        prompt_args = {k: v for k, v in kwargs.items() if k == k.upper()}
        api_args = {k: v for k, v in kwargs.items() if k != k.upper()}

        prior_messages = kwargs.get("messages", [])

        response = await self.query_response(
            messages=[
                *prior_messages,
                *await generate_obj_query_messages(response_model, prompt_args),
            ],
            **api_args,
        )

        return parse_obj_response(response_model, response)

    async def query_block(self, block_type: str, **kwargs) -> str:
        prompt_args = {k: v for k, v in kwargs.items() if k == k.upper()}
        api_args = {k: v for k, v in kwargs.items() if k != k.upper()}

        prior_messages = kwargs.get("messages", [])

        response = await self.query_response(
            messages=[
                *prior_messages,
                *await _generate_block_query_messages(block_type, prompt_args),
            ],
            **api_args,
        )

        return parse_block_response(block_type, response)

    async def query_continuation(self, prompt: str, **kwargs) -> str:
        prompt_args = {k: v for k, v in kwargs.items() if k == k.upper()}
        api_args = {k: v for k, v in kwargs.items() if k != k.upper()}

        prior_messages = kwargs.get("messages", [])

        prompt_references = await compile_user_prompt(**prompt_args)

        response = await self.query_response(
            messages=[
                *prior_messages,
                {"role": "user", "content": prompt_references},
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
        "Provide the final response in a single fenced code block and nothing else. Provide "
        f"the response within: ```{block_type}\ncontent\n```.\n\n"
        f"The content should be {block_type}-formatted."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
