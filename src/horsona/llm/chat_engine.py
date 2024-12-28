import json
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Type, TypeVar, Union

from pydantic import BaseModel, TypeAdapter

from .base_engine import AsyncLLMEngine
from .engine_utils import compile_user_prompt, parse_block_response, parse_obj_response

__all__ = ["AsyncChatEngine"]

T = TypeVar("T", bound=BaseModel)
S = TypeVar("S", bound=Union[str, T])


class AsyncChatEngine(AsyncLLMEngine, ABC):
    def __init__(self, conversational=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.conversational = conversational

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

        await self._update_messages_with_prompt_args(
            api_args.setdefault("messages", []), prompt_args
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

        await self._update_messages_with_prompt_args(
            api_args.setdefault("messages", []), prompt_args
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

        await self._update_messages_with_prompt_args(
            api_args.setdefault("messages", []), prompt_args
        )
        api_args.setdefault("messages", []).extend(
            await _generate_obj_query_messages(response_model)
        )

        response = await self.query_response(**api_args)

        return parse_obj_response(response_model, response)

    async def query_block(self, block_type: str, **kwargs) -> str:
        prompt_args = {k: v for k, v in kwargs.items() if k == k.upper()}
        api_args = {k: v for k, v in kwargs.items() if k != k.upper()}

        await self._update_messages_with_prompt_args(
            api_args.setdefault("messages", []), prompt_args
        )
        api_args.setdefault("messages", []).extend(
            await _generate_block_query_messages(block_type, prompt_args)
        )

        response = await self.query_response(**api_args)

        return parse_block_response(block_type, response)

    async def query_continuation(self, prompt: str, **kwargs) -> str:
        prompt_args = {k: v for k, v in kwargs.items() if k == k.upper()}
        api_args = {k: v for k, v in kwargs.items() if k != k.upper()}

        await self._update_messages_with_prompt_args(
            api_args.setdefault("messages", []), prompt_args
        )
        api_args.setdefault("messages", []).extend(
            [
                {"role": "assistant", "content": prompt},
                {
                    "role": "user",
                    "content": "Please continue. Just the continuation, nothing else.",
                },
            ]
        )

        return await self.query_response(**api_args)

    async def _update_messages_with_prompt_args(
        self, messages: list[dict[str, str]], prompt_args: dict[str, Any]
    ) -> None:
        if not prompt_args:
            return

        if self.conversational:
            messages.insert(
                0, {"role": "user", "content": await compile_user_prompt(**prompt_args)}
            )
        else:
            messages.append(
                {"role": "user", "content": await compile_user_prompt(**prompt_args)}
            )


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


async def _generate_obj_query_messages(
    response_model: Type[BaseModel] | Type[Any],
) -> list[dict[str, Any]]:
    """
    Generate messages for an object query.

    This function creates a system message and a user message for querying
    an LLM to generate a response matching a specific model.

    Args:
        response_model (BaseModel): The expected response model.

    Returns:
        list: A list of message dictionaries for the LLM query.
    """
    user_prompt = (
        "Return the correct JSON response within a ```json codeblock, not the "
        "JSON_SCHEMA. Use only fields specified by the JSON_SCHEMA and nothing else."
    )

    schema = None
    try:
        if issubclass(response_model, BaseModel):
            schema = response_model.model_json_schema()
    except TypeError:
        pass

    if schema is None:
        schema = TypeAdapter(response_model).json_schema()

    system_prompt = (
        "Your task is to understand the content and provide "
        "the parsed objects in json that matches the following json_schema:\n\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        "Make sure to return an instance of the JSON, not the schema itself."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
