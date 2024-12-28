import json
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Type, TypeVar, Union

from pydantic import BaseModel

from horsona.llm.base_engine import AsyncLLMEngine, LLMMetrics

T = TypeVar("T", bound=BaseModel)
S = TypeVar("S", bound=Union[str, T])


class WrapperLLMEngine(AsyncLLMEngine):
    def __init__(self, underlying_llm: AsyncLLMEngine, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.underlying_llm = underlying_llm

    async def query_response(self, metrics: LLMMetrics = None, **kwargs) -> str:
        if "TASK" not in kwargs:
            kwargs["TASK"] = (
                "Respond as the assistant based on the most recent user message above."
            )

        args = await self._hook_all_args(**kwargs)

        prompt_args = {k: v for k, v in args.items() if k == k.upper()}

        result = await self.underlying_llm.query_response(metrics=metrics, **args)
        await self.capture_response(
            translate_to_prompt_args(args),
            result,
        )

        return result

    async def query_object(self, response_model: Type[T], **kwargs) -> T:
        args = await self._hook_all_args(**kwargs)
        prompt_args = {k: v for k, v in args.items() if k == k.upper()}

        result = await self.underlying_llm.query_object(response_model, **args)
        await self.capture_response(
            translate_to_prompt_args(args),
            result,
        )

        return result

    async def query_block(self, block_type: str, **kwargs) -> str:
        args = await self._hook_all_args(**kwargs)
        prompt_args = {k: v for k, v in args.items() if k == k.upper()}

        result = await self.underlying_llm.query_block(block_type, **args)
        await self.capture_response(
            translate_to_prompt_args(args),
            result,
        )

        return result

    async def query_continuation(self, prompt: str, **kwargs) -> str:
        kwargs["__USER_PROMPT"] = prompt
        args = await self._hook_all_args(**kwargs)
        prompt_args = {k: v for k, v in args.items() if k == k.upper()}

        result = await self.underlying_llm.query_continuation(prompt, **args)
        await self.capture_response(
            translate_to_prompt_args(args),
            result,
        )

        return result

    async def query_stream(
        self, metrics: LLMMetrics = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        if "TASK" not in kwargs:
            kwargs["TASK"] = (
                "Respond as the assistant based on the most recent user message above."
            )

        args = await self._hook_all_args(**kwargs)
        prompt_args = {k: v for k, v in args.items() if k == k.upper()}

        result = []
        async for chunk in self.underlying_llm.query_stream(metrics=metrics, **args):
            result.append(chunk)
            yield chunk

        await self.capture_response(
            translate_to_prompt_args(args),
            "".join(result),
        )

    async def capture_response(self, prompt_args: dict, response: str) -> str:
        pass

    async def hook_prompt_args(self, **prompt_args) -> dict:
        return prompt_args

    async def _hook_all_args(self, **kwargs) -> str:
        prompt_args = {k: v for k, v in kwargs.items() if k == k.upper()}
        api_args = {k: v for k, v in kwargs.items() if k != k.upper()}

        added_task = False
        added_history = False

        if "messages" in api_args:
            prior_messages = api_args["messages"]
            prompt_args["CHAT_HISTORY"] = prior_messages
            added_history = True

        prompt_args = await self.hook_prompt_args(**prompt_args)

        if added_history:
            prompt_args.pop("CHAT_HISTORY")

        return {**prompt_args, **api_args}


def translate_to_prompt_args(kwargs: dict) -> dict:
    prompt_args = {k: v for k, v in kwargs.items() if k == k.upper()}
    api_args = {k: v for k, v in kwargs.items() if k != k.upper()}

    if "messages" in api_args:
        prompt_args["CHAT_HISTORY"] = api_args["messages"]

    return prompt_args
