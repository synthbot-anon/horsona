from typing import AsyncGenerator, Type, TypeVar, Union

from pydantic import BaseModel

from horsona.llm.base_engine import AsyncLLMEngine, LLMMetrics
from horsona.llm.chat_engine import AsyncChatEngine

T = TypeVar("T", bound=BaseModel)
S = TypeVar("S", bound=Union[str, T])


class CustomLLMEngine(AsyncChatEngine):
    def __init__(self, underlying_llm: AsyncLLMEngine, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.underlying_llm = underlying_llm

    async def query(self, metrics: LLMMetrics = None, **kwargs) -> str:
        args = await self._hook_all_args(**kwargs)
        return await self.underlying_llm.query(metrics=metrics, **args)

    async def query_object(self, response_model: Type[T], **kwargs) -> T:
        args = await self._hook_all_args(**kwargs)
        return await self.underlying_llm.query_object(response_model, **args)

    async def query_block(self, block_type: str, **kwargs) -> str:
        args = await self._hook_all_args(**kwargs)
        return await self.underlying_llm.query_block(block_type, **args)

    async def query_continuation(self, prompt: str, **kwargs) -> str:
        kwargs["__USER_PROMPT"] = prompt
        args = await self._hook_all_args(**kwargs)
        prompt = args.pop("__USER_PROMPT")

        return await self.underlying_llm.query_continuation(prompt, **args)

    async def query_stream(
        self, metrics: LLMMetrics = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        args = await self._hook_all_args(**kwargs)
        return await self.underlying_llm.query_stream(metrics=metrics, **args)

    async def hook_prompt_args(self, **prompt_args) -> str:
        return prompt_args

    async def _hook_all_args(self, **kwargs) -> str:
        prompt_args = {k: v for k, v in kwargs.items() if k == k.upper()}
        api_args = {k: v for k, v in kwargs.items() if k != k.upper()}

        prompt_args = await self.hook_prompt_args(**prompt_args)

        return {**prompt_args, **api_args}
