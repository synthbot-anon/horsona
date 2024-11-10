from typing import Type, TypeVar, Union

from pydantic import BaseModel

from horsona.llm.base_engine import AsyncLLMEngine
from horsona.memory.list_module import ListModule

T = TypeVar("T", bound=BaseModel)
S = TypeVar("S", bound=Union[str, T])


class HistoryLLMEngine(AsyncLLMEngine):
    def __init__(
        self,
        underlying_llm: AsyncLLMEngine,
        history_module: ListModule,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.underlying_llm = underlying_llm
        self.history_module = history_module

    async def query(self, **kwargs) -> tuple[str, int]:
        return await self.underlying_llm.query(**kwargs), 0

    async def query_object(self, response_model: Type[T], **kwargs) -> T:
        return await self.underlying_llm.query_object(
            response_model,
            HISTORY_CONTEXT=self.history_module.get_items(),
            **kwargs,
        )

    async def query_block(self, block_type: str, **kwargs) -> str:
        return await self.underlying_llm.query_block(
            block_type,
            HISTORY_CONTEXT=self.history_module.get_items(),
            **kwargs,
        )

    async def query_continuation(self, prompt: str, **kwargs) -> str:
        return await self.underlying_llm.query_continuation(
            prompt,
            HISTORY_CONTEXT=self.history_module.get_items(),
            **kwargs,
        )
