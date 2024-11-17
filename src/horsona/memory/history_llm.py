from typing import TypeVar, Union

from pydantic import BaseModel

from horsona.llm.base_engine import AsyncLLMEngine, LLMMetrics
from horsona.llm.chat_engine import AsyncChatEngine
from horsona.memory.list_module import ListModule

T = TypeVar("T", bound=BaseModel)
S = TypeVar("S", bound=Union[str, T])


class HistoryLLMEngine(AsyncChatEngine):
    def __init__(
        self,
        underlying_llm: AsyncLLMEngine,
        history_module: ListModule,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.underlying_llm = underlying_llm
        self.history_module = history_module

    async def hook_prompt_args(self, **prompt_args) -> str:
        return {
            **prompt_args,
            "HISTORY_CONTEXT": self.history_module.get_items(),
        }

    async def query(
        self, metrics: LLMMetrics | None = None, **kwargs
    ) -> tuple[str, int]:
        return await self.underlying_llm.query(metrics=metrics, **kwargs)
