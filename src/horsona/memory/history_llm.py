from typing import TypeVar, Union

from pydantic import BaseModel

from horsona.llm.base_engine import AsyncLLMEngine, LLMMetrics
from horsona.llm.custom_llm import CustomLLMEngine
from horsona.memory.list_module import ListModule

T = TypeVar("T", bound=BaseModel)
S = TypeVar("S", bound=Union[str, T])


class HistoryLLMEngine(CustomLLMEngine):
    def __init__(
        self,
        underlying_llm: AsyncLLMEngine,
        history_module: ListModule,
        **kwargs,
    ):
        super().__init__(underlying_llm, **kwargs)
        self.history_module = history_module

    async def hook_prompt_args(self, **prompt_args) -> str:
        return {
            "HISTORY_CONTEXT": self.history_module.get_items(),
            **prompt_args,
        }
