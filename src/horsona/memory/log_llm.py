from typing import TypeVar, Union

from pydantic import BaseModel

from horsona.llm.base_engine import AsyncLLMEngine
from horsona.memory.history_llm import HistoryLLMEngine
from horsona.memory.log_module import LogModule
from horsona.memory.readagent_llm import ReadAgentLLMEngine

T = TypeVar("T", bound=BaseModel)
S = TypeVar("S", bound=Union[str, T])


class LogLLMEngine(ReadAgentLLMEngine):
    def __init__(
        self,
        underlying_llm: AsyncLLMEngine,
        conversation_module: LogModule,
        max_pages: int = 3,
        **kwargs,
    ):
        self.conversation_module = conversation_module

        # LLM with recent messages context
        self.recent_messages_llm = HistoryLLMEngine(
            underlying_llm, conversation_module.recent_messages_module
        )

        # LLM with overview context
        self.overview_llm = ReadAgentLLMEngine(
            underlying_llm,
            conversation_module.overview_module,
            max_pages=max_pages,
        )

        super().__init__(
            self.recent_messages_llm,
            self.conversation_module.overview_module,
            max_pages=max_pages,
        )
