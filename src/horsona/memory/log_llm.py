from typing import Type, TypeVar, Union

from horsona.llm.base_engine import AsyncLLMEngine
from horsona.memory.history_llm import HistoryLLMEngine
from horsona.memory.log_module import LogModule
from horsona.memory.readagent_llm import ReadAgentLLMEngine
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)
S = TypeVar("S", bound=Union[str, T])


class LogLLMEngine(AsyncLLMEngine):
    def __init__(
        self,
        underlying_llm: AsyncLLMEngine,
        conversation_module: LogModule,
        max_pages: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.underlying_llm = underlying_llm
        self.conversation_module = conversation_module
        self.max_pages = max_pages

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

        # LLM with both recent messages and overview context
        self.conversation_llm = ReadAgentLLMEngine(
            self.recent_messages_llm,
            conversation_module.overview_module,
            max_pages=max_pages,
        )

    async def query(self, **kwargs) -> tuple[str, int]:
        return await self.underlying_llm.query(**kwargs), 0

    async def query_object(self, response_model: Type[T], **kwargs) -> T:
        """
        Query conversation LLM and parse response into a pydantic model, with conversation context.

        Args:
            response_model: Pydantic model class to parse response into
            **kwargs: Additional query arguments

        Returns:
            Parsed response as specified pydantic model
        """
        return await self.conversation_llm.query_object(
            response_model,
            **kwargs,
        )

    async def query_block(self, block_type: str, **kwargs) -> str:
        """
        Query conversation LLM for a specific block type, with conversation context.

        Args:
            block_type: Type of block to generate (e.g. "text", "md")
            **kwargs: Additional query arguments

        Returns:
            Generated block as string
        """
        return await self.conversation_llm.query_block(
            block_type,
            **kwargs,
        )

    async def query_continuation(self, prompt: str, **kwargs) -> str:
        return await self.conversation_llm.query_continuation(
            prompt,
            **kwargs,
        )
