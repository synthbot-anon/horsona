from typing import Generic, TypeVar

from horsona.autodiff.basic import HorseData, HorseModule
from horsona.autodiff.variables import ListValue
from horsona.llm.base_engine import AsyncLLMEngine
from horsona.memory.gist_module import GistModule
from horsona.memory.list_module import ListModule

T = TypeVar("T", bound=HorseData)


class LogModule(HorseModule, Generic[T]):
    """
    Maintains memory of a conversation using recent memory and overview modules.

    Attributes:
        recent_messages_module (ListModule): Stores recent conversation history
        overview_module (GistModule): Stores high-level conversation overview
    """

    def __init__(
        self,
        llm: AsyncLLMEngine,
        recent_messages_module: ListModule = None,
        overview_module: GistModule = None,
        **kwargs,
    ):
        """
        Initialize conversation memory modules.

        Args:
            llm (AsyncLLMEngine): Base LLM engine to use for memory modules
            recent_messages_module (ListModule, optional): Module for storing recent messages
            overview_module (GistModule, optional): Module for storing conversation overview
            **kwargs: Additional keyword arguments for parent HorseModule
        """
        super().__init__(**kwargs)
        self.recent_messages_module = recent_messages_module or ListModule()
        self.overview_module = overview_module or GistModule(llm)

    async def append(self, item: T) -> T | ListValue:
        """
        Add text to memory modules.

        Args:
            item (T): Item to add to memory
        """
        chunked_item = await self.recent_messages_module.append(item)
        if chunked_item is not None:
            await self.overview_module.append(chunked_item)

        return chunked_item
