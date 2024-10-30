import asyncio
from typing import Generic, TypeVar

from horsona.autodiff.basic import HorseData, HorseModule
from horsona.autodiff.variables import Value
from horsona.llm.engine_utils import compile_user_prompt

T = TypeVar("T", bound=HorseData)


class ListModule(HorseModule, Generic[T]):
    """
    A module for maintaining a list of HorseData items of a specific type.

    This module provides functionality to store and retrieve typed items in a list,
    similar to how GistModule maintains a list of summaries.

    Attributes:
        items (list[T]): The list of stored items of type T
        **kwargs: Additional keyword arguments for parent HorseModule
    """

    def __init__(
        self,
        items: list[T] = None,
        max_length=2048,
        item_lengths: list[int] = None,
        **kwargs,
    ):
        """
        Initialize the ListCache module.

        Args:
            items (list[T], optional): Initial list of items to store
            **kwargs: Additional keyword arguments for parent HorseModule
        """
        super().__init__(**kwargs)
        self.items = items if items is not None else []
        self.max_length = max_length
        self.item_lengths = item_lengths

    async def append(self, item: T, **kwargs) -> T:
        """
        Add an item to the list cache.

        Args:
            item (T): The item to append to the cache
            **kwargs: Additional context when appending the item

        Returns:
            T: The appended item
        """
        if self.item_lengths is None:
            item_prompts = await asyncio.gather(
                *[compile_user_prompt(ITEM=item) for item in self.items]
            )
            self.item_lengths = [len(prompt) for prompt in item_prompts]

        item_length = len(await compile_user_prompt(ITEM=item))
        self.items.append(item)
        self.item_lengths.append(item_length)

        while sum(self.item_lengths) > self.max_length:
            self.items.pop(0)
            self.item_lengths.pop(0)

        if len(self.items) == 0:
            return None

        return item

    def get_items(self) -> list[T]:
        """
        Get all items currently stored in the cache.

        Returns:
            list[T]: List of all stored items
        """
        return self.items

    def clear(self):
        """Clear all items from the cache."""
        self.items = []
        self.item_lengths = None
