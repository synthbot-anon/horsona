import asyncio
from typing import Generic, TypeVar

from horsona.autodiff.basic import HorseData, HorseModule
from horsona.autodiff.variables import ListValue, Value
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
        min_item_length=256,
        item_lengths: list[int] = None,
        pending_items: list[T] = None,
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
        self.pending_items = pending_items if pending_items is not None else []
        self.item_lengths = item_lengths if item_lengths is not None else []
        self.max_length = max_length
        self.min_item_length = min(max_length, min_item_length)


    async def append(self, item: T, **kwargs) -> T:
        """
        Add an item to the list cache.

        Args:
            item (T): The item to append to the cache
            **kwargs: Additional context when appending the item

        Returns:
            T: The appended item
        """
        if self.items and (not self.item_lengths or len(self.item_lengths) != len(self.items)):
            self.item_lengths = [len(await compile_user_prompt(ITEM=item)) for item in self.items]

        self.pending_items.append(item)

        # Aggregate items if minimum length is reached
        pending_str = await compile_user_prompt(ITEM=self.pending_items)
        pending_length = len(pending_str)
        if pending_length >= self.min_item_length:
            if len(self.pending_items) == 1:
                new_item = self.pending_items[0]
            else:
                new_item = ListValue("Item list", self.pending_items)
            self.items.append(new_item)
            self.item_lengths.append(pending_length)
            self.pending_items = []

        # Remove oldest items until under max length
        while sum(self.item_lengths) + pending_length > self.max_length:
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
        return self.items + self.pending_items

    def clear(self):
        """Clear all items from the cache."""
        self.items = []
        self.pending_items = []
        self.item_lengths = []
