from abc import ABC, abstractmethod
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel

from horsona.autodiff.basic import HorseGradient, HorseVariable
from horsona.llm.base_engine import AsyncLLMEngine


class DatabaseUpdate(BaseModel):
    operation: Literal["UPDATE"] = "UPDATE"
    key: str
    original_data: Optional[str] = None
    errata: Optional[str] = None
    corrected_data: str


class DatabaseDelete(BaseModel):
    operation: Literal["DELETE"] = "DELETE"
    key: str


class DatabaseNoChange(BaseModel):
    operation: Literal["NO_CHANGE"] = "NO_CHANGE"
    key: str


class DatabaseOpGradient(HorseGradient):
    changes: list[Union[DatabaseUpdate, DatabaseDelete, DatabaseNoChange]]


class DatabaseTextGradient(HorseGradient):
    context: Any
    change: Any


class DatabaseInsertGradient(HorseGradient):
    rows: Any


class Database(HorseVariable, ABC):
    """
    An class representing a database.

    This class provides an interface for basic database operations
    and gradient application. It uses an AsyncLLMEngine for processing gradients.

    Attributes:
        llm (AsyncLLMEngine): An asynchronous LLM engine used for processing gradients.

    Parameters:
        llm (AsyncLLMEngine): The LLM engine to be used for processing updates.
        **kwargs: Additional keyword arguments to be passed to the parent HorseVariable class (required_grad, name).

    Gradients:
        This class supports three types of gradients:
        1. DatabaseTextGradient: For context-based text updates.
        2. DatabaseOpGradient: For specific update, delete, or no-change operations.
        3. DatabaseInsertGradient: For inserting new rows into the database.

    Example:
        >>> class MyDatabase(Database):
        ...     # Implement abstract methods
        ...     pass
        >>> llm_engine = AsyncLLMEngine()
        >>> db = MyDatabase(llm_engine)
        >>> await db.insert({"key": "old_value"})
        >>> await db.apply_gradients([DatabaseTextGradient(context={"key": "old_value"}, change="new_value")])
    """

    def __init__(self, llm: AsyncLLMEngine, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm

    @abstractmethod
    async def insert(self, data): ...

    @abstractmethod
    async def query(self, query, **kwargs) -> dict: ...

    @abstractmethod
    async def delete(self, index): ...

    @abstractmethod
    async def contains(self, key): ...

    @abstractmethod
    async def update(self, key, value): ...

    @abstractmethod
    async def get(self, key): ...

    async def apply_gradients(self, gradients: list[HorseGradient]):
        if not gradients:
            return

        all_contexts = dict()
        all_gradients = []
        all_changes = []

        for gradient in gradients:
            if isinstance(gradient, DatabaseTextGradient):
                all_contexts.update(gradient.context)
                all_gradients.append(gradient.change)
            elif isinstance(gradient, DatabaseOpGradient):
                for change in gradient.changes:
                    query, _ = (await self.query(change.key)).popitem()
                    all_changes.append((query, change))
            elif isinstance(gradient, DatabaseInsertGradient):
                await self.insert(gradient.rows.value)

        response = await self.llm.query_object(
            DatabaseOpGradient,
            ERRATA=all_gradients,
            DATASET=all_contexts,
            TASK=(
                "You are maintaining the DATASET with the latest information. "
                "A user provided ERRATA to the DATASET. "
                "Edit the DATASET to address the ERRATA. "
            ),
        )

        for change in response.changes:
            result = await self.query(change.key)
            if not result:
                continue
            query, _ = result.popitem()
            all_changes.append((query, change))

        for query, change in all_changes:
            if not isinstance(change, DatabaseUpdate):
                continue
            await self.update(query, change.corrected_data)

        for query, change in all_changes:
            if not isinstance(change, DatabaseDelete):
                continue
            await self.delete(query)
