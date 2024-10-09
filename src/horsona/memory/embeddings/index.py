import asyncio
from abc import ABC, abstractmethod
from typing import Literal, Union

from pydantic import BaseModel

from horsona.autodiff.basic import HorseGradient, HorseVariable


class IndexInsert(BaseModel):
    operation: Literal["INSERT"]
    value: str


class IndexDelete(BaseModel):
    operation: Literal["DELETE"]
    index: int


class IndexChanges(HorseGradient):
    changes: list[Union[IndexInsert, IndexDelete]]


class EmbeddingIndex(HorseVariable, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def apply_gradients(self, gradients: list[IndexChanges]):
        insertions = []
        deletions = []

        for grad in gradients:
            for change in grad.changes:
                if isinstance(change, IndexInsert):
                    insertions.append(change.value)
                elif isinstance(change, IndexDelete):
                    deletions.append(change.index)

        await asyncio.gather(
            self.delete(deletions),
            self.extend(insertions),
        )

    @abstractmethod
    async def query(self, query: str, topk: int) -> dict:
        ...

    @abstractmethod
    async def extend(self, data: list[str]):
        ...

    @abstractmethod
    async def delete(self, indices: list[int | str] = []):
        ...
