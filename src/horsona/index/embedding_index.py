from abc import ABC, abstractmethod

from horsona.index.base_index import BaseIndex


class EmbeddingIndex(BaseIndex[str], ABC):
    @abstractmethod
    async def query(self, query: str, topk: int) -> dict: ...

    @abstractmethod
    async def extend(self, data: list[str]): ...

    @abstractmethod
    async def delete(self, indices: list[int | str] = []): ...
