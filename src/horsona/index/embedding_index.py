from abc import ABC, abstractmethod

from horsona.index.base_index import BaseIndex


class EmbeddingIndex(BaseIndex[str], ABC):
    async def query(self, query: str, topk: int) -> dict[str, str]:
        result_with_weights = await self.query_with_weights(query, topk)
        return {k: v[0] for k, v in result_with_weights.items()}

    @abstractmethod
    async def query_with_weights(
        self, query: str, topk: int
    ) -> dict[str, tuple[str, float]]: ...

    @abstractmethod
    async def extend(self, data: list[str]) -> None: ...

    @abstractmethod
    async def delete(self, indices: list[int | str] = []) -> None: ...
