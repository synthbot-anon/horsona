from collections import defaultdict
from typing import Any

from horsona.database.base_database import Database
from horsona.index.base_index import BaseIndex
from horsona.llm.base_engine import AsyncLLMEngine


class EmbeddingDatabase(Database):
    def __init__(
        self, llm: AsyncLLMEngine, index: BaseIndex, data: dict = None, **kwargs
    ):
        super().__init__(llm, **kwargs)
        self.index = index
        if data is not None:
            self.data = defaultdict(lambda: [], data)
        else:
            self.data = defaultdict(lambda: [])

    async def insert(self, data: dict) -> None:
        await self.index.extend(list(data.keys()))
        for key, value in data.items():
            self.data[key].append(value)

    async def query(self, query: str, topk: int = 1) -> dict:
        result_with_weights = await self.query_with_weights(query, topk)
        return {k: v[0] for k, v in result_with_weights.items()}

    async def query_with_weights(self, query: str, topk: int = 1) -> dict:
        indices = await self.index.query_with_weights(query, topk)
        return {
            key: (self.data[key], weight)
            for key, weight in indices.values()
            if key in self.data
        }

    async def delete(self, index: str) -> None:
        deleted_keys = await self.index.delete([index])
        for key in deleted_keys:
            self.data.pop(key)

    async def contains(self, key: str) -> bool:
        return key in self.data

    async def update(self, key: str, value: Any) -> Any:
        if key not in self.data:
            return
        result = self.data[key]
        self.data[key] = value
        return result

    async def get(self, key: str) -> Any:
        return self.data.get(key)

    async def json(self) -> dict:
        raise NotImplementedError
