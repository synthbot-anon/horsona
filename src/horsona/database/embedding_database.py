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
            self.data = data
        else:
            self.data = {}

    async def insert(self, data: dict) -> None:
        await self.index.extend(list(data.keys()))
        self.data.update(data)

    async def query(self, query: str, topk: int = 1) -> dict:
        indices = await self.index.query(query, topk)
        return {key: self.data[key] for key in indices.values() if key in self.data}

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
