from horsona.memory.database import Database
from horsona.memory.embeddings.index import EmbeddingIndex


class EmbeddingDatabase(Database):
    def __init__(self, index: EmbeddingIndex):
        self.data = {}
        self.index = index

    async def update(self, data):
        await self.index.extend(list(data.keys()))
        self.data.update(data)

    async def query(self, query, topk=1) -> dict:
        return await self.index.query(query, topk)

    async def delete(self, index):
        deleted_keys = await self.index.delete([index])
        for key in deleted_keys:
            self.data.pop(key)

    async def contains(self, key):
        return key in self.data

    async def replace(self, key, value):
        if key not in self.data:
            return
        result = self.data[key]
        self.data[key] = value
        return result

    async def get(self, key):
        return self.data.get(key)
