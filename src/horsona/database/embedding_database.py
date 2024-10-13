from horsona.autodiff.basic import load_state_dict, state_dict
from horsona.database.base_database import Database
from horsona.index.base_index import BaseIndex


class EmbeddingDatabase(Database):
    def __init__(self, llm, index: BaseIndex, data=None, **kwargs):
        super().__init__(llm, **kwargs)
        self.index = index
        if data is not None:
            self.data = data
        else:
            self.data = {}

    async def insert(self, data):
        await self.index.extend(list(data.keys()))
        self.data.update(data)

    async def query(self, query, topk=1) -> dict:
        indices = await self.index.query(query, topk)
        return {key: self.data[key] for key in indices.values() if key in self.data}

    async def delete(self, index):
        deleted_keys = await self.index.delete([index])
        for key in deleted_keys:
            self.data.pop(key)

    async def contains(self, key):
        return key in self.data

    async def update(self, key, value):
        if key not in self.data:
            return
        result = self.data[key]
        self.data[key] = value
        return result

    async def get(self, key):
        return self.data.get(key)

    async def json(self):
        raise NotImplementedError
