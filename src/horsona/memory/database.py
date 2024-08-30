from abc import ABC, abstractmethod


class Database(ABC):
    @abstractmethod
    async def update(self, data):
        pass

    @abstractmethod
    async def query(self, query, topk=1) -> dict:
        pass

    @abstractmethod
    async def delete(self, index):
        pass

    @abstractmethod
    async def contains(self, key):
        pass

    @abstractmethod
    async def replace(self, key, value):
        pass

    @abstractmethod
    async def get(self, key):
        pass
