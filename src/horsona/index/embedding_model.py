from abc import ABC, abstractmethod


class EmbeddingModel(ABC):
    @abstractmethod
    async def get_data_embeddings(self, sentences):
        pass

    @abstractmethod
    async def get_query_embeddings(self, sentences):
        pass
