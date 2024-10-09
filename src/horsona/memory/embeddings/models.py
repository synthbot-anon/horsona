from abc import ABC, abstractmethod

from ollama import AsyncClient


class EmbeddingModel(ABC):
    @abstractmethod
    async def get_data_embeddings(self, sentences):
        pass

    @abstractmethod
    async def get_query_embeddings(self, sentences):
        pass


class OllamaEmbeddingModel(EmbeddingModel):
    def __init__(self, model):
        super().__init__()
        self.model = model

    async def get_data_embeddings(self, sentences):
        client = AsyncClient()
        response = await client.embed(model=self.model, input=sentences)
        return response["embeddings"]

    async def get_query_embeddings(self, sentences):
        return await self.get_data_embeddings(sentences)
