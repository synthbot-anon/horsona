from typing import List, Optional

from ollama import AsyncClient

from horsona.index.embedding_model import EmbeddingModel


class OllamaEmbeddingModel(EmbeddingModel):
    def __init__(
        self, model: str, url: Optional[str] = None, name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)
        self.model = model
        self.url = url

    async def get_data_embeddings(self, sentences: List[str]) -> List[List[float]]:
        client = AsyncClient(host=self.url)
        response = await client.embed(model=self.model, input=sentences)
        return response["embeddings"]

    async def get_query_embeddings(self, sentences: List[str]) -> List[List[float]]:
        return await self.get_data_embeddings(sentences)
