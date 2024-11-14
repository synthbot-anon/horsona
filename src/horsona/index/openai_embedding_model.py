from typing import Any, List, Optional

from openai import AsyncOpenAI

from horsona.index.embedding_model import EmbeddingModel


class OpenAIEmbeddingModel(EmbeddingModel):
    def __init__(self, model: str, name: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(name=name)
        self.model = model
        self.kwargs = kwargs

    async def get_data_embeddings(self, sentences: List[str]) -> List[List[float]]:
        client = AsyncOpenAI(**self.kwargs)
        response = await client.embeddings.create(model=self.model, input=sentences)
        return [embedding.embedding for embedding in response.data]

    async def get_query_embeddings(self, sentences: List[str]) -> List[List[float]]:
        return await self.get_data_embeddings(sentences)
