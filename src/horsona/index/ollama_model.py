from ollama import AsyncClient

from horsona.index.embedding_model import EmbeddingModel


class OllamaEmbeddingModel(EmbeddingModel):
    def __init__(self, model, url=None):
        super().__init__()
        self.model = model
        self.url = url

    async def get_data_embeddings(self, sentences):
        client = AsyncClient(host=self.url)
        response = await client.embed(model=self.model, input=sentences)
        return response["embeddings"]

    async def get_query_embeddings(self, sentences):
        return await self.get_data_embeddings(sentences)
