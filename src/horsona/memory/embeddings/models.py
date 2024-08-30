import torch
from transformers import AutoModel, AutoTokenizer

from horsona.memory.embeddings.index import EmbeddingModel


class HuggingFaceBGEModel(EmbeddingModel):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.embedding_model = AutoModel.from_pretrained(model)
        self.embedding_model.eval()

    async def get_data_embeddings(self, sentences):
        encoded_input = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        )

        with torch.no_grad():
            model_output = self.embedding_model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(
            sentence_embeddings, p=2, dim=1
        )
        return sentence_embeddings

    async def get_query_embeddings(self, sentences):
        return await self.get_data_embeddings(sentences)
