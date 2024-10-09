import hnswlib

from horsona.memory.embeddings.index import EmbeddingIndex
from horsona.memory.embeddings.models import EmbeddingModel


class HnswEmbeddingIndex(EmbeddingIndex):
    def __init__(self, model: EmbeddingModel, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.index_size = 0

        self.index_to_value = {}
        self.value_to_index = {}
        self.indices = []

        self.embeddings = None
        self.deleted_indices = set()

        self._next_index = 0

    def _ensure_capacity(self, example_embeddings):
        dim = len(example_embeddings[0])
        num_elements = len(example_embeddings) + len(self.index_to_value)
        max_elements = max(4096, num_elements * 2)

        if self.embeddings is None:
            self.embeddings = hnswlib.Index(space="l2", dim=dim)
            self.embeddings.init_index(
                max_elements=max_elements,
                ef_construction=200,
                M=16,
                allow_replace_deleted=True,
            )
        else:
            if num_elements > self.index_size:
                self.embeddings.resize_index(max_elements)

    async def query(self, query: str, topk: int) -> dict:
        if self.embeddings is None:
            return {}

        if not query:
            return {}

        if topk == 0:
            return {}

        query_emb = await self.model.get_query_embeddings([query])
        indices, distances = self.embeddings.knn_query(query_emb, k=topk)
        values = [self.index_to_value[i] for i in indices[0]]

        return dict(zip(indices[0], values))

    async def extend(self, data: list[str]):
        if not data:
            return

        new_indices = []
        for value in data:
            if value in self.value_to_index:
                index = self.value_to_index[value]
                new_indices.append(index)
            else:
                new_indices.append(self._next_index)
                self._next_index += 1

        # Update values, indices, and embeddings
        new_embeddings = await self.model.get_data_embeddings(data)
        self._ensure_capacity(new_embeddings)

        self.index_to_value.update(dict(zip(new_indices, data)))
        self.value_to_index.update(dict(zip(data, new_indices)))
        for i in new_indices:
            if i in self.deleted_indices:
                self.deleted_indices.remove(i)

        self.embeddings.add_items(
            new_embeddings,
            new_indices,
            replace_deleted=True,
        )

    async def delete(self, indices: list[int | str] = []):
        if not indices:
            return []

        deleted_values = []
        for value in indices:
            if isinstance(value, int):
                if value in self.deleted_indices:
                    continue
                self.embeddings.mark_deleted(value)
                self.deleted_indices.add(value)
                deleted_values.append(self.index_to_value[value])
            if value in self.value_to_index:
                index = self.value_to_index[value]
                if index in self.deleted_indices:
                    continue
                self.embeddings.mark_deleted(index)
                self.deleted_indices.add(index)
                deleted_values.append(value)

        return deleted_values
