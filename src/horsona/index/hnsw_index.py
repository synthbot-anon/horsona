import os
import tempfile
from typing import Type

import hnswlib

from horsona.autodiff.basic import state_dict
from horsona.index.embedding_index import EmbeddingIndex
from horsona.index.embedding_model import EmbeddingModel


class HnswEmbeddingIndex(EmbeddingIndex):
    def __init__(
        self,
        model: EmbeddingModel,
        index_size: int = None,
        index_to_value: dict = None,
        value_to_index: dict = None,
        indices: list = None,
        embeddings: hnswlib.Index = None,
        deleted_indices: set = None,
        next_index: int = None,
        space: str = None,
        dim: int = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model

        self.index_to_value = index_to_value or {}
        self.value_to_index = value_to_index or {}
        self.indices = indices or []
        self.index_size = index_size or 0
        self.deleted_indices = deleted_indices or set()
        self.next_index = next_index or len(self.index_to_value)
        self.space = space or "l2"
        self.dim = dim or None
        self.embeddings = embeddings

    def state_dict(self) -> dict:
        if self.embeddings is None:
            return super().state_dict()

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Save to the temporary file
            self.embeddings.save_index(temp_path)

            # Read the file contents
            with open(temp_path, "rb") as file:
                index_bytes = file.read()

            return super().state_dict(embeddings=index_bytes)
        finally:
            # Ensure the temporary file is removed
            os.unlink(temp_path)

    @classmethod
    def load_state_dict(
        cls, state_dict: dict, args: dict = {}, debug_prefix: list = []
    ) -> "HnswEmbeddingIndex":
        if state_dict["embeddings"]["data"] is None:
            return super().load_state_dict(state_dict, args, debug_prefix=debug_prefix)

        embeddings_data = state_dict["embeddings"]["data"]
        space = state_dict["space"]["data"]
        dim = state_dict["dim"]["data"]
        index_size = state_dict["index_size"]["data"]

        with tempfile.NamedTemporaryFile(delete=False, mode="wb") as temp_file:
            # Write the bytes to the temporary file
            temp_file.write(embeddings_data)
            temp_path = temp_file.name

        try:
            # Load the index from the temporary file
            embeddings = hnswlib.Index(space=space, dim=dim)
            embeddings.load_index(temp_path, max_elements=index_size)
        finally:
            # Ensure the temporary file is removed
            os.unlink(temp_path)

        return super().load_state_dict(
            state_dict,
            {
                "embeddings": embeddings,
                **args,
            },
            debug_prefix=debug_prefix,
        )

    def _ensure_capacity(self, example_embeddings: list) -> None:
        self.dim = len(example_embeddings[0])
        num_elements = len(example_embeddings) + len(self.index_to_value)
        max_elements = max(4096, num_elements * 2)

        if self.embeddings is None:
            self.embeddings = hnswlib.Index(space=self.space, dim=self.dim)
            self.embeddings.init_index(
                max_elements=max_elements,
                ef_construction=200,
                M=16,
                allow_replace_deleted=True,
            )
        else:
            if num_elements > self.index_size:
                self.embeddings.resize_index(max_elements)
                self.index_size = max_elements

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

    async def extend(self, data: list[str]) -> None:
        if not data:
            return

        new_indices = []
        for value in data:
            if value in self.value_to_index:
                index = self.value_to_index[value]
                new_indices.append(index)
            else:
                new_indices.append(self.next_index)
                self.next_index += 1

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

    async def delete(self, indices: list[int | str] = []) -> list[str]:
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
