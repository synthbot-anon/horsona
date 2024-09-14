import asyncio
from abc import ABC, abstractmethod
from typing import Literal, Union

import torch
from pydantic import BaseModel

from horsona.autodiff.basic import HorseGradient, HorseVariable


class IndexInsert(BaseModel):
    operation: Literal["INSERT"]
    value: str


class IndexDelete(BaseModel):
    operation: Literal["DELETE"]
    index: int


class IndexChanges(HorseGradient):
    changes: list[Union[IndexInsert, IndexDelete]]


class EmbeddingModel(ABC):
    @abstractmethod
    async def get_data_embeddings(self, sentences):
        pass

    @abstractmethod
    async def get_query_embeddings(self, sentences):
        pass


class EmbeddingIndex(HorseVariable):
    def __init__(self, description, model: EmbeddingModel, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.values = {}
        self.reverse_values = {}
        self.embeddings = None
        self.indices = []
        self._known_data = set()
        self._next_index = 0
        self.description = description
        self.recent_queries = []
        self.recent_changes = []

    async def json(self):
        return {
            "count": len(self.value),
            "index_type": "cosine similarity",
            "description": self.description,
            "recent_queries": self.recent_queries,
            "recent_changes": self.recent_changes,
        }

    async def apply_gradients(self, gradients: list[IndexChanges]):
        insertions = []
        deletions = []

        print("gradients:", gradients)

        for grad in gradients:
            for change in grad.changes:
                if isinstance(change, IndexInsert):
                    insertions.append(change.value)
                elif isinstance(change, IndexDelete):
                    deletions.append(change.index)

        await asyncio.gather(
            self.delete(deletions),
            self.extend(insertions),
        )

    async def query(self, query: str, topk: int) -> dict:
        if not query:
            return {}

        if not self.values:
            return {}

        query_emb = await self.model.get_query_embeddings([query])
        matches = (self.embeddings @ query_emb.T).squeeze()
        topk = min(topk, len(matches))
        locations = torch.topk(matches, topk).indices.tolist()
        indices = [self.indices[i] for i in locations]
        values = [self.values[i] for i in indices]

        return dict(zip(indices, values))

    async def extend(self, data: list[str]):
        if not data:
            return

        # Only add the new values
        new_data = {}
        for value in data:
            if value in self._known_data:
                continue
            new_data[self._next_index] = value
            self._next_index += 1

        if not new_data:
            return

        # Update values, indices, and embeddings
        new_indices = []
        new_values = []
        for item in new_data.items():
            new_indices.append(item[0])
            new_values.append(item[1])
            self.reverse_values[item[1]] = item[0]

        self.values.update(new_data)
        self.indices.extend(new_indices)

        new_embeddings = await self.model.get_data_embeddings(new_values)

        if self.embeddings == None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = torch.cat([self.embeddings, new_embeddings])

    async def delete(self, indices: list[int | str] = []):
        if not indices:
            return []

        if not self.values:
            return []

        for i, value in enumerate(indices):
            if isinstance(value, int):
                continue
            if value in self.reverse_values:
                indices[i] = self.reverse_values[value]

        # Figure out which items to delete
        indices = indices or []
        locations = []
        for i in indices:
            try:
                locations.append(self.indices.index(i))
            except:
                pass

        # Delete the values and indices at the specified locations
        deleted_values = []
        for idx, loc in sorted(
            zip(indices, locations), reverse=True, key=lambda x: x[1]
        ):
            deleted_values.append(self.values[idx])
            del self.reverse_values[self.values[idx]]
            del self.values[idx]
            del self.indices[loc]

        # Delete the embeddings at the specified locations
        locations = torch.tensor(locations)
        retain_indices = torch.ones(self.embeddings.shape[0], dtype=torch.bool)
        retain_indices[locations] = False
        self.embeddings = self.embeddings[retain_indices]

        return deleted_values

    async def update(self, values: dict[int, str]):
        raise NotImplementedError
