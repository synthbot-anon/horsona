import json
from abc import ABC, abstractmethod

from horsona.autodiff.basic import HorseData
from horsona.config import indices, load_indices


class EmbeddingModel(HorseData, ABC):
    def __init__(self, name: str | None = None) -> None:
        super().__init__()
        self.name = name

    def state_dict(self, **override: dict) -> dict:
        if self.name is not None:
            if override:
                raise ValueError(
                    "Cannot override fields when saving an EmbeddingModel by name"
                )
            return {
                "name": self.name,
            }
        else:
            return super().state_dict(**override)

    @classmethod
    def load_state_dict(
        cls, state_dict: dict, args: dict = {}, debug_prefix: list = []
    ) -> "EmbeddingModel":
        if isinstance(state_dict["name"], str):
            if args:
                raise ValueError(
                    "Cannot override fields when creating an EmbeddingModel by name"
                )
            load_indices()  # Ensure indices are loaded
            return indices[state_dict["name"]]
        else:
            return super().load_state_dict(state_dict, args, debug_prefix)

    @abstractmethod
    async def get_data_embeddings(self, sentences: list[str]) -> list[list[float]]:
        pass

    @abstractmethod
    async def get_query_embeddings(self, sentences: list[str]) -> list[list[float]]:
        pass
