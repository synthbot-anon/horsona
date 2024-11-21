import json
from abc import ABC, abstractmethod

from horsona.autodiff.basic import HorseData
from horsona.config.json_with_comments import load_json_with_comments
from horsona.index.base_index import BaseIndex

INDEX_CONFIG_PATH = "index_config.json"

indices = {}
_loaded_indices = False


def load_indices() -> dict[str, BaseIndex]:
    global _loaded_indices, indices
    from horsona.index.hnsw_index import HnswEmbeddingIndex

    if _loaded_indices:
        return

    with open(INDEX_CONFIG_PATH, "r") as f:
        config = load_json_with_comments(f)

    indices.clear()

    for item in config:
        for name, params in item.items():
            index_type = params["type"]

            if index_type == "HnswEmbeddingIndex":
                embedding = embedding_model_from_config(params["embedding"])
                indices[name] = HnswEmbeddingIndex(model=embedding)
            else:
                raise ValueError(f"Unknown index type: {index_type}")

    _loaded_indices = True
    return indices


def embedding_model_from_config(config: dict) -> "EmbeddingModel":
    from horsona.index.ollama_model import OllamaEmbeddingModel
    from horsona.index.openai_embedding_model import OpenAIEmbeddingModel

    if config["type"] == "OllamaEmbeddingModel":
        model = config["model"]
        url = config.get("url")
        return OllamaEmbeddingModel(model, url=url)
    elif config["type"] == "OpenAIEmbeddingModel":
        model = config["model"]
        return OpenAIEmbeddingModel(model)
    else:
        raise ValueError(f"Unknown embedding model type: {config['type']}")


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
