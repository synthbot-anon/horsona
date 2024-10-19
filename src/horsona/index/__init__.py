from horsona.index.base_index import BaseIndex
from horsona.index.hnsw_index import HnswEmbeddingIndex
from horsona.index.ollama_model import OllamaEmbeddingModel
from horsona.index.openai_embedding_model import OpenAIEmbeddingModel


def indices_from_config(config: dict) -> dict[str, BaseIndex]:
    indices = {}

    for item in config:
        for name, params in item.items():
            index_type = params["type"]

            if index_type == "HnswEmbeddingIndex":
                embedding = embedding_model_from_config(params["embedding"])
                indices[name] = HnswEmbeddingIndex(model=embedding)
            else:
                raise ValueError(f"Unknown index type: {index_type}")

    return indices


def embedding_model_from_config(config: dict):
    if config["type"] == "OllamaEmbeddingModel":
        model = config["model"]
        url = config.get("url")
        return OllamaEmbeddingModel(model, url=url)
    elif config["type"] == "OpenAIEmbeddingModel":
        model = config["model"]
        return OpenAIEmbeddingModel(model)
    else:
        raise ValueError(f"Unknown embedding model type: {config['type']}")
