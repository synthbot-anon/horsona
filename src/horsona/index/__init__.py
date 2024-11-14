from horsona.index.base_index import BaseIndex
from horsona.index.embedding_model import EmbeddingModel, indices, load_indices


def configure_config_path(path: str) -> None:
    import horsona.index

    horsona.index.INDEX_CONFIG_PATH = path


def get_index(name: str) -> BaseIndex | EmbeddingModel:
    load_indices({})
    return indices[name]
