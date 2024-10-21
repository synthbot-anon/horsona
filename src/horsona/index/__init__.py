from horsona.index.embedding_model import indices, load_indices


def configure_config_path(path: str):
    import horsona.index

    horsona.index.INDEX_CONFIG_PATH = path


def get_index(name: str):
    load_indices({})
    return indices[name]
