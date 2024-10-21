from horsona.llm.base_engine import engines, load_engines


def configure_config_path(path: str):
    import horsona.llm

    horsona.llm.LLM_CONFIG_PATH = path


def get_llm_engine(name: str):
    load_engines()
    return engines[name]
