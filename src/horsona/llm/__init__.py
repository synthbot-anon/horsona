from horsona.llm.base_engine import AsyncLLMEngine, engines, load_engines


def configure_config_path(path: str) -> None:
    import horsona.llm

    horsona.llm.LLM_CONFIG_PATH = path


def get_llm_engine(name: str) -> AsyncLLMEngine:
    load_engines()
    return engines[name]
