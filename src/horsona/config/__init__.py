from typing import TYPE_CHECKING

from horsona.config.json_with_comments import load_json_with_comments

if TYPE_CHECKING:
    from horsona.index.base_index import BaseIndex
    from horsona.index.embedding_model import EmbeddingModel
    from horsona.llm.base_engine import AsyncLLMEngine


LLM_CONFIG_PATH = "llm_config.json"
INDEX_CONFIG_PATH = "index_config.json"

indices: dict[str, "BaseIndex"] = {}
_loaded_indices: bool = False

llms: dict[str, "AsyncLLMEngine"] = {}
_loaded_llms: bool = False


def load_llms() -> dict[str, "AsyncLLMEngine"]:
    """
    Load LLM engine configurations from the config file and instantiate engine objects.

    Returns:
        dict[str, AsyncLLMEngine]: Dictionary mapping engine names to engine instances
    """
    global llms, _loaded_llms

    if _loaded_llms:
        return llms

    from horsona.llm.anthropic_engine import AsyncAnthropicEngine
    from horsona.llm.cerebras_engine import AsyncCerebrasEngine
    from horsona.llm.fireworks_engine import AsyncFireworksEngine
    from horsona.llm.grok_engine import AsyncGrokEngine
    from horsona.llm.groq_engine import AsyncGroqEngine
    from horsona.llm.multi_engine import create_multi_engine
    from horsona.llm.openai_engine import AsyncOpenAIEngine
    from horsona.llm.openrouter_engine import AsyncOpenRouterEngine
    from horsona.llm.perplexity_engine import AsyncPerplexityEngine
    from horsona.llm.together_engine import AsyncTogetherEngine

    with open(LLM_CONFIG_PATH, "r") as f:
        config = load_json_with_comments(f)

    llms.clear()

    # Create engine instances based on config
    for item in config:
        for name, params in item.items():
            engine_type = params["type"]
            model = params.get("model")
            rate_limits = params.get("rate_limits", [])

            if engine_type == "AsyncCerebrasEngine":
                llms[name] = AsyncCerebrasEngine(
                    model=model,
                    rate_limits=rate_limits,
                    name=name,
                )
            elif engine_type == "AsyncGroqEngine":
                llms[name] = AsyncGroqEngine(
                    model=model, rate_limits=rate_limits, name=name
                )
            elif engine_type == "AsyncFireworksEngine":
                llms[name] = AsyncFireworksEngine(
                    model=model, rate_limits=rate_limits, name=name
                )
            elif engine_type == "AsyncOpenAIEngine":
                llms[name] = AsyncOpenAIEngine(
                    model=model, rate_limits=rate_limits, name=name
                )
            elif engine_type == "AsyncAnthropicEngine":
                llms[name] = AsyncAnthropicEngine(
                    model=model, rate_limits=rate_limits, name=name
                )
            elif engine_type == "AsyncTogetherEngine":
                llms[name] = AsyncTogetherEngine(
                    model=model, rate_limits=rate_limits, name=name
                )
            elif engine_type == "AsyncGrokEngine":
                llms[name] = AsyncGrokEngine(
                    model=model, rate_limits=rate_limits, name=name
                )
            elif engine_type == "AsyncOpenRouterEngine":
                llms[name] = AsyncOpenRouterEngine(
                    model=model,
                    url=params.get("url", "https://openrouter.ai/api/v1"),
                    rate_limits=rate_limits,
                    name=name,
                )
            elif engine_type == "AsyncPerplexityEngine":
                llms[name] = AsyncPerplexityEngine(
                    model=model, rate_limits=rate_limits, name=name
                )
            elif engine_type == "MultiEngine":
                sub_engines = [llms[engine_name] for engine_name in params["engines"]]
                llms[name] = create_multi_engine(*sub_engines, name=name)
            elif engine_type == "ReferenceEngine":
                sub_engine = params["reference"]
                llms[name] = llms[sub_engine]
            else:
                raise ValueError(f"Unknown engine type: {engine_type}")

    _loaded_llms = True
    return llms


def load_indices() -> dict[str, "BaseIndex"]:
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
                embedding = _embedding_model_from_config(params["embedding"])
                indices[name] = HnswEmbeddingIndex(model=embedding)
            else:
                raise ValueError(f"Unknown index type: {index_type}")

    _loaded_indices = True
    return indices


def _embedding_model_from_config(config: dict) -> "EmbeddingModel":
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


def get_index(name: str) -> "BaseIndex | EmbeddingModel":
    load_indices()
    return indices[name]


def get_llm(name: str) -> "AsyncLLMEngine":
    load_llms()
    return llms[name]
