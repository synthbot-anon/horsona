from horsona.llm.anthropic_engine import AsyncAnthropicEngine
from horsona.llm.base_engine import AsyncLLMEngine
from horsona.llm.cerebras_engine import AsyncCerebrasEngine
from horsona.llm.fireworks_engine import AsyncFireworksEngine
from horsona.llm.groq_engine import AsyncGroqEngine
from horsona.llm.multi_engine import create_multi_engine
from horsona.llm.openai_engine import AsyncOpenAIEngine
from horsona.llm.together_engine import AsyncTogetherEngine


def engines_from_config(config: dict) -> dict[str, AsyncLLMEngine]:
    engines = {}

    for item in config:
        for name, params in item.items():
            engine_type = params["type"]
            model = params.get("model")
            rate_limits = params.get("rate_limits", [])

            if engine_type == "AsyncCerebrasEngine":
                engines[name] = AsyncCerebrasEngine(
                    model=model, rate_limits=rate_limits
                )
            elif engine_type == "AsyncGroqEngine":
                engines[name] = AsyncGroqEngine(model=model, rate_limits=rate_limits)
            elif engine_type == "AsyncFireworksEngine":
                engines[name] = AsyncFireworksEngine(
                    model=model, rate_limits=rate_limits
                )
            elif engine_type == "AsyncOpenAIEngine":
                engines[name] = AsyncOpenAIEngine(model=model, rate_limits=rate_limits)
            elif engine_type == "AsyncAnthropicEngine":
                engines[name] = AsyncAnthropicEngine(
                    model=model, rate_limits=rate_limits
                )
            elif engine_type == "AsyncTogetherEngine":
                engines[name] = AsyncTogetherEngine(
                    model=model, rate_limits=rate_limits
                )
            elif engine_type == "MultiEngine":
                sub_engines = [
                    engines[engine_name] for engine_name in params["engines"]
                ]
                engines[name] = create_multi_engine(*sub_engines)
            else:
                raise ValueError(f"Unknown engine type: {engine_type}")

    return engines
