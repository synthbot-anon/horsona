import json

import pytest
from dotenv import load_dotenv
from horsona.llm import engines_from_config
from horsona.llm.anthropic_engine import AsyncAnthropicEngine
from horsona.llm.base_engine import AsyncLLMEngine
from horsona.llm.cerebras_engine import AsyncCerebrasEngine
from horsona.llm.fireworks_engine import AsyncFireworksEngine
from horsona.llm.groq_engine import AsyncGroqEngine
from horsona.llm.multi_engine import create_multi_engine
from horsona.llm.openai_engine import AsyncOpenAIEngine
from horsona.llm.together_engine import AsyncTogetherEngine

load_dotenv()


@pytest.fixture(scope="session", autouse=False)
def llm_engines() -> dict[str, AsyncLLMEngine]:
    with open("llm_config.json") as f:
        config = json.load(f)
    return engines_from_config(config)


@pytest.fixture(scope="session", autouse=False)
def cerebras_llama31_70b(llm_engines) -> AsyncLLMEngine:
    return llm_engines["cerebras_llama31_70b"]


@pytest.fixture(scope="session", autouse=False)
def cerebras_llama31_8b(llm_engines) -> AsyncLLMEngine:
    return llm_engines["cerebras_llama31_8b"]


@pytest.fixture(scope="session", autouse=False)
def groq_llama31_70b(llm_engines) -> AsyncLLMEngine:
    return llm_engines["groq_llama31_70b"]


@pytest.fixture(scope="session", autouse=False)
def fireworks_llama31_70b(llm_engines) -> AsyncLLMEngine:
    return llm_engines["fireworks_llama31_70b"]


@pytest.fixture(scope="session", autouse=False)
def openai_gpt4o_mini(llm_engines) -> AsyncLLMEngine:
    return llm_engines["openai_gpt4o_mini"]


@pytest.fixture(scope="session", autouse=False)
def anthropic_claude3_haiku(llm_engines) -> AsyncLLMEngine:
    return llm_engines["anthropic_claude3_haiku"]


@pytest.fixture(scope="session", autouse=False)
def together_llama31_8b(llm_engines) -> AsyncLLMEngine:
    return llm_engines["together_llama31_8b"]


@pytest.fixture(scope="session", autouse=False)
def reasoning_llm(llm_engines) -> AsyncLLMEngine:
    return llm_engines["reasoning_llm"]
