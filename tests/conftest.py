import pytest
from dotenv import load_dotenv

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
def cerebras_llama31_70b() -> AsyncLLMEngine:
    return AsyncCerebrasEngine(
        model="llama3.1-70b",
        rate_limits=[
            # Interval, Max Calls, Max Tokens
            (1, 3, 240000),
            (60, 120, 240000),
            (3600, 2600, 4000000),
            (3600 * 24, 57600, 4000000),
        ],
    )


@pytest.fixture(scope="session", autouse=False)
def cerebras_llama31_8b() -> AsyncLLMEngine:
    return AsyncCerebrasEngine(
        model="llama3.1-8b",
        rate_limits=[
            # Interval, Max Calls, Max Tokens
            (1, 3, 240000),
            (60, 120, 240000),
            (3600, 2600, 4000000),
            (3600 * 24, 57600, 4000000),
        ],
    )


@pytest.fixture(scope="session", autouse=False)
def groq_llama31_70b() -> AsyncLLMEngine:
    return AsyncGroqEngine(
        model="llama3-70b-8192",
        rate_limits=[
            # Interval, Max Calls, Max Tokens
            (1, 3, 6000),
            (60, 30, 6000),
            (3600 * 24, 14400, None),
        ],
    )


@pytest.fixture(scope="session", autouse=False)
def fireworks_llama31_70b() -> AsyncLLMEngine:
    return AsyncFireworksEngine(
        model="accounts/fireworks/models/llama-v3p1-70b-instruct",
        rate_limits=[
            # Interval (seconds), Max Calls, Max Tokens
            (1, 3, None),
            (60, 600, None),
        ],
    )


@pytest.fixture(scope="session", autouse=False)
def openai_gpt4o_mini() -> AsyncLLMEngine:
    return AsyncOpenAIEngine(
        model="gpt-4o-mini",
        rate_limits=[
            # Interval (seconds), Max Calls, Max Tokens
            (60, 500, 200000),
            (3600 * 24, 10000, None),
        ],
    )


@pytest.fixture(scope="session", autouse=False)
def anthropic_claude3_haiku() -> AsyncLLMEngine:
    return AsyncAnthropicEngine(
        model="claude-3-haiku-20240307",
        rate_limits=[
            # Interval (seconds), Max Calls, Max Tokens
            (60, 50, 50000),
            (3600 * 24, None, 5000000),
        ],
    )


@pytest.fixture(scope="session", autouse=False)
def together_llama31_8b() -> AsyncLLMEngine:
    return AsyncTogetherEngine(model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")


@pytest.fixture(scope="session", autouse=False)
def reasoning_llm(
    cerebras_llama31_70b,
    fireworks_llama31_70b,
    openai_gpt4o_mini,
    anthropic_claude3_haiku,
) -> AsyncLLMEngine:
    return create_multi_engine(
        cerebras_llama31_70b,
        fireworks_llama31_70b,
        openai_gpt4o_mini,
        anthropic_claude3_haiku,
    )
