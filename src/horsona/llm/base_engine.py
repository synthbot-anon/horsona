import asyncio
import functools
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Optional, Type, TypeVar, Union

from pydantic import BaseModel

from horsona.autodiff.basic import HorseData
from horsona.config.json_with_comments import load_json_with_comments
from horsona.llm.limits import CallLimit, TokenLimit

__all__ = ["AsyncLLMEngine", "LLM_CONFIG_PATH"]

LLM_CONFIG_PATH = "llm_config.json"

T = TypeVar("T", bound=BaseModel)
S = TypeVar("S", bound=Union[str, T])

engines: dict[str, "AsyncLLMEngine"] = {}
_loaded_engines: bool = False


def load_engines() -> dict[str, "AsyncLLMEngine"]:
    """
    Load LLM engine configurations from the config file and instantiate engine objects.

    Returns:
        dict[str, AsyncLLMEngine]: Dictionary mapping engine names to engine instances
    """
    global engines, _loaded_engines

    if _loaded_engines:
        return engines

    from horsona.llm.anthropic_engine import AsyncAnthropicEngine
    from horsona.llm.cerebras_engine import AsyncCerebrasEngine
    from horsona.llm.fireworks_engine import AsyncFireworksEngine
    from horsona.llm.grok_engine import AsyncGrokEngine
    from horsona.llm.groq_engine import AsyncGroqEngine
    from horsona.llm.multi_engine import create_multi_engine
    from horsona.llm.openai_engine import AsyncOpenAIEngine
    from horsona.llm.perplexity_engine import AsyncPerplexityEngine
    from horsona.llm.together_engine import AsyncTogetherEngine

    with open(LLM_CONFIG_PATH, "r") as f:
        config = load_json_with_comments(f)

    engines.clear()

    # Create engine instances based on config
    for item in config:
        for name, params in item.items():
            engine_type = params["type"]
            model = params.get("model")
            rate_limits = params.get("rate_limits", [])

            if engine_type == "AsyncCerebrasEngine":
                engines[name] = AsyncCerebrasEngine(
                    model=model,
                    rate_limits=rate_limits,
                    name=name,
                )
            elif engine_type == "AsyncGroqEngine":
                engines[name] = AsyncGroqEngine(
                    model=model, rate_limits=rate_limits, name=name
                )
            elif engine_type == "AsyncFireworksEngine":
                engines[name] = AsyncFireworksEngine(
                    model=model, rate_limits=rate_limits, name=name
                )
            elif engine_type == "AsyncOpenAIEngine":
                engines[name] = AsyncOpenAIEngine(
                    model=model, rate_limits=rate_limits, name=name
                )
            elif engine_type == "AsyncAnthropicEngine":
                engines[name] = AsyncAnthropicEngine(
                    model=model, rate_limits=rate_limits, name=name
                )
            elif engine_type == "AsyncTogetherEngine":
                engines[name] = AsyncTogetherEngine(
                    model=model, rate_limits=rate_limits, name=name
                )
            elif engine_type == "AsyncGrokEngine":
                engines[name] = AsyncGrokEngine(
                    model=model, rate_limits=rate_limits, name=name
                )
            elif engine_type == "AsyncPerplexityEngine":
                engines[name] = AsyncPerplexityEngine(
                    model=model, rate_limits=rate_limits, name=name
                )
            elif engine_type == "MultiEngine":
                sub_engines = [
                    engines[engine_name] for engine_name in params["engines"]
                ]
                engines[name] = create_multi_engine(*sub_engines, name=name)
            elif engine_type == "ReferenceEngine":
                sub_engine = params["reference"]
                engines[name] = engines[sub_engine]
            else:
                raise ValueError(f"Unknown engine type: {engine_type}")

    _loaded_engines = True
    return engines


class RateLimits(HorseData):
    """
    Manages both call-based and token-based rate limits.
    """

    def __init__(
        self,
        limits: list[dict[str, float]],
        call_limits: Optional[list[CallLimit]] = None,
        token_limits: Optional[list[TokenLimit]] = None,
    ) -> None:
        self.limits = limits
        self.call_limits: list[CallLimit] = []
        self.token_limits: list[TokenLimit] = []

        # Initialize rate limits from config
        for rate_limit in limits:
            interval = rate_limit["interval"]
            calls = rate_limit.get("max_calls")
            tokens = rate_limit.get("max_tokens")
            if calls is not None:
                self.call_limits.append(CallLimit(calls, interval))
            if tokens is not None:
                self.token_limits.append(TokenLimit(tokens, interval))

    async def consume_call(self) -> None:
        """Record consumption of one API call across all limits."""
        await asyncio.gather(*[limit.consume() for limit in self.call_limits])

    def report_tokens_consumed(self, count: int) -> None:
        """Record token consumption across all limits."""
        for limit in self.token_limits:
            limit.report_consumed(count)

    async def wait_for(self, expected_tokens: Optional[int] = None) -> None:
        """Wait until both call and token consumption is allowed."""
        await asyncio.gather(
            *[limit.wait_for() for limit in self.call_limits],
            *[limit.wait_for(expected_tokens) for limit in self.token_limits],
        )

    def next_allowed(self, expected_tokens: Optional[int] = None) -> float:
        """Return timestamp when both call and token consumption will be allowed."""
        if self.call_limits:
            next_call = max(limit.next_allowed() for limit in self.call_limits)
        else:
            next_call = time.time()

        if not expected_tokens:
            return next_call

        if self.token_limits:
            next_token = max(
                limit.next_allowed(expected_tokens) for limit in self.token_limits
            )
        else:
            next_token = time.time()

        return max(next_call, next_token)


class TokenLimitException(Exception):
    """Raised when token rate limits are exceeded."""

    pass


class CallLimitException(Exception):
    """Raised when API call rate limits are exceeded."""

    pass


@dataclass
class LLMMetrics:
    """Tracks metrics for LLM API usage."""

    tokens_consumed: int = 0


def tracks_metrics(fn):
    """
    Decorator that tracks token consumption metrics for LLM API calls.
    Updates both the rate limiter and optional metrics object.
    """

    @functools.wraps(fn)
    async def wrapper(
        self: "AsyncLLMEngine", *args: Any, **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        orig_metrics = kwargs.pop("metrics", None)
        new_metrics = LLMMetrics()

        async for chunk in fn(self, *args, metrics=new_metrics, **kwargs):
            new_consumed = new_metrics.tokens_consumed - new_metrics.tokens_consumed
            self.rate_limit.report_tokens_consumed(new_consumed)
            if orig_metrics is not None:
                orig_metrics.tokens_consumed += new_consumed
            yield chunk

    return wrapper


class AsyncLLMEngine(HorseData, ABC):
    """
    Base class for asynchronous Language Learning Model (LLM) engines.

    Provides a standardized interface for:
    - Making raw queries to LLM APIs
    - Getting responses as structured objects
    - Getting responses in specific block formats
    - Rate limiting and usage tracking

    Subclasses must implement the abstract query methods for specific LLM APIs.
    """

    def __init__(
        self,
        rate_limits: list[dict[str, float]] = [],
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the engine with rate limits and optional name.

        Args:
            rate_limits: List of rate limit configurations
            name: Optional name for the engine instance
            **kwargs: Additional engine-specific arguments
        """
        super().__init__()
        self.rate_limit = RateLimits(rate_limits)
        self.name = name

    def state_dict(self, **override: Any) -> dict[str, Any]:
        """
        Get serializable state of the engine.
        Named engines are serialized by reference.
        """
        if self.name is not None:
            if override:
                raise ValueError(
                    "Cannot override fields when saving an AsyncLLMEngine by name"
                )
            return {
                "name": self.name,
            }
        else:
            return super().state_dict(**override)

    @classmethod
    def load_state_dict(
        cls,
        state_dict: dict[str, Any],
        args: dict[str, Any] = {},
        debug_prefix: list[str] = [],
    ) -> "AsyncLLMEngine":
        """
        Load engine from serialized state.
        Named engines are loaded from the global engines registry.
        """
        if isinstance(state_dict["name"], str):
            if args:
                raise ValueError(
                    "Cannot override fields when creating an AsyncLLMEngine by name"
                )
            load_engines()
            return engines[state_dict["name"]]
        else:
            return super().load_state_dict(state_dict, args, debug_prefix)

    @abstractmethod
    async def query_response(self, **kwargs: Any) -> tuple[str, int]:
        """
        Send a query to the LLM and get the complete response.

        Args:
            **kwargs: API-specific arguments (e.g. max_tokens, temperature)

        Returns:
            tuple[str, int]: Response text and tokens consumed
        """
        ...

    @abstractmethod
    async def query_stream(self, **kwargs: Any) -> AsyncGenerator[str, None]:
        """
        Send a query to the LLM and stream the response chunks.

        Args:
            **kwargs: API-specific arguments

        Yields:
            str: Response text chunks as they arrive
        """
        ...

    @abstractmethod
    async def query_object(self, response_model: Type[T], **kwargs: Any) -> T:
        """
        Query the LLM and parse the response into a structured object.

        Args:
            response_model: Pydantic model class to parse response into
            **kwargs: Prompt args (UPPERCASE) and API args (lowercase)

        Returns:
            T: Response parsed into response_model instance
        """
        ...

    @abstractmethod
    async def query_block(self, block_type: str, **kwargs: Any) -> str:
        """
        Query the LLM for a specific block type response.

        Args:
            block_type: Type of block to request (e.g. "python", "json")
            **kwargs: Prompt args (UPPERCASE) and API args (lowercase)

        Returns:
            str: Response formatted as requested block type
        """
        ...

    @abstractmethod
    async def query_continuation(self, prompt: str, **kwargs: Any) -> str:
        """
        Query the LLM to continue a prompt.

        Args:
            prompt: Text prompt to continue
            **kwargs: API-specific arguments

        Returns:
            str: Generated continuation of the prompt
        """
        ...
