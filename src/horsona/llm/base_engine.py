import asyncio
import functools
import json
import time
from abc import ABC, abstractmethod
from typing import Type, TypeVar, Union

from pydantic import BaseModel

from horsona.autodiff.basic import HorseData

__all__ = ["AsyncLLMEngine", "LLM_CONFIG_PATH"]

LLM_CONFIG_PATH = "llm_config.json"

T = TypeVar("T", bound=BaseModel)
S = TypeVar("S", bound=Union[str, T])

engines = {}
_loaded_engines = False


def load_engines() -> dict[str, "AsyncLLMEngine"]:
    global engines, _loaded_engines

    if _loaded_engines:
        return

    from horsona.llm.anthropic_engine import AsyncAnthropicEngine
    from horsona.llm.cerebras_engine import AsyncCerebrasEngine
    from horsona.llm.fireworks_engine import AsyncFireworksEngine
    from horsona.llm.groq_engine import AsyncGroqEngine
    from horsona.llm.multi_engine import create_multi_engine
    from horsona.llm.openai_engine import AsyncOpenAIEngine
    from horsona.llm.perplexity_engine import AsyncPerplexityEngine
    from horsona.llm.together_engine import AsyncTogetherEngine

    with open(LLM_CONFIG_PATH, "r") as f:
        config = json.load(f)

    engines.clear()

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
            elif engine_type == "AsyncPerplexityEngine":
                engines[name] = AsyncPerplexityEngine(
                    model=model, rate_limits=rate_limits, name=name
                )
            elif engine_type == "MultiEngine":
                sub_engines = [
                    engines[engine_name] for engine_name in params["engines"]
                ]
                engines[name] = create_multi_engine(*sub_engines, name=name)
            else:
                raise ValueError(f"Unknown engine type: {engine_type}")

    _loaded_engines = True
    return engines


class CallLimit(HorseData):
    def __init__(self, limit: float, interval: float) -> None:
        assert limit is not None and limit > 0, "Call limit must be a positive float"
        assert interval >= 0, "Rate interval must be a non-negative float"

        self.limit = limit
        self.interval = interval
        self.last_blocked = time.time() - self.interval / self.limit

    async def consume(self) -> None:
        if self.limit == None:
            return

        await self.wait_for()
        self.last_blocked = max(
            self.last_blocked, time.time() - self.interval / self.limit
        )
        self.last_blocked += self.interval / self.limit

    def next_allowed(self) -> float:
        return max(self.last_blocked + self.interval / self.limit, time.time())

    async def wait_for(self) -> None:
        next_allowed = self.next_allowed()
        now = time.time()
        if next_allowed > now:
            await asyncio.sleep(next_allowed - now)


class TokenLimit(HorseData):
    def __init__(self, limit: float, interval: float) -> None:
        assert limit is not None and limit > 0, "Token limit must be a positive float"
        assert interval >= 0, "Rate interval must be a non-negative float"

        self.limit = limit
        self.interval = interval
        self.last_blocked = time.time() - self.interval / self.limit

    def report_consumed(self, count: int) -> None:
        self.last_blocked = max(
            self.last_blocked, time.time() - self.interval + self.interval / self.limit
        )
        self.last_blocked += self.interval / self.limit * count

    def next_allowed(self, count: int) -> float:
        if self.limit == None:
            return time.time()

        return max(
            self.last_blocked + self.interval / self.limit * count,
            time.time() + self.interval / self.limit * (count - 1),
        )

    async def wait_for(self, count: int) -> None:
        if count == None:
            count = 1

        next_allowed = self.next_allowed(count)
        now = time.time()
        if next_allowed > now:
            await asyncio.sleep(next_allowed - now)


class RateLimits(HorseData):
    def __init__(
        self,
        limits: list[dict],
        call_limits: CallLimit = None,
        token_limits: TokenLimit = None,
    ) -> None:
        self.limits = limits
        self.call_limits: list[CallLimit] = []
        self.token_limits: list[TokenLimit] = []

        for rate_limit in limits:
            interval = rate_limit["interval"]
            calls = rate_limit.get("max_calls", None)
            tokens = rate_limit.get("max_tokens", None)
            if calls != None:
                self.call_limits.append(CallLimit(calls, interval))
            if tokens != None:
                self.token_limits.append(TokenLimit(tokens, interval))

    async def consume_call(self) -> None:
        await asyncio.gather(*[limit.consume() for limit in self.call_limits])

    def report_tokens_consumed(self, count: int) -> None:
        for limit in self.token_limits:
            limit.report_consumed(count)

    async def wait_for(self, expected_tokens: int = None) -> None:
        await asyncio.gather(
            *[limit.wait_for() for limit in self.call_limits],
            *[limit.wait_for(expected_tokens) for limit in self.token_limits],
        )

    def next_allowed(self, expected_tokens: int = None) -> float:
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
    pass


class CallLimitException(Exception):
    pass


class AsyncLLMEngine(HorseData, ABC):
    """
    A class representing an engine for interacting with Language Learning Models
    (LLMs).

    This class provides an interface for getting structured outputs.

    Attributes:

    Methods:
        query: Abstract method to be implemented by subclasses for sending queries to
               the LLM.
        query_object: Query the LLM and parse the response into a specified object
                      type.
        query_block: Query the LLM for a specific block type and parse the response.

    Usage:
        Subclass AsyncLLMEngine and implement the `query` method to use with a specific
        LLM API.

        Use `query_object` to get responses parsed into pydantic object types.
        Use `query_block` to get responses for markdown block types.
    """

    def __init__(self, rate_limits: list = [], name: str = None, **kwargs) -> None:
        """
        Initialize the AsyncLLMEngine.

        """
        super().__init__()
        self.rate_limit = RateLimits(rate_limits)
        self.kwargs = kwargs
        self.name = name

    def __new__(cls, *args, **kwargs) -> "AsyncLLMEngine":
        self = super().__new__(cls)

        original_query = self.query

        @functools.wraps(original_query)
        async def wrapped_query(*args, **kwargs) -> tuple[str, int]:
            await self.rate_limit.consume_call()
            content, tokens_consumed = await original_query(
                *args, **{**self.kwargs, **kwargs}
            )
            self.rate_limit.report_tokens_consumed(tokens_consumed)
            return content

        self.query = wrapped_query
        return self

    def state_dict(self, **override) -> dict:
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
        cls, state_dict: dict, args: dict = {}, debug_prefix: list = []
    ) -> "AsyncLLMEngine":
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
    async def query(self, **kwargs) -> tuple[str, int]:
        """
        Send a query to the Language Learning Model.

        The implementation should return the response from the LLM and the number of
        tokens consumed by the query. Callers will receive only the response.

        Args:
            **kwargs: Arbitrary keyword arguments for the query. Example: max_tokens.
        Returns:
            content: The response from the LLM.
            tokens_consumed: The number of tokens consumed by the query.
        """
        ...

    @abstractmethod
    async def query_object(self, response_model: Type[T], **kwargs) -> T:
        """
        Query the LLM and parse the response into a specified object type.

        This method separates the input kwargs into prompt arguments and API arguments,
        generates the query messages, sends the query, and parses the response.

        Args:
            response_model (type[T]): The type of object to parse the response into. It
                                      should be a Pydantic BaseModel subclass.
            **kwargs: Arbitrary keyword arguments. Arguments with all-uppercase keys
                      will be passed to the LLM via the prompt. Others as LLM API
                      arguments.

        Returns:
            T: An instance of the response_model type, populated with the parsed
               response.

        Raises:
            Exception: If the query fails.
        """
        ...

    @abstractmethod
    async def query_block(self, block_type: str, **kwargs) -> str:
        """
        Query the LLM for a specific block type and parse the response.

        This method separates the input kwargs into prompt arguments and API arguments,
        generates the query messages for the specified block type, sends the query,
        and parses the response.

        Args:
            block_type (str): The type of block to query for.
            **kwargs: Arbitrary keyword arguments. Arguments with all-uppercase keys
                      will be passed to the LLM via the prompt. Others as LLM API
                      arguments.

        Returns:
            T: The parsed response for the specified block type.

        Raises:
            Exception: If the query fails.
        """
        ...

    @abstractmethod
    async def query_continuation(self, prompt: str, **kwargs) -> str:
        """
        Query the LLM to continue the prompt.

        All kwargs are passed to the underlying API.

        Args:
            prompt (str): The prompt to continue.
            **kwargs: Arbitrary keyword arguments for the query.

        Returns:
            str: The continuation of the prompt.

        Raises:
            Exception: If the query fails.
        """
        ...
