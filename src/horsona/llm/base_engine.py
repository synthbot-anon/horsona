import asyncio
import functools
import time
from abc import ABC, abstractmethod
from typing import Type, TypeVar, Union

from pydantic import BaseModel

__all__ = ["AsyncLLMEngine"]

T = TypeVar("T", bound=BaseModel)
S = TypeVar("S", bound=Union[str, T])


class CallLimit:
    def __init__(self, limit: float, interval: float):
        assert limit is not None and limit > 0, "Call limit must be a positive float"
        assert interval >= 0, "Rate interval must be a non-negative float"

        self.limit = limit
        self.interval = interval
        self.last_blocked = time.time() - self.interval / self.limit

    async def consume(self):
        if self.limit == None:
            return

        await self.wait_for()
        self.last_blocked = max(
            self.last_blocked, time.time() - self.interval / self.limit
        )
        self.last_blocked += self.interval / self.limit

    def next_allowed(self):
        return max(self.last_blocked + self.interval / self.limit, time.time())

    async def wait_for(self):
        next_allowed = self.next_allowed()
        now = time.time()
        if next_allowed > now:
            await asyncio.sleep(next_allowed - now)


class TokenLimit:
    def __init__(self, limit: float, interval: float):
        assert limit is not None and limit > 0, "Token limit must be a positive float"
        assert interval >= 0, "Rate interval must be a non-negative float"

        self.limit = limit
        self.interval = interval
        self.last_blocked = time.time() - self.interval / self.limit

    def report_consumed(self, count):
        self.last_blocked = max(
            self.last_blocked, time.time() - self.interval + self.interval / self.limit
        )
        self.last_blocked += self.interval / self.limit * count

    def next_allowed(self, count):
        if self.limit == None:
            return time.time()

        return max(
            self.last_blocked + self.interval / self.limit * count,
            time.time() + self.interval / self.limit * (count - 1),
        )

    async def wait_for(self, count):
        if count == None:
            count = 1

        next_allowed = self.next_allowed(count)
        now = time.time()
        if next_allowed > now:
            await asyncio.sleep(next_allowed - now)


class RateLimits:
    def __init__(self, limits: list[dict]):
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

    async def consume_call(self):
        await asyncio.gather(*[limit.consume() for limit in self.call_limits])

    def report_tokens_consumed(self, count):
        for limit in self.token_limits:
            limit.report_consumed(count)

    async def wait_for(self, expected_tokens=None):
        await asyncio.gather(
            *[limit.wait_for() for limit in self.call_limits],
            *[limit.wait_for(expected_tokens) for limit in self.token_limits],
        )

    def next_allowed(self, expected_tokens=None):
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


class AsyncLLMEngine(ABC):
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

    def __init__(self, rate_limits=[], **kwargs):
        """
        Initialize the AsyncLLMEngine.

        """
        self.rate_limit = RateLimits(rate_limits)
        self.kwargs = kwargs

    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls)

        original_query = self.query

        @functools.wraps(original_query)
        async def wrapped_query(**kwargs):
            await self.rate_limit.consume_call()
            content, tokens_consumed = await original_query(**{**self.kwargs, **kwargs})
            self.rate_limit.report_tokens_consumed(tokens_consumed)
            return content

        self.query = wrapped_query
        return self

    @abstractmethod
    async def query(self, **kwargs):
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
    async def query_block(self, block_type: str, **kwargs) -> T:
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

    async def query_structured(self, structure: S, **kwargs):
        """
        Query the LLM and parse the response into a specified structure.

        This method separates the input kwargs into prompt arguments and API arguments,
        generates the query messages, sends the query, and parses the response.

        Args:
            structure (Union[str, BaseModel]): The structure to parse the response into,
                                               either a string for markdown block types
                                               or a Pydantic model for object types.
            **kwargs: Arbitrary keyword arguments. Arguments with all-uppercase keys
                      will be passed to the LLM via the prompt. Others as LLM API
                      arguments.

        Returns:
            Union[str, BaseModel]: The parsed response in the specified structure.

        Raises:
            Exception: If the query fails.
        """
        if isinstance(structure, str):
            return await self.query_block(structure, **kwargs)
        elif issubclass(structure, BaseModel):
            return await self.query_object(structure, **kwargs)
        else:
            raise ValueError(
                "Invalid structure type. Must be a string or a Pydantic model. "
                f"Got: {type(structure)}"
            )

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
