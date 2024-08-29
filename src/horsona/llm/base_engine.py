from abc import ABC, abstractmethod
from typing import TypeVar, Union

from pydantic import BaseModel

__all__ = ["AsyncLLMEngine", "LLMEngine"]

T = TypeVar("T", bound=BaseModel)
S = TypeVar("S", bound=Union[str, T])


class LLMEngine(ABC):
    """
    A class representing an engine for interacting with Language Learning Models
    (LLMs).

    This class provides an interface for getting structured outputs.

    Attributes:
        fallback (LLMEngine): An optional fallback LLMEngine to use if queries fail.

    Usage:
        Subclass LLMEngine and implement the `query` method to use with a specific LLM API.
        Use `query_object` to get responses parsed into pydantic object types.
        Use `query_block` to get responses for markdown block types.
    """

    def __init__(self, fallback: "LLMEngine" = None):
        """
        Initialize the LLMEngine.

        Args:
            fallback (LLMEngine, optional): Another LLMEngine instance to use as a fallback
                                            if this engine's queries fail. Defaults to None.
        """
        self.fallback = fallback

    @abstractmethod
    def query(self, **kwargs):
        """
        Send a query to the Language Learning Model.

        This is an abstract method that should be implemented by subclasses to interact
        with specific LLM APIs.

        Args:
            **kwargs: Arbitrary keyword arguments for the query. Example: max_tokens.
        """
        pass

    @abstractmethod
    def query_object(self, response_model: type[T], **kwargs) -> T:
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
            Exception: If the query fails and there's no fallback engine.
        """
        pass

    @abstractmethod
    def query_block(self, block_type: str, **kwargs) -> str:
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
            Exception: If the query fails and there's no fallback engine.
        """
        pass

    def query_structured(self, structure: S, **kwargs):
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
            Exception: If the query fails and there's no fallback engine.
        """
        if isinstance(structure, str):
            return self.query_block(structure, **kwargs)
        elif issubclass(structure, BaseModel):
            return self.query_object(structure, **kwargs)
        else:
            raise ValueError(
                "Invalid structure type. Must be a string or a Pydantic model. "
                f"Got: {type(structure)}"
            )


class AsyncLLMEngine(ABC):
    """
    A class representing an engine for interacting with Language Learning Models
    (LLMs).

    This class provides an interface for getting structured outputs.

    Attributes:
        fallback (AsyncLLMEngine): An optional fallback AsyncLLMEngine to use if
                                   queries fail.

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

    def __init__(self, fallback: "AsyncLLMEngine" = None):
        """
        Initialize the AsyncLLMEngine.

        Args:
            fallback (AsyncLLMEngine, optional): Another LLMEngine instance to use as a
                                                 fallback if this engine's queries
                                                 fail. Defaults to None.
        """
        self.fallback = fallback

    @abstractmethod
    async def query(self, **kwargs):
        """
        Send a query to the Language Learning Model.

        This is an abstract method that should be implemented by subclasses to interact
        with specific LLM APIs.

        Args:
            **kwargs: Arbitrary keyword arguments for the query. Example: max_tokens.
        """
        pass

    @abstractmethod
    async def query_object(self, response_model: type[T], **kwargs) -> T:
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
            Exception: If the query fails and there's no fallback engine.
        """
        pass

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
            Exception: If the query fails and there's no fallback engine.
        """
        pass

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
            Exception: If the query fails and there's no fallback engine.
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
            Exception: If the query fails and there's no fallback engine.
        """
        pass
