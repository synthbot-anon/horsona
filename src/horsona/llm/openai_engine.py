from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from .oai_engine import AsyncOAIEngine


class AsyncOpenAIEngine(AsyncOAIEngine):
    """
    An asynchronous implementation of ChatEngine for interacting with OpenAI models.

    This class provides an asynchronous interface for querying OpenAI language models.

    Attributes:
        model (str): The name of the OpenAI model to use.
        client (AsyncOpenAI): An instance of the asynchronous OpenAI client for API interactions.

    Inherits from:
        AsyncOAIEngine: Base class for OpenAI-compatible API engines
    """

    def __init__(self, model: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.client = AsyncOpenAI()

    async def create(
        self, **kwargs
    ) -> AsyncStream[ChatCompletionChunk] | ChatCompletion:
        kwargs["model"] = self.model
        return await self.client.chat.completions.create(**kwargs)
