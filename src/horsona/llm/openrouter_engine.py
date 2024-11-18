import os
from typing import AsyncGenerator

from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from .oai_engine import AsyncOAIEngine


class AsyncOpenRouterEngine(AsyncOAIEngine):
    """
    An asynchronous implementation of ChatEngine for interacting with OpenRouter models.

    This class provides an asynchronous interface for querying OpenRouter language models.

    Attributes:
        model (str): The name of the OpenRouter model to use.
        client (AsyncOpenAI): An instance of the asynchronous OpenAI client configured for OpenRouter.

    Inherits from:
        AsyncOAIEngine: Base class for OpenAI-compatible API engines
    """

    def __init__(self, model: str, *args, url: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.client = AsyncOpenAI(
            base_url=url,
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )

    async def create(
        self, **kwargs
    ) -> AsyncStream[ChatCompletionChunk] | ChatCompletion:
        kwargs["model"] = self.model
        return await self.client.chat.completions.create(**kwargs)
