import os
from typing import AsyncGenerator

from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from .oai_engine import AsyncOAIEngine


class AsyncGrokEngine(AsyncOAIEngine):
    """
    An asynchronous implementation of ChatEngine for interacting with Grok models.

    This class provides an asynchronous interface for querying Grok language models.

    Attributes:
        model (str): The name of the Grok model to use.
        client (AsyncOpenAI): An instance of the asynchronous OpenAI client configured for Grok.

    Inherits from:
        AsyncOAIEngine: Base class for OpenAI-compatible API engines
    """

    def __init__(self, model: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.client = AsyncOpenAI(
            base_url="https://api.x.ai/v1",
            api_key=os.environ.get("GROK_API_KEY"),
        )

    async def create(
        self, **kwargs
    ) -> AsyncStream[ChatCompletionChunk] | ChatCompletion:
        kwargs["model"] = self.model
        return await self.client.chat.completions.create(**kwargs)
