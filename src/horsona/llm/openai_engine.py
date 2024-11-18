from openai import AsyncOpenAI
from openai.types.completion import Completion

from .oai_engine import AsyncOAIEngine


class AsyncOpenAIEngine(AsyncOAIEngine):
    """
    An asynchronous implementation of ChatEngine for interacting with Cerebras models.

    This class provides an asynchronous interface for querying Cerebras language models.

    Attributes:
        model (str): The name of the Cerebras model to use.
        client (AsyncCerebras): An instance of the asynchronous Cerebras client for API interactions.

    Inherits from:
        AsyncChatEngine
    """

    def __init__(self, model: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.client = AsyncOpenAI()

    async def create(self, **kwargs) -> Completion:
        kwargs["model"] = self.model
        return await self.client.chat.completions.create(**kwargs)
