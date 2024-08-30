from cerebras.cloud.sdk import AsyncCerebras

from .chat_engine import AsyncChatEngine


class AsyncCerebrasEngine(AsyncChatEngine):
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
        self.client = AsyncCerebras()

    async def query(self, **kwargs):
        response = await self.client.chat.completions.create(
            model=self.model, timeout=1, **kwargs
        )
        return response.choices[0].message.content
