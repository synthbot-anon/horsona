from cerebras.cloud.sdk import AsyncCerebras, Cerebras

from .chat_engine import AsyncChatEngine, ChatEngine


class CerebrasEngine(ChatEngine):
    """
    A concrete implementation of ChatEngine for interacting with Cerebras models.

    This class provides a synchronous interface for querying Cerebras language models.

    Attributes:
        model (str): The name of the Cerebras model to use.
        client (Cerebras): An instance of the Cerebras client for API interactions.

    Inherits from:
        ChatEngine
    """

    def __init__(self, model: str, *args, **kwargs):
        """
        Initialize the CerebrasEngine.

        Args:
            model (str): The name of the Cerebras model to use.
            *args: Variable length argument list to pass to the parent constructor.
            **kwargs: Arbitrary keyword arguments to pass to the parent constructor.
        """
        super().__init__(*args, **kwargs)
        self.model = model
        self.client = Cerebras()

    def query(self, **kwargs):
        response = self.client.chat.completions.create(model=self.model, **kwargs)
        return response.choices[0].message.content


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
        response = await self.client.chat.completions.create(model=self.model, **kwargs)
        return response.choices[0].message.content
