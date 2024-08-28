from groq import AsyncGroq, Groq

from .chat_engine import AsyncChatEngine, ChatEngine


class GroqEngine(ChatEngine):
    """
    A concrete implementation of ChatEngine for interacting with Groq models.

    This class provides a synchronous interface for querying Groq language models.

    Attributes:
        model (str): The name of the Groq model to use.
        client (Groq): An instance of the Groq client for API interactions.

    Inherits from:
        ChatEngine
    """

    def __init__(self, model: str, *args, **kwargs):
        """
        Initialize the GroqEngine.

        Args:
            model (str): The name of the Groq model to use.
            *args: Variable length argument list to pass to the parent constructor.
            **kwargs: Arbitrary keyword arguments to pass to the parent constructor.
        """
        super().__init__(*args, **kwargs)
        self.model = model
        self.client = Groq()

    def query(self, **kwargs):
        response = self.client.chat.completions.create(
            model=self.model, stream=False, **kwargs
        )
        return response.choices[0].message.content


class AsyncGroqEngine(AsyncChatEngine):
    """
    An asynchronous implementation of ChatEngine for interacting with Groq models.

    This class provides an asynchronous interface for querying Groq language models.

    Attributes:
        model (str): The name of the Groq model to use.
        client (AsyncGroq): An instance of the asynchronous Groq client for API interactions.

    Inherits from:
        AsyncChatEngine
    """

    def __init__(self, model: str, *args, **kwargs):
        """
        Initialize the AsyncGroqEngine.

        Args:
            model (str): The name of the Groq model to use.
            *args: Variable length argument list to pass to the parent constructor.
            **kwargs: Arbitrary keyword arguments to pass to the parent constructor.
        """
        super().__init__(*args, **kwargs)
        self.model = model
        self.client = AsyncGroq()

    async def query(self, **kwargs):
        response = await self.client.chat.completions.create(
            model=self.model, stream=False, **kwargs
        )
        return response.choices[0].message.content
