from groq import AsyncGroq

from horsona.llm.base_engine import LLMMetrics
from horsona.llm.oai_engine import AsyncOAIEngine

from .chat_engine import AsyncChatEngine


class AsyncGroqEngine(AsyncOAIEngine):
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

    async def create(self, **kwargs):
        kwargs["model"] = self.model

        if "stream_options" in kwargs:
            del kwargs["stream_options"]

        return await self.client.chat.completions.create(**kwargs)
