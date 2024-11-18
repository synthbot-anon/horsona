from together import AsyncTogether

from horsona.llm.base_engine import LLMMetrics
from horsona.llm.oai_engine import AsyncOAIEngine

from .chat_engine import AsyncChatEngine


class AsyncTogetherEngine(AsyncOAIEngine):
    """
    An asynchronous implementation of ChatEngine for interacting with Together models.

    This class provides an asynchronous interface for querying Together language models.

    Attributes:
        model (str): The name of the Together model to use.
        client (AsyncTogether): An instance of the asynchronous Together client for API interactions.

    Inherits from:
        AsyncChatEngine
    """

    def __init__(self, model: str, *args, **kwargs):
        """
        Initialize the AsyncTogetherEngine.

        Args:
            model (str): The name of the Together model to use.
            *args: Variable length argument list to pass to the parent constructor.
            **kwargs: Arbitrary keyword arguments to pass to the parent constructor.
        """
        super().__init__(*args, **kwargs)
        self.model = model
        self.client = AsyncTogether()

    async def create(self, **kwargs):
        kwargs["model"] = self.model
        return await self.client.chat.completions.create(**kwargs)
