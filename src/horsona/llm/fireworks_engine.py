import os
import warnings
from typing import AsyncGenerator

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from fireworks.client import AsyncFireworks

from openai.types.completion import Completion

from horsona.llm.oai_engine import AsyncOAIEngine


class AsyncFireworksEngine(AsyncOAIEngine):
    """
    An asynchronous implementation of ChatEngine for interacting with Fireworks models.

    This class provides an asynchronous interface for querying Fireworks language models.

    Attributes:
        model (str): The name of the Fireworks model to use.
        client (AsyncFireworks): An instance of the asynchronous Fireworks client for API interactions.

    Inherits from:
        AsyncChatEngine
    """

    def __init__(self, model: str, *args, **kwargs):
        """
        Initialize the AsyncFireworksEngine.

        Args:
            model (str): The name of the Fireworks model to use.
            *args: Variable length argument list to pass to the parent constructor.
            **kwargs: Arbitrary keyword arguments to pass to the parent constructor.
        """
        super().__init__(*args, **kwargs)
        self.model = model
        self.client = AsyncFireworks(api_key=os.environ["FIREWORKS_API_KEY"])

    async def create(self, **kwargs) -> Completion:
        kwargs["model"] = self.model
        if "stream" not in kwargs:
            kwargs["stream"] = False

        if "stream_options" in kwargs:
            del kwargs["stream_options"]

        result = self.client.chat.completions.acreate(**kwargs)

        if isinstance(result, AsyncGenerator):
            return result
        else:
            return await result
