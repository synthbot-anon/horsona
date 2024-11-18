import os
import warnings
from typing import AsyncGenerator

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from fireworks.client import AsyncFireworks

from fireworks.client.api import ChatCompletionResponse, CompletionStreamResponse

from horsona.llm.oai_engine import AsyncOAIEngine


class AsyncFireworksEngine(AsyncOAIEngine):
    """
    An asynchronous implementation of ChatEngine for interacting with Fireworks models.

    This class provides an asynchronous interface for querying Fireworks language models.
    It inherits from AsyncOAIEngine to maintain compatibility with OpenAI-style interfaces
    while providing access to Fireworks' models.

    Attributes:
        model (str): The name of the Fireworks model to use.
        client (AsyncFireworks): An instance of the asynchronous Fireworks client for API interactions.

    Inherits from:
        AsyncOAIEngine: Base class providing OpenAI-compatible interface
    """

    def __init__(self, model: str, *args, **kwargs):
        """
        Initialize the AsyncFireworksEngine.

        Args:
            model (str): The name of the Fireworks model to use (e.g. "accounts/fireworks/models/llama-v2-7b-chat").
            *args: Variable length argument list to pass to the parent constructor.
            **kwargs: Arbitrary keyword arguments to pass to the parent constructor.
        """
        super().__init__(*args, **kwargs)
        self.model = model
        self.client = AsyncFireworks(api_key=os.environ["FIREWORKS_API_KEY"])

    async def create(
        self, **kwargs
    ) -> AsyncGenerator[CompletionStreamResponse, None] | ChatCompletionResponse:
        """
        Create a chat completion using the Fireworks API.

        Args:
            **kwargs: Arbitrary keyword arguments passed to the Fireworks chat completion API.
                     Common arguments include:
                     - messages (List[Dict]): List of chat messages
                     - stream (bool): Whether to stream the response
                     - temperature (float): Sampling temperature
                     - max_tokens (int): Maximum tokens to generate

        Returns:
            Union[AsyncGenerator[CompletionStreamResponse, None], ChatCompletionResponse]:
                If streaming is enabled, returns an async generator yielding completion chunks.
                Otherwise, returns the complete chat completion response.
        """
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
