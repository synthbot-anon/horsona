import os

import httpx

from .chat_engine import AsyncChatEngine, ChatEngine


class PerplexityEngine(ChatEngine):
    """
    A concrete implementation of ChatEngine for interacting with Perplexity AI models.

    This class provides a synchronous interface for querying Perplexity AI language models.

    Attributes:
        model (str): The name of the Perplexity AI model to use.
        apikey (str): The API key for authenticating with Perplexity AI, retrieved from environment variables.

    Inherits from:
        ChatEngine
    """

    def __init__(self, model, *args, **kwargs):
        """
        Initialize the PerplexityEngine.

        Args:
            model (str): The name of the Perplexity AI model to use.
            *args: Variable length argument list to pass to the parent constructor.
            **kwargs: Arbitrary keyword arguments to pass to the parent constructor.
        """
        super().__init__(*args, **kwargs)
        self.model = model
        self.apikey = os.environ["PERPLEXITY_API_KEY"]

    def query(self, **kwargs):
        url = "https://api.perplexity.ai/chat/completions"

        payload = {
            "model": self.model,
            "return_citations": True,
            **kwargs,
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self.apikey}",
        }

        response = httpx.post(url, json=payload, headers=headers)

        content = response.json()["choices"][0]["message"]["content"]
        return content


class AsyncPerplexityEngine(AsyncChatEngine):
    """
    An asynchronous implementation of ChatEngine for interacting with Perplexity AI models.

    This class provides an asynchronous interface for querying Perplexity AI language models.

    Attributes:
        model (str): The name of the Perplexity AI model to use.
        apikey (str): The API key for authenticating with Perplexity AI, retrieved from environment variables.

    Inherits from:
        AsyncChatEngine
    """

    def __init__(self, model, *args, **kwargs):
        """
        Initialize the AsyncPerplexityEngine.

        Args:
            model (str): The name of the Perplexity AI model to use.
            *args: Variable length argument list to pass to the parent constructor.
            **kwargs: Arbitrary keyword arguments to pass to the parent constructor.
        """
        super().__init__(*args, **kwargs)
        self.model = model
        self.apikey = os.environ["PERPLEXITY_API_KEY"]

    async def query(self, **kwargs):
        url = "https://api.perplexity.ai/chat/completions"

        payload = {
            "model": self.model,
            "return_citations": True,
            **kwargs,
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self.apikey}",
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url, json=payload, headers=headers, timeout=None
            )

        content = response.json()["choices"][0]["message"]["content"]
        return content
