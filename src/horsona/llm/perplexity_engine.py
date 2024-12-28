import json
import os
from typing import Any, AsyncGenerator

import httpx

from horsona.llm.base_engine import LLMMetrics, tracks_metrics

from .chat_engine import AsyncChatEngine
from .engine_utils import clean_json_string


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

    def __init__(self, model: str, **kwargs):
        """
        Initialize the AsyncPerplexityEngine.

        Args:
            model (str): The name of the Perplexity AI model to use.
            *args: Variable length argument list to pass to the parent constructor.
            **kwargs: Arbitrary keyword arguments to pass to the parent constructor.
        """
        super().__init__(**kwargs)
        self.model = model
        self.apikey = os.environ["PERPLEXITY_API_KEY"]

    @tracks_metrics
    async def query(
        self, *, metrics: LLMMetrics, **kwargs
    ) -> AsyncGenerator[str, None]:
        url = "https://api.perplexity.ai/chat/completions"
        kwargs["messages"] = _clean_messages(kwargs.get("messages", []))
        print(kwargs["messages"])

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

        raw_content = clean_json_string(response.content.decode("utf-8"))
        response_json = json.loads(raw_content)
        content = response_json["choices"][0]["message"]["content"]
        total_tokens = response_json["usage"]["total_tokens"]

        metrics.tokens_consumed = total_tokens

        yield content


def _clean_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    new_messages = []

    # System messages all come first
    system_messages = [m for m in messages if m["role"] == "system"]
    if system_messages:
        new_messages.append(
            {
                "role": "system",
                "content": "\n\n".join([m["content"] for m in system_messages]),
            }
        )

    # User and assistant messages must alternate

    previous_role = None
    for m in messages:
        if m["role"] not in ["user", "assistant"]:
            continue

        if m["role"] == previous_role:
            new_messages[-1]["content"] += "\n\n" + m["content"]
        else:
            new_messages.append(m)
        previous_role = m["role"]

    return new_messages
