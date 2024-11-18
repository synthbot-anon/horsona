from typing import AsyncGenerator

from anthropic import AsyncAnthropic

from horsona.llm.base_engine import LLMMetrics
from horsona.llm.chat_engine import AsyncChatEngine


class AsyncAnthropicEngine(AsyncChatEngine):
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
        self.client = AsyncAnthropic()

    async def query(self, metrics: LLMMetrics = None, **kwargs) -> str:
        system_msg = []
        messages = kwargs["messages"]
        for i in reversed(range(len(messages))):
            msg = messages[i]
            if msg.get("role") == "system":
                system_msg.append(msg["content"])
                del messages[i]

        if "max_tokens" not in kwargs or kwargs["max_tokens"] is None:
            kwargs["max_tokens"] = 2**12

        kwargs["model"] = self.model

        response = await self.client.messages.create(
            system=system_msg,
            **kwargs,
        )

        total_tokens = response.usage.input_tokens + response.usage.output_tokens
        metrics.tokens_consumed = total_tokens
        return response.content[0].text

    async def query_stream(
        self, metrics: LLMMetrics = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        system_msg = []
        messages = kwargs["messages"]
        for i in reversed(range(len(messages))):
            msg = messages[i]
            if msg.get("role") == "system":
                system_msg.append(msg["content"])
                del messages[i]

        if "max_tokens" not in kwargs or kwargs["max_tokens"] is None:
            kwargs["max_tokens"] = 2**12

        kwargs["model"] = self.model

        if "stream" in kwargs:
            del kwargs["stream"]

        if "stream_options" in kwargs:
            del kwargs["stream_options"]

        input_tokens = 0
        output_tokens = 0

        async with self.client.messages.stream(system=system_msg, **kwargs) as stream:
            async for chunk in stream:
                if hasattr(chunk, "usage"):
                    if hasattr(chunk.usage, "input_tokens"):
                        input_tokens = chunk.usage.input_tokens
                    if hasattr(chunk.usage, "output_tokens"):
                        output_tokens = chunk.usage.output_tokens
                    metrics.tokens_consumed = input_tokens + output_tokens

                if chunk.type not in ("content_block_start", "content_block_delta"):
                    continue

                if hasattr(chunk, "content_block"):
                    if (
                        hasattr(chunk.content_block, "text")
                        and chunk.content_block.text
                    ):
                        yield chunk.content_block.text

                if hasattr(chunk, "delta"):
                    if hasattr(chunk.delta, "text") and chunk.delta.text:
                        yield chunk.delta.text
