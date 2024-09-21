from anthropic import AsyncAnthropic

from .chat_engine import AsyncChatEngine


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

    async def query(self, **kwargs):
        system_msg = []
        messages = kwargs["messages"]
        for i in reversed(range(len(messages))):
            msg = messages[i]
            if msg.get("role") == "system":
                system_msg.append(msg["content"])
                del messages[i]

        response = await self.client.messages.create(
            system="\n\n".join(system_msg) if system_msg else None,
            model=self.model,
            max_tokens=2**13,
            **kwargs,
        )

        return response.content[0].text, response.usage.total_tokens
