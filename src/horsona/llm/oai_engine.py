from abc import ABC, abstractmethod
from typing import AsyncGenerator

from openai.types.chat.chat_completion import ChatCompletion

from horsona.llm.base_engine import LLMMetrics, tracks_metrics

from .chat_engine import AsyncChatEngine
from .engine_utils import compile_user_prompt


class AsyncOAIEngine(AsyncChatEngine, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    async def create(self, **kwargs) -> ChatCompletion: ...

    @tracks_metrics
    async def query(
        self, *, metrics: LLMMetrics, **kwargs
    ) -> AsyncGenerator[str, None]:
        api_args = {k: v for k, v in kwargs.items() if k.upper() != k}
        prompt_args = {k: v for k, v in kwargs.items() if k == k.upper()}

        if "tool_choice" in api_args:
            raise NotImplementedError("Tool choice not implemented for OAIEngine")

        if prompt_args:
            api_args.setdefault("messages", []).extend(
                {"role": "user", "content": await compile_user_prompt(**prompt_args)}
            )

        if not api_args.get("stream", False):
            response: ChatCompletion = await self.create(**api_args)

            tokens_consumed = response.usage.total_tokens

            # Check if the conversation was too long for the context window
            finish_reason = response.choices[0].finish_reason
            if finish_reason == "length":
                raise Exception("The conversation was too long for the context window.")

            # Check if the model's output included copyright material (or similar)
            if finish_reason == "content_filter":
                raise Exception("Content was filtered due to policy violations.")

            # Else the model is responding directly to the user
            if finish_reason in ("stop", "eos"):
                metrics.tokens_consumed += tokens_consumed
                yield response.choices[0].message.content

            # Catch any other case, this is unexpected
            else:
                raise Exception("Unexpected API finish_reason:", finish_reason)
        else:
            api_args["stream_options"] = {"include_usage": True}

            async for chunk in await self.create(**api_args):
                if metrics is not None:
                    if hasattr(chunk, "usage") and chunk.usage is not None:
                        # With include_usage, the final chunk object should include the total tokens consumed
                        # So we can override our default assumption of 1 token on the final chunk
                        metrics.tokens_consumed = chunk.usage.total_tokens
                    else:
                        # By default, assume 1 token per chunk
                        metrics.tokens_consumed += 1

                if chunk.choices:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
