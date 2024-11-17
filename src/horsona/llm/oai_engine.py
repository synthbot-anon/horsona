import json
from abc import ABC, abstractmethod
from typing import Type, Union

from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel

from horsona.llm.base_engine import LLMMetrics

from .chat_engine import AsyncChatEngine
from .engine_utils import compile_user_prompt


class AsyncOAIEngine(AsyncChatEngine, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    async def create(self, **kwargs) -> ChatCompletion: ...

    async def query(
        self, prompt: str = None, metrics: LLMMetrics = None, **kwargs
    ) -> tuple[str, int]:
        if prompt is not None:
            kwargs.setdefault("messages", []).append(
                {"role": "user", "content": prompt}
            )

        response: ChatCompletion = await self.create(**kwargs)
        tokens_consumed = response.usage.total_tokens

        tool_required = kwargs.get("tool_choice", "auto") != "auto"

        # Check if the conversation was too long for the context window
        finish_reason = response.choices[0].finish_reason
        if finish_reason == "length":
            raise Exception("The conversation was too long for the context window.")

        # Check if the model's output included copyright material (or similar)
        if finish_reason == "content_filter":
            raise Exception("Content was filtered due to policy violations.")

        # Check if the model has made a tool_call
        if (
            finish_reason == "tool_calls"
            or
            # Edge case where if we forced the model to call one of our functions
            (tool_required and finish_reason == "stop")
        ):
            metrics.tokens_consumed += tokens_consumed
            return [x.function for x in response.choices[0].message.tool_calls]

        # Else the model is responding directly to the user
        elif finish_reason == "stop":
            metrics.tokens_consumed += tokens_consumed
            return response.choices[0].message.content

        # Catch any other case, this is unexpected
        else:
            raise Exception("Unexpected API finish_reason:", finish_reason)

    async def query_tool(
        self,
        tools: list[Type[BaseModel]],
        requires_tool: Union[bool, Type[BaseModel]] = True,
        **kwargs,
    ) -> Union[str, list[BaseModel]]:
        """
        Query the LLM to use a tool.

        This method separates the input kwargs into prompt arguments and API arguments,
        generates the query messages, sends the query, and parses the response.

        It returns a list of instantiated BaseModels, one for each tool call.

        Args:
            tools (list[Type[BaseModel]]): The list of allowed tools, each of which
                                      should be a Pydantic BaseModel subclass.
            requires_tool (bool): If True, the model must use a tool. If a BaseModel
                                    subclass, the model must use that tool. If False,
                                    the model may return a string response.
            **kwargs: Arbitrary keyword arguments. Arguments with all-uppercase keys
                      will be passed to the LLM via the prompt. Others as LLM API
                      arguments.

        Returns:
            T: An instance of the response_model type, populated with the parsed
               response.

        Raises:
            Exception: If the query fails.
        """
        prompt_args = {k: v for k, v in kwargs.items() if k == k.upper()}
        api_args = {k: v for k, v in kwargs.items() if k != k.upper()}

        tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": tool.__name__,
                    "description": tool.__doc__ or tool.__name__,
                    "parameters": tool.model_json_schema(),
                },
            }
            for tool in tools
        ]

        prior_messages = kwargs.get("messages", [])

        args = {
            "messages": [
                *prior_messages,
                {"role": "user", "content": await compile_user_prompt(**prompt_args)},
            ],
            "tools": tool_schemas,
        }

        if requires_tool == True:
            args["tool_choice"] = "required"
        elif requires_tool:
            args["tool_choice"] = {
                "type": "function",
                "function": {"name": requires_tool.__name__},
            }
        else:
            args["tool_choice"] = "auto"

        result = await self.query(**args, **api_args)
        if isinstance(result, str):
            return result

        tool_map = {tool.__name__: tool for tool in tools}
        response = []

        for call in result:
            fn = tool_map[call.name]
            args = json.loads(call.arguments)
            response.append(fn(**args))

        return response
