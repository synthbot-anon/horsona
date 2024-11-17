from typing import Optional, Union

from pydantic import BaseModel, Field


class ChatCompletionMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatCompletionMessage]
    temperature: Optional[float] = Field(default=1.0)
    top_p: Optional[float] = Field(default=1.0)
    n: Optional[int] = Field(default=1)
    stream: Optional[bool] = Field(default=False)
    stop: Optional[Union[str, list[str]]] = Field(default=None)
    max_tokens: Optional[int] = Field(default=None)
    presence_penalty: Optional[float] = Field(default=0)
    frequency_penalty: Optional[float] = Field(default=0)
    logit_bias: Optional[dict[str, float]] = Field(default=None)
    user: Optional[str] = Field(default=None)


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(
        default_factory=lambda: "chatcmpl-" + str(id(ChatCompletionResponse))
    )
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(__import__("time").time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage
