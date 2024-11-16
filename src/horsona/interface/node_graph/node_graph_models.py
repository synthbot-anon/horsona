from enum import StrEnum, auto
from typing import Any, Generic, Literal, Optional, Self, Union

from pydantic import BaseModel, field_validator, model_validator


class ArgumentType(StrEnum):
    NONE = auto()
    UNSUPPORTED = auto()
    STR = auto()
    FLOAT = auto()
    INT = auto()
    BOOL = auto()
    LIST = auto()
    DICT = auto()
    TUPLE = auto()
    SET = auto()
    NODE = auto()


class NoneArgument(BaseModel):
    type: Literal[ArgumentType.NONE] = ArgumentType.NONE
    value: Optional[Any] = None


class UnsupportedArgument(BaseModel):
    type: Literal[ArgumentType.UNSUPPORTED] = ArgumentType.UNSUPPORTED
    value: Optional[Any] = None


class StrArgument(BaseModel):
    type: Literal[ArgumentType.STR] = ArgumentType.STR
    value: str


class FloatArgument(BaseModel):
    type: Literal[ArgumentType.FLOAT] = ArgumentType.FLOAT
    value: float


class IntArgument(BaseModel):
    type: Literal[ArgumentType.INT] = ArgumentType.INT
    value: int


class BoolArgument(BaseModel):
    type: Literal[ArgumentType.BOOL] = ArgumentType.BOOL
    value: bool


class ListArgument(BaseModel):
    type: Literal[ArgumentType.LIST] = ArgumentType.LIST
    value: list[Any]

    @model_validator(mode="after")
    def validate_list(self) -> Self:
        self.value = [create_argument(**x) for x in self.value]
        return self


class DictArgument(BaseModel):
    type: Literal[ArgumentType.DICT] = ArgumentType.DICT
    value: dict[str, Any]

    @model_validator(mode="after")
    def validate_dict(self) -> Self:
        self.value = {k: create_argument(**v) for k, v in self.value.items()}
        return self


class TupleArgument(BaseModel):
    type: Literal[ArgumentType.TUPLE] = ArgumentType.TUPLE
    value: list[Any]

    @model_validator(mode="after")
    def validate_tuple(self) -> Self:
        self.value = tuple([create_argument(**x) for x in self.value])
        return self


class SetArgument(BaseModel):
    type: Literal[ArgumentType.SET] = ArgumentType.SET
    value: list[Any]

    @model_validator(mode="after")
    def validate_set(self) -> Self:
        self.value = set([create_argument(**x) for x in self.value])
        return self


class NodeArgument(BaseModel):
    type: Literal[ArgumentType.NODE] = ArgumentType.NODE
    value: int


Argument = Union[
    NoneArgument,
    UnsupportedArgument,
    StrArgument,
    FloatArgument,
    IntArgument,
    BoolArgument,
    ListArgument,
    DictArgument,
    TupleArgument,
    SetArgument,
    NodeArgument,
]


def create_argument(type: ArgumentType, value: Any) -> Argument:
    if type == ArgumentType.NONE:
        return NoneArgument(value=value)
    elif type == ArgumentType.UNSUPPORTED:
        return UnsupportedArgument(value=value)
    elif type == ArgumentType.STR:
        return StrArgument(value=value)
    elif type == ArgumentType.FLOAT:
        return FloatArgument(value=value)
    elif type == ArgumentType.INT:
        return IntArgument(value=value)
    elif type == ArgumentType.BOOL:
        return BoolArgument(value=value)
    elif type == ArgumentType.LIST:
        return ListArgument(value=value)
    elif type == ArgumentType.DICT:
        return DictArgument(value=value)
    elif type == ArgumentType.TUPLE:
        return TupleArgument(value=value)
    elif type == ArgumentType.SET:
        return SetArgument(value=value)
    elif type == ArgumentType.NODE:
        return NodeArgument(value=value)
    else:
        raise ValueError(f"Unsupported argument type: {type}")


class SessionInfo(BaseModel):
    session_id: str
    last_active: float
    remaining_ttl: float


class SessionListResponse(BaseModel):
    sessions: list[SessionInfo]


class CreateSessionResponse(BaseModel):
    session_id: str
    message: str


class KeepAliveResponse(BaseModel):
    message: str


class ResourceResponse(BaseModel):
    result: Argument
    data: Optional[dict[str, Argument]] = None


class ListResourcesResponse(BaseModel):
    resources: list[ResourceResponse]


class DeleteSessionResponse(BaseModel):
    message: str


class GetResourceResponse(ResourceResponse):
    pass
