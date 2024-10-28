from enum import StrEnum, auto
from types import NoneType
from typing import Any, Optional, TypeAlias, Union

from pydantic import BaseModel, field_validator

PrimitiveType: TypeAlias = NoneType | str | float | int | bool

ValueType: TypeAlias = (
    PrimitiveType
    | list[PrimitiveType]
    | dict[str, PrimitiveType]
    | tuple[PrimitiveType, ...]
    | set[PrimitiveType]
)


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


class Argument(BaseModel):
    type: ArgumentType
    value: ValueType


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
    id: int
    result: dict[str, Argument]


class ListResourcesResponse(BaseModel):
    resources: list[ResourceResponse]


class DeleteSessionResponse(BaseModel):
    message: str


class GetResourceResponse(ResourceResponse):
    pass


class PostResourceRequest(BaseModel):
    module_name: str
    class_name: Optional[str] = None
    function_name: str
    kwargs: dict[str, Argument] = {}


class PostResourceResponse(BaseModel):
    id: Optional[int]
    result: Union[dict[str, Argument], Any]

    @field_validator("result")
    @classmethod
    def validate_result(cls, v):
        try:
            # First attempt to validate as dictionary
            return {k: Argument.model_validate(v) for k, v in v.items()}
        except:
            # If that fails, return as-is
            return Argument.model_validate(v)
