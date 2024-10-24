from types import NoneType
from typing import Any, Literal, Optional, TypeAlias

from pydantic import BaseModel

PrimitiveType: TypeAlias = NoneType | str | float | int | bool

ValueType: TypeAlias = (
    PrimitiveType
    | list[PrimitiveType]
    | dict[str, PrimitiveType]
    | tuple[PrimitiveType, ...]
    | set[PrimitiveType]
)


class Argument(BaseModel):
    type: Literal[
        "none",
        "unsupported",
        "str",
        "float",
        "int",
        "bool",
        "list",
        "dict",
        "tuple",
        "set",
        "node",
    ]
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
    result: dict[str, Argument] | Any
