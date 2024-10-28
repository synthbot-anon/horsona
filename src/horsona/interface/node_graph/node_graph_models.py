from enum import StrEnum, auto
from typing import Any, Optional, Self, Union

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


class Argument(BaseModel):
    type: ArgumentType
    value: Any

    @model_validator(mode="after")
    def validate(self) -> Self:
        if self.type in (
            ArgumentType.NONE,
            ArgumentType.UNSUPPORTED,
            ArgumentType.STR,
            ArgumentType.FLOAT,
            ArgumentType.INT,
            ArgumentType.BOOL,
        ):
            pass
        elif self.type == ArgumentType.LIST:
            for i, x in enumerate(self.value):
                if not isinstance(x, Argument):
                    self.value[i] = Argument(**x)
        elif self.type == ArgumentType.DICT:
            for k, v in self.value.items():
                if not isinstance(v, Argument):
                    self.value[k] = Argument(**v)
        elif self.type == ArgumentType.TUPLE:
            new_tuple = []
            for x in self.value:
                if isinstance(x, Argument):
                    new_tuple.append(x)
                else:
                    new_tuple.append(Argument(**x))
        elif self.type == ArgumentType.SET:
            new_set = set()
            for x in self.value:
                if isinstance(x, Argument):
                    new_set.add(x)
                else:
                    new_set.add(Argument(**x))
            self.value = new_set
        return self


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
    result: Union[dict[str, Argument], Argument]

    @model_validator(mode="after")
    def validate_result_matches_id(self) -> Self:
        if self.id is not None:
            assert isinstance(self.result, dict)
            for value in self.result.values():
                assert isinstance(value, Argument)
        else:
            assert isinstance(self.result, Argument)
            assert self.result.type != ArgumentType.NODE

        return self

    @field_validator("result")
    @classmethod
    def validate_result(cls, v):
        try:
            # First attempt to validate as dictionary
            return {k: Argument.model_validate(v) for k, v in v.items()}
        except:
            # If that fails, return as-is
            return Argument.model_validate(v)
