from collections import OrderedDict
from typing import (
    AsyncGenerator,
    Generic,
    ItemsView,
    KeysView,
    Optional,
    Type,
    TypeVar,
    Union,
    ValuesView,
)

from pydantic import BaseModel

from horsona.autodiff.basic import (
    GradContext,
    HorseGradient,
    HorseVariable,
    horsefunction,
)
from horsona.llm.base_engine import AsyncLLMEngine

HorseType = TypeVar("HorseType", bound=Union[BaseModel, dict, int, float, bool, list])


class Value(HorseVariable, Generic[HorseType]):
    """
    A generic class representing a HorseType, which can be a BaseModel, dict, int, float, bool, or list.
    For constant values, use class Value. For parameters that can be updated, use class Parameter.

    Attributes:
        value (HorseType): The wrapped value.
        VALUE_TYPE (Optional[Type[HorseType]]): The type of the value.

    Parameters:
        value (HorseType): The initial value to be wrapped.
        **kwargs: Additional keyword arguments to be passed to the parent HorseVariable class.

    Gradients:
        Value does not support gradients.
        Parameter gradients can be any JSON-compatible type. The LLM updater interprets
        these gradients to update the parameter's value.

    Example:
        >>> v = Value(42)
        >>> print(v.value)
        42
        >>> v = Value({"x": 1, "y": 2})
        >>> print(v.value)
        {'x': 1, 'y': 2}

    Example:
        >>> llm = AsyncLLMEngine(...)
        >>> param = Parameter(42, llm)
        >>> await param.apply_gradients([{"change": 5}])
    """

    datatype: str
    value: HorseType
    VALUE_TYPE: Optional[Type[HorseType]] = None

    def __init__(
        self,
        datatype: str,
        value: HorseType,
        llm: Optional[AsyncLLMEngine] = None,
        **kwargs,
    ) -> None:
        if self.VALUE_TYPE and isinstance(value, dict):
            value = self.VALUE_TYPE(value)
        if self.VALUE_TYPE is None:
            self.VALUE_TYPE = type(value)

        if "name" not in kwargs:
            kwargs["name"] = datatype.replace(" ", "_").lower()

        super().__init__(**kwargs)

        self.datatype = datatype
        self.value = value
        self.llm = llm

    async def derive(self, value: HorseType, **kwargs) -> "Value[HorseType]":
        """
        Create a new Value instance with the given value.

        Args:
            value (HorseType): The new value to be wrapped.

        Returns:
            Value: A new Value instance with the given value.
        """
        if self.VALUE_TYPE and isinstance(value, dict):
            value = self.VALUE_TYPE(value)
        if self.VALUE_TYPE is None:
            self.VALUE_TYPE = type(value)
        return Value(self.datatype, value, **kwargs)

    async def apply_gradients(self, gradients: list[HorseGradient]) -> None:
        if not self.llm:
            raise ValueError(
                f"Cannot apply gradients to {self} without an updater LLM."
            )

        class UpdatedValue(BaseModel):
            final_value: type(self.value)

        update = await self.llm.query_object(
            UpdatedValue,
            DATA=self,
            ERRATA=gradients,
            DATATYPE=self.datatype,
            TASK=(
                "You are maintaining the DATA, which is an instance of DATATYPE. "
                "The ERRATA applies the DATA. "
                "Revise the DATA to resolve the ERRATA. "
                "Make sure the revised DATA is an instance of the same DATATYPE."
            ),
        )

        self.value = update.final_value

    async def json(self) -> HorseType:
        return self.value

    def __iter__(self):
        return iter(self.value)

    def __getitem__(self, key) -> HorseType | "Value":
        return self.value[key]

    def __setitem__(self, key, value) -> None:
        self.value[key] = value


class DictValue(Value[dict]):
    def __init__(
        self,
        datatype: str,
        value: Optional[dict] = None,
        llm: Optional[AsyncLLMEngine] = None,
        **kwargs,
    ) -> None:
        if value is not None:
            value = OrderedDict(value)
        else:
            value = OrderedDict()
        super().__init__(datatype, value, llm, **kwargs)

    async def json(self) -> dict:
        return self.value

    def __getitem__(self, key) -> HorseType | Value:
        return self.value[key]

    def __setitem__(self, key, value) -> None:
        self.value[key] = value

    def __delitem__(self, key) -> None:
        del self.value[key]

    def __contains__(self, key) -> bool:
        return key in self.value

    def update(self, other: "DictValue") -> None:
        self.value.update(other.value)

    def __len__(self) -> int:
        return len(self.value)

    def keys(self) -> KeysView:
        return self.value.keys()

    def values(self) -> ValuesView:
        return self.value.values()

    def items(self) -> ItemsView:
        return self.value.items()

    def popitem(self, last: bool = True) -> tuple:
        return self.value.popitem(last=last)


class ListValue(Value[list]):
    def __init__(
        self,
        datatype: str,
        value: Optional[list] = None,
        llm: Optional[AsyncLLMEngine] = None,
        **kwargs,
    ) -> None:
        if value is not None:
            value = value
        else:
            value = []
        super().__init__(datatype, value, llm, **kwargs)

    async def json(self) -> list:
        return self.value

    def __getitem__(self, key) -> HorseType | Value:
        return self.value[key]

    def __setitem__(self, key, value) -> None:
        self.value[key] = value

    def __delitem__(self, key) -> None:
        del self.value[key]

    def __contains__(self, key) -> bool:
        return key in self.value

    def __len__(self) -> int:
        return len(self.value)

    @horsefunction
    async def extend(
        self, other: "ListValue"
    ) -> AsyncGenerator[HorseVariable, GradContext]:
        result = ListValue(
            self.datatype,
            self.value + other.value,
            self.llm or other.llm,
            predecessors=[self, other],
        )
        yield result

    @horsefunction
    async def append(
        self, item: HorseType
    ) -> AsyncGenerator[HorseVariable, GradContext]:
        result = ListValue(
            self.datatype,
            self.value + [item],
            self.llm,
            predecessors=[self],
        )
        yield result

    async def apply_gradients(self, gradients: list[HorseGradient]) -> None:
        pass
