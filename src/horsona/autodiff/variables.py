from collections import OrderedDict
from typing import Generic, Optional, Type, TypeVar, Union

from pydantic import BaseModel

from horsona.autodiff.basic import HorseGradient, HorseVariable, horsefunction
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
        updater_llm: AsyncLLMEngine = None,
        **kwargs,
    ):
        if self.VALUE_TYPE and isinstance(value, dict):
            value = self.VALUE_TYPE(value)
        if self.VALUE_TYPE is None:
            self.VALUE_TYPE = type(value)

        if "name" not in kwargs:
            kwargs["name"] = datatype.replace(" ", "_").lower()

        super().__init__(**kwargs)

        self.datatype = datatype
        self.value = value
        self.updater_llm = updater_llm

    async def derive(self, value: HorseType, **kwargs):
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

    async def apply_gradients(self, gradients: list[HorseGradient]):
        if not self.updater_llm:
            raise ValueError(
                f"Cannot apply gradients to {self} without an updater LLM."
            )

        class UpdatedValue(BaseModel):
            final_value: type(self.value)

        update = await self.updater_llm.query_object(
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

    async def json(self):
        return self.value

    def __iter__(self):
        return iter(self.value)

    def __getitem__(self, key):
        return self.value[key]

    def __setitem__(self, key, value):
        self.value[key] = value


class DatabaseValue(HorseVariable):
    def __init__(self, data=None, **kwargs):
        super().__init__(**kwargs)
        if data is not None:
            self.data = OrderedDict(data)
        else:
            self.data = OrderedDict()

    async def json(self):
        return self.data

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def __contains__(self, key):
        return key in self.data

    def update(self, other):
        self.data.update(other.data)

    def __len__(self):
        return len(self.data)

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def popitem(self, last=True):
        return self.data.popitem(last=last)

    def state_dict(self):
        return super().state_dict(data=list(self.data.items()))


class ListValue(HorseVariable):
    def __init__(self, data=None, **kwargs):
        super().__init__(**kwargs)
        if data is not None:
            self.data = data
        else:
            self.data = []

    async def json(self):
        return self.data

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def __contains__(self, key):
        return key in self.data

    def __len__(self):
        return len(self.data)

    @horsefunction
    def extend(self, other):
        result = ListValue(self.data + other.data, predecessors=[self, other])
        yield result

    @horsefunction
    def append(self, item):
        result = ListValue(self.data + [item], predecessors=[self])
        yield result

    async def apply_gradients(self, gradients: list[HorseGradient]):
        pass
