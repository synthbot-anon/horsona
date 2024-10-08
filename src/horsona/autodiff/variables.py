from typing import Generic, Optional, Type, TypeVar, Union

from pydantic import BaseModel

from horsona.autodiff.basic import HorseGradient, HorseVariable
from horsona.llm.base_engine import AsyncLLMEngine

HorseType = TypeVar("HorseType", bound=Union[BaseModel, dict, int, float, bool, list])


class Value(HorseVariable, Generic[HorseType]):
    value: HorseType
    VALUE_TYPE: Optional[Type[HorseType]] = None

    def __init__(self, value: HorseType, **kwargs):
        super().__init__(**kwargs)
        if self.VALUE_TYPE and isinstance(value, dict):
            value = self.VALUE_TYPE(value)
        if self.VALUE_TYPE is None:
            self.VALUE_TYPE = type(value)
        self.value = value

    async def derive(self, value: HorseType, **kwargs):
        if self.VALUE_TYPE and isinstance(value, dict):
            value = self.VALUE_TYPE(value)
        if self.VALUE_TYPE is None:
            self.VALUE_TYPE = type(value)
        return Value(value, **kwargs)

    async def json(self):
        return self.value


class Parameter(Value):
    def __init__(self, value: HorseType, updater_llm: AsyncLLMEngine, **kwargs):
        super().__init__(value, requires_grad=True, **kwargs)
        self.updater_llm = updater_llm

    async def apply_gradients(self, gradients: list[HorseGradient]):
        class UpdatedValue(BaseModel):
            final_value: type(self.value)

        update = await self.updater_llm.query_object(
            UpdatedValue,
            DATA=self,
            ERRATA=gradients,
            TASK=(
                "You are maintaining the DATA with the latest information. "
                "A user provided ERRATA to the DATA. "
                "Correct the DATA to address the ERRATA."
            ),
        )

        self.value = update.final_value
