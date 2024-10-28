from typing import AsyncGenerator, Generic, TypeVar

from horsona.autodiff.basic import GradContext, horsefunction
from horsona.autodiff.variables import HorseType, ListValue, Value

T = TypeVar("T", bound=HorseType)


class ListCache(ListValue, Generic[T]):
    def __init__(self, max_size, value=None, **kwargs):
        super_kwargs = kwargs.copy()
        datatype = super_kwargs.pop("datatype", "Recent items")
        ListValue.__init__(
            self,
            datatype,
            value,
            **super_kwargs,
        )
        self.max_size = max_size

    async def load(self, item: Value[T]) -> AsyncGenerator[ListValue, GradContext]:
        new_data = self.value + [item]
        if len(new_data) > self.max_size:
            new_data = new_data[-self.max_size :]

        new_context = ListCache(
            max_size=self.max_size,
            value=new_data,
            name=self.name,
            predecessors=[self, item],
        )

        return new_context
