from typing import AsyncGenerator, Generic, TypeVar

from horsona.autodiff.basic import GradContext, horsefunction
from horsona.autodiff.variables import HorseType, ListValue, Value
from horsona.cache.base_cache import BaseCache

T = TypeVar("T", bound=HorseType)


class ListCache(ListValue, BaseCache[ListValue, Value[T]], Generic[T]):
    def __init__(self, max_size, data=None, **kwargs):
        ListValue.__init__(self, data=data, **kwargs)
        BaseCache.__init__(self)
        self.max_size = max_size

    @horsefunction
    async def load(self, item: Value[T]) -> AsyncGenerator[ListValue, GradContext]:
        new_data = self.data + [item]
        if len(new_data) > self.max_size:
            new_data = new_data[-self.max_size :]

        new_context = ListCache(
            max_size=self.max_size,
            data=new_data,
            name=self.name,
            predecessors=[self, item],
        )

        yield new_context

        # TODO: This should backprop gradients to items in the cache

    async def sync(self) -> ListValue:
        return self
