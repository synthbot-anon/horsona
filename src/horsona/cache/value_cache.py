from typing import TypeVar

from horsona.autodiff.basic import horsefunction
from horsona.autodiff.variables import HorseType, Value
from horsona.cache.base_cache import BaseCache

T = TypeVar("T", bound=HorseType)


class ValueCache(BaseCache[Value[T], Value[T]]):
    def __init__(self, context: Value[T], **kwargs):
        super().__init__(context, **kwargs)

    @horsefunction
    async def load(self, value: Value[T]):
        old_context = self.context
        new_context = value
        self.context = new_context
        yield value

        # Backprop to the old context
