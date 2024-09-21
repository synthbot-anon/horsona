from typing import TypeVar

from horsona.autodiff.basic import horsefunction
from horsona.autodiff.variables import HorseType, Value
from horsona.memory.caches.cache import Cache

T = TypeVar("T", bound=HorseType)


class ValueCache(Cache[Value[T], Value[T]]):
    def __init__(self, initial_value: Value[T]):
        super().__init__(initial_value)

    @horsefunction
    async def load(self, value: Value[T]):
        old_context = self.context
        new_context = value
        self.context = new_context
        yield value

        # Backprop to the old context
