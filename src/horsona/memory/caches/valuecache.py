from horsona.autodiff.basic import horsefunction
from horsona.autodiff.variables import Value
from horsona.memory.caches.cache import Cache


class ValueCache(Cache[Value]):
    def __init__(self, initial_value: Value):
        super().__init__(initial_value)

    @horsefunction
    async def load(self, value: Value):
        old_context = self.context
        new_context = value
        self.context = new_context
        yield value

        # Backprop to the old context
