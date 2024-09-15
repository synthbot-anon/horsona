from typing import AsyncGenerator

from horsona.autodiff.basic import (
    GradContext,
    HorseVariable,
    horsefunction,
)
from horsona.autodiff.variables import Value
from horsona.memory.caches.cache import Cache


class ListCacheContext(HorseVariable):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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

    def extend(self, other):
        return self.data.extend(other.data)

    def append(self, item):
        return self.data.append(item)

    async def apply_gradients(self):
        raise NotImplementedError("ListCacheContext does not support gradients")


class ListCache(Cache[ListCacheContext]):
    context: ListCacheContext

    def __init__(self, size):
        super().__init__(ListCacheContext())
        self.size = size

    @horsefunction
    async def load(self, item: Value) -> AsyncGenerator[ListCacheContext, GradContext]:
        old_context = self.context
        new_context = ListCacheContext(predecessors=[old_context, item])
        new_context.extend(old_context)

        new_context.append(item)

        while len(new_context) > self.size:
            del new_context[0]

        self.context = new_context
        yield new_context

        # TODO: This should backprop gradients to items in the cache
