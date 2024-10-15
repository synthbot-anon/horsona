from typing import AsyncGenerator, Generic, TypeVar

from horsona.autodiff.basic import (
    GradContext,
    HorseVariable,
    horsefunction,
    load_state_dict,
    state_dict,
)
from horsona.autodiff.variables import HorseType, Value
from horsona.cache.base_cache import BaseCache


class ListCacheContext(HorseVariable):
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

    def extend(self, other):
        return self.data.extend(other.data)

    def append(self, item):
        return self.data.append(item)

    async def apply_gradients(self):
        pass


T = TypeVar("T", bound=HorseType)


class ListCache(BaseCache[ListCacheContext, Value[T]], Generic[T]):
    context: ListCacheContext

    def __init__(self, size, context=None, name=None, **kwargs):
        if context == None:
            if name is not None:
                context = ListCacheContext(name=f"{name}Context")
            else:
                context = ListCacheContext()

        super().__init__(context, name=name, **kwargs)
        self.size = size

    @horsefunction
    async def load(
        self, item: Value[T]
    ) -> AsyncGenerator[ListCacheContext, GradContext]:
        old_context = self.context
        new_context = ListCacheContext(predecessors=[old_context, item])
        new_context.extend(old_context)

        new_context.append(item)

        while len(new_context) > self.size:
            del new_context[0]

        self.context = new_context
        yield new_context

        # TODO: This should backprop gradients to items in the cache
