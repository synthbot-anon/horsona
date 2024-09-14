from horsona.autodiff.basic import (HorseFunction, HorseGradient, HorseModule,
                                    HorseVariable)
from horsona.autodiff.variables import Value


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


class LoadDataFunction(HorseFunction):
    async def forward(
        self, item: HorseVariable, cache_context: ListCacheContext, cache_size: int
    ) -> HorseVariable:
        new_context = ListCacheContext()
        new_context.extend(cache_context)

        new_context.append(item)

        while len(new_context) > cache_size:
            del new_context[0]

        return new_context

    async def backward(
        self,
        context: dict[HorseVariable, list[HorseGradient]],
        result: HorseVariable,
        item: HorseVariable,
        cache_context: ListCacheContext,
        cache_size: int,
    ):
        return


class ListCache(HorseModule):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit
        self.load_fn = LoadDataFunction()
        self.context = ListCacheContext()

    async def load(
        self, item: Value, context: ListCacheContext = ListCacheContext()
    ) -> ListCacheContext:
        self.context = await self.load_fn(item, self.context, self.limit)
        return self.context
