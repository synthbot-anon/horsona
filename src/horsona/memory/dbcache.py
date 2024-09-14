from collections import OrderedDict, defaultdict

from horsona.autodiff.basic import (HorseFunction, HorseGradient, HorseModule,
                                    HorseVariable)
from horsona.autodiff.variables import Value
from horsona.llm.base_engine import AsyncLLMEngine
from horsona.memory.database import (Database, DatabaseInsertGradient,
                                     DatabaseOpGradient, DatabaseTextGradient)


class DatabaseCacheContext(HorseVariable):
    gradients: list[None]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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


class LoadDataFunction(HorseFunction):
    async def forward(
        self,
        llm: AsyncLLMEngine,
        query: Value,
        cache_context: DatabaseCacheContext,
        database: Database,
        cache_size: int,
        **kwargs
    ) -> HorseVariable:
        if not isinstance(query.value, str):
            raise ValueError("Query must be a string")

        result = await database.query(query.value, **kwargs)
        new_context = DatabaseCacheContext(
            predecessors=[query, cache_context, database]
        )
        new_context.update(cache_context)

        for key, value in result.items():
            if key in new_context:
                # Remove the key since we'll be adding it to the end
                del new_context[key]
            new_context[key] = value

        while len(new_context) > cache_size:
            new_context.popitem(last=False)

        return new_context

    async def backward(
        self,
        context: dict[HorseVariable, list[HorseGradient]],
        result: DatabaseCacheContext,
        llm: AsyncLLMEngine,
        query: Value,
        cache_context: DatabaseCacheContext,
        database: Database,
        cache_size: int,
        **kwargs
    ):
        if result not in context:
            return

        unsorted_gradients = []

        g: dict[HorseVariable, list] = defaultdict(list)

        for gradient in context[result]:
            if isinstance(
                gradient,
                (DatabaseTextGradient, DatabaseOpGradient, DatabaseInsertGradient),
            ):
                g[database].append(gradient)
            else:
                unsorted_gradients.append(gradient)

        return g


"""
Proper cache implementation:

load: database, cache, query -> cache
sync: database, cache -> cache
"""


class DatabaseCache(HorseModule):
    def __init__(
        self,
        llm: AsyncLLMEngine,
        database: Database,
        cache_size: int,
    ):
        super().__init__()
        self.llm = llm
        self.database: Database = database
        self.cache_size = cache_size

        self.context = DatabaseCacheContext()
        self.load_fn = LoadDataFunction()

    async def load(self, query: Value, **kwargs) -> DatabaseCacheContext:
        self.context = await self.load_fn(
            self.llm, query, self.context, self.database, self.cache_size, **kwargs
        )
        return self.context

    async def sync(self) -> DatabaseCacheContext:
        pass

    async def json(self):
        return self.context
