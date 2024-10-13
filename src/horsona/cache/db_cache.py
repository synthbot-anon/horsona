from collections import OrderedDict
from typing import AsyncGenerator, Type

from horsona.autodiff.basic import (
    GradContext,
    HorseVariable,
    horsefunction,
    load_state_dict,
    state_dict,
)
from horsona.autodiff.variables import Value
from horsona.cache.base_cache import BaseCache
from horsona.database.base_database import (
    Database,
    DatabaseInsertGradient,
    DatabaseOpGradient,
    DatabaseTextGradient,
)
from horsona.llm.base_engine import AsyncLLMEngine


class DatabaseCacheContext(HorseVariable):
    def __init__(self, data=None, **kwargs):
        if "requires_grad" in kwargs:
            assert not kwargs["requires_grad"]

        super().__init__(**kwargs)
        if data is not None:
            self.data = OrderedDict(data)
        else:
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

    def popitem(self, last=True):
        return self.data.popitem(last=last)

    def state_dict(self):
        return super().state_dict(data=list(self.data.items()))


class DatabaseCache(BaseCache[DatabaseCacheContext, Value[str]]):
    context: DatabaseCacheContext

    def __init__(
        self,
        llm: AsyncLLMEngine,
        database: Database,
        cache_size: int,
        context=None,
        name: str = None,
        db_query_args={},
        **kwargs,
    ):
        if context is None:
            if name is not None:
                context = DatabaseCacheContext(name=f"{name}Context")
            else:
                context = DatabaseCacheContext()
        super().__init__(context, name=name, **kwargs)
        self.llm = llm
        self.database: Database = database
        self.cache_size = cache_size
        self.db_query_args = db_query_args

    @horsefunction
    async def load(
        self, query: Value[str]
    ) -> AsyncGenerator[DatabaseCacheContext, None]:
        if not isinstance(query.value, str):
            raise ValueError("Query must be a string")

        old_context = self.context
        result = await self.database.query(query.value, **self.db_query_args)
        new_context = DatabaseCacheContext(
            predecessors=[old_context, query, self.database]
        )
        new_context.update(old_context)

        for key, value in result.items():
            if key in new_context:
                # Remove the key since we'll be adding it to the end
                del new_context[key]
            new_context[key] = value

        while len(new_context) > self.cache_size:
            new_context.popitem(last=False)

        self.context = new_context
        grad_context = yield new_context

        if new_context not in grad_context:
            return

        unsorted_gradients = []

        for gradient in grad_context[new_context]:
            if isinstance(
                gradient,
                (DatabaseTextGradient, DatabaseOpGradient, DatabaseInsertGradient),
            ):
                grad_context[self.database].append(gradient)
            else:
                unsorted_gradients.append(gradient)

        # TODO: backprop gradients to the query and old_context

    @horsefunction
    async def sync(self) -> AsyncGenerator[DatabaseCacheContext, GradContext]:
        old_context = self.context
        new_context = DatabaseCacheContext(
            predecessors=[self.database, old_context],
        )

        for key in self.context.keys():
            value = await self.database.query(key, **self.db_query_args)
            new_context[key] = value

        grad_context = yield new_context

        if old_context in grad_context:
            grad_context[old_context].extend(grad_context[new_context])
