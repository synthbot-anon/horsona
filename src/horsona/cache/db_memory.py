from typing import AsyncGenerator, Type

from horsona.autodiff.basic import GradContext, horsefunction
from horsona.autodiff.variables import DictValue, Value
from horsona.database.base_database import (
    Database,
    DatabaseInsertGradient,
    DatabaseOpGradient,
    DatabaseTextGradient,
)
from horsona.llm.base_engine import AsyncLLMEngine
from horsona.memory.base_memory import BaseMemory


class DatabaseCache(DictValue, BaseMemory[DictValue, Value[str]]):
    def __init__(
        self,
        llm: AsyncLLMEngine,
        database: Database,
        cache_size: int,
        value=None,
        db_query_args={},
        **kwargs,
    ):
        BaseMemory.__init__(self)

        super_kwargs = kwargs.copy()
        datatype = super_kwargs.pop("datatype", "Database cache")
        DictValue.__init__(
            self,
            datatype,
            value,
            llm,
            **super_kwargs,
        )

        self.database: Database = database
        self.cache_size = cache_size
        self.db_query_args = db_query_args

    @horsefunction
    async def load(
        self, query: Value[str], **kwargs
    ) -> AsyncGenerator["DatabaseCache", None]:
        result = await self.database.query(query.value, **self.db_query_args)
        new_data = self.value.copy()

        for key, value in result.items():
            if key in new_data:
                # Remove the key since we'll be adding it to the end
                del new_data[key]
            new_data[key] = value

        while len(new_data) > self.cache_size:
            new_data.popitem(last=False)

        new_cache = DatabaseCache(
            llm=self.llm,
            database=self.database,
            cache_size=self.cache_size,
            value=new_data,
            predecessors=[self, query, self.database],
        )

        grad_context = yield new_cache

        unsorted_gradients = []

        for gradient in grad_context[new_cache]:
            if isinstance(
                gradient,
                (DatabaseTextGradient, DatabaseOpGradient, DatabaseInsertGradient),
            ):
                grad_context[self.database].append(gradient)
            else:
                unsorted_gradients.append(gradient)

        # TODO: backprop gradients to the query and old_context

    @horsefunction
    async def sync(self) -> AsyncGenerator["DatabaseCache", GradContext]:
        new_data = self.value.copy()

        for key in self.keys():
            value = await self.database.query(key, **self.db_query_args)
            new_data[key] = value

        new_cache = DatabaseCache(
            llm=self.llm,
            database=self.database,
            cache_size=self.cache_size,
            value=new_data,
            predecessors=[self, self.database],
        )

        grad_context = yield new_cache

        if self in grad_context:
            grad_context[self].extend(grad_context[new_cache])
