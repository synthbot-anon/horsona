from typing import Literal, OrderedDict, Union

from pydantic import BaseModel

from horsona.autodiff.basic import HorseVariable
from horsona.llm.base_engine import AsyncLLMEngine
from horsona.memory.database import Database


class CacheUpdate(BaseModel):
    operation: Literal["UPDATE"]
    key: str
    updated_data: str


class CacheDelete(BaseModel):
    operation: Literal["DELETE"]
    key: str


class CacheNoChange(BaseModel):
    operation: Literal["NO_CHANGE"]
    key: str


class CacheChanges(BaseModel):
    changes: list[Union[CacheUpdate, CacheDelete, CacheNoChange]]


class DatabaseCache(HorseVariable):
    def __init__(
        self,
        llm: AsyncLLMEngine,
        database: Database,
        cache_size: int,
        context=None,
        requires_grad=True,
    ):
        super().__init__(requires_grad=requires_grad)
        self.llm = llm
        self.database: Database = database
        self.cache_size = cache_size

        if context is None:
            context = OrderedDict()
        self.context = context

    async def load_data(self, query, **kwargs):
        result = await self.database.query(query, **kwargs)
        new_context = self.context.copy()

        for _, key in result.items():
            if not await self.database.contains(key):
                continue
            if key in new_context:
                # Remove the key since we'll be adding it to the end
                del new_context[key]
            new_context[key] = await self.database.get(key)

        while len(new_context) > self.cache_size:
            new_context.popitem(last=False)

        self.context = new_context

    async def json(self):
        return self.context

    async def apply_gradients(self):
        if not self.gradients:
            return

        response = await self.llm.query_object(
            CacheChanges,
            ERRATA=self.gradients,
            DATASET=self.context,
            TASK=(
                # "The DATASET consists of lookups and corresponding data. "
                "You are maintaining the DATASET with the latest information. "
                "A user provided ERRATA to the DATASET. "
                "Clean up the underlying DATASET. Only update or delete "
                "rows that are directly relevant to the ERRATA. "
                "Change the DATASET as little as possible "
                "to address the ERRATA. Include all of the information from the ERRATA verbatim. "
            ),
        )

        changes = []
        for change in response.changes:
            result = await self.database.query(change.key)
            if not result:
                continue
            doc_index, quesy = result.popitem()
            if quesy in self.context:
                changes.append((doc_index, quesy, change))

        for doc_index, query, change in changes:
            if not isinstance(change, CacheUpdate):
                continue
            await self.database.replace(query, change.updated_data)
            self.context[query] = change.updated_data
        for doc_index, query, change in changes:
            if not isinstance(change, CacheDelete):
                continue
            if query in self.context:
                del self.context[query]
                await self.database.delete(doc_index)
