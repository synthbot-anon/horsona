from collections import defaultdict
from typing import TypeVar, Union

from pydantic import BaseModel

from horsona.autodiff.variables import Value
from horsona.database.embedding_database import EmbeddingDatabase
from horsona.llm.base_engine import AsyncLLMEngine, LLMMetrics
from horsona.llm.chat_engine import AsyncChatEngine
from horsona.llm.custom_llm import CustomLLMEngine

T = TypeVar("T", bound=BaseModel)
S = TypeVar("S", bound=Union[str, T])


class EmbeddingLLMEngine(CustomLLMEngine):
    def __init__(
        self,
        underlying_llm: AsyncLLMEngine,
        database: EmbeddingDatabase,
        database_query_kwargs: dict = {},
        **kwargs,
    ):
        super().__init__(underlying_llm, **kwargs)
        self.database = database
        self.database_query_kwargs = database_query_kwargs

    async def _get_search_results(self, **kwargs) -> dict[str, str]:
        searches = await get_relevant_queries(self.underlying_llm, **kwargs)

        search_queries = Value(
            "Search queries",
            [Value("Search query", x, predecessors=[]) for x in searches.keys()],
            predecessors=[],
        )

        # Look up responses for each search query
        search_results = defaultdict(lambda: [])
        for q in search_queries.value:
            result = await self.database.query(q.value, **self.database_query_kwargs)
            for key, value in result.items():
                search_results[key].append(value)

        return search_results

    async def hook_prompt_args(self, **prompt_args) -> T:
        return {
            "RETRIEVED_CONTEXT": await self._get_search_results(
                **prompt_args,
            ),
            **prompt_args,
        }


async def get_relevant_queries(llm: AsyncLLMEngine, **kwargs) -> dict[str, int]:
    # Convert prompt into search queries
    class Search(BaseModel):
        queries: dict[str, int]

    if "TASK" in kwargs:
        kwargs["__USER_TASK"] = kwargs.pop("TASK")
        prompt_key = "__USER_TASK"
    elif kwargs:
        prompt_key = list(kwargs.keys())[-1]
    else:
        prompt_key = "request"

    search = await llm.query_object(
        Search,
        **kwargs,
        TASK=(
            f"You are trying to understand the given {prompt_key}. "
            "You have access to a search engine that can retrieve relevant information. "
            f"Suggest keyword search queries that would provide better context for understanding the {prompt_key}. "
            "For each query, also specify a weight between 0 and 10 that indicates its importance."
        ),
    )

    return search.queries
