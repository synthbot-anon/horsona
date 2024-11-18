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
        # Convert prompt into search queries
        class Search(BaseModel):
            queries: list[str]

        kwargs_clone = kwargs.copy()
        if "TASK" in kwargs_clone:
            kwargs_clone["__USER_TASK"] = kwargs_clone.pop("TASK")
            prompt_key = "__USER_TASK"
        else:
            prompt_key = list(kwargs_clone.keys())[-1]

        search = await self.underlying_llm.query_object(
            Search,
            **kwargs_clone,
            TASK=(
                f"You are trying to understand the given {prompt_key}. "
                "You have access to a search engine that can retrieve relevant information. "
                f"Suggest keyword search queries that would provide better context for understanding the {prompt_key}."
            ),
        )

        search_queries = Value(
            "Search queries",
            [Value("Search query", x, predecessors=[]) for x in search.queries],
            predecessors=[],
        )

        # Look up responses for each search query
        search_results = {}
        for q in search_queries.value:
            result = await self.database.query(q.value, **self.database_query_kwargs)
            search_results.update(result)

        return search_results

    async def hook_prompt_args(self, **prompt_args) -> T:
        return {
            "RETRIEVED_CONTEXT": await self._get_search_results(
                **prompt_args,
            ),
            **prompt_args,
        }
