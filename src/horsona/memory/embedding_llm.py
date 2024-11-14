from typing import AsyncGenerator, Type, TypeVar, Union

from pydantic import BaseModel

from horsona.autodiff.basic import GradContext, horsefunction
from horsona.autodiff.variables import Value
from horsona.database.base_database import Database
from horsona.database.embedding_database import EmbeddingDatabase
from horsona.llm.base_engine import AsyncLLMEngine
from horsona.llm.engine_utils import (
    compile_user_prompt,
    generate_obj_query_messages,
    parse_block_response,
    parse_obj_response,
)

T = TypeVar("T", bound=BaseModel)
S = TypeVar("S", bound=Union[str, T])


class EmbeddingLLMEngine(AsyncLLMEngine):
    def __init__(
        self,
        underlying_llm: AsyncLLMEngine,
        database: EmbeddingDatabase,
        database_query_kwargs: dict = {},
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.underlying_llm = underlying_llm
        self.database = database
        self.database_query_kwargs = database_query_kwargs

    async def query(self, **kwargs) -> tuple[str, int]:
        return await self.underlying_llm.query(**kwargs), 0

    async def _get_search_results(
        self, **kwargs
    ) -> AsyncGenerator[tuple[str, int], GradContext]:
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

    async def query_object(self, response_model: Type[T], **kwargs) -> T:
        prompt_args = {k: v for k, v in kwargs.items() if k == k.upper()}
        api_args = {k: v for k, v in kwargs.items() if k != k.upper()}

        search_results = await self._get_search_results(
            **prompt_args,
        )

        # Compile responses using the underlying LLM
        return await self.underlying_llm.query_object(
            response_model,
            RETRIEVED_CONTEXT=search_results,
            **kwargs,
        )

    async def query_block(self, block_type: str, **kwargs) -> str:
        prompt_args = {k: v for k, v in kwargs.items() if k == k.upper()}
        api_args = {k: v for k, v in kwargs.items() if k != k.upper()}

        search_results = await self._get_search_results(
            **prompt_args,
        )

        return await self.underlying_llm.query_block(
            block_type,
            RETRIEVED_CONTEXT=search_results,
            **kwargs,
        )

    async def query_continuation(self, prompt: str, **kwargs) -> str:
        api_args = {k: v for k, v in kwargs.items() if k != k.upper()}
        prompt_args = {k: v for k, v in kwargs.items() if k == k.upper()}

        search_results = await self._get_search_results(
            PROMPT=prompt,
            **prompt_args,
        )

        return await self.underlying_llm.query_continuation(
            prompt,
            RETRIEVED_CONTEXT=search_results,
            **kwargs,
        )
