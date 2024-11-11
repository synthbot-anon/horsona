from typing import Type, TypeVar, Union

from pydantic import BaseModel

from horsona.llm.base_engine import AsyncLLMEngine
from horsona.memory.gist_module import GistModule

T = TypeVar("T", bound=BaseModel)
S = TypeVar("S", bound=Union[str, T])


class ReadAgentLLMEngine(AsyncLLMEngine):
    def __init__(
        self,
        underlying_llm: AsyncLLMEngine,
        gist_module: GistModule,
        max_pages: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.underlying_llm = underlying_llm
        self.gist_module = gist_module
        self.max_pages = max_pages

    async def query(self, **kwargs) -> tuple[str, int]:
        return await self.underlying_llm.query(**kwargs), 0

    async def _get_gist_context(self, **kwargs) -> dict:
        # Get the main prompt/task from kwargs
        kwargs_clone = kwargs.copy()
        if "TASK" in kwargs_clone:
            kwargs_clone["__USER_TASK"] = kwargs_clone.pop("TASK")
            prompt_key = "__USER_TASK"
        else:
            prompt_key = list(kwargs_clone.keys())[-1]

        # Retrieve relevant pages from gists
        class RelevantPages(BaseModel):
            pages: list[int]

        relevant_pages = await self.underlying_llm.query_object(
            RelevantPages,
            GISTS=self.gist_module.available_gists,
            **kwargs_clone,
            TASK=(
                "You have access to a list of available gists and pages in GISTS. "
                f"Select 0 to {self.max_pages} items that are relevant to the {prompt_key}. "
            ),
        )

        target_pages = []
        for i in reversed(sorted(relevant_pages.pages)):
            if i > 0 and i < len(self.gist_module.available_pages):
                target_pages.append(i)
            if len(target_pages) == self.max_pages:
                break

        return [self.gist_module.available_pages[i] for i in target_pages]

    async def query_object(self, response_model: Type[T], **kwargs) -> T:
        api_args = {k: v for k, v in kwargs.items() if k != k.upper()}
        prompt_args = {k: v for k, v in kwargs.items() if k == k.upper()}

        # Get gist context
        retrieved_pages = await self._get_gist_context(**prompt_args)

        # Add gist context to kwargs and query underlying LLM
        return await self.underlying_llm.query_object(
            response_model,
            GIST_CONTEXT=self.gist_module.available_gists,
            POTENTIALLY_RELEVANT_PAGES=retrieved_pages,
            **kwargs,
        )

    async def query_block(self, block_type: str, **kwargs) -> str:
        api_args = {k: v for k, v in kwargs.items() if k != k.upper()}
        prompt_args = {k: v for k, v in kwargs.items() if k == k.upper()}

        # Get gist context
        retrieved_pages = await self._get_gist_context(**prompt_args)

        # Add gist context to kwargs and query underlying LLM
        return await self.underlying_llm.query_block(
            block_type,
            GIST_CONTEXT=self.gist_module.available_gists,
            POTENTIALLY_RELEVANT_PAGES=retrieved_pages,
            **kwargs,
        )

    async def query_continuation(self, prompt: str, **kwargs) -> str:
        api_args = {k: v for k, v in kwargs.items() if k != k.upper()}
        prompt_args = {k: v for k, v in kwargs.items() if k == k.upper()}

        # Get gist context
        retrieved_pages = await self._get_gist_context(PROMPT=prompt, **prompt_args)

        # Add gist context to kwargs and query underlying LLM
        return await self.underlying_llm.query_continuation(
            prompt,
            GIST_CONTEXT=self.gist_module.available_gists,
            POTENTIALLY_RELEVANT_PAGES=retrieved_pages,
            **kwargs,
        )
