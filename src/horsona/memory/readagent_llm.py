from typing import Type, TypeVar, Union

from pydantic import BaseModel

from horsona.llm.base_engine import AsyncLLMEngine, LLMMetrics
from horsona.llm.custom_llm import CustomLLMEngine
from horsona.memory.gist_module import GistModule

T = TypeVar("T", bound=BaseModel)
S = TypeVar("S", bound=Union[str, T])


class ReadAgentLLMEngine(CustomLLMEngine):
    def __init__(
        self,
        underlying_llm: AsyncLLMEngine,
        gist_module: GistModule,
        max_pages: int = 3,
        **kwargs,
    ):
        super().__init__(underlying_llm, **kwargs)
        self.gist_module = gist_module
        self.max_pages = max_pages

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
            pages: list[int | str | None] | None

        relevant_pages = await self.underlying_llm.query_object(
            RelevantPages,
            GISTS=self.gist_module.available_gists,
            **kwargs_clone,
            TASK=(
                "You have access to a list of available gists and pages in GISTS. "
                f"Select 0 to {self.max_pages} items that are relevant to the {prompt_key}. "
                "The result should only include page indices (integers). "
                "If selecting 0 pages, return an empty list."
            ),
        )

        # Clean the result
        cleaned_pages: list[int] = []
        for page in relevant_pages.pages:
            if page is None:
                continue

            try:
                cleaned_pages.append(int(page))
                continue
            except ValueError:
                pass

            if "." in str(page):
                for piece in str(page).split(".")[::-1]:
                    try:
                        cleaned_pages.append(int(piece))
                        break
                    except ValueError:
                        continue

        target_pages = []
        for i in reversed(sorted(cleaned_pages)):
            if i > 0 and i < len(self.gist_module.available_pages):
                target_pages.append(i)
            if len(target_pages) == self.max_pages:
                break

        return [self.gist_module.available_pages[i] for i in target_pages]

    async def hook_prompt_args(self, **prompt_args) -> str:
        return {
            "GIST_CONTEXT": self.gist_module.available_gists,
            "POTENTIALLY_RELEVANT_PAGES": await self._get_gist_context(**prompt_args),
            **prompt_args,
        }
