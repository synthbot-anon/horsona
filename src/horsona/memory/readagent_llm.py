from typing import Type, TypeVar, Union

from pydantic import BaseModel

from horsona.llm.base_engine import AsyncLLMEngine, LLMMetrics
from horsona.llm.engine_utils import compile_user_prompt
from horsona.llm.wrapper_llm import WrapperLLMEngine
from horsona.memory.gist_module import GistModule

T = TypeVar("T", bound=BaseModel)
S = TypeVar("S", bound=Union[str, T])


class ReadAgentLLMEngine(WrapperLLMEngine):
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

    async def hook_prompt_args(self, **prompt_args) -> str:
        return {
            "GIST_CONTEXT": self.gist_module.available_gists,
            "POTENTIALLY_RELEVANT_PAGES": await get_relevant_pages(
                self.underlying_llm,
                self.gist_module.available_gists,
                self.gist_module.available_pages,
                self.max_pages,
                **prompt_args,
            ),
            **prompt_args,
        }


async def get_relevant_pages(
    llm: AsyncLLMEngine, gists: list[str], pages: list[str], max_results: int, **kwargs
) -> dict:
    assert len(gists) == len(pages)

    # Get the main prompt/task from kwargs
    assert "TASK" in kwargs
    kwargs["READAGENT_TASK"] = kwargs.pop("TASK")
    # Retrieve relevant pages from gists

    class RelevantPages(BaseModel):
        pages: list[int | str | None] | None

    relevant_pages = await llm.query_object(
        RelevantPages,
        GISTS=gists,
        **kwargs,
        TASK=(
            "You have access to a list of available gists and pages in GISTS. "
            f"Select 0 to {max_results} gist indices whose pages might be relevant to the READAGENT_TASK. "
            "The result should only include gist indices (integers). "
            "If selecting 0 gists, return an empty list."
        ),
    )

    if relevant_pages.pages is None or len(relevant_pages.pages) == 0:
        return []

    # Clean the result
    cleaned_pages: list[int] = []
    for page in relevant_pages.pages:
        if page is None:
            continue

        if len(cleaned_pages) >= max_results:
            break

        page_index = None
        try:
            page_index = int(page)
        except ValueError:
            if "." in str(page):
                for piece in str(page).split(".")[::-1]:
                    try:
                        page_index = int(piece)
                        break
                    except ValueError:
                        continue

        if page_index is not None and page_index > 0 and page_index < len(pages):
            cleaned_pages.append(page_index)

    return [pages[i] for i in cleaned_pages]
