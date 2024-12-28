from collections import defaultdict
from typing import Type, TypeVar, Union

from pydantic import BaseModel

from horsona.database.embedding_database import EmbeddingDatabase
from horsona.llm.base_engine import AsyncLLMEngine, LLMMetrics
from horsona.llm.engine_utils import compile_user_prompt
from horsona.llm.wrapper_llm import WrapperLLMEngine
from horsona.memory.embedding_llm import get_relevant_queries
from horsona.memory.gist_module import GistModule
from horsona.memory.readagent_llm import get_relevant_pages
from horsona.memory.wiki_module import WikiModule

T = TypeVar("T", bound=BaseModel)
S = TypeVar("S", bound=Union[str, T])


class WikiLLMEngine(WrapperLLMEngine):
    """
    A LLM engine that augments prompts with relevant context from a WikiModule.
    Retrieves both high-level gists and detailed content from files based on semantic search.

    Attributes:
        wiki_module (WikiModule): Module containing the indexed wiki
        max_gist_chars (int): Maximum total characters of gists to include in context
        max_page_chars (int): Maximum total characters of detailed content to include
    """

    def __init__(
        self,
        underlying_llm: AsyncLLMEngine,
        wiki_module: WikiModule,
        max_gist_chars: int = 8096,
        max_page_chars: int = 8096,
        **kwargs,
    ):
        super().__init__(underlying_llm, **kwargs)
        self.wiki_module = wiki_module
        self.max_gist_chars = max_gist_chars
        self.max_page_chars = max_page_chars

    async def _get_relevant_gists(self, **kwargs) -> dict:
        """
        Retrieves relevant gists and content from the filesystem based on the prompt.

        Returns:
            tuple: (sorted_gists, sorted_content) where:
                - sorted_gists is a list of gist summaries in chronological order
                - sorted_content is a list of detailed file contents in chronological order
        """
        # Generate semantic search queries based on the user's prompt
        relevant_queries = await get_relevant_queries(self.underlying_llm, **kwargs)

        # Search the embedding database with each query and combine results with weights
        all_results = defaultdict(lambda: [None, 0])
        for query, weight in relevant_queries.items():
            results = await self.wiki_module.embedding_db.query_with_weights(
                query, topk=100
            )
            for weighted_file in results.values():
                file, distance = weighted_file
                for file in file:
                    all_results[file["path"]][0] = file
                    all_results[file["path"]][1] += max(
                        all_results[file["path"]][1], distance / max(1, weight)
                    )

        weights_by_path = {x[0]["path"]: x[1] for x in all_results.values()}

        # Select files to include in gist context, up to max_gist_chars
        selected_files = []
        selected_paths = set()
        total_gist_length = 0
        for file, weight in sorted(all_results.values(), key=lambda x: x[1]):
            if file["path"] in selected_paths:
                continue

            total_gist_length += file["gist_length"]

            if total_gist_length < self.max_gist_chars:
                selected_files.append(file)
                selected_paths.add(file["path"])
            else:
                break

        # Sort selected files chronologically for better LLM comprehension
        chrono_selected_files = [
            x for x in sorted(selected_files, key=lambda x: x["path"])
        ]

        # Find most relevant detailed content from selected files
        selected_pages = await get_relevant_pages(
            self.underlying_llm,
            [x["gist"] for x in chrono_selected_files],
            chrono_selected_files,
            10,
            **kwargs,
        )

        # Select pages to include
        sorted_pages = sorted(selected_pages, key=lambda x: weights_by_path[x["path"]])

        final_content = []
        total_content_length = 0
        final_gists = []
        total_gist_length = 0

        for page in sorted_pages:
            total_content_length += page["content_length"]
            total_gist_length += page["gist_length"]

            if total_content_length <= self.max_page_chars:
                final_content.append(page)

            if total_gist_length <= self.max_gist_chars:
                final_gists.append(page)

            if (
                total_content_length >= self.max_page_chars
                and total_gist_length >= self.max_gist_chars
            ):
                break

        for file, weight in sorted(all_results.values(), key=lambda x: x[1]):
            if file in final_gists:
                continue

            total_gist_length += file["gist_length"]

            if total_gist_length <= self.max_gist_chars:
                final_gists.append(file)
            else:
                break

        # Sort the results chronologically
        return [
            [x["gist"] for x in sorted(final_gists, key=lambda x: x["path"])],
            [x["content"] for x in sorted(final_content, key=lambda x: x["path"])],
        ]

    async def hook_prompt_args(self, **prompt_args) -> str:
        """
        Augments the prompt arguments with relevant filesystem context.

        Returns:
            dict: Original prompt args plus:
                - GIST_CONTEXT: List of relevant file gists
                - POTENTIALLY_RELEVANT_PAGES: List of relevant detailed content
        """
        assert "TASK" in prompt_args

        sorted_gists, sorted_content = await self._get_relevant_gists(**prompt_args)
        prompt_args["FS_BANK_TASK"] = prompt_args.pop("TASK")
        return {
            "GIST_CONTEXT": sorted_gists,
            "POTENTIALLY_RELEVANT_PAGES": sorted_content,
            **prompt_args,
            "TASK": "Use the GIST_CONTEXT and POTENTIALLY_RELEVANT_PAGES to respond to the FS_BANK_TASK.",
        }
