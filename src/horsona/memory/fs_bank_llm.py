from typing import Type, TypeVar, Union

from pydantic import BaseModel

from horsona.database.embedding_database import EmbeddingDatabase
from horsona.llm.base_engine import AsyncLLMEngine, LLMMetrics
from horsona.llm.custom_llm import CustomLLMEngine
from horsona.memory.embedding_llm import get_relevant_queries
from horsona.memory.fs_bank_module import FilesystemBankModule
from horsona.memory.gist_module import GistModule
from horsona.memory.readagent_llm import get_relevant_pages

T = TypeVar("T", bound=BaseModel)
S = TypeVar("S", bound=Union[str, T])


class FilesystemBankLLMEngine(CustomLLMEngine):
    """
    A LLM engine that augments prompts with relevant context from a FilesystemBankModule.
    Retrieves both high-level gists and detailed content from files based on semantic search.

    Attributes:
        fs_bank_module (FilesystemBankModule): Module containing the indexed filesystem
        max_gist_chars (int): Maximum total characters of gists to include in context
        max_page_chars (int): Maximum total characters of detailed content to include
    """

    def __init__(
        self,
        underlying_llm: AsyncLLMEngine,
        fs_bank_module: FilesystemBankModule,
        max_gist_chars: int = 1024,
        max_page_chars: int = 2048,
        **kwargs,
    ):
        super().__init__(underlying_llm, **kwargs)
        self.fs_bank_module = fs_bank_module
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
        all_results = []
        for query, weight in relevant_queries.items():
            results = await self.fs_bank_module.embedding_db.query_with_weights(
                query, topk=100
            )
            for content, weighted_files in results.items():
                files, distance = weighted_files
                for file in files:
                    all_results.append((file, weight * distance))

        # Create lookup mappings for file metadata
        results_by_chronology = sorted(all_results, key=lambda x: x[0]["path"])
        chrono_path_indices = {
            x[0]["path"]: i for i, x in enumerate(results_by_chronology)
        }

        weights_by_path = {x[0]["path"]: x[1] for x in all_results}
        pages_by_path = {x[0]["path"]: x[0] for x in all_results}

        # Select files to include in gist context, up to max_gist_chars
        selected_files = []
        total_gist_length = 0
        for file, weight in all_results:
            total_gist_length += file["gist_length"]

            if total_gist_length < self.max_gist_chars:
                selected_files.append(file)
            else:
                break

        # Sort selected files chronologically for better LLM comprehension
        sorted_files = [
            x
            for x in sorted(
                selected_files, key=lambda x: chrono_path_indices[x["path"]]
            )
        ]

        # Find most relevant detailed content from selected files
        selected_content = await get_relevant_pages(
            self.underlying_llm,
            [x["gist"] for x in sorted_files],
            [{"path": x["path"], "content": x["content"]} for x in sorted_files],
            10,
            **kwargs,
        )

        # Select content to include, up to max_page_chars
        sorted_content = sorted(
            selected_content, key=lambda x: weights_by_path[x["path"]]
        )
        selected_content = []
        total_content_length = 0
        for content in sorted_content:
            page = pages_by_path[content["path"]]
            total_content_length += page["content_length"]

            if total_content_length < self.max_page_chars:
                selected_content.append(page)
            else:
                break

        # Sort final selections chronologically
        sorted_content = [
            x["content"] for x in sorted(selected_content, key=lambda x: x["path"])
        ]
        sorted_gists = [x["gist"] for x in sorted_files]

        return sorted_gists, sorted_content

    async def hook_prompt_args(self, **prompt_args) -> str:
        """
        Augments the prompt arguments with relevant filesystem context.

        Returns:
            dict: Original prompt args plus:
                - GIST_CONTEXT: List of relevant file gists
                - POTENTIALLY_RELEVANT_PAGES: List of relevant detailed content
        """
        sorted_gists, sorted_content = await self._get_relevant_gists(**prompt_args)
        return {
            "GIST_CONTEXT": sorted_gists,
            "POTENTIALLY_RELEVANT_PAGES": sorted_content,
            **prompt_args,
        }
