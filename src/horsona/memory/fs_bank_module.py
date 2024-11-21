import asyncio
from collections import defaultdict
from typing import Dict

from pydantic import BaseModel

from horsona.autodiff.basic import HorseModule
from horsona.autodiff.variables import Value
from horsona.database.embedding_database import EmbeddingDatabase
from horsona.llm.base_engine import AsyncLLMEngine
from horsona.llm.engine_utils import compile_user_prompt
from horsona.memory.gist_module import GistModule, paginate


class FilesystemBankModule(HorseModule):
    """
    A module that manages multiple GistModules organized like a filesystem.
    Each "folder" corresponds to a GistModule that creates summaries of files added to it.
    Files are indexed using an EmbeddingDatabase for retrieval.

    Attributes:
        llm (AsyncLLMEngine): The language model engine used for generating gists and keywords
        embedding_db (EmbeddingDatabase): Database for indexing and retrieving files by semantic similarity
        page_size (int): Maximum number of characters per page when splitting files
        folders (Dict[str, GistModule]): Mapping of folder names to their GistModules
        all_paths (list[str]): Sorted list of all file paths in the filesystem
        **kwargs: Additional keyword arguments for parent HorseModule
    """

    def __init__(
        self,
        llm: AsyncLLMEngine,
        embedding_db: EmbeddingDatabase,
        page_size: int = 1000,
        files: dict[str, GistModule] = None,
        all_paths: list[str] = None,
        guidelines: str | None = None,
        **kwargs,
    ):
        """
        Initialize the FilesystemBankModule.

        Args:
            llm (AsyncLLMEngine): The language model engine to use
            embedding_db (EmbeddingDatabase): Database for indexing and retrieving files
            page_size (int): Maximum characters per page when splitting files
            files (dict[str, GistModule], optional): Pre-existing folder mapping
            all_paths (list[str], optional): Pre-existing sorted list of file paths
            guidelines (str | None): Optional guidelines for how files should be summarized
            **kwargs: Additional keyword arguments for parent HorseModule
        """
        super().__init__(**kwargs)
        self.llm = llm
        self.embedding_db = embedding_db
        self.page_size = page_size
        self.files: dict[str, GistModule] = files or {}
        self.all_paths = all_paths or []
        self.guidelines = guidelines

    def insert_path(self, path: str):
        """
        Insert a new path into the sorted all_paths list.
        Maintains lexicographical ordering and prevents duplicates.

        Args:
            path (str): The file path to insert
        """
        insert_idx = 0
        for i, existing_path in enumerate(self.all_paths):
            if path < existing_path:
                insert_idx = i
                break
            elif path == existing_path:
                # Path already exists, no need to insert
                return
            insert_idx = i + 1

        self.all_paths.insert(insert_idx, path)

    def create_file(
        self,
        file_path: str,
    ) -> GistModule:
        """
        Create a new folder with its own GistModule if it doesn't exist.

        Args:
            folder_name (str): Name of the folder to create
            guidelines (Value[str] | None): Optional guidelines for how files in this folder should be summarized

        Returns:
            GistModule: The new or existing GistModule for this folder
        """
        if file_path not in self.files:
            self.files[file_path] = GistModule(
                llm=self.llm,
                guidelines=self.guidelines,
            )

        return self.files[file_path]

    async def add_file(
        self,
        filepath: str,
        content: Value[str],
        **kwargs,
    ) -> GistModule | None:
        """
        Add a file to a folder, create its gist, and index it for search.
        The file is split into pages if it exceeds page_size.
        Each page gets its own gist and search keywords.

        Args:
            folder_name (str): Name of the folder to add the file to
            file_name (str): Name of the file being added
            content (Value[str]): Content of the file
            **kwargs: Additional context when creating gists and keywords

        Returns:
            Value[str]: List of gists for each page of the file

        Raises:
            ValueError: If the specified folder doesn't exist
        """
        gist_module = self.create_file(filepath)

        stored_contents = "".join(gist_module.available_pages)
        if "".join(stored_contents.split()) == "".join(content.value.split()):
            return None

        tasks: list[asyncio.Task] = []

        async def exec_index_file(page: str, gist: Value[str], i: int):
            page_path = f"{filepath} ({i:04d})"

            # Insert path while maintaining sorted order
            self.insert_path(page_path)

            # Index the file content with its metadata
            await self.embedding_db.insert(
                {
                    gist.value: {
                        "content": page,
                        "gist": gist.value,
                        "path": page_path,
                        "gist_length": len(await compile_user_prompt(ITEM=gist.value)),
                        "content_length": len(await compile_user_prompt(ITEM=page)),
                    }
                }
            )

        # Process each page of the file separately
        for i, page in enumerate(paginate(content.value, self.page_size)):
            gist = await gist_module.append(page, **kwargs)
            index_task = asyncio.create_task(exec_index_file(page, gist, i))
            tasks.append(index_task)

        await asyncio.gather(*tasks)

        return gist_module

    async def get_file(self, folder_name: str, file_name: str) -> dict:
        """
        Retrieve a file's content and metadata by its path.

        Args:
            folder_name (str): Name of the folder containing the file
            file_name (str): Name of the file to retrieve

        Returns:
            dict: File data including content, gist, and metadata
        """
        return await self.embedding_db.get(f"{folder_name}/{file_name}")

    async def search_files(self, query: str, topk: int = 1) -> dict:
        """
        Search for files across all folders using semantic similarity.

        Args:
            query (str): Search query to match against file content and keywords
            topk (int): Maximum number of results to return

        Returns:
            dict: Matching files with their content, gists and metadata, ordered by relevance
        """
        return await self.embedding_db.query(query, topk)
