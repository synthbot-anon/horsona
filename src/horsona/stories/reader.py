import asyncio
from typing import AsyncGenerator, Optional, Protocol

from dotenv import load_dotenv
from pydantic import BaseModel

from horsona.autodiff.basic import (
    GradContext,
    HorseModule,
    HorseVariable,
    horsefunction,
)
from horsona.autodiff.variables import Value
from horsona.llm.base_engine import AsyncLLMEngine
from horsona.memory.caches.cache import Cache
from horsona.memory.caches.dbcache import DatabaseCache
from horsona.memory.caches.listcache import ListCache
from horsona.memory.caches.valuecache import ValueCache
from horsona.memory.database import (
    Database,
    DatabaseInsertGradient,
    DatabaseTextGradient,
)
from horsona.memory.embeddings.database import EmbeddingDatabase
from horsona.memory.embeddings.index import EmbeddingIndex
from horsona.memory.embeddings.models import HuggingFaceBGEModel

load_dotenv()


class LiveState(BaseModel):
    current_location: str
    characters_in_scene: list[str]
    last_speaker: str
    expected_next_speaker: str
    memory_corrections: list[str]


class ReadResult(HorseVariable):
    def __init__(
        self,
        database_context: DatabaseCache,
        buffer_context: ListCache,
        state_context: Value,
        new_information: Value,
        corrections: Value,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.database_context = database_context
        self.buffer_context = buffer_context
        self.state_context = state_context
        self.new_information = new_information
        self.corrections = corrections

    async def json(self):
        return self.new_information


class StoryReader(HorseModule):
    def __init__(
        self,
        llm: AsyncLLMEngine,
        setting_db: Database = None,
        buffer_cache: Cache = None,
        database_cache: Cache = None,
        state_cache: ValueCache = None,
    ):
        super().__init__()
        self.llm = llm

        if setting_db is None:
            setting_db = setting_db or EmbeddingDatabase(
                self.llm,
                EmbeddingIndex(
                    "Current state of the story setting",
                    HuggingFaceBGEModel(model="BAAI/bge-large-en-v1.5"),
                ),
                requires_grad=True,
            )
        self.setting_db = setting_db

        if buffer_cache is None:
            buffer_cache = ListCache(5)
        self.buffer_memory = buffer_cache

        if database_cache is None:
            database_cache = DatabaseCache(llm, setting_db, 10)
        self.database_memory = database_cache

        if state_cache is None:
            state_cache = ValueCache(
                Value(
                    LiveState(
                        current_location="",
                        characters_in_scene=[],
                        last_speaker="",
                        expected_next_speaker="",
                        memory_corrections=[],
                    )
                )
            )
        self.state_cache: ValueCache = state_cache

    @horsefunction
    async def read(self, paragraph: Value) -> AsyncGenerator[ReadResult, GradContext]:
        class UpdatedState(BaseModel):
            new_state: self.state_cache.context.VALUE_TYPE
            memory_corrections: list[str]

        buffer_context = await self.buffer_memory.sync()
        database_context = await self.database_memory.sync()
        state_context = await self.state_cache.sync()

        class Search(BaseModel):
            queries: list[str]

        search = await self.llm.query_structured(
            Search,
            PREVIOUS_PARAGRAPHS=buffer_context,
            MEMORY_STATE=database_context,
            STORY_STATE=state_context,
            PARAGRAPH=paragraph,
            TASK=(
                "You are trying to understand the current PARAGRAPH in a story. "
                "The MEMORY_STATE gives context for reading the PARAGRAPH. "
                "You have access to a search engine that can retrieve past information about the story state. "
                "Suggest keyword search queries about the story state that would provide better context for the PARAGRAPH."
            ),
        )

        for q in search.queries:
            database_context = await self.database_memory.load(
                Value(
                    q,
                    predecessors=[
                        paragraph,
                        database_context,
                        buffer_context,
                        state_context,
                    ],
                )
            )

        class QueryInfo(BaseModel):
            query: str
            result: str

        class Information(BaseModel):
            information: list[QueryInfo]

        update, new_info = await asyncio.gather(
            self.llm.query_structured(
                UpdatedState,
                MEMORY_STATE=database_context,
                PREVIOUS_PARAGRAPHS=buffer_context,
                STORY_STATE=state_context,
                PARAGRAPH=paragraph,
                TASK=(
                    "Modify the STORY_STATE based on the PARAGRAPH. "
                    "If anything in MEMORY_STATE is incorrect, provide memory_corrections. "
                    "Your memory_corrections will be given to someone without context of where you are "
                    "in the story, so provide enough details in every individual memory_correction."
                ),
            ),
            self.llm.query_structured(
                Information,
                MEMORY_STATE=database_context,
                PREVIOUS_PARAGRAPHS=buffer_context,
                STORY_STATE=state_context,
                PARAGRAPH=paragraph,
                TASK=(
                    "You are trying to understand the current PARAGRAPH in a story. "
                    "The MEMORY_STATE gives context for reading the PARAGRAPH. "
                    "You are adding information to a keyword search engine that can retrieve past information about the story state. "
                    "Suggest keyword queries and responses so that future searches would retrieve relevant information from PARAGRAPH. "
                    "The responses will be given to someone without context, so provide enough details in every individual response."
                ),
            ),
        )

        new_info = Value(
            {i.query: i.result for i in new_info.information},
            predecessors=[paragraph, database_context, buffer_context, state_context],
        )

        corrections = Value(
            update.memory_corrections,
            predecessors=[paragraph, database_context, buffer_context, state_context],
        )

        new_state_value = Value(
            update.new_state,
            predecessors=[
                paragraph,
                database_context,
                buffer_context,
                state_context,
            ],
        )

        buffer_context, state_context = await asyncio.gather(
            self.buffer_memory.load(paragraph),
            self.state_cache.load(new_state_value),
        )

        result = ReadResult(
            database_context,
            buffer_context,
            state_context,
            new_info,
            corrections,
            predecessors=[paragraph, database_context, buffer_context, state_context],
        )

        grad_context = yield result

        if database_context in grad_context:
            if result.corrections.value:
                grad_context[result.database_context].append(
                    DatabaseTextGradient(
                        context=result.database_context, change=result.corrections
                    )
                )
            if result.new_information.value:
                grad_context[result.database_context].append(
                    DatabaseInsertGradient(rows=result.new_information)
                )
