import asyncio
from typing import AsyncGenerator

from pydantic import BaseModel

from horsona.autodiff.basic import (
    GradContext,
    HorseModule,
    HorseVariable,
    horsefunction,
    load_state_dict,
    state_dict,
)
from horsona.autodiff.variables import Value
from horsona.cache.base_cache import BaseCache
from horsona.cache.db_cache import DatabaseCache, DatabaseCacheContext
from horsona.cache.list_cache import ListCache
from horsona.cache.value_cache import ValueCache
from horsona.database.base_database import (
    Database,
    DatabaseInsertGradient,
    DatabaseTextGradient,
)
from horsona.llm.base_engine import AsyncLLMEngine


class LiveState(BaseModel):
    current_location: str = "unknown"
    characters_in_scene: list[str] = []
    last_speaker: str = "none"
    expected_next_speaker: str = "none"
    memory_corrections: list[str] = []


class ReadContext(HorseVariable):
    def __init__(
        self,
        database_context: DatabaseCacheContext,
        buffer_context: HorseVariable,
        state_context: Value,
    ):
        super().__init__(predecessors=[database_context, buffer_context, state_context])
        self.database_context = database_context
        self.buffer_context = buffer_context
        self.state_context = state_context

    async def json(self):
        return {
            "setting_info": self.database_context,
            "recent_paragraphs": self.buffer_context,
            "story_state": self.state_context,
        }


class ReadContextLoss(HorseVariable):
    def __init__(self, new_information: Value, corrections: Value):
        super().__init__(predecessors=[new_information, corrections])
        self.new_information = new_information
        self.corrections = corrections


class ReaderModule(HorseModule):
    def __init__(
        self,
        llm: AsyncLLMEngine,
        setting_db: Database = None,
        buffer_cache: BaseCache = None,
        database_cache: BaseCache = None,
        live_cache: ValueCache = None,
        **kwargs,
    ):
        if setting_db is None and database_cache is None:
            raise ValueError(
                "At least one of setting_db or database_cache must be provided"
            )

        super().__init__(**kwargs)
        self.llm = llm

        if buffer_cache is None:
            buffer_cache = ListCache(5)
        self.buffer_cache = buffer_cache

        if database_cache is None:
            database_cache = DatabaseCache(llm, setting_db, 10)
        self.database_cache = database_cache

        if live_cache is None:
            live_cache = ValueCache(Value("Live state", LiveState()))
        self.live_cache: ValueCache = live_cache

    async def read(self, paragraph: Value) -> tuple[ReadContext, ReadContextLoss]:
        class UpdatedState(BaseModel):
            new_state: self.live_cache.context.VALUE_TYPE
            memory_corrections: list[str]

        buffer_context = await self.buffer_cache.sync()
        database_context = await self.database_cache.sync()
        state_context = await self.live_cache.sync()

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
            database_context = await self.database_cache.load(
                Value(
                    "Search query",
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
            "New information",
            {i.query: i.result for i in new_info.information},
            predecessors=[database_context, buffer_context, state_context, paragraph],
        )

        corrections = Value(
            "Data corrections",
            update.memory_corrections,
            predecessors=[database_context, buffer_context, state_context, paragraph],
        )

        new_state_value = Value(
            self.live_cache.context.datatype,
            update.new_state,
            predecessors=[
                paragraph,
                database_context,
                buffer_context,
                state_context,
            ],
        )

        new_buffer_context, state_context = await asyncio.gather(
            self.buffer_cache.load(paragraph),
            self.live_cache.load(new_state_value),
        )

        read_context = ReadContext(
            database_context,
            buffer_context,
            state_context,
        )

        @horsefunction
        async def read_loss() -> AsyncGenerator[ReadContextLoss, GradContext]:
            result = ReadContextLoss(new_info, corrections)
            grad_context = yield result

            if read_context.database_context in grad_context:
                if result.corrections.value:
                    grad_context[read_context.database_context].append(
                        DatabaseTextGradient(
                            context=read_context.database_context,
                            change=result.corrections,
                        )
                    )
                if result.new_information.value:
                    grad_context[read_context.database_context].append(
                        DatabaseInsertGradient(rows=result.new_information)
                    )

        return read_context, await read_loss()
