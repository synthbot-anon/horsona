from typing import AsyncGenerator

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
from horsona.memory.database import DatabaseInsertGradient, DatabaseTextGradient
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


class Search(BaseModel):
    queries: list[str]


class QueryInfo(BaseModel):
    query: str
    result: str


class Information(BaseModel):
    information: list[QueryInfo]


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


class StoryStateContext(Value):
    def __init__(self, **kwargs):
        value = LiveState(
            current_location="",
            characters_in_scene=[],
            last_speaker="",
            expected_next_speaker="",
            memory_corrections=[],
        )
        super().__init__(value, **kwargs)


class StoryState(Cache):
    def __init__(self):
        super().__init__(StoryStateContext())

    async def load(self, value: Value):
        self.context = value
        return self.context


class StoryReader(HorseModule):
    def __init__(self, llm: AsyncLLMEngine):
        super().__init__()
        self.llm = llm
        self.setting_database = EmbeddingDatabase(
            self.llm,
            EmbeddingIndex(
                "Current state of the story setting",
                HuggingFaceBGEModel(model="BAAI/bge-large-en-v1.5"),
            ),
            requires_grad=True,
        )

        self.buffer_memory = ListCache(5)
        self.database_memory = DatabaseCache(llm, self.setting_database, 10)
        self.current_state = StoryState()

    @horsefunction
    async def read(self, paragraph) -> AsyncGenerator[ReadResult, GradContext]:
        class UpdatedState(BaseModel):
            new_state: type(self.current_state.context.value)
            memory_corrections: list[str]

        buffer_context = await self.buffer_memory.sync()
        database_context = await self.database_memory.sync()
        state_context = await self.current_state.sync()

        search = await self.llm.query_structured(
            Search,
            PREVIOUS_PARAGRAPHS=buffer_context,
            MEMORY_STATE=database_context,
            CURRENT_STATE=state_context,
            PARAGRAPH=paragraph,
            TASK=(
                "You are trying to understand the current PARAGRAPH in a story. "
                "The MEMORY_STATE gives context for reading the PARAGRAPH. "
                "You have access to a search engine that can retrieve past information about the story state. "
                "Suggest keyword search queries about the story state that would provide better context for the PARAGRAPH."
            ),
        )

        for q in search.queries:
            database_context = await self.database_memory.load(Value(q))

        update: UpdatedState = await self.llm.query_structured(
            UpdatedState,
            MEMORY_STATE=database_context,
            PREVIOUS_PARAGRAPHS=buffer_context,
            CURRENT_STATE=state_context,
            PARAGRAPH=paragraph,
            TASK=(
                "Modify the CURRENT_STATE based on the PARAGRAPH. "
                "If anything in MEMORY_STATE is incorrect, provide memory_corrections. "
                "Your memory_corrections will be given to someone without context of where you are "
                "in the story, so provide enough details in every individual memory_correction."
            ),
        )

        new_info: Information = await self.llm.query_structured(
            Information,
            MEMORY_STATE=database_context,
            PREVIOUS_PARAGRAPHS=buffer_context,
            CURRENT_STATE=state_context,
            PARAGRAPH=paragraph,
            TASK=(
                "You are trying to understand the current PARAGRAPH in a story. "
                "The MEMORY_STATE gives context for reading the PARAGRAPH. "
                "You are adding information to a keyword search engine that can retrieve past information about the story state. "
                "Suggest keyword queries and responses so that future searches would retrieve relevant information from PARAGRAPH. "
                "The responses will be given to someone without context, so provide enough details in every individual response."
            ),
        )

        state_context = await self.current_state.load(Value(update.new_state))
        buffer_context = await self.buffer_memory.load(paragraph)

        new_info = Value({i.query: i.result for i in new_info.information})
        corrections = Value(update.memory_corrections)

        result = ReadResult(
            database_context,
            buffer_context,
            state_context,
            new_info,
            corrections,
            predecessors=[paragraph, database_context, buffer_context, state_context],
        )
        grad_context = yield result

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
