from collections import defaultdict

from dotenv import load_dotenv
from pydantic import BaseModel

from horsona.autodiff.basic import (HorseFunction, HorseGradient, HorseModule,
                                    HorseVariable)
from horsona.autodiff.variables import Value
from horsona.llm.base_engine import AsyncLLMEngine
from horsona.memory.database import (DatabaseInsertGradient,
                                     DatabaseTextGradient)
from horsona.memory.dbcache import DatabaseCache
from horsona.memory.embeddings.database import EmbeddingDatabase
from horsona.memory.embeddings.index import EmbeddingIndex
from horsona.memory.embeddings.models import HuggingFaceBGEModel
from horsona.memory.listcache import ListCache

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


class ReadFunction(HorseFunction):
    async def forward(
        self,
        llm: AsyncLLMEngine,
        buffer_memory: ListCache,
        database_memory: DatabaseCache,
        current_state: "StoryState",
        paragraph: Value,
    ):
        class UpdatedState(BaseModel):
            new_state: type(current_state.context.value)
            memory_corrections: list[str]

        search = await llm.query_structured(
            Search,
            PREVIOUS_PARAGRAPHS=buffer_memory.context,
            MEMORY_STATE=database_memory.context,
            CURRENT_STATE=current_state.context,
            PARAGRAPH=paragraph,
            TASK=(
                "You are trying to understand the current PARAGRAPH in a story. "
                "The MEMORY_STATE gives context for reading the PARAGRAPH. "
                "You have access to a search engine that can retrieve past information about the story state. "
                "Suggest keyword search queries about the story state that would provide better context for the PARAGRAPH."
            ),
        )

        for q in search.queries:
            database_context = await database_memory.load(Value(q))

        update: UpdatedState = await llm.query_structured(
            UpdatedState,
            MEMORY_STATE=database_memory.context,
            PREVIOUS_PARAGRAPHS=buffer_memory.context,
            CURRENT_STATE=current_state.context,
            PARAGRAPH=paragraph,
            TASK=(
                "Modify the CURRENT_STATE based on the PARAGRAPH. "
                "If anything in MEMORY_STATE is incorrect, provide memory_corrections. "
                "Your memory_corrections will be given to someone without context of where you are "
                "in the story, so provide enough details in every individual memory_correction."
            ),
        )

        new_info: Information = await llm.query_structured(
            Information,
            MEMORY_STATE=database_memory.context,
            PREVIOUS_PARAGRAPHS=buffer_memory.context,
            CURRENT_STATE=current_state.context,
            PARAGRAPH=paragraph,
            TASK=(
                "You are trying to understand the current PARAGRAPH in a story. "
                "The MEMORY_STATE gives context for reading the PARAGRAPH. "
                "You are adding information to a keyword search engine that can retrieve past information about the story state. "
                "Suggest keyword queries and responses so that future searches would retrieve relevant information from PARAGRAPH. "
                "The responses will be given to someone without context, so provide enough details in every individual response."
            ),
        )

        state_context = await current_state.load(Value(update.new_state))
        buffer_context = await buffer_memory.load(paragraph)

        new_info = Value({i.query: i.result for i in new_info.information})
        corrections = Value(update.memory_corrections)
        return ReadResult(
            database_context,
            buffer_context,
            state_context,
            new_info,
            corrections,
            predecessors=[paragraph, database_context, buffer_context, state_context],
        )

    async def backward(
        self,
        context: dict[HorseVariable, list[HorseGradient]],
        result: ReadResult,
        llm: AsyncLLMEngine,
        buffer_memory: ListCache,
        database_memory: DatabaseCache,
        current_state: "StoryState",
        paragraph: Value,
    ):
        g = defaultdict(list)
        if result.corrections.value:
            g[result.database_context].append(
                DatabaseTextGradient(
                    context=result.database_context, change=result.corrections
                )
            )
        if result.new_information.value:
            g[result.database_context].append(
                DatabaseInsertGradient(rows=result.new_information)
            )

        return g


class StoryState(HorseModule):
    def __init__(self):
        super().__init__()
        self.context = Value(
            LiveState(
                current_location="",
                characters_in_scene=[],
                last_speaker="",
                expected_next_speaker="",
                memory_corrections=[],
            )
        )

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

        self.new_info_loss = ReadFunction()

    async def read(self, paragraph):
        return await self.new_info_loss(
            self.llm,
            self.buffer_memory,
            self.database_memory,
            self.current_state,
            paragraph,
        )
