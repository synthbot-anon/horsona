import json
from typing import Literal, Union

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from horsona.autodiff.basic import (HorseFunction, HorseModule, HorseOptimizer,
                                    HorseVariable)
from horsona.autodiff.variables import Result
from horsona.llm.base_engine import AsyncLLMEngine
from horsona.llm.cerebras_engine import AsyncCerebrasEngine
from horsona.llm.fireworks_engine import AsyncFireworksEngine
from horsona.memory.caches.dbcache import DatabaseCache
from horsona.memory.embeddings.databases import EmbeddingDatabase
from horsona.memory.embeddings.index import EmbeddingIndex
from horsona.memory.embeddings.models import HuggingFaceBGEModel

load_dotenv()


class StoryState(BaseModel):
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


class NewInfoResult(Result):
    def __init__(
        self,
        setting_database,
        long_term_memory,
        short_term_memory,
        current_state,
        new_information,
        **kwargs
    ):
        super().__init__(new_information, **kwargs)
        self.setting_database = setting_database
        self.long_term_memory = long_term_memory
        self.short_term_memory = short_term_memory
        self.current_state = current_state
        self.new_information = new_information

    async def json(self):
        return self.new_information


class NewInfoLoss(HorseFunction):
    async def forward(self, reader: "StoryReader", paragraph: str):
        reader.current_state.memory_corrections = ["Fill this in"]
        long_term_memory = reader.long_term_memory
        short_term_memory = reader.short_term_memory
        setting_database = reader.setting_database
        current_state = reader.current_state

        search = await reader.llm.query_structured(
            Search,
            PREVIOUS_PARAGRAPHS=short_term_memory,
            MEMORY_STATE=long_term_memory,
            CURRENT_STATE=current_state,
            PARAGRAPH=paragraph,
            TASK=(
                "You are trying to understand the current PARAGRAPH in a story. "
                "The MEMORY_STATE gives context for reading the PARAGRAPH. "
                "You have access to a search engine that can retrieve past information about the story state. "
                "Suggest keyword search queries about the story state that would provide better context for the PARAGRAPH."
            ),
        )

        for q in search.queries:
            await long_term_memory.load_data(q, topk=3)

        new_info = await reader.llm.query_structured(
            Information,
            MEMORY_STATE=long_term_memory,
            PREVIOUS_PARAGRAPHS=short_term_memory,
            CURRENT_STATE=current_state,
            PARAGRAPH=paragraph,
            TASK=(
                "You are trying to understand the current PARAGRAPH in a story. "
                "The MEMORY_STATE gives context for reading the PARAGRAPH. "
                "You are adding information to a keyword search engine that can retrieve past information about the story state. "
                "Suggest keyword queries and responses so that future searches would retrieve relevant information from PARAGRAPH. "
                "The responses will be given to someone without context, so provide enough details in every individual response."
            ),
        )
        new_info = {i.query: i.result for i in new_info.information}

        return NewInfoResult(
            setting_database,
            long_term_memory,
            short_term_memory,
            current_state,
            new_info,
        )

    async def backward(
        self, result: NewInfoResult, reader: "StoryReader", paragraph: str
    ):
        # TODO: update everything through gradients to make it more general

        await reader.setting_database.update(result.new_information)

        updated_state = await reader.llm.query_structured(
            StoryState,
            MEMORY_STATE=result.long_term_memory,
            PREVIOUS_PARAGRAPHS=result.short_term_memory,
            CURRENT_STATE=result.current_state,
            PARAGRAPH=paragraph,
            TASK=(
                "Modify the CURRENT_STATE based on the PARAGRAPH. "
                "If anything in MEMORY_STATE is incorrect, provide memory_corrections. "
                "Your memory_corrections will be given to someone without context of where you are "
                "in the story, so provide enough details in every individual memory_correction."
            ),
        )

        reader.current_state = updated_state
        reader.long_term_memory.gradients = updated_state.memory_corrections

        reader.short_term_memory.gradients.append(paragraph)


class ListBuffer(HorseVariable):
    def __init__(self, limit, buffer: list = None):
        super().__init__()
        if buffer == None:
            buffer = []
        self.limit = limit
        self.buffer = buffer

    async def apply_gradients(self):
        self.buffer.extend(self.gradients)
        self.buffer = self.buffer[-self.limit :]

    async def json(self):
        return self.buffer


class StoryReader(HorseModule):
    def __init__(self, llm: AsyncLLMEngine):
        super().__init__()
        self.llm = llm
        self.setting_database = EmbeddingDatabase(
            EmbeddingIndex(
                "Current state of the story setting",
                HuggingFaceBGEModel(model="BAAI/bge-large-en-v1.5"),
            )
        )

        self.short_term_memory = ListBuffer(5)
        self.long_term_memory = DatabaseCache(llm, self.setting_database, 10)

        self.current_state = StoryState(
            current_location="",
            characters_in_scene=[],
            last_speaker="",
            expected_next_speaker="",
            memory_corrections=[],
        )

        self.new_info_loss = NewInfoLoss()

    async def read(self, paragraph):
        return await self.new_info_loss(self, paragraph)
