from typing import Generator

import pytest
from dotenv import load_dotenv

from horsona.autodiff.basic import HorseOptimizer
from horsona.autodiff.losses import ConstantLoss
from horsona.llm.base_engine import AsyncLLMEngine
from horsona.llm.cerebras_engine import AsyncCerebrasEngine
from horsona.memory.embeddings.index import EmbeddingIndex, IndexChanges
from horsona.memory.embeddings.models import HuggingFaceBGEModel

load_dotenv()

SAMPLE_DATA = [
    "James is shown the Earth pony creation screen",
    "A gray earth pony with navy blue hair is displayed on the monitor",
    "A blue unicorn appears on screen",
    "James's pony on screen has an astonished expression on his face",
    "James notices the flat panel monitor has an embedded webcam",
    "James and his pony are making faces at the camera",
    "James realizes his pony is a modified version of himself",
    "A red earth pony mare, Honeycrisp, appeared on screen",
]


@pytest.fixture(scope="module")
def llm() -> AsyncLLMEngine:
    return AsyncCerebrasEngine(model="llama3.1-70b")


@pytest.fixture(scope="module")
def embedding_model() -> EmbeddingIndex:
    return HuggingFaceBGEModel(model="BAAI/bge-large-en-v1.5")



@pytest.mark.asyncio
async def test_query(llm, embedding_model):
    index = EmbeddingIndex("State of the story setting", embedding_model)
    await index.extend(SAMPLE_DATA[:])

    result = await index.query("Who is Honeycrisp", topk=1)
    assert "A red earth pony mare, Honeycrisp, appeared on screen" in result.values()

@pytest.mark.asyncio
async def test_delete(llm, embedding_model):
    index = EmbeddingIndex("State of the story setting", embedding_model)
    await index.extend(SAMPLE_DATA[:])

    await index.delete([1, 2])

    remaining_data = set(index.values.values())
    assert "A gray earth pony with navy blue hair is displayed on the monitor" not in remaining_data
    assert "A blue unicorn appears on screen" not in remaining_data

    assert len(remaining_data) == len(SAMPLE_DATA) - 2

@pytest.mark.asyncio
async def test_apply_gradients(embedding_model):
    index = EmbeddingIndex("State of the story setting", embedding_model)
    await index.extend(SAMPLE_DATA[:])

    index.gradients.append(IndexChanges(**{
        'changes': [
        {'operation':"DELETE", 'index':1},
        {'operation':"DELETE", 'index':7,},
        {'operation':"INSERT", 'value':"A blue earth pony mare, Honeycrisp, appeared on screen"},
        {'operation':'INSERT', 'value':"James is an awful pony"},
    ]}).changes)

    await index.apply_gradients()

    assert len(index.values) == len(SAMPLE_DATA)
    
    values = set(index.values.values())
    print(values)
    assert "A gray earth pony with navy blue hair is displayed on the monitor" not in values
    assert "A red earth pony mare, Honeycrisp, appeared on screen" not in values
    assert "A blue earth pony mare, Honeycrisp, appeared on screen" in values
    assert "James is an awful pony" in values