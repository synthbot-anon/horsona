from typing import Generator

import pytest
from dotenv import load_dotenv

from horsona.autodiff.basic import HorseOptimizer
from horsona.autodiff.losses import ConstantLoss
from horsona.llm.base_engine import AsyncLLMEngine
from horsona.llm.cerebras_engine import AsyncCerebrasEngine
from horsona.memory.models.rag import HuggingFaceBGEModel
from horsona.memory.rag import RAGDataset, RAGModule

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
def llm() -> Generator[AsyncLLMEngine, None, None]:
    yield AsyncCerebrasEngine(model="llama3.1-70b")


@pytest.fixture(scope="module")
def rag_model() -> HuggingFaceBGEModel:
    return HuggingFaceBGEModel(model="BAAI/bge-large-en-v1.5")


@pytest.mark.asyncio
async def test_rag(llm, rag_model):
    # Populate the dataset
    dataset = RAGDataset("State of the story setting", rag_model)
    await dataset.insert(SAMPLE_DATA[:])

    # Create the RAG module
    rag = RAGModule(dataset, llm)
    optimizer = HorseOptimizer(rag.parameters())

    # Query the model
    result = await rag.query("What color is Honeycrisp?")

    # Update the underlying data based on the fact that Honeycrisp is blue
    loss_fn = ConstantLoss("Honeycrisp is blue")
    loss = await loss_fn(result)
    await loss.backward()
    await optimizer.step()

    assert "A blue earth pony mare, Honeycrisp, appeared on screen" in rag.db.data
