import pytest
from pydantic import BaseModel

from horsona.database.embedding_database import EmbeddingDatabase
from horsona.index.base_index import BaseIndex
from horsona.llm.base_engine import AsyncLLMEngine
from horsona.memory.embedding_llm import EmbeddingLLMEngine

SAMPLE_DATA = {
    "What is James shown?": "James is shown the Earth pony creation screen",
    "What does the pony on screen look like?": "A gray earth pony with navy blue hair is displayed on the monitor",
    "What is on the screen?": "A blue unicorn appears on screen",
    "What is James' reaction?": "James's pony on screen has an astonished expression on his face",
    "What is monitoring James?": "James notices the flat panel monitor has an embedded webcam",
    "What is James doing?": "James and his pony are making faces at the camera",
    "What does James' pony look like?": "James realizes his pony is a modified version of himself",
    "Who is Honeycrisp?": "A red earth pony mare, Honeycrisp, appeared on screen",
}


@pytest.fixture
async def embedding_llm(reasoning_llm, query_index):
    print("Using reasoning llm:", reasoning_llm)
    database = EmbeddingDatabase(reasoning_llm, query_index)
    await database.insert(SAMPLE_DATA)
    return EmbeddingLLMEngine(
        reasoning_llm, database, database_query_kwargs={"topk": 3}
    )


@pytest.mark.asyncio
async def test_query_block(embedding_llm):
    response = await embedding_llm.query_block(
        "text",
        PROMPT="What is James doing with the camera?",
    )

    print(response)

    assert isinstance(response, str)
    assert "faces" in response


@pytest.mark.asyncio
async def test_query_object(embedding_llm):
    class Response(BaseModel):
        color: str

    response = await embedding_llm.query_object(
        Response,
        PROMPT="What color is Honeycrisp?",
    )

    print(response)

    assert isinstance(response, Response)
    assert response.color.lower() == "red"


@pytest.mark.asyncio
async def test_load_embedding_llm(reasoning_llm, embedding_llm):
    state_dict = embedding_llm.state_dict()
    restored = EmbeddingLLMEngine.load_state_dict(state_dict)

    assert isinstance(restored, EmbeddingLLMEngine)
    assert isinstance(restored.underlying_llm, type(reasoning_llm))
    assert isinstance(restored.database, EmbeddingDatabase)


@pytest.mark.asyncio
async def test_query_with_context(embedding_llm):
    response = await embedding_llm.query_block(
        "text",
        PROMPT="What does the monitor have attached to it?",
    )

    print(response)

    assert isinstance(response, str)
    assert "webcam" in response.lower()
