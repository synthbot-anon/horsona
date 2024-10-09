import pytest
from horsona.autodiff.basic import step
from horsona.autodiff.losses import apply_loss
from horsona.llm.base_engine import AsyncLLMEngine
from horsona.llm.cerebras_engine import AsyncCerebrasEngine
from horsona.memory.embeddings.hnsw_index import HnswEmbeddingIndex
from horsona.memory.embeddings.index import EmbeddingIndex, IndexChanges
from horsona.memory.embeddings.models import OllamaEmbeddingModel

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
    return OllamaEmbeddingModel(model="imcurie/bge-large-en-v1.5")


@pytest.mark.asyncio
async def test_query(embedding_model):
    index = HnswEmbeddingIndex(embedding_model)
    await index.extend(SAMPLE_DATA[:])

    result = await index.query("Who is Honeycrisp", topk=1)
    assert "A red earth pony mare, Honeycrisp, appeared on screen" in result.values()


@pytest.mark.asyncio
async def test_delete(embedding_model):
    index = HnswEmbeddingIndex(embedding_model)
    await index.extend(SAMPLE_DATA[:])

    await index.delete([1, 2])

    q1 = "A gray earth pony with navy blue hair is displayed on the monitor"
    test1 = await index.query(q1, topk=1)
    assert q1 not in test1.values()

    q2 = "A blue unicorn appears on screen"
    test2 = await index.query(q2, topk=1)
    assert q2 not in test2.values()


@pytest.mark.asyncio
async def test_apply_gradients(embedding_model):
    index = HnswEmbeddingIndex(embedding_model, requires_grad=True)

    await index.extend(SAMPLE_DATA[:])

    loss = await apply_loss(
        index,
        IndexChanges(
            **{
                "changes": [
                    {"operation": "DELETE", "index": 1},
                    {
                        "operation": "DELETE",
                        "index": 7,
                    },
                    {
                        "operation": "INSERT",
                        "value": "A blue earth pony mare, Honeycrisp, appeared on screen",
                    },
                    {"operation": "INSERT", "value": "James is an awful pony"},
                ]
            }
        ),
    )

    gradients = await loss.backward([index])
    await step(gradients)

    q1 = "A gray earth pony with navy blue hair is displayed on the monitor"
    test1 = await index.query(q1, topk=1)
    assert q1 not in test1.values()

    q2 = "A red earth pony mare, Honeycrisp, appeared on screen"
    test2 = await index.query(q2, topk=1)
    assert q2 not in test2.values()

    q3 = "A blue earth pony mare, Honeycrisp, appeared on screen"
    test3 = await index.query(q3, topk=1)
    assert q3 in test3.values()

    q4 = "James is an awful pony"
    test4 = await index.query(q4, topk=1)
    assert q4 in test4.values()
