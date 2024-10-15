import pytest
from horsona.autodiff.basic import step
from horsona.autodiff.losses import apply_loss
from horsona.autodiff.variables import Value
from horsona.cache.db_cache import DatabaseCache
from horsona.database.base_database import (
    DatabaseOpGradient,
    DatabaseTextGradient,
    DatabaseUpdate,
)
from horsona.database.embedding_database import EmbeddingDatabase

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


@pytest.mark.asyncio
async def test_update_database(reasoning_llm, query_index):
    database = EmbeddingDatabase(reasoning_llm, query_index)
    print(database)
    await database.insert(SAMPLE_DATA)

    cache = DatabaseCache(reasoning_llm, database, 5, name="test_db")

    context = await cache.load(Value("Search query", "Who is Honeycrisp?"))
    loss = await apply_loss(
        context,
        DatabaseTextGradient(
            context=context, change=Value("Data correction", "Honeycrisp is blue")
        ),
    ) + await apply_loss(
        context,
        DatabaseOpGradient(
            changes=[
                DatabaseUpdate(
                    key="What is James shown?",
                    corrected_data="James is shown a beautiful mare.",
                )
            ]
        ),
    )

    gradients = await loss.backward([database])
    await step(gradients)

    result = await database.query("Who is Honeycrisp")
    assert "A blue earth pony mare, Honeycrisp, appeared on screen" in result.values()

    result = await database.query("What is James shown?")
    assert "James is shown a beautiful mare." in result.values()
