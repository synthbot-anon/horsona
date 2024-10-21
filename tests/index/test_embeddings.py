import pytest

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


@pytest.mark.asyncio
async def test_query(query_index):
    await query_index.extend(SAMPLE_DATA[:])

    result = await query_index.query("Who is Honeycrisp", topk=1)
    assert "A red earth pony mare, Honeycrisp, appeared on screen" in result.values()


@pytest.mark.asyncio
async def test_delete(query_index):
    await query_index.extend(SAMPLE_DATA[:])

    await query_index.delete([1, 2])

    q1 = "A gray earth pony with navy blue hair is displayed on the monitor"
    test1 = await query_index.query(q1, topk=1)
    assert q1 not in test1.values()

    q2 = "A blue unicorn appears on screen"
    test2 = await query_index.query(q2, topk=1)
    assert q2 not in test2.values()
