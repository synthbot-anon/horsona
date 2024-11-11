import pytest
from horsona.autodiff.variables import ListValue, Value
from horsona.memory.list_module import ListModule
from horsona.memory.log_module import LogModule


@pytest.fixture
async def log_module(reasoning_llm):
    recent_messages_module = ListModule(min_item_length=0)
    return LogModule(reasoning_llm, recent_messages_module=recent_messages_module)


@pytest.mark.asyncio
async def test_append(log_module: LogModule):
    # Test appending item without adding to overview
    test_item = Value("Test", "Hello world")
    result = await log_module.append(test_item)

    assert result is not None

    # Should return chunked item if added to recent_messages
    assert isinstance(result, (Value, ListValue))
    assert len(log_module.recent_messages_module.items) == 1

    # Should be added to overview
    assert len(log_module.overview_module.available_gists) == 1


@pytest.mark.asyncio
async def test_append_multiple_items(log_module: LogModule):
    # Test appending multiple items
    items = [Value("Test", f"Message {i}") for i in range(3)]

    for item in items:
        await log_module.append(item)

    total_items = len(log_module.recent_messages_module.items) + len(
        log_module.recent_messages_module.pending_items
    )
    assert total_items == 3
