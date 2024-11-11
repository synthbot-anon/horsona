import pytest
from horsona.autodiff.variables import Value
from horsona.memory.history_llm import HistoryLLMEngine
from horsona.memory.list_module import ListModule
from pydantic import BaseModel


@pytest.fixture
async def history_llm(reasoning_llm):
    history_module = ListModule()
    return HistoryLLMEngine(reasoning_llm, history_module)


@pytest.mark.asyncio
async def test_query_block(history_llm):
    # Add some history items
    await history_llm.history_module.append(Value("History", "The sky is red"))
    await history_llm.history_module.append(Value("History", "Grass is green"))

    # Query should have access to history context
    response = await history_llm.query_block(
        "text",
        PROMPT="What color was the sky mentioned earlier?",
    )

    assert "red" in response.lower()


@pytest.mark.asyncio
async def test_query_object(history_llm):
    class ColorResponse(BaseModel):
        color: str

    # Add history item
    await history_llm.history_module.append(Value("History", "The sky is red"))

    # Query object should have access to history
    response = await history_llm.query_object(
        ColorResponse,
        PROMPT="What color was the sky mentioned in the history?",
    )

    assert isinstance(response, ColorResponse)
    assert response.color.lower() == "red"


@pytest.mark.asyncio
async def test_load_history_llm(reasoning_llm, history_llm):
    # Add some history
    await history_llm.history_module.append(Value("History", "Test history item"))

    # Save and restore
    state_dict = history_llm.state_dict()
    restored = HistoryLLMEngine.load_state_dict(state_dict)

    assert isinstance(restored, HistoryLLMEngine)
    assert isinstance(restored.underlying_llm, type(reasoning_llm))
    assert isinstance(restored.history_module, ListModule)
    assert len(restored.history_module.get_items()) == 1
    assert restored.history_module.get_items()[0].value == "Test history item"
