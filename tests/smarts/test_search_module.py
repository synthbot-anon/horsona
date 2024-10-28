import pytest
from horsona.autodiff.losses import apply_loss
from horsona.autodiff.variables import Value
from horsona.llm.base_engine import AsyncLLMEngine
from horsona.smarts.search_module import (
    Evaluation,
    SearchModule,
    SearchResult,
    ValidationResult,
)


@pytest.mark.asyncio
async def test_search(search_llm: AsyncLLMEngine, reasoning_llm: AsyncLLMEngine):
    search_module = SearchModule(search_llm, reasoning_llm)

    topic = Value(
        "Search Topic",
        "The impact of artificial intelligence on job markets",
        reasoning_llm,
    )

    # Test gather_info method
    search_result = await search_module.gather_info(topic)
    assert isinstance(search_result.value, SearchResult)
    assert search_result.value.information != ""
    assert len(search_result.value.sources) > 0

    # Apply a loss to test gradient flow
    loss = await apply_loss(search_result, "The information lacks specific statistics.")
    await loss.step([topic])

    # Check if the topic has been updated
    assert "statistic" in topic.value.lower()


@pytest.mark.asyncio
async def test_search_module_state_dict(
    search_llm: AsyncLLMEngine, reasoning_llm: AsyncLLMEngine
):
    # Create original SearchModule
    original_module = SearchModule(search_llm, reasoning_llm, name="search_engine")

    # Save state dict
    saved_state = original_module.state_dict()

    # Reload SearchModule from state dict
    restored_module = SearchModule.load_state_dict(saved_state)

    assert restored_module.name == "search_engine"
    assert restored_module.search_llm.name == search_llm.name
    assert restored_module.reasoning_llm.name == reasoning_llm.name


# Test validate_info method with different input types
@pytest.mark.parametrize(
    "topic_value, info_value, expected_evaluation",
    [
        (
            "My Little Pony: Equestria's Geography",
            "Equestria is a magical land with diverse regions like Ponyville, Canterlot, and the Everfree Forest.",
            (Evaluation.VALID, Evaluation.PARTIALLY_VALID),
        ),
        (
            "My Little Pony: Alicorn Princesses",
            "Twilight Sparkle was born as an alicorn and immediately gained full magical powers.",
            (Evaluation.INVALID,),
        ),
        (
            "My Little Pony: Cutie Marks",
            "Cutie marks appear when a pony discovers their special talent, but the process can be different for each pony.",
            (Evaluation.VALID, Evaluation.PARTIALLY_VALID),
        ),
    ],
)
async def test_validate_info(
    search_llm: AsyncLLMEngine,
    reasoning_llm: AsyncLLMEngine,
    topic_value,
    info_value,
    expected_evaluation,
):
    search_module = SearchModule(search_llm, reasoning_llm)
    topic = Value("Topic", topic_value, reasoning_llm)
    info = Value("Information", info_value, reasoning_llm)

    validation_result = await search_module.validate_info(topic, info)

    assert isinstance(validation_result.value, ValidationResult)
    assert isinstance(validation_result.value.evaluation, Evaluation)
    assert validation_result.value.evaluation in expected_evaluation
