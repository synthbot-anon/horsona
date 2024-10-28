import pytest
from horsona.autodiff.losses import apply_loss
from horsona.autodiff.variables import Value
from horsona.llm.base_engine import AsyncLLMEngine
from horsona.smarts.mece import MECEModule, MECEStructure


@pytest.mark.asyncio
async def test_mece_module(reasoning_llm: AsyncLLMEngine):
    mece_module = MECEModule(reasoning_llm)

    topic = Value(
        "Topic",
        "Software Development",
        reasoning_llm,
    )

    mece_value = await mece_module.generate_mece(topic)

    assert isinstance(mece_value.value, MECEStructure)
    assert mece_value.value.topic == "Software Development"
    assert len(mece_value.value.categories) > 0
    for category in mece_value.value.categories:
        assert category.name != ""
        assert category.description != ""

    # Check if the generated MECE structure is relevant to the topic
    assert any(
        "coding" in category.name.lower()
        or "coding" in category.description.lower()
        or "implementation" in category.name.lower()
        or "implementation" in category.description.lower()
        or "develop" in mece_value.value.topic.lower()
        or "develop" in mece_value.value.description.lower()
        for category in mece_value.value.categories
    )

    topic_loss = await apply_loss(
        mece_value,
        "This is supposed to be about software deployment, not software development.",
    )

    await topic_loss.step([topic])

    print(topic.value)

    assert "deployment" in topic.value.lower()


@pytest.mark.asyncio
async def test_mece_module_state_dict(reasoning_llm: AsyncLLMEngine):
    # Create original MECEModule
    original_module = MECEModule(reasoning_llm, name="mece_generator")

    # Save state dict
    saved_state = original_module.state_dict()

    # Reload MECEModule from state dict
    restored_module = MECEModule.load_state_dict(saved_state)

    assert restored_module.name == "mece_generator"
