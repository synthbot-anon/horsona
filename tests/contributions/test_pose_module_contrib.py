import pytest

from horsona.autodiff.losses import apply_loss
from horsona.autodiff.variables import Value
from horsona.contributions.sample.pose import PoseDescription, PoseModule
from horsona.llm.base_engine import AsyncLLMEngine


@pytest.mark.asyncio
async def test_pose_module(reasoning_llm: AsyncLLMEngine):
    pose_module = PoseModule(reasoning_llm)

    character_info = Value(
        "Character Info",
        {
            "name": "Luna",
            "species": "Unicorn",
            "personality": "Introverted, mysterious",
        },
        reasoning_llm,
    )

    context = Value(
        "Story sentence",
        "Luna is standing tall at a royal gala, feeling slightly uncomfortable.",
        reasoning_llm,
    )

    pose_value = await pose_module.generate_pose(character_info, context)

    assert isinstance(pose_value.value, PoseDescription)
    assert pose_value.value.pose != ""
    assert pose_value.value.facial_expression != ""
    assert pose_value.value.body_language != ""

    # Check if the generated pose fits the character and context
    assert (
        "stand" in pose_value.value.pose.lower()
        or "stand" in pose_value.value.body_language.lower()
    )

    context_loss = await apply_loss(pose_value, "Luna should be sitting, not standing.")
    character_loss = await apply_loss(pose_value, "Luna is an Alicorn, not a Unicorn.")

    loss = context_loss + character_loss
    await loss.step([context, character_info])

    assert "sit" in context.value.lower() or "sat" in context.value.lower()
    assert "alicorn" in character_info.value["species"].lower()


@pytest.mark.asyncio
async def test_pose_module_state_dict(reasoning_llm: AsyncLLMEngine):
    # Create original PoseModule
    original_module = PoseModule(reasoning_llm, name="pose_generator")

    # Save state dict
    saved_state = original_module.state_dict()

    # Reload PoseModule from state dict
    restored_module = PoseModule.load_state_dict(saved_state)

    assert restored_module.name == "pose_generator"
