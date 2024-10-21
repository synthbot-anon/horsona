import pytest

from horsona.autodiff.losses import apply_loss
from horsona.autodiff.variables import Value
from horsona.character.dialogue import DialogueModule, DialogueResponse
from horsona.llm.base_engine import AsyncLLMEngine


@pytest.mark.asyncio
async def test_dialogue_module(reasoning_llm: AsyncLLMEngine):
    dialogue_module = DialogueModule(reasoning_llm)

    character_sheet = Value(
        "Character Sheet",
        {
            "name": "Twilight Sparkle",
            "occupation": "Princess of Friendship",
            "personality": "Intelligent, organized, and studious",
            "background": "Student and protege of Princess Celestia",
            "species": "Alicorn",
        },
        reasoning_llm,
    )

    context = Value(
        "Story context",
        "Twilight Sparkle has just discovered an ancient magical artifact in the Everfree Forest.",
        reasoning_llm,
    )

    dialogue_value = await dialogue_module.generate_dialogue(character_sheet, context)

    assert isinstance(dialogue_value.value, DialogueResponse)
    assert dialogue_value.value.dialogue != ""
    assert dialogue_value.value.tone != ""
    assert dialogue_value.value.subtext != ""

    # Check if the generated dialogue fits the character and context
    # Not sure how to check this yet

    context_loss = await apply_loss(
        dialogue_value,
        "The artifact was found in the Tenochtitlan Basin, not the Everfree Forest.",
    )
    character_loss = await apply_loss(
        dialogue_value, "Twilight is a unicorn, not an alicorn."
    )

    loss = context_loss + character_loss
    await loss.step([context, character_sheet])

    assert "tenochtitlan" in context.value.lower()
    assert "unicorn" in character_sheet.value["species"].lower()


@pytest.mark.asyncio
async def test_dialogue_module_state_dict(reasoning_llm: AsyncLLMEngine):
    # Create original DialogueModule
    original_module = DialogueModule(reasoning_llm, name="dialogue_generator")

    # Save state dict
    saved_state = original_module.state_dict()

    # Reload DialogueModule from state dict
    restored_module = DialogueModule.load_state_dict(
        saved_state, args={"llm": reasoning_llm}
    )

    assert restored_module.name == "dialogue_generator"
