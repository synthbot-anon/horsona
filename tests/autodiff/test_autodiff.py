import pytest
from pydantic import BaseModel

from horsona.autodiff.functions import extract_object
from horsona.autodiff.losses import apply_loss
from horsona.autodiff.variables import Value


@pytest.mark.asyncio
async def test_autodiff(reasoning_llm):
    false_dialogue_memory = Value("Story dialogue", "Hello Luna.", reasoning_llm)

    class PonyName(BaseModel):
        name: str

    extracted_name = await extract_object(
        reasoning_llm,
        PonyName,
        TEXT=false_dialogue_memory,
        TASK="Extract the name from the TEXT.",
    )

    loss = await apply_loss(
        extracted_name, "The name should have been Celestia"
    ) + await apply_loss(
        extracted_name, "They should have been addressed as Princess [...]"
    )

    await loss.step([false_dialogue_memory])

    assert false_dialogue_memory.value == "Hello Princess Celestia."
