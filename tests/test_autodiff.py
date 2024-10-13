import pytest
from horsona.autodiff.basic import step
from horsona.autodiff.functions import extract_object
from horsona.autodiff.losses import apply_loss
from horsona.autodiff.variables import Parameter
from pydantic import BaseModel


@pytest.mark.asyncio
async def test_autodiff(reasoning_llm):
    input_text = Parameter("My name is Luna", reasoning_llm)

    class PonyName(BaseModel):
        name: str

    extracted_name = await extract_object(
        reasoning_llm,
        PonyName,
        TEXT=input_text,
        TASK="Extract the name from the TEXT.",
    )

    loss = await apply_loss(
        extracted_name, "The name should be Celestia"
    ) + await apply_loss(extracted_name, "They should be addressed as Princess")

    gradients = await loss.backward([input_text])
    await step(gradients)

    assert input_text.value == "My name is Princess Celestia"
