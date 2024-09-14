import pytest
from pydantic import BaseModel

from horsona.autodiff.basic import HorseOptimizer
from horsona.autodiff.functions import TextExtractor
from horsona.autodiff.losses import ConstantLoss
from horsona.autodiff.variables import Parameter
from horsona.llm.cerebras_engine import AsyncCerebrasEngine


@pytest.fixture(scope="module")
def reasoning_llm():
    return AsyncCerebrasEngine(
        model="llama3.1-70b",
    )


@pytest.mark.asyncio
async def test_autodiff(reasoning_llm):
    input_text = Parameter("My name is Luna", reasoning_llm)
    name_extractor_fn = TextExtractor(reasoning_llm)

    class PonyName(BaseModel):
        name: str

    extracted_name = await name_extractor_fn(
        PonyName,
        TEXT=input_text,
        TASK="Extract the name from the TEXT.",
    )

    name_loss_fn = ConstantLoss("The name should be Celestia")
    title_loss_fn = ConstantLoss("They should be addressed as Princess")
    optimizer = HorseOptimizer([input_text])

    loss = await name_loss_fn(extracted_name) + await title_loss_fn(extracted_name)

    gradients = await loss.backward()
    await optimizer.step(gradients)

    assert input_text.value == "My name is Princess Celestia"
