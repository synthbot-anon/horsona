import functools

import pytest
from pydantic import BaseModel

from horsona.autodiff import HorseFunction, HorseOptimizer, HorseVariable
from horsona.autodiff.losses import ConstantLoss
from horsona.llm import AsyncLLMEngine
from horsona.llm.cerebras_engine import AsyncCerebrasEngine


@pytest.fixture(scope="module")
def extraction_llm():
    yield AsyncCerebrasEngine(model="llama3.1-70b")


class ConstantText(HorseVariable):
    def __init__(self, llm: AsyncLLMEngine, value: str):
        super().__init__(value=value)
        self.llm = llm

    async def apply_gradients(self, gradients):
        class UpdatedText(BaseModel):
            updated_text: str

        updated_text = await self.llm.query_object(
            UpdatedText,
            TEXT=self,
            FEEDBACK=gradients,
            TASK="Update the TEXT based on the FEEDBACK.",
        )

        self.value = updated_text.updated_text


class NameExtractor(HorseFunction):
    async def forward(self, llm: AsyncLLMEngine, text: HorseVariable) -> HorseVariable:
        class PonyName(BaseModel):
            name: str

        name = await llm.query_object(
            PonyName,
            TEXT=text,
            TASK="Extract the name from the text.",
        )

        result = HorseVariable(
            value=name.name,
            predecessors=[text],
        )

        result.grad_fn = functools.partial(self.backward, llm, result, text)

        return result

    async def backward(
        self, llm: AsyncLLMEngine, name: HorseVariable, input_text: HorseVariable
    ):
        if not name.requires_grad:
            return

        class NameMatches(BaseModel):
            requires_update: bool

        response = await llm.query_object(
            NameMatches,
            NAME=name,
            FEEDBACK=name.gradients,
            TASK="Based on the FEEDBACK, check if the NAME requires an update.",
        )

        if not response.requires_update:
            return

        class TextUpdate(BaseModel):
            suggested_changes: str

        updated_text = await llm.query_object(
            TextUpdate,
            TEXT=input_text,
            EXTRACTED_NAME=name,
            FEEDBACK=name.gradients,
            TASK=(
                "The FEEDBACK was given when extracting EXTRACTED_NAME from the TEXT. "
                "Give a list of minimal changes that should be made to the TEXT to "
                "address the FEEDBACK.",
            ),
        )

        input_text.gradients.append(updated_text.suggested_changes)


@pytest.mark.asyncio
async def test_autodiff(extraction_llm):
    input_text = ConstantText(extraction_llm, value="My name is Luna")
    extracted_name = await NameExtractor().forward(extraction_llm, input_text)
    name_loss_fn = ConstantLoss("The name should be Celestia")
    title_loss_fn = ConstantLoss("The title should be Princess")
    optimizer = HorseOptimizer([input_text])

    loss = (
        await name_loss_fn.forward(extracted_name) +
        await title_loss_fn.forward(extracted_name)
    )
    
    await loss.backward()
    await optimizer.step()

    assert input_text.value == "My name is Princess Celestia"
