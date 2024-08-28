from pydantic import BaseModel

from horsona.llm.base_engine import AsyncLLMEngine

from .basic import HorseVariable


class TextVariable(HorseVariable):
    def __init__(self, value: str, updater_llm: AsyncLLMEngine = None, required_grad=True):
        if required_grad:
            assert updater_llm is not None, "updater_llm must be provided if required_grad is True."

        super().__init__(value=value, requires_grad=required_grad)
        self.updater_llm = updater_llm

    async def apply_gradients(self):
        class UpdatedText(BaseModel):
            final_text: str
            addresses_feedback: bool

        updated_text = await self.updater_llm.query_object(
            UpdatedText,
            ORIGINAL_TEXT=self,
            FEEDBACK=self.gradients,
            TASK="The FEEDBACK was given for the ORIGINAL_TEXT. Replace the ORIGINAL_TEXT to address the FEEDBACK. Try to address all of the FEEDBACK, and address only the FEEDBACK. Then state whether the updated text addresses the feedback.",
        )

        print(updated_text.model_dump_json(indent=2))

        self.value = updated_text.final_text
