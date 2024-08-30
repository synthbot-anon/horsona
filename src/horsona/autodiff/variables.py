from pydantic import BaseModel

from horsona.llm.base_engine import AsyncLLMEngine

from .basic import HorseType, HorseVariable


class Value(HorseVariable):
    def __init__(
        self,
        value: HorseType,
        updater_llm: AsyncLLMEngine = None,
        required_grad=True,
        **kwargs
    ):
        if required_grad:
            assert (
                updater_llm is not None
            ), "updater_llm must be provided if required_grad is True."

        super().__init__(requires_grad=required_grad, **kwargs)
        self.value = value
        self.updater_llm = updater_llm

    async def json(self):
        return self.value

    async def apply_gradients(self):
        class UpdatedValue(BaseModel):
            final_value: type(self.value)
            addresses_feedback: bool

        print("applying gradient to", self.value)

        update = await self.updater_llm.query_object(
            UpdatedValue,
            ORIGINAL_VALUE=self,
            FEEDBACK=self.gradients,
            TASK=(
                "The FEEDBACK was given for the ORIGINAL_VALUE. "
                "Replace the ORIGINAL_VALUE to address the FEEDBACK. "
                "Try to address all of the FEEDBACK, and address only the FEEDBACK. "
                "Then state whether the updated text addresses the feedback."
            ),
        )

        print(update.model_dump_json(indent=2))

        self.value = update.final_value


class Result(HorseVariable):
    def __init__(self, value: HorseType, required_grad=True, **kwargs):
        super().__init__(requires_grad=required_grad, **kwargs)
        self.value = value

    async def json(self):
        return self.value

    async def apply_gradients(self):
        raise NotImplementedError(
            "Result objects cannot be updated with gradients. Did you mean to use Value?"
        )
