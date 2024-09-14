from horsona.autodiff.basic import HorseFunction, HorseGradient, HorseVariable
from horsona.autodiff.utils import assign_feedback
from horsona.autodiff.variables import Value
from horsona.llm.base_engine import AsyncLLMEngine


class TextExtractor(HorseFunction):
    def __init__(self, llm: AsyncLLMEngine = None):
        self.llm = llm

    async def forward(self, model_cls: type[HorseVariable], **kwargs) -> Value:
        extraction = await self.llm.query_object(
            model_cls,
            **kwargs,
        )

        return Value(
            value=extraction,
            predecessors=[x for x in kwargs.values() if isinstance(x, HorseVariable)],
        )

    async def backward(
        self,
        context: dict[HorseVariable, list[HorseGradient]],
        result: HorseVariable,
        model_cls,
        **kwargs,
    ):
        if not context[result]:
            return

        inputs = {k: v for k, v in kwargs.items() if isinstance(v, HorseVariable)}
        return await assign_feedback(
            self.llm,
            context,
            result,
            inputs,
            TASK=(
                "The FEEDBACK was given when extracting RESULT from INPUTS. "
                f"Based on the errors, determine which list of FEEDBACK items applies for each INPUT {list(inputs.keys())}."
            ),
        )
