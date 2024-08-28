import functools

from pydantic import BaseModel

from horsona.llm.base_engine import AsyncLLMEngine

from .basic import HorseFunction, HorseVariable


class TextExtractor(HorseFunction):
    def __init__(self, llm: AsyncLLMEngine = None):
        self.llm = llm

    async def forward(self, model_cls, **kwargs) -> HorseVariable:
        extraction = await self.llm.query_object(
            model_cls,
            **kwargs,
        )

        result = HorseVariable(
            value=extraction,
            predecessors=[x for x in kwargs.values() if isinstance(x, HorseVariable)],
        )

        result.grad_fn = functools.partial(self.backward, result, kwargs)

        return result

    async def backward(
        self, result: HorseVariable, kwargs
    ):
        if not result.requires_grad:
            return

        if not result.gradients:
            return

        class ResultMatches(BaseModel):
            requires_update: bool

        response = await self.llm.query_object(
            ResultMatches,
            RESULT=result,
            FEEDBACK=result.gradients,
            TASK="Based on the FEEDBACK, check if the RESULT requires an update.",
        )

        if not response.requires_update:
            return

        class SuggestedAssignment(BaseModel):
            input_name: str
            relevant_feedback: list[str]

        class FeedbackAssignmnets(BaseModel):
            assignments: list[SuggestedAssignment]
        
        gradients = await self.llm.query_object(
            FeedbackAssignmnets,
            INPUTS=[{'name': k, 'value': v} for k,v in kwargs.items() if isinstance(v, HorseVariable) and v.requires_grad],
            RESULT=result,
            FEEDBACK=result.gradients,
            TASK=(
                "The FEEDBACK was given when extracting RESULT from INPUTS. "
                "Based on the errors, determine which list of FEEDBACK items applies for each INPUT."
            ),
        )

        print(gradients.model_dump_json(indent=2))

        for change in gradients.assignments:
            if change.input_name not in kwargs:
                continue

            variable = kwargs[change.input_name]
            
            if not isinstance(variable, HorseVariable):
                continue
            if not variable.requires_grad:
                continue
            
            variable.gradients.extend(change.relevant_feedback)
