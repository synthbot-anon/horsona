from pydantic import BaseModel

from horsona.autodiff.basic import HorseGradient, HorseVariable
from horsona.llm.base_engine import AsyncLLMEngine


async def assign_feedback(
    llm: AsyncLLMEngine,
    context: dict[HorseVariable, list[HorseGradient]],
    result: HorseVariable,
    inputs,
    **kwargs
):
    class SuggestedAssignment(BaseModel):
        input_name: str
        relevant_feedback: list[str]

    class FeedbackAssignments(BaseModel):
        assignments: list[SuggestedAssignment]

    gradients = await llm.query_object(
        FeedbackAssignments,
        INPUTS=inputs,
        RESULT=result,
        FEEDBACK=context[result],
        **kwargs,
    )

    result = {}
    for change in gradients.assignments:
        if change.input_name not in inputs:
            continue

        variable = inputs[change.input_name]

        if not isinstance(variable, HorseVariable):
            continue

        result[variable] = change.relevant_feedback

    return result
