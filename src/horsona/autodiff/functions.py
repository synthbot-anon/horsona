from typing import AsyncGenerator

from pydantic import BaseModel

from horsona.autodiff.basic import (
    GradContext,
    HorseGradient,
    HorseVariable,
    horsefunction,
)
from horsona.autodiff.variables import Value
from horsona.llm.base_engine import AsyncLLMEngine


@horsefunction
async def extract_object(
    llm: AsyncLLMEngine,
    model_cls: type[HorseVariable],
    **kwargs,
) -> AsyncGenerator[Value, GradContext]:
    extraction = await llm.query_object(
        model_cls,
        **kwargs,
    )

    variable_inputs = {k: v for k, v in kwargs.items() if isinstance(v, HorseVariable)}
    result = Value(
        value=extraction,
        predecessors=list(variable_inputs.values()),
    )

    context = yield result

    if result not in context:
        return

    await assign_feedback(
        llm,
        context,
        result,
        variable_inputs,
    )


async def assign_feedback(
    llm: AsyncLLMEngine,
    context: dict[HorseVariable, list[HorseGradient]],
    result: HorseVariable,
    inputs,
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
        TASK=(
            "The FEEDBACK was given when extracting RESULT from INPUTS. "
            f"Based on the errors, determine which list of FEEDBACK items applies for each INPUT {list(inputs.keys())}."
        ),
    )

    for change in gradients.assignments:
        if change.input_name not in inputs:
            continue

        variable = inputs[change.input_name]

        if not isinstance(variable, HorseVariable):
            continue

        context[variable].extend([Value(x) for x in change.relevant_feedback])
