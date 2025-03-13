from typing import Any, AsyncGenerator

from pydantic import BaseModel

from horsona.autodiff.basic import (
    GradContext,
    HorseData,
    HorseGradient,
    HorseVariable,
    horsefunction,
)
from horsona.autodiff.variables import Value
from horsona.llm.base_engine import AsyncLLMEngine


@horsefunction
async def extract_object(
    llm: AsyncLLMEngine,
    model_cls: type[BaseModel],
    **kwargs,
) -> AsyncGenerator[Value, GradContext]:
    extraction = await llm.query_object(
        model_cls,
        **kwargs,
    )

    variable_inputs = {k: v for k, v in kwargs.items() if isinstance(v, HorseVariable)}
    result = Value(
        datatype=model_cls.__name__,
        value=extraction,
        predecessors=list(variable_inputs.values()),
    )

    grad_context = yield result

    if result not in grad_context:
        return

    await assign_feedback(
        llm,
        grad_context,
        result,
        variable_inputs,
    )


async def assign_feedback(
    llm: AsyncLLMEngine,
    grad_context: dict[HorseVariable, list[HorseGradient]],
    result: HorseVariable,
    inputs: Any,
) -> None:
    class SuggestedAssignment(BaseModel):
        input_name: str
        relevant_feedback: list[str]

    class FeedbackAssignments(BaseModel):
        assignments: list[SuggestedAssignment]

    gradients = await llm.query_object(
        FeedbackAssignments,
        INPUTS=inputs,
        RESULT=result,
        FEEDBACK=grad_context[result],
        TASK=(
            "The INPUTS have the format <name>value</name>. "
            "The FEEDBACK was given when extracting RESULT from INPUTS. "
            f"Based on the errors, determine which list of FEEDBACK items applies for each INPUT {list(inputs.keys())}. "
            "Use the FEEDBACK values, not indices."
        ),
    )

    for change in gradients.assignments:
        if change.input_name not in inputs:
            continue

        variable = inputs[change.input_name]
        if not isinstance(variable, HorseVariable):
            continue
        if variable not in grad_context:
            continue

        grad_context[variable].extend(
            [Value("Feedback", x) for x in change.relevant_feedback]
        )
