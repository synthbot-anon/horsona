from typing import AsyncGenerator

from horsona.autodiff.variables import Value

from .basic import GradContext, HorseGradient, HorseVariable, horsefunction


@horsefunction
async def apply_loss(
    arg: HorseVariable, loss: HorseGradient
) -> AsyncGenerator[Value, GradContext]:
    context = yield Value(loss, predecessors=[arg])
    context[arg].append(loss)
