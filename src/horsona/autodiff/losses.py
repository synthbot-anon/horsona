from typing import AsyncGenerator

from horsona.autodiff.variables import Value

from .basic import GradContext, HorseGradient, HorseVariable, horsefunction


@horsefunction
async def apply_loss(
    arg: HorseVariable, loss: HorseGradient
) -> AsyncGenerator[HorseVariable, GradContext]:
    grad_context = yield Value("Errata", loss, predecessors=[arg])
    if arg not in grad_context:
        return
    grad_context[arg].append(loss)
