from horsona.autodiff.variables import Value

from .basic import HorseFunction, HorseGradient, HorseVariable


class ConstantLoss(HorseFunction):
    def __init__(self, loss: HorseGradient = None):
        self.loss = loss

    async def forward(
        self, arg: HorseVariable, loss: HorseGradient = None
    ) -> HorseVariable:
        return Value(loss or self.loss, predecessors=[arg])

    async def backward(
        self,
        context: dict[HorseVariable, list[HorseGradient]],
        result: HorseVariable,
        arg: HorseVariable,
        loss: HorseGradient = None,
    ):
        return {arg: [loss or self.loss]}
