from horsona.autodiff.variables import Result

from .basic import HorseFunction, HorseType, HorseVariable


class ConstantLoss(HorseFunction):
    def __init__(self, loss: HorseType = None):
        self.loss = loss

    async def forward(
        self, arg: HorseVariable, loss: HorseType = None
    ) -> HorseVariable:
        return Result(loss or self.loss, predecessors=[arg])

    async def backward(
        self, result: HorseVariable, arg: HorseVariable, loss: HorseType = None
    ):
        if arg.requires_grad:
            arg.gradients.append(loss or self.loss)
