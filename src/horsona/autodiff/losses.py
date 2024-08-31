from .basic import HorseFunction, HorseType, HorseVariable


class ConstantLoss(HorseFunction):
    def __init__(self, loss: HorseType):
        self.task = loss

    async def forward(self, *args: HorseVariable) -> HorseVariable:
        return HorseVariable(
            value=self.task,
            predecessors=list(args),
        )

    async def backward(self, response: HorseVariable, *args: HorseVariable):
        for var in args:
            if var.requires_grad:
                var.gradients.append(self.task)
