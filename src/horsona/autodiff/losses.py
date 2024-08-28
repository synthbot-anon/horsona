import functools

from pydantic import BaseModel

from .basic import HorseModule, HorseType, HorseVariable


class ConstantLoss(HorseModule):
    def __init__(self, loss: HorseType):
        self.task = loss
    
    async def __call__(self, *args: list[HorseVariable]) -> HorseVariable:
        return await self.forward(*args)

    async def forward(self, *args: list[HorseVariable]) -> HorseVariable:
        result = HorseVariable(
            value=self.task,
            predecessors=list(args),
        )
        result.grad_fn = functools.partial(self.backward, response=result)
        return result

    async def backward(self, response: HorseVariable):
        for var in response.predecessors:
            if var.requires_grad:
                var.gradients.append(self.task)
