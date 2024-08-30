import functools
import json
from abc import ABC, abstractmethod
from typing import Optional, TypeVar, Union

from pydantic import BaseModel

HorseType = TypeVar("HorseType", bound=Union[BaseModel, dict, int, float, bool, list])

class HorseVariable:
    def __init__(
        self,
        value: HorseType,
        predecessors: set["HorseVariable"] = set(),
        requires_grad: bool = True,
    ):
        self.value = value

        _predecessor_requires_grad = [v for v in predecessors if v.requires_grad]
        if (not requires_grad) and (len(_predecessor_requires_grad) > 0):
            raise Exception(
                "If the variable does not require grad, none of its predecessors "
                "should require grad. In this case, following predecessors require "
                f"grad: {_predecessor_requires_grad}"
            )

        self.gradients: list[HorseVariable] = list()
        self.grad_fn = None
        self.predecessors = set(predecessors)
        self.requires_grad = requires_grad

    def __str__(self):
        if isinstance(self.value, BaseModel):
            return self.value.model_dump_json(indent=2)
        else:
            return json.dumps(self.value, indent=2)

    async def apply_gradients(self):
        raise NotImplementedError

    async def reset_gradients(self):
        self.gradients = []

    async def backward(self):
        """
        Backpropagate gradients through the computation graph starting from this
        variable.
        """
        topo: list[HorseVariable] = []
        visited = set()

        def build_topo(v: HorseVariable):
            if v not in visited:
                visited.add(v)
                for predecessor in v.predecessors:
                    build_topo(predecessor)
                topo.append(v)

        build_topo(self)

        for v in reversed(topo):
            if v.requires_grad:
                if v.grad_fn is not None:
                    await v.grad_fn()

    async def forward(self):
        return self

    def __add__(self, other: "HorseVariable"):
        async def sum_grad_fn(a: HorseVariable, b: HorseVariable):
            a.gradients.extend(self.gradients)
            b.gradients.extend(self.gradients)

        result = HorseVariable(
            value=[self.value, other.value],
            predecessors=set([self, other]),
        )
        result.grad_fn = functools.partial(sum_grad_fn, self, other)
        return result


class HorseFunction(ABC):
    """
    The class to define a function that can be called and backpropagated through.
    """

    def __init__(self):
        super().__init__()

    async def __call__(self, *args, **kwargs):
        result = await self.forward(*args, **kwargs)
        result.grad_fn = functools.partial(self.backward, result, *args, **kwargs)
        return result

    @abstractmethod
    async def forward(self, *args, **kwargs) -> HorseVariable:
        """Execute the function call and return a Variable result.
        
        This function should return a Variable object that represents the result of the
        function call."""
        pass

    @abstractmethod
    async def backward(self, result: HorseVariable, *args, **kwargs):
        """Set the gradient for all input variables that require a gradient.

        This function should add to predecessor.gradients for all predecessors where
        .required_grad is True.

        This function should be an async function.
        """
        pass


class HorseModule(ABC):
    """Abstract module class with parameters akin to PyTorch's nn.Module."""

    def parameters(self):
        visited = set([self])
        for value in self.__dict__.values():
            if isinstance(value, HorseVariable):
                yield value
            elif isinstance(value, HorseModule):
                if value in visited:
                    continue
                yield from value.parameters()


    async def zero_grad(self):
        for p in self.parameters():
            await p.reset_gradients()


class HorseOptimizer:
    def __init__(self, parameters):
        self.parameters: set[HorseVariable] = set(parameters)

    async def step(self):
        for param in self.parameters:
            if param.requires_grad and param.gradients:
                await param.apply_gradients()

    async def zero_grad(self):
        for param in self.parameters:
            await param.reset_gradients()
