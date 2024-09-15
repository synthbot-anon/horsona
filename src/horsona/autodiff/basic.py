import asyncio
import inspect
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import wraps
from types import MappingProxyType
from typing import (
    AsyncGenerator,
    Awaitable,
    Callable,
    Collection,
    Generator,
    Optional,
    ParamSpec,
    TypeVar,
    Union,
    overload,
)

from pydantic import BaseModel


class HorseGradient(BaseModel, ABC):
    pass


class HorseVariable(ABC):
    def __init__(
        self,
        predecessors: set["HorseVariable"] = set(),
        requires_grad: bool = False,
    ):
        self.grad_fn = None
        self.predecessors = set(predecessors)
        self.requires_grad = requires_grad

    @abstractmethod
    async def json(self):
        pass

    async def apply_gradients(self, gradients: list[HorseGradient]):
        raise NotImplementedError

    async def backward(
        self, leaves: Collection["HorseVariable"]
    ) -> dict["HorseVariable", list[HorseGradient]]:
        """
        Backpropagate gradients through the computation graph starting from this
        variable.
        """
        leaf_variables = set(leaves)
        topo: list[HorseVariable] = []
        visited = set()
        in_path = {}
        pending_parents = defaultdict(set)
        children = defaultdict(set)

        def build_topo(v: HorseVariable) -> bool:
            if v in visited:
                # If already visited, return whether it's on a path to a leaf variable
                return in_path.get(v, False)

            # Check if the current variable is a leaf variable
            is_in_path = v in leaf_variables

            # Recursively visit predecessors
            if v not in leaf_variables:
                for predecessor in v.predecessors:
                    if build_topo(predecessor):
                        is_in_path = True
                        pending_parents[predecessor].add(v)
                        children[v].add(predecessor)

            # If the current variable is on a path to any leaf variable, add it to topo
            if is_in_path:
                topo.append(v)

            # Record whether the current variable is on a path to a leaf variable
            in_path[v] = is_in_path
            visited.add(v)

            return is_in_path

        build_topo(self)

        grad_context = {k: [] for k in topo}
        grad_context = MappingProxyType(grad_context)

        async def calculate_gradients(v: HorseVariable):
            if v.grad_fn is not None:
                await v.grad_fn(grad_context)
            tasks = []
            for child in children[v]:
                pending_parents[child].remove(v)
                if not pending_parents[child]:
                    tasks.append(calculate_gradients(child))
            await asyncio.gather(*tasks)

        await calculate_gradients(self)

        return grad_context

    def __add__(self, other: "HorseVariable"):
        class Sum(HorseVariable):
            async def json(self):
                return [await x.json() for x in self.predecessors]

        @horsefunction
        def sum_variables(
            a: HorseVariable, b: HorseVariable
        ) -> Generator[Sum, GradContext, None]:
            result = Sum(predecessors=[a, b])
            context = yield result
            context[a].extend(context[result])
            context[b].extend(context[result])

        return sum_variables(self, other)


P = ParamSpec("P")
T = TypeVar("T")
GradContext = defaultdict[HorseVariable, list[HorseGradient]]


class GradContext:
    pass


@overload
def horsefunction(func: Callable[P, Generator[T, GradContext, None]]) -> Callable[P, T]:
    ...


@overload
def horsefunction(
    func: Callable[P, AsyncGenerator[T, GradContext]],
) -> Callable[P, Awaitable[T]]:
    ...


def horsefunction(
    func: Callable[
        P, Union[Generator[T, GradContext, None], AsyncGenerator[T, GradContext]]
    ],
) -> Union[Callable[P, T], Callable[P, Awaitable[T]]]:
    if inspect.isasyncgenfunction(func):

        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            generator = func(*args, **kwargs)
            result: HorseVariable = await generator.__anext__()

            async def backward(context):
                try:
                    await generator.asend(context)
                except StopAsyncIteration:
                    pass

            result.grad_fn = backward
            return result

        return wrapper
    elif inspect.isgeneratorfunction(func):

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            generator = func(*args, **kwargs)
            result: HorseVariable = next(generator)

            async def backward(context):
                try:
                    generator.send(context)
                except StopIteration:
                    pass

            result.grad_fn = backward
            return result

        return wrapper
    else:
        raise TypeError(
            "Function must be a generator function or async generator function"
        )


class HorseModule(ABC):
    """Abstract module class with parameters akin to PyTorch's nn.Module."""

    def parameters(self):
        visited = set([self])
        for value in self.__dict__.values():
            if isinstance(value, HorseVariable):
                if value.requires_grad:
                    yield value
            elif isinstance(value, HorseModule):
                if value in visited:
                    continue
                yield from value.parameters()


async def step(gradients: dict[HorseVariable, list[HorseGradient]]):
    tasks = []
    for v, g in gradients.items():
        if v.requires_grad:
            tasks.append(v.apply_gradients(g))
    await asyncio.gather(*tasks)
