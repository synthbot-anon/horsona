import asyncio
import importlib
import inspect
from abc import ABC
from collections import defaultdict
from functools import wraps
from types import MappingProxyType
from typing import (
    AsyncGenerator,
    Awaitable,
    Callable,
    Collection,
    Generator,
    ParamSpec,
    Type,
    TypeVar,
    Union,
    overload,
)

from anthropic import NoneType
from pydantic import BaseModel


class HorseGradient(BaseModel, ABC):
    pass


M = TypeVar("M", bound=Union["HorseModule", "HorseVariable"])


class HorseData:
    def state_dict(self, **override):
        fields = self.__dict__.copy()
        fields.update(override)
        return state_dict(fields)["data"]

    @classmethod
    def load_state_dict(cls: Type[M], state_dict, args={}, debug_prefix=[]) -> M:
        if hasattr(args, "__call__"):
            args = args()

        kwargs = {}
        remaining_args = args.copy()

        for k, v in state_dict.items():
            kwargs[k] = load_state_dict(
                v, remaining_args.pop(k, {}), debug_prefix=debug_prefix + [k]
            )

        kwargs.update(remaining_args)
        try:
            return cls(**kwargs)
        except Exception as e:
            raise Exception(f"Error loading {'.'.join(debug_prefix)} ({cls})") from e


class HorseVariable(HorseData, ABC):
    def __init__(
        self,
        predecessors: set["HorseVariable"] = set(),
        requires_grad: bool = False,
        name: str = None,
        grad_fn: Callable[["GradContext"], Awaitable[None]] = None,
    ):
        super().__init__()
        self.grad_fn = grad_fn
        self.predecessors = set(predecessors)
        self.requires_grad = requires_grad
        self.name = name

    async def json(self):
        raise NotImplementedError(
            f"Class {self.__class__.__name__} can't be passed to LLMEngines since it doesn't implement json"
        )

    def state_dict(self, **override):
        fields = self.__dict__.copy()
        del fields["predecessors"]
        fields.update(override)
        return state_dict(fields)["data"]

    def __repr__(self):
        if self.name is not None:
            return f"{self.name}(class={self.__class__.__name__}, requires_grad={self.requires_grad})"
        else:
            return f"{self.__class__.__name__}(requires_grad={self.requires_grad})"

    async def apply_gradients(self, gradients: list[HorseGradient]):
        raise NotImplementedError(
            f"Class {self.__class__.__name__} can't accept gradients since it doesn't implement apply_gradients"
        )

    async def backward(
        self, leaves: Collection["HorseVariable"]
    ) -> dict["HorseVariable", list[HorseGradient]]:
        """
        Backpropagate gradients through the computation graph starting from this
        variable.
        """
        assert (
            x.requires_grad for x in leaves
        ), f"Can't call backward on a variable that doesn't require gradients"

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


class HorseModule(HorseData, ABC):
    """Abstract module class with parameters akin to PyTorch's nn.Module."""

    def __init__(self, name=None):
        super().__init__()
        self.name = name

    def __repr__(self):
        if self.name is not None:
            return f"{self.name}(class={self.__class__.__name__})"
        else:
            return f"{self.__class__.__name__}()"

    def parameters(self):
        def _parameters(obj):
            visited = set([obj])
            for value in obj.__dict__.values():
                if isinstance(value, HorseVariable):
                    if value.requires_grad:
                        yield value
                elif isinstance(value, HorseModule):
                    if value in visited:
                        continue
                    yield from value.parameters()

        return list(_parameters(self))


def load_state_dict(state_dict, args, debug_prefix=[]):
    if not isinstance(args, dict):
        return args

    package_name = state_dict["package"]
    type_name = state_dict["type"]
    data = state_dict["data"]

    if data is None:
        return None

    field_module = importlib.import_module(package_name)
    field_class = getattr(field_module, type_name)

    if issubclass(field_class, HorseData):
        return field_class.load_state_dict(data, args, debug_prefix=debug_prefix)
    elif issubclass(field_class, BaseModel):
        kwargs = data.copy()
        kwargs.update(args)
        return field_class(**kwargs)
    elif package_name == "builtins" and issubclass(
        field_class, (bool, int, float, str, bytes)
    ):
        return field_class(data)
    elif package_name == "builtins" and issubclass(field_class, (list, tuple, set)):
        return [load_state_dict(v, args, debug_prefix=debug_prefix) for v in data]
    elif package_name == "builtins" and issubclass(field_class, dict):
        return {
            k: load_state_dict(v, args.get(k, {}), debug_prefix=debug_prefix + [k])
            for k, v in data.items()
        }
    else:
        raise ValueError(
            f"Invalid class passed to load_state_dict: {field_class}. Must be a HorseModule, HorseVariable, or a json-compatible type."
        )


def state_dict(value):
    if isinstance(value, dict):
        result = {}
        for k, v in value.items():
            v_dict = state_dict(v)
            if v_dict is None:
                continue
            result[k] = v_dict
        return {
            "package": "builtins",
            "type": "dict",
            "data": result,
        }
    elif value is None:
        return {
            "package": "builtins",
            "type": "NoneType",
            "data": None,
        }
    elif isinstance(value, HorseData):
        return {
            "package": value.__class__.__module__,
            "type": value.__class__.__name__,
            "data": value.state_dict(),
        }
    elif isinstance(value, BaseModel):
        return {
            "package": value.__class__.__module__,
            "type": value.__class__.__name__,
            "data": value.model_dump(),
        }
    elif isinstance(value, (bool, int, float, str, bytes)):
        return {
            "package": "builtins",
            "type": type(value).__name__,
            "data": value,
        }
    elif isinstance(value, (list, tuple, set)):
        result = []
        for v in value:
            v_dict = state_dict(v)
            if v_dict is None:
                continue
            result.append(v_dict)
        return {
            "package": "builtins",
            "type": type(value).__name__,
            "data": result,
        }
    else:
        return None


async def step(gradients: dict[HorseVariable, list[HorseGradient]]):
    tasks = []
    for v, g in gradients.items():
        if v.requires_grad:
            tasks.append(v.apply_gradients(g))
    await asyncio.gather(*tasks)
