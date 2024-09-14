from abc import ABC, abstractmethod
from collections import defaultdict
from reprlib import recursive_repr

from pydantic import BaseModel


class partial:
    """New function with partial application of the given arguments
    and keywords. This is a modification of the functools.partial class.
    Positional args provided during the call will be prepended to the arguments
    instead of appended.
    """

    __slots__ = "func", "args", "keywords", "__dict__", "__weakref__"

    def __new__(cls, func, /, *args, **keywords):
        if not callable(func):
            raise TypeError("the first argument must be callable")

        if hasattr(func, "func"):
            args = func.args + args
            keywords = {**func.keywords, **keywords}
            func = func.func

        self = super(partial, cls).__new__(cls)

        self.func = func
        self.args = args
        self.keywords = keywords
        return self

    def __call__(self, /, *args, **keywords):
        keywords = {**self.keywords, **keywords}
        return self.func(*args, *self.args, **keywords)

    @recursive_repr()
    def __repr__(self):
        qualname = type(self).__qualname__
        args = [repr(self.func)]
        args.extend(repr(x) for x in self.args)
        args.extend(f"{k}={v!r}" for (k, v) in self.keywords.items())
        if type(self).__module__ == "functools":
            return f"functools.{qualname}({', '.join(args)})"
        return f"{qualname}({', '.join(args)})"

    def __reduce__(self):
        return (
            type(self),
            (self.func,),
            (self.func, self.args, self.keywords or None, self.__dict__ or None),
        )

    def __setstate__(self, state):
        if not isinstance(state, tuple):
            raise TypeError("argument to __setstate__ must be a tuple")
        if len(state) != 4:
            raise TypeError(f"expected 4 items in state, got {len(state)}")
        func, args, kwds, namespace = state
        if (
            not callable(func)
            or not isinstance(args, tuple)
            or (kwds is not None and not isinstance(kwds, dict))
            or (namespace is not None and not isinstance(namespace, dict))
        ):
            raise TypeError("invalid partial state")

        args = tuple(args)  # just in case it's a subclass
        if kwds is None:
            kwds = {}
        elif type(kwds) is not dict:  # XXX does it need to be *exactly* dict?
            kwds = dict(kwds)
        if namespace is None:
            namespace = {}

        self.__dict__ = namespace
        self.func = func
        self.args = args
        self.keywords = kwds


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

        grad_context: dict[HorseVariable, list[HorseGradient]] = defaultdict(list)
        for v in reversed(topo):
            if v.grad_fn is not None:
                new_gradients = await v.grad_fn(grad_context)
                if not new_gradients:
                    continue
                for k, g in new_gradients.items():
                    grad_context[k].extend(g)

        return grad_context

    async def forward(self):
        return self

    def __add__(self, other: "HorseVariable"):
        class Sum(HorseVariable):
            async def json(self):
                return [await x.json() for x in self.predecessors]

        async def sum_grad_fn(
            context, r: HorseVariable, a: HorseVariable, b: HorseVariable
        ):
            result = {
                a: context[r],
                b: context[r],
            }
            return result

        result = Sum(predecessors=[self, other])
        result.grad_fn = partial(sum_grad_fn, result, self, other)
        return result


class HorseFunction(ABC):
    """
    The class to define a function that can be called and backpropagated through.
    """

    def __init__(self):
        super().__init__()

    async def __call__(self, *args, **kwargs) -> HorseVariable:
        result = await self.forward(*args, **kwargs)

        result.grad_fn = partial(self.backward, result, *args, **kwargs)
        return result

    @abstractmethod
    async def forward(self, *args, **kwargs) -> HorseVariable:
        """Execute the function call and return a Variable result.

        This function should return a Variable object that represents the result of the
        function call."""
        pass

    @abstractmethod
    async def backward(self, result: HorseVariable, *args, **kwargs):
        """Return a dict of gradients for all input variables."""
        pass


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


class HorseOptimizer:
    def __init__(self, parameters):
        self.parameters: set[HorseVariable] = set(parameters)
        for p in self.parameters:
            if not p.requires_grad:
                raise ValueError(f"{p} must require gradients")

        if len(self.parameters) == 0:
            raise ValueError("Optimizer got an empty parameter set")

    async def step(self, gradients: dict[HorseVariable, list[HorseGradient]]):
        for v, g in gradients.items():
            if v.requires_grad:
                await v.apply_gradients(g)
