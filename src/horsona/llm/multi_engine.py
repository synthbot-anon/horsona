import asyncio
import inspect
import sys
from random import random
from typing import Any, Generic, Type, TypeVar

from horsona.autodiff.basic import HorseData
from horsona.llm.base_engine import AsyncLLMEngine, engines, load_llms

T = TypeVar("T", bound=AsyncLLMEngine)


def get_mro_hierarchy(cls: Type) -> tuple[Type, ...]:
    """Returns the Method Resolution Order (MRO) for a class"""
    return cls.__mro__


def get_type(obj: Any) -> Type:
    if hasattr(obj, "get_type"):
        return obj.get_type()
    else:
        return type(obj)


def find_greatest_common_ancestor(objects: list[Any]) -> Type:
    """
    Find the most specific common ancestor class for a list of objects.

    Args:
        objects (list): List of Python objects

    Returns:
        type: The most specific common ancestor class

    Example:
        >>> class Animal: pass
        >>> class Mammal(Animal): pass
        >>> class Dog(Mammal): pass
        >>> class Cat(Mammal): pass
        >>> find_greatest_common_ancestor([Dog(), Cat()])
        <class 'Mammal'>
    """
    if not objects:
        raise ValueError("List of objects cannot be empty")

    # Get the MRO (Method Resolution Order) for the first object's class
    mros = [get_mro_hierarchy(get_type(obj)) for obj in objects]

    # Find common classes among all objects
    common_classes = set(mros[0])
    for mro in mros[1:]:
        common_classes.intersection_update(mro)

    if not common_classes:
        return object  # If no common ancestor found, return base object class

    # Find the most specific common ancestor
    # This will be the class that appears earliest in the first object's MRO
    for cls in mros[0]:
        if cls in common_classes:
            return cls

    return object  # Fallback to object class if no common ancestor found


class MultiEngine(HorseData, Generic[T]):
    def __init__(
        self,
        engines: list[T],
        max_retries: int = 3,
        backoff_exp: float = 2,
        backoff_multiplier: float = 1,
        name: str = None,
    ):
        super().__init__()
        self.engines = engines
        self.max_retries = max_retries
        self.backoff_exp = backoff_exp
        self.backoff_multiplier = backoff_multiplier
        self.name = name

    def __new__(
        cls,
        engines: list[T],
        max_retries: int = 3,
        backoff_exp: float = 2,
        backoff_multiplier: float = 1,
        name: str = None,
    ):
        self = super().__new__(cls)

        _engines = list(engines)
        _max_retries = max_retries
        _backoffs = {engine: -1 for engine in _engines}

        def select_engine():
            if len(_engines) == 0:
                raise Exception("No engines available")
            return sorted(_engines, key=lambda x: x.rate_limit.next_allowed())[0]

        def async_wrapper(name):
            async def wrapper(*args, **kwargs):
                last_exception = None

                for _ in range(_max_retries + 1):
                    selection = select_engine()
                    if _backoffs[selection] >= 0:
                        await asyncio.sleep(
                            random()
                            * backoff_multiplier
                            * (backoff_exp ** _backoffs[selection])
                        )

                    fn = getattr(selection, name)

                    try:
                        result = await fn(*args, **kwargs)
                        _backoffs[selection] = -1
                        return result
                    except Exception as e:
                        last_exception = e

                        if selection in _backoffs:
                            _backoffs[selection] += 1
                            if _backoffs[selection] == max_retries:
                                sys.stderr.write(
                                    f"Engine {selection.__class__.__name__} failed too many times, removing it\n"
                                )
                                _engines.remove(selection)
                                del _backoffs[selection]

                raise last_exception

            return wrapper

        self.select_engine = select_engine
        self.async_wrapper = async_wrapper
        return self

    def __getattr__(self, name: str):
        if name in ("state_dict", "load_state_dict", "get_type"):
            return getattr(self, name)

        selection = getattr(self, "select_engine")()
        result = getattr(selection, name)
        if inspect.iscoroutinefunction(result):
            return getattr(self, "async_wrapper")(name)
        else:
            return result

    def get_type(self) -> Type[T]:
        return find_greatest_common_ancestor(self.engines)

    def state_dict(self, **override: dict) -> dict:
        if self.name is not None:
            if override:
                raise ValueError(
                    "Cannot override fields when saving an AsyncLLMEngine by name"
                )
            return {
                "name": self.name,
            }
        else:
            return super().state_dict(**override)

    @classmethod
    def load_state_dict(
        cls, state_dict: dict, args: dict = {}, debug_prefix: list = []
    ) -> T:
        if isinstance(state_dict["name"], str):
            if args:
                raise ValueError(
                    "Cannot override fields when creating an AsyncLLMEngine by name"
                )
            load_llms()
            return engines[state_dict["name"]]
        else:
            return super().load_state_dict(state_dict, args, debug_prefix)


def create_multi_engine(
    *engines: T, max_retries: int = 3, backoff_exp=2, backoff_multiplier=1, name=None
) -> MultiEngine:
    return MultiEngine(engines, max_retries, backoff_exp, backoff_multiplier, name)
