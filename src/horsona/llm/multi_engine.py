import asyncio
import inspect
import sys
from random import random
from typing import Type, TypeVar

from horsona.autodiff.basic import HorseData
from horsona.llm.base_engine import AsyncLLMEngine, engines, load_engines

T = TypeVar("T", bound=AsyncLLMEngine)


class MultiEngine(HorseData):
    def __init__(
        self,
        engines: list[T],
        max_retries: int = 3,
        backoff_exp=2,
        backoff_multiplier=1,
        name=None,
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
        backoff_exp=2,
        backoff_multiplier=1,
        name=None,
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

    def __getattr__(self, name):
        if name in ("state_dict", "load_state_dict"):
            return getattr(self, name)

        selection = getattr(self, "select_engine")()
        result = getattr(selection, name)
        if inspect.iscoroutinefunction(result):
            return getattr(self, "async_wrapper")(name)
        else:
            return result

    def state_dict(self, **override):
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
    def load_state_dict(cls, state_dict, args={}, debug_prefix=[]):
        if isinstance(state_dict["name"], str):
            if args:
                raise ValueError(
                    "Cannot override fields when creating an AsyncLLMEngine by name"
                )
            load_engines()
            return engines[state_dict["name"]]
        else:
            return super().load_state_dict(state_dict, args, debug_prefix)


def create_multi_engine(
    *engines: T, max_retries: int = 3, backoff_exp=2, backoff_multiplier=1, name=None
):
    return MultiEngine(engines, max_retries, backoff_exp, backoff_multiplier, name)
