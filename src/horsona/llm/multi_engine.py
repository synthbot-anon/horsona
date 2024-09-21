import asyncio
import inspect
import sys
from random import random
from typing import TypeVar

from horsona.llm.base_engine import AsyncLLMEngine

T = TypeVar("T", bound=AsyncLLMEngine)


def create_multi_engine(
    *engines: T, max_retries: int = 3, backoff_exp=2, backoff_multiplier=1
) -> T:
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
                print("using", selection)
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

    class MultiEngine:
        def __getattr__(self, name):
            selection = select_engine()
            print("using", selection)
            result = getattr(selection, name)

            if inspect.iscoroutinefunction(result):
                return async_wrapper(name)
            else:
                return result

    return MultiEngine()
