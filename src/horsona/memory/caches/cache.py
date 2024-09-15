from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from horsona.autodiff.basic import HorseModule, HorseVariable

C = TypeVar("C", bound=HorseVariable)
Q = TypeVar("Q", bound=HorseVariable)


class Cache(HorseModule, Generic[C], ABC):
    context: C

    def __init__(self, context: C):
        self.context = context

    @abstractmethod
    async def load(self, query: Q, **kwargs) -> C:
        pass

    async def sync(self) -> C:
        return self.context
