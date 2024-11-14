from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from horsona.autodiff.basic import HorseData

Q = TypeVar("Q")


class BaseIndex(Generic[Q], HorseData, ABC):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @abstractmethod
    async def query(self, query: Q, *args, **kwargs) -> dict[str, list]: ...

    @abstractmethod
    async def extend(self, data: list[Q]) -> None: ...

    @abstractmethod
    async def delete(self, indices: list[int | Q] = []) -> None: ...
