from abc import ABC, abstractmethod

from horsona.autodiff.basic import HorseVariable


class Dataset(HorseVariable, ABC):
    async def dict(self):
        return await self.summary()

    @abstractmethod
    async def summary(self):
        pass
