from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from horsona.autodiff.basic import HorseModule, HorseVariable

C = TypeVar("C", bound=HorseVariable)
Q = TypeVar("Q", bound=HorseVariable)


class BaseCache(HorseModule, Generic[C, Q], ABC):
    """
    An abstract base class for a cache module in the Horse framework.

    This class extends HorseModule and is parameterized with two generic types:
    C for the context (cache content) type, and Q for the query type.

    Attributes:
        context (C): The current state of the cache.

    Parameters:
        context (C): The initial context (cache content) to be stored.
        **kwargs: Additional keyword arguments to be passed to the parent HorseModule class.

    Methods:
        load(query: Q, **kwargs) -> C:
            Abstract method to load new data into the cache based on a query.
        sync() -> C:
            Synchronize the cache with its data source and return the updated cache.

    Note:
        This is an abstract base class. Concrete implementations must provide
        an implementation for the abstract 'load' method.

    Example:
        >>> class MyCache(BaseCache[dict, str]):
        ...     async def load(self, query: str, **kwargs) -> dict:
        ...         # Implementation to load data based on the query
        ...         pass
        >>> cache = MyCache({"initial": "data"})
        >>> updated_cache = await cache.load("new_query")
        >>> synced_cache = await cache.sync()
    """

    context: C

    def __init__(self, context: C, **kwargs):
        """
        Initialize a BaseCache instance with the given context.

        Args:
            context (C): The initial context (cache content) to be stored.
            **kwargs: Additional keyword arguments to be passed to the parent HorseModule class.
        """
        super().__init__(**kwargs)
        self.context = context

    @abstractmethod
    async def load(self, query: Q, **kwargs) -> C:
        """
        Load new data into the cache based on the provided query.

        This method updates the cache with new data fetched or computed
        based on the query, and returns the updated cache.

        Args:
            query (Q): The query used to determine what data to load.
            **kwargs: Additional keyword arguments for the load operation.

        Returns:
            C: The updated cache after loading the new data.
        """
        ...

    async def sync(self) -> C:
        """
        Synchronize the cache with its data source.

        This method ensures that the cache is up-to-date with respect to
        its data source, performing any necessary updates or reconciliations.

        Returns:
            C: The updated cache after synchronization.
        """
        return self.context
