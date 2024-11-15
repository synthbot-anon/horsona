import asyncio
from abc import abstractmethod
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional


class ResourceStateLock:
    """
    A lock implementation for managing concurrent access to resources based on state.

    This class ensures that only one state can be active at a time, while allowing
    multiple concurrent tasks requiring the same state. The lock is only released when
    all tasks for the current state are completed.
    """

    def __init__(self):
        self._state_change = asyncio.Event()
        self._active_state: Optional[str] = None
        self._task_counter: int = 0
        self._state_set = asyncio.Event()
        self._state_set.set()

    @asynccontextmanager
    async def acquire(
        self, resource: str, required_state: str
    ) -> AsyncGenerator[None, None]:
        """
        Context manager for acquiring the resource lock.

        Args:
            resource: The resource to acquire the lock for
            required_state: The required state for the resource

        Raises:
            TimeoutError: If the lock cannot be acquired within the timeout period
            RuntimeError: If the lock cannot be acquired
        """
        try:
            # Wait until the state can be or has been claimed
            while self._active_state != required_state:
                if self._active_state is None:
                    # No active state, so we can claim it
                    self._active_state = required_state
                    # Notify anyone waiting so they can check the new state
                    self._state_change.set()
                    self._state_change.clear()
                    break
                await self._state_change.wait()

            self._task_counter += 1

            await self.set_state(resource, required_state)

            yield

        finally:
            self._task_counter -= 1
            if self._task_counter == 0:
                self._active_state = None
                self._state_change.set()
                self._state_change.clear()

    @abstractmethod
    async def set_state(self, resource: str, state: str) -> None:
        pass
