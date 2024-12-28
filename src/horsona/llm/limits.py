import asyncio
import time
from typing import Optional

from horsona.autodiff.basic import HorseData


class CallLimit(HorseData):
    """
    Tracks and enforces rate limits based on number of API calls.
    """

    def __init__(self, limit: float, interval: float) -> None:
        """
        Args:
            limit: Maximum number of calls allowed per interval
            interval: Time interval in seconds
        """
        assert limit is not None and limit > 0, "Call limit must be a positive float"
        assert interval >= 0, "Rate interval must be a non-negative float"

        self.limit = limit
        self.interval = interval
        self.last_blocked = time.time() - self.interval / self.limit

    async def consume(self) -> None:
        """Record consumption of one call and wait if needed."""
        if self.limit is None:
            return

        await self.wait_for()
        self.last_blocked = max(
            self.last_blocked, time.time() - self.interval / self.limit
        )
        self.last_blocked += self.interval / self.limit

    def next_allowed(self) -> float:
        """Return timestamp when next call will be allowed."""
        return max(self.last_blocked + self.interval / self.limit, time.time())

    async def wait_for(self) -> None:
        """Wait until next call is allowed."""
        next_allowed = self.next_allowed()
        now = time.time()
        if next_allowed > now:
            await asyncio.sleep(next_allowed - now)


class TokenLimit(HorseData):
    """
    Tracks and enforces rate limits based on number of tokens.
    """

    def __init__(self, limit: float, interval: float) -> None:
        """
        Args:
            limit: Maximum number of tokens allowed per interval
            interval: Time interval in seconds
        """
        assert limit is not None and limit > 0, "Token limit must be a positive float"
        assert interval >= 0, "Rate interval must be a non-negative float"

        self.limit = limit
        self.interval = interval
        self.last_blocked = time.time() - self.interval / self.limit

    def report_consumed(self, count: int) -> None:
        """Record consumption of tokens."""
        self.last_blocked = max(
            self.last_blocked, time.time() - self.interval + self.interval / self.limit
        )
        self.last_blocked += self.interval / self.limit * count

    def next_allowed(self, count: int) -> float:
        """Return timestamp when consuming given number of tokens will be allowed."""
        if self.limit is None:
            return time.time()

        return max(
            self.last_blocked + self.interval / self.limit * count,
            time.time() + self.interval / self.limit * (count - 1),
        )

    async def wait_for(self, count: Optional[int]) -> None:
        """Wait until consuming given number of tokens is allowed."""
        if count is None:
            count = 1

        next_allowed = self.next_allowed(count)
        now = time.time()
        if next_allowed > now:
            await asyncio.sleep(next_allowed - now)
