import time

import pytest

from horsona.llm.base_engine import RateLimits


@pytest.mark.asyncio
async def test_call_limit():
    limits = RateLimits(
        [
            {"interval": 0.2, "max_calls": 2, "max_tokens": 10},
            {"interval": 0.4, "max_calls": 3, "max_tokens": 10},
        ]
    )

    start = time.time()

    # Should take no time to reach the limits
    await limits.consume_call()
    assert time.time() - start < 0.05
    await limits.consume_call()
    await limits.consume_call()
    assert time.time() - start >= 0.2

    await limits.consume_call()
    assert time.time() - start > 0.4
    assert time.time() - start < 0.5


@pytest.mark.asyncio
async def test_token_limit():
    limits = RateLimits(
        [
            {"interval": 0.1, "max_calls": 100, "max_tokens": 5},
            {"interval": 0.3, "max_calls": 100, "max_tokens": 9},
        ]
    )

    start = time.time()

    for i in range(5):
        await limits.wait_for(9)
        limits.report_tokens_consumed(9)
    await limits.wait_for(1)

    assert time.time() - start > 1.5
    assert time.time() - start < 1.8
