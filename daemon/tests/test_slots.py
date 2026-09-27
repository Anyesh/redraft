import asyncio

import pytest

from daemon.slots import Busy, SlotPool


async def test_preferred_free_slot_is_reused():
    pool = SlotPool(total=2, queue_depth=1, queue_timeout_s=1.0)
    assert await pool.acquire(preferred=1) == 1
    pool.release(1)
    assert await pool.acquire(preferred=1) == 1


async def test_busy_preferred_slot_falls_back_to_a_free_one():
    pool = SlotPool(total=2, queue_depth=1, queue_timeout_s=1.0)
    held = await pool.acquire(preferred=0)
    assert await pool.acquire(preferred=held) == 1 - held


async def test_waiter_gets_the_released_slot_in_fifo_order():
    pool = SlotPool(total=1, queue_depth=2, queue_timeout_s=1.0)
    await pool.acquire(preferred=None)
    first = asyncio.create_task(pool.acquire(preferred=None))
    second = asyncio.create_task(pool.acquire(preferred=None))
    await asyncio.sleep(0)
    assert pool.stats() == {"total": 1, "busy": 1, "queued": 2}
    pool.release(0)
    assert await first == 0
    assert not second.done()
    pool.release(0)
    assert await second == 0


async def test_full_queue_refuses_at_once():
    pool = SlotPool(total=1, queue_depth=1, queue_timeout_s=1.0)
    await pool.acquire(preferred=None)
    waiter = asyncio.create_task(pool.acquire(preferred=None))
    await asyncio.sleep(0)
    with pytest.raises(Busy) as exc:
        await pool.acquire(preferred=None)
    assert exc.value.reason == "queue_full"
    waiter.cancel()


async def test_wait_past_timeout_is_refused_and_leaves_the_queue():
    pool = SlotPool(total=1, queue_depth=1, queue_timeout_s=0.01)
    await pool.acquire(preferred=None)
    with pytest.raises(Busy) as exc:
        await pool.acquire(preferred=None)
    assert exc.value.reason == "queue_timeout"
    assert pool.stats()["queued"] == 0


async def test_cancelled_waiter_does_not_leak_a_handed_over_slot():
    pool = SlotPool(total=1, queue_depth=1, queue_timeout_s=1.0)
    await pool.acquire(preferred=None)
    waiter = asyncio.create_task(pool.acquire(preferred=None))
    await asyncio.sleep(0)
    pool.release(0)
    waiter.cancel()
    await asyncio.gather(waiter, return_exceptions=True)
    if not waiter.cancelled():
        pool.release(waiter.result())
    assert pool.stats()["busy"] == 0
    assert await pool.acquire(preferred=None) == 0
