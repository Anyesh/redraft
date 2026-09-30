import asyncio

from daemon.slots import SlotPool


def pool(total: int = 2, queue_depth: int = 1) -> SlotPool:
    return SlotPool(total, queue_depth, queue_timeout_s=1.0)


async def test_claim_takes_exactly_the_named_free_slot():
    slots = pool()
    assert await slots.claim(1, timeout_s=0.1)
    assert slots.stats()["busy"] == 1
    assert await slots.acquire(None) == 0
    slots.release(1)
    assert await slots.acquire(1) == 1


async def test_claim_waits_for_the_slot_and_beats_queued_refreshes():
    slots = pool(total=1, queue_depth=2)
    held = await slots.acquire(None)
    queued = asyncio.create_task(slots.acquire(None))
    await asyncio.sleep(0)
    claim = asyncio.create_task(slots.claim(held, timeout_s=1.0))
    await asyncio.sleep(0)
    slots.release(held)
    assert await claim is True
    assert not queued.done()
    slots.release(held)
    assert await queued == held


async def test_claim_timeout_returns_false_and_leaks_nothing():
    slots = pool(total=1)
    held = await slots.acquire(None)
    assert await slots.claim(held, timeout_s=0.05) is False
    slots.release(held)
    assert slots.stats() == {"total": 1, "busy": 0, "queued": 0}
