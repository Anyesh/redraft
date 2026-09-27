import asyncio

from daemon.jobs import track


async def test_a_job_cancelled_before_it_starts_still_releases_and_ends():
    queue: asyncio.Queue = asyncio.Queue()
    released = []

    async def body():
        await queue.put(("open", {}))

    task = asyncio.create_task(body())
    track(task, queue, lambda: released.append(True))
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)
    assert released == [True]
    assert await queue.get() == ("cancelled", {"reason": "superseded"})
    assert await queue.get() is None


async def test_a_finished_job_releases_and_ends_without_extra_events():
    queue: asyncio.Queue = asyncio.Queue()
    released = []

    async def body():
        await queue.put(("done", {}))

    task = asyncio.create_task(body())
    track(task, queue, lambda: released.append(True))
    await task
    await asyncio.sleep(0)
    assert released == [True]
    assert await queue.get() == ("done", {})
    assert await queue.get() is None


async def test_an_unexpected_failure_becomes_an_error_event():
    queue: asyncio.Queue = asyncio.Queue()

    async def body():
        raise RuntimeError("boom")

    task = asyncio.create_task(body())
    track(task, queue, lambda: None)
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)
    name, data = await queue.get()
    assert (name, data["error"]) == ("error", "internal")
    assert await queue.get() is None
