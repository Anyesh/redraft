import asyncio
import logging
from collections.abc import Callable

log = logging.getLogger("redraftd")


def track(task: asyncio.Task, queue: asyncio.Queue, release: Callable[[], None]) -> None:
    """Guarantee a refresh job frees its slot and ends its stream however it stops.

    This lives in a done-callback, not in the job's own finally, because a task
    cancelled before its first step never runs its body at all.
    """

    def finish(t: asyncio.Task) -> None:
        release()
        if t.cancelled():
            queue.put_nowait(("cancelled", {"reason": "superseded"}))
        elif t.exception() is not None:
            log.error("refresh job failed", exc_info=t.exception())
            queue.put_nowait(("error", {"error": "internal"}))
        queue.put_nowait(None)

    task.add_done_callback(finish)
