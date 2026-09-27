import asyncio
import contextlib
from collections import deque


class Busy(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class SlotPool:
    """Hands out engine slot ids so at most `total` generations run at once.

    A caller names the slot that served it last; that slot is used when free so
    the engine's per-slot prompt cache is reused, otherwise any free slot is.
    With none free, callers wait FIFO in a queue bounded by depth and by time,
    and are refused with Busy past either bound.
    """

    def __init__(self, total: int, queue_depth: int, queue_timeout_s: float):
        self.total = total
        self.queue_depth = queue_depth
        self.queue_timeout_s = queue_timeout_s
        self._free: list[int] = list(range(total))
        self._waiters: deque[asyncio.Future[int]] = deque()

    async def acquire(self, preferred: int | None) -> int:
        if self._free:
            slot = preferred if preferred in self._free else self._free[0]
            self._free.remove(slot)
            return slot
        if len(self._waiters) >= self.queue_depth:
            raise Busy("queue_full")
        fut: asyncio.Future[int] = asyncio.get_running_loop().create_future()
        self._waiters.append(fut)
        try:
            return await asyncio.wait_for(asyncio.shield(fut), self.queue_timeout_s)
        except TimeoutError:
            self._abandon(fut)
            raise Busy("queue_timeout") from None
        except asyncio.CancelledError:
            self._abandon(fut)
            raise

    def release(self, slot: int) -> None:
        while self._waiters:
            fut = self._waiters.popleft()
            if not fut.done():
                fut.set_result(slot)
                return
        self._free.append(slot)

    def stats(self) -> dict:
        return {
            "total": self.total,
            "busy": self.total - len(self._free),
            "queued": len(self._waiters),
        }

    def _abandon(self, fut: asyncio.Future[int]) -> None:
        # A slot handed over in the same tick the waiter gave up must go back to
        # the pool, or it is lost for the life of the daemon.
        with contextlib.suppress(ValueError):
            self._waiters.remove(fut)
        if fut.done() and not fut.cancelled():
            self.release(fut.result())
        else:
            fut.cancel()
