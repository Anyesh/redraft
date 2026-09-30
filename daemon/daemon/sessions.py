import asyncio
import contextlib
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field

SEGMENT = re.compile(r"[A-Za-z0-9._~:@+-]{1,128}")


class BadSessionId(ValueError):
    pass


def parse_session_id(session_id: str) -> tuple[str, str, str]:
    parts = session_id.split("/")
    if len(parts) != 3 or not all(SEGMENT.fullmatch(p) for p in parts):
        raise BadSessionId(session_id)
    return parts[0], parts[1], parts[2]


def parse_prefix(prefix: str) -> tuple[str, ...]:
    """A prefix names a tenant (`t/`) or a document (`t/d/`); the trailing slash
    is required so `t/doc-1/` can never match `t/doc-10/`."""
    parts = prefix.split("/")
    if (
        len(parts) not in (2, 3)
        or parts[-1] != ""
        or not all(SEGMENT.fullmatch(p) for p in parts[:-1])
    ):
        raise BadSessionId(prefix)
    return tuple(parts[:-1])


@dataclass
class Source:
    name: str
    text: str


@dataclass
class Draft:
    """Token ids of a derived text, with the UTF-16 end offset after each token,
    so held spans can be mapped back onto the text the person sees."""

    text: str
    ids: list[int]
    ends: list[int]


@dataclass
class Session:
    session_id: str
    instruction: str
    sources: list[Source]
    derived: str
    pinned: list[str]
    max_tokens: int
    revision: int = 1
    derived_dirty: bool = False
    draft: Draft | None = None
    slot: int | None = None
    updated_at: float = field(default_factory=time.time)

    @property
    def tenant(self) -> str:
        return self.session_id.split("/", 1)[0]

    def touch(self) -> None:
        self.revision += 1
        self.updated_at = time.time()


class SessionStore:
    """LRU store of sections. A tenant at its cap evicts its own oldest section,
    so one tenant cannot push out another's until the global cap binds."""

    def __init__(self, cap: int, tenant_cap: int):
        self.cap = cap
        self.tenant_cap = tenant_cap
        self._sessions: OrderedDict[str, Session] = OrderedDict()

    def put(self, session: Session) -> None:
        self._sessions[session.session_id] = session
        self._sessions.move_to_end(session.session_id)
        self._evict(session.tenant)

    def get(self, session_id: str) -> Session | None:
        session = self._sessions.get(session_id)
        if session is not None:
            self._sessions.move_to_end(session_id)
        return session

    def delete(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def ids_with_prefix(self, prefix: str) -> list[str]:
        return [sid for sid in self._sessions if sid.startswith(prefix)]

    def __contains__(self, session_id: str) -> bool:
        return session_id in self._sessions

    def __len__(self) -> int:
        return len(self._sessions)

    def _evict(self, tenant: str) -> None:
        owned = [sid for sid, s in self._sessions.items() if s.tenant == tenant]
        for sid in owned[: max(0, len(owned) - self.tenant_cap)]:
            del self._sessions[sid]
        while len(self._sessions) > self.cap:
            self._sessions.popitem(last=False)


class DropStaleRunner:
    """Keeps at most one in-flight generation task per session.

    A refresh arriving while a prior one is still running for the same session
    cancels it first, so the daemon never streams output for a context the
    client has already moved past.
    """

    def __init__(self):
        self._tasks: dict[str, asyncio.Task] = {}

    def start(self, session_id: str, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._tasks[session_id] = task
        task.add_done_callback(lambda t: self._forget(session_id, t))
        return task

    async def replace(self, session_id: str, coro) -> asyncio.Task:
        await self.cancel(session_id)
        return self.start(session_id, coro)

    async def cancel(self, session_id: str) -> None:
        task = self._tasks.pop(session_id, None)
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def _forget(self, session_id: str, task: asyncio.Task) -> None:
        if self._tasks.get(session_id) is task:
            del self._tasks[session_id]
