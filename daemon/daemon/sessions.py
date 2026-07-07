import asyncio
import contextlib
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass
class Session:
    instruction: str
    context: str
    max_tokens: int
    last_text: str = ""
    tokens: list[int] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)


class SessionStore:
    def __init__(self, cap: int):
        self.cap = cap
        self._sessions: OrderedDict[str, Session] = OrderedDict()

    def create(
        self, instruction: str, context: str, max_tokens: int
    ) -> tuple[str, Session]:
        session_id = uuid.uuid4().hex
        session = Session(
            instruction=instruction, context=context, max_tokens=max_tokens
        )
        self._sessions[session_id] = session
        self._evict_if_needed()
        return session_id, session

    def get(self, session_id: str) -> Session | None:
        session = self._sessions.get(session_id)
        if session is not None:
            self._sessions.move_to_end(session_id)
        return session

    def update(
        self,
        session_id: str,
        *,
        text: str,
        tokens: list[int],
        context: str | None = None,
        instruction: str | None = None,
    ) -> None:
        session = self._sessions[session_id]
        session.last_text = text
        session.tokens = tokens
        session.updated_at = time.time()
        if context is not None:
            session.context = context
        if instruction is not None:
            session.instruction = instruction
        self._sessions.move_to_end(session_id)

    def delete(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def __len__(self) -> int:
        return len(self._sessions)

    def _evict_if_needed(self) -> None:
        while len(self._sessions) > self.cap:
            self._sessions.popitem(last=False)


class DropStaleRunner:
    """Keeps at most one in-flight generation task per session.

    A refresh arriving while a prior one is still running for the same session
    cancels it first, so the daemon never streams output for a context the
    client has already moved past (the live-pane UX contract).
    """

    def __init__(self):
        self._tasks: dict[str, asyncio.Task] = {}

    async def replace(self, session_id: str, coro) -> asyncio.Task:
        await self.cancel(session_id)
        task = asyncio.create_task(coro)
        self._tasks[session_id] = task
        return task

    async def cancel(self, session_id: str) -> None:
        task = self._tasks.pop(session_id, None)
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
