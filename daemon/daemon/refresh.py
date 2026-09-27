import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from redraft_client import RedraftClient

from daemon.engine import EngineState
from daemon.prompts import Kind, render
from daemon.sessions import Draft, Session
from daemon.spans import Event, SpanBuilder, piece_ends, utf16_len


@dataclass
class Prepared:
    kind: Kind
    prompt_ids: list[int]
    draft: Draft | None
    use_redraft: bool


async def ensure_draft(client: RedraftClient, session: Session) -> Draft | None:
    if not session.derived:
        return None
    if session.draft is None or session.draft.text != session.derived:
        ids, pieces = await client.tokenize_with_pieces(session.derived)
        session.draft = Draft(session.derived, ids, piece_ends(pieces))
    return session.draft


async def prepare(
    client: RedraftClient, engine: EngineState, session: Session, force_baseline: bool
) -> Prepared:
    kind: Kind = "revise" if session.derived_dirty else "rederive"
    context, question = render(
        kind, session.instruction, session.sources, session.derived, session.pinned
    )
    prompt_ids = await client.tokenize(await client.apply_template(context, question))
    draft = await ensure_draft(client, session)
    use_redraft = bool(engine.redraft and draft and draft.ids and not force_baseline)
    return Prepared(kind, prompt_ids, draft, use_redraft)


def missing_pinned(text: str, pinned: list[str]) -> list[str]:
    lines = {line.rstrip() for line in text.split("\n")}
    return [p for p in pinned if p.rstrip() not in lines]


class Generation:
    """One refresh against the engine: streams delta and span events, then
    exposes the new text, its draft and the numbers for the done event."""

    def __init__(
        self,
        client: RedraftClient,
        engine: EngineState,
        session: Session,
        prepared: Prepared,
        slot: int,
    ):
        self.client = client
        self.engine = engine
        self.session = session
        self.prepared = prepared
        self.slot = slot
        self.text = ""
        self.draft: Draft | None = None
        self.metrics: dict = {}

    async def run(self) -> AsyncIterator[Event]:
        prep, session = self.prepared, self.session
        builder = SpanBuilder(session.derived, prep.draft.ends if prep.draft else [])
        started = time.perf_counter()
        if prep.use_redraft:
            stream = self.client.stream_redraft(
                prep.prompt_ids,
                prep.draft.ids,
                self.engine.eos,
                session.max_tokens,
                id_slot=self.slot,
            )
        else:
            stream = self.client.stream_baseline(
                prep.prompt_ids, session.max_tokens, id_slot=self.slot
            )
        final: dict = {}
        async for chunk in stream:
            final = chunk
            srcs = chunk.get("redraft_src") if prep.use_redraft else None
            for event in builder.feed(chunk.get("content", ""), srcs):
                yield event
        for event in builder.finish():
            yield event
        wall_ms = (time.perf_counter() - started) * 1000

        self.text = builder.new_text
        if prep.use_redraft:
            self.draft = self._draft_from_stream(final, builder)
        else:
            self.draft = await self._draft_from_text()
        if self.draft is None:
            self.draft = await self._draft_from_text()
        self.metrics = {
            "mode": "redraft" if prep.use_redraft else "baseline",
            "reused": final.get("redraft_held_fraction", 0.0)
            if prep.use_redraft
            else 0.0,
            "reused_chars": builder.reused_chars,
            "wall_ms": round(wall_ms, 1),
            "prompt_ms": final.get("timings", {}).get("prompt_ms"),
            "tokens": len(self.draft.ids),
            "divergences": final.get("redraft_divergences", 0),
            "pinned_missing": missing_pinned(self.text, session.pinned),
        }

    def _draft_from_stream(self, final: dict, builder: SpanBuilder) -> Draft | None:
        # Tokens ending in an incomplete UTF-8 sequence at the very end are never
        # streamed, so the emitted list can run past the streamed offsets; they
        # carry no text and end where the text ends.
        ids = final.get("redraft_emitted", [])
        ends = builder.token_ends
        if len(ends) > len(ids):
            return None
        ends = ends + [utf16_len(self.text)] * (len(ids) - len(ends))
        return Draft(self.text, list(ids), ends)

    async def _draft_from_text(self) -> Draft:
        if not self.text:
            return Draft("", [], [])
        ids, pieces = await self.client.tokenize_with_pieces(self.text)
        return Draft(self.text, ids, piece_ends(pieces))
