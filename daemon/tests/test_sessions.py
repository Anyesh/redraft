import asyncio

import pytest

from daemon.sessions import (
    BadSessionId,
    DropStaleRunner,
    Session,
    SessionStore,
    Source,
    parse_session_id,
)


def session(sid: str) -> Session:
    return Session(
        session_id=sid,
        instruction="summarize",
        sources=[Source(name="notes", text="n")],
        derived="",
        pinned=[],
        max_tokens=64,
    )


def test_session_id_has_three_segments():
    assert parse_session_id("acme/doc-1/intro") == ("acme", "doc-1", "intro")


@pytest.mark.parametrize(
    "sid", ["acme/doc", "acme/doc/a/b", "acme//s", "acme/d/s p", "a/" + "x" * 129 + "/s"]
)
def test_bad_session_ids_are_rejected(sid):
    with pytest.raises(BadSessionId):
        parse_session_id(sid)


def test_put_get_delete_round_trip():
    store = SessionStore(cap=4, tenant_cap=4)
    store.put(session("acme/d/s"))
    assert store.get("acme/d/s").instruction == "summarize"
    assert store.delete("acme/d/s")
    assert store.get("acme/d/s") is None
    assert not store.delete("acme/d/s")


def test_tenant_cap_evicts_that_tenants_oldest_section_only():
    store = SessionStore(cap=10, tenant_cap=2)
    store.put(session("globex/d/s"))
    store.put(session("acme/d/a"))
    store.put(session("acme/d/b"))
    store.put(session("acme/d/c"))
    assert store.get("acme/d/a") is None
    assert store.get("acme/d/b") is not None
    assert store.get("globex/d/s") is not None


def test_global_cap_evicts_least_recently_used():
    store = SessionStore(cap=2, tenant_cap=2)
    store.put(session("acme/d/a"))
    store.put(session("globex/d/b"))
    store.get("acme/d/a")
    store.put(session("initech/d/c"))
    assert store.get("globex/d/b") is None
    assert store.get("acme/d/a") is not None
    assert len(store) == 2


def test_put_replaces_an_existing_section_without_evicting():
    store = SessionStore(cap=1, tenant_cap=1)
    store.put(session("acme/d/a"))
    store.put(session("acme/d/a"))
    assert len(store) == 1


async def test_drop_stale_runner_replace_cancels_previous_task():
    runner = DropStaleRunner()
    ran_to_completion = False

    async def slow():
        nonlocal ran_to_completion
        await asyncio.sleep(10)
        ran_to_completion = True

    async def fast():
        pass

    await runner.replace("s1", slow())
    await runner.replace("s1", fast())
    await asyncio.sleep(0.01)

    assert ran_to_completion is False


async def test_drop_stale_runner_cancel_on_unknown_session_is_a_no_op():
    runner = DropStaleRunner()
    await runner.cancel("does-not-exist")


async def test_drop_stale_runner_lets_completed_task_finish_untouched():
    runner = DropStaleRunner()
    result = []

    async def quick():
        result.append(1)

    task = await runner.replace("s1", quick())
    await task
    await runner.cancel("s1")
    assert result == [1]
