import asyncio

from daemon.sessions import DropStaleRunner, SessionStore


def test_create_and_get_round_trips():
    store = SessionStore(cap=10)
    session_id, session = store.create("summarize", "doc text", 512)
    assert store.get(session_id) is session
    assert session.instruction == "summarize"
    assert session.context == "doc text"
    assert session.max_tokens == 512
    assert session.tokens == []
    assert session.last_text == ""


def test_get_unknown_session_returns_none():
    store = SessionStore(cap=10)
    assert store.get("nope") is None


def test_update_sets_fields_and_carries_tokens_forward():
    store = SessionStore(cap=10)
    session_id, _ = store.create("summarize", "doc v1", 512)
    store.update(session_id, text="answer v1", tokens=[1, 2, 3])
    session = store.get(session_id)
    assert session.last_text == "answer v1"
    assert session.tokens == [1, 2, 3]

    store.update(session_id, text="answer v2", tokens=[4, 5], context="doc v2")
    session = store.get(session_id)
    assert session.last_text == "answer v2"
    assert session.tokens == [4, 5]
    assert session.context == "doc v2"


def test_delete_removes_session():
    store = SessionStore(cap=10)
    session_id, _ = store.create("summarize", "doc", 512)
    assert store.delete(session_id) is True
    assert store.get(session_id) is None
    assert store.delete(session_id) is False


def test_lru_eviction_pops_oldest_untouched_session():
    store = SessionStore(cap=2)
    first, _ = store.create("a", "ctx-a", 512)
    second, _ = store.create("b", "ctx-b", 512)
    third, _ = store.create("c", "ctx-c", 512)

    assert len(store) == 2
    assert store.get(first) is None
    assert store.get(second) is not None
    assert store.get(third) is not None


def test_get_promotes_recency_and_saves_from_eviction():
    store = SessionStore(cap=2)
    first, _ = store.create("a", "ctx-a", 512)
    second, _ = store.create("b", "ctx-b", 512)

    store.get(first)
    store.create("c", "ctx-c", 512)

    assert store.get(first) is not None
    assert store.get(second) is None


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
