from redraft_client import RedraftClient


async def probe_redraft_support(client: RedraftClient) -> bool:
    """Best-effort check for whether the backing server understands
    redraft_stabilize. An unpatched server ignores the unknown field and
    returns a normal completion with no redraft_emitted on the final chunk;
    any transport failure is treated the same as "unsupported" so the daemon
    degrades to baseline instead of failing to start.
    """
    try:
        eos = await client.eos_ids()
        events = [e async for e in client.stream_redraft([0], [0], eos, 1)]
    except Exception:
        return False
    final = events[-1] if events else {}
    return "redraft_emitted" in final
