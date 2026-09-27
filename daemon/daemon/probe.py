from redraft_client import RedraftClient


async def probe_redraft_support(client: RedraftClient) -> bool:
    """Whether the engine understands redraft_stabilize. An unpatched server
    ignores the unknown field and returns a normal completion with no
    redraft_emitted on the final chunk. Transport errors propagate so the
    caller can retry the probe later instead of pinning "unsupported".
    """
    eos = await client.eos_ids()
    events = [e async for e in client.stream_redraft([0], [0], eos, 1)]
    final = events[-1] if events else {}
    return "redraft_emitted" in final
