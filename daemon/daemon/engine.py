from dataclasses import dataclass, field
from pathlib import PurePath

import httpx
from redraft_client import RedraftClient

from daemon.probe import probe_redraft_support


@dataclass
class EngineState:
    """What redraftd knows about the llama-server behind it.

    Refreshed on startup, on every /healthz and whenever a refresh finds the
    engine not ready, so a daemon started before its engine (or across a model
    unload) recovers without a restart. The redraft probe runs a one-token
    completion, so it runs once per daemon and only once the engine is up.
    """

    reachable: bool = False
    resident: bool = False
    redraft: bool | None = None
    total_slots: int | None = None
    n_ctx: int | None = None
    model_name: str | None = None
    eos: set[int] = field(default_factory=set)

    @property
    def ready(self) -> bool:
        return self.resident and self.n_ctx is not None

    async def refresh(self, client: RedraftClient) -> None:
        try:
            health = await client.http.get("/health")
        except httpx.HTTPError:
            self.reachable = self.resident = False
            return
        self.reachable = True
        self.resident = health.status_code == 200
        if not self.resident:
            return
        try:
            props = await client.props()
            self.total_slots = props.get("total_slots")
            self.n_ctx = props.get("default_generation_settings", {}).get("n_ctx")
            model_path = props.get("model_path") or props.get("model_alias") or ""
            self.model_name = PurePath(model_path).stem or None
            if not self.eos:
                self.eos = await client.eos_ids()
            if self.redraft is None:
                self.redraft = await probe_redraft_support(client)
        except httpx.HTTPError:
            self.resident = False
