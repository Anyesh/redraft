import os
from dataclasses import dataclass

DEFAULT_REDRAFT_BASE = "http://127.0.0.1:8080"
DEFAULT_REDRAFT_MODEL = "default"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787
DEFAULT_SESSION_CAP = 256
DEFAULT_TENANT_SESSION_CAP = 64
DEFAULT_MAX_TOKENS = 512
DEFAULT_SLOTS = 1
DEFAULT_QUEUE_DEPTH = 8
DEFAULT_QUEUE_TIMEOUT_MS = 5000


@dataclass(frozen=True)
class Settings:
    redraft_base: str = DEFAULT_REDRAFT_BASE
    redraft_model: str = DEFAULT_REDRAFT_MODEL
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    session_cap: int = DEFAULT_SESSION_CAP
    tenant_session_cap: int = DEFAULT_TENANT_SESSION_CAP
    default_max_tokens: int = DEFAULT_MAX_TOKENS
    slots: int = DEFAULT_SLOTS
    queue_depth: int = DEFAULT_QUEUE_DEPTH
    queue_timeout_ms: int = DEFAULT_QUEUE_TIMEOUT_MS
    tokens_file: str | None = None
    no_auth: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ.get
        return cls(
            redraft_base=env("REDRAFT_BASE", DEFAULT_REDRAFT_BASE),
            redraft_model=env("REDRAFT_MODEL", DEFAULT_REDRAFT_MODEL),
            host=env("REDRAFTD_HOST", DEFAULT_HOST),
            port=int(env("REDRAFTD_PORT", DEFAULT_PORT)),
            session_cap=int(env("REDRAFTD_SESSION_CAP", DEFAULT_SESSION_CAP)),
            tenant_session_cap=int(
                env("REDRAFTD_TENANT_SESSION_CAP", DEFAULT_TENANT_SESSION_CAP)
            ),
            default_max_tokens=int(
                env("REDRAFTD_DEFAULT_MAX_TOKENS", DEFAULT_MAX_TOKENS)
            ),
            slots=int(env("REDRAFTD_SLOTS", DEFAULT_SLOTS)),
            queue_depth=int(env("REDRAFTD_QUEUE_DEPTH", DEFAULT_QUEUE_DEPTH)),
            queue_timeout_ms=int(
                env("REDRAFTD_QUEUE_TIMEOUT_MS", DEFAULT_QUEUE_TIMEOUT_MS)
            ),
            tokens_file=env("REDRAFTD_TOKENS_FILE"),
            no_auth=env("REDRAFTD_NO_AUTH", "") == "1",
        )
