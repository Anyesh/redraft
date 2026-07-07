import os
from dataclasses import dataclass

DEFAULT_REDRAFT_BASE = "http://127.0.0.1:8080"
DEFAULT_REDRAFT_MODEL = "default"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787
DEFAULT_SESSION_CAP = 256
DEFAULT_MAX_TOKENS = 512


@dataclass(frozen=True)
class Settings:
    redraft_base: str = DEFAULT_REDRAFT_BASE
    redraft_model: str = DEFAULT_REDRAFT_MODEL
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    session_cap: int = DEFAULT_SESSION_CAP
    default_max_tokens: int = DEFAULT_MAX_TOKENS

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            redraft_base=os.environ.get("REDRAFT_BASE", DEFAULT_REDRAFT_BASE),
            redraft_model=os.environ.get("REDRAFT_MODEL", DEFAULT_REDRAFT_MODEL),
            host=os.environ.get("REDRAFTD_HOST", DEFAULT_HOST),
            port=int(os.environ.get("REDRAFTD_PORT", DEFAULT_PORT)),
            session_cap=int(
                os.environ.get("REDRAFTD_SESSION_CAP", DEFAULT_SESSION_CAP)
            ),
            default_max_tokens=int(
                os.environ.get("REDRAFTD_DEFAULT_MAX_TOKENS", DEFAULT_MAX_TOKENS)
            ),
        )
