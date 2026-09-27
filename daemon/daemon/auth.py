import hashlib
import hmac
from pathlib import Path


class AuthError(Exception):
    def __init__(self, status: int, code: str):
        super().__init__(code)
        self.status = status
        self.code = code


class TokenTable:
    """Bearer tokens stored as sha256 digests, each scoped to one tenant or `*`.

    The file holds one `sha256:<hex> <scope>` per line; `#` starts a comment.
    Plaintext tokens never touch disk on the daemon side.
    """

    def __init__(self, scopes: dict[str, str] | None):
        self._scopes = scopes

    @classmethod
    def open(cls) -> "TokenTable":
        return cls(None)

    @classmethod
    def from_file(cls, path: str | Path) -> "TokenTable":
        scopes: dict[str, str] = {}
        for n, raw in enumerate(Path(path).read_text().splitlines(), start=1):
            text = raw.split("#", 1)[0].strip()
            if not text:
                continue
            parts = text.split()
            if len(parts) != 2 or not parts[0].startswith("sha256:"):
                raise ValueError(f"{path}:{n}: expected 'sha256:<hex> <scope>'")
            scopes[parts[0].removeprefix("sha256:").lower()] = parts[1]
        return cls(scopes)

    @property
    def enabled(self) -> bool:
        return self._scopes is not None

    def authorize(self, header: str | None, tenant: str | None) -> None:
        if self._scopes is None:
            return
        scheme, _, token = (header or "").partition(" ")
        if scheme != "Bearer" or not token:
            raise AuthError(401, "unauthorized")
        digest = hashlib.sha256(token.strip().encode()).hexdigest()
        scope = None
        for known, known_scope in self._scopes.items():
            if hmac.compare_digest(known, digest):
                scope = known_scope
        if scope is None:
            raise AuthError(401, "unauthorized")
        if tenant is not None and scope != "*" and scope != tenant:
            raise AuthError(403, "forbidden_tenant")
