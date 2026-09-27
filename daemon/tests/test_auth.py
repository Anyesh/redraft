import hashlib

import pytest

from daemon.auth import AuthError, TokenTable


def line(token: str, scope: str) -> str:
    return f"sha256:{hashlib.sha256(token.encode()).hexdigest()} {scope}"


@pytest.fixture
def table(tmp_path):
    path = tmp_path / "tokens"
    path.write_text(
        "# loomd service token\n"
        + line("svc", "*")
        + "\n\n"
        + line("acme-only", "acme")
        + "\n"
    )
    return TokenTable.from_file(path)


def test_wildcard_token_reaches_any_tenant(table):
    table.authorize("Bearer svc", "acme")
    table.authorize("Bearer svc", "globex")


def test_scoped_token_reaches_only_its_tenant(table):
    table.authorize("Bearer acme-only", "acme")
    with pytest.raises(AuthError) as exc:
        table.authorize("Bearer acme-only", "globex")
    assert exc.value.status == 403


@pytest.mark.parametrize("header", [None, "", "Bearer", "Bearer nope", "Basic svc"])
def test_missing_or_unknown_token_is_unauthorized(table, header):
    with pytest.raises(AuthError) as exc:
        table.authorize(header, "acme")
    assert exc.value.status == 401


def test_malformed_line_fails_loudly(tmp_path):
    path = tmp_path / "tokens"
    path.write_text("plaintext-token acme\n")
    with pytest.raises(ValueError):
        TokenTable.from_file(path)


def test_open_table_allows_everything():
    TokenTable.open().authorize(None, "anyone")
