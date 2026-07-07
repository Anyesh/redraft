"""Live smoke test against a real patched llama-server, driven directly
through redraftd's ASGI app (no need for the daemon process to be running
separately). Needs REDRAFT_BASE pointed at a live server with the
redraft_stabilize patch applied; see engine/BUILD.md. Not run in CI or by
`pytest`, since it requires a real model and takes real wall-clock time.

Usage:
    REDRAFT_BASE=http://<host>:<port> uv run python smoke.py
"""

import asyncio
import json

import httpx

from daemon.app import create_app
from daemon.config import Settings

DOCUMENT = " ".join(
    f"Paragraph {i} discusses the migration of the accounts service to the "
    "new billing pipeline, covering rollout sequencing, on-call ownership, "
    "and the rollback plan should the new pipeline underperform in staging."
    for i in range(1, 11)
)
COSMETIC_EDIT = DOCUMENT.replace("Paragraph 5", "paragraph 5", 1)
FULL_REWRITE = (
    "The quarterly earnings call is scheduled for next Tuesday and will "
    "cover revenue growth, margin pressure from the new datacenter build-out, "
    "and guidance for the coming fiscal year."
)


def parse_sse(content: bytes) -> list[dict]:
    events = []
    for line in content.decode().splitlines():
        if line.startswith("data:"):
            events.append(json.loads(line[len("data:") :].strip()))
    return events


async def main() -> None:
    settings = Settings.from_env()
    app = create_app(settings)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://smoke"
        ) as client:
            health = await client.get("/v1/health")
            print("health:", health.json())

            seed = await client.post(
                "/v1/sessions",
                json={
                    "instruction": "Summarize this document.",
                    "context": DOCUMENT,
                    "max_tokens": 512,
                },
            )
            seed_final = parse_sse(seed.content)[-1]
            session_id = seed_final["session_id"]
            seed_wall_s = seed_final["wall_s"]
            print(f"seed: wall_s={seed_wall_s:.3f}")

            cosmetic = await client.post(
                f"/v1/sessions/{session_id}/refresh",
                json={"context": COSMETIC_EDIT},
            )
            cosmetic_final = parse_sse(cosmetic.content)[-1]
            print(
                f"cosmetic edit: mode={cosmetic_final['mode']} "
                f"held_fraction={cosmetic_final.get('held_fraction')} "
                f"wall_s={cosmetic_final['wall_s']:.3f}"
            )
            assert cosmetic_final["mode"] == "redraft", (
                "server did not run redraft mode"
            )
            assert cosmetic_final["held_fraction"] > 0.5, (
                "expected a high held fraction"
            )
            assert cosmetic_final["wall_s"] < seed_wall_s, (
                "redraft should be faster than the seed"
            )

            rewrite = await client.post(
                f"/v1/sessions/{session_id}/refresh",
                json={"context": FULL_REWRITE},
            )
            rewrite_final = parse_sse(rewrite.content)[-1]
            print(
                f"full rewrite: mode={rewrite_final['mode']} "
                f"held_fraction={rewrite_final.get('held_fraction')} "
                f"wall_s={rewrite_final['wall_s']:.3f}"
            )
            assert rewrite_final["held_fraction"] < 0.5, (
                "rewrite should collapse the held fraction"
            )

    print("smoke test passed")


if __name__ == "__main__":
    asyncio.run(main())
