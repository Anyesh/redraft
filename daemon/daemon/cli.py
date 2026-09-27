import argparse
from dataclasses import replace

import uvicorn

from daemon.app import create_app
from daemon.config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(prog="redraftd")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--redraft-base")
    parser.add_argument("--redraft-model")
    parser.add_argument("--session-cap", type=int)
    parser.add_argument("--tenant-session-cap", type=int)
    parser.add_argument("--slots", type=int, help="must equal the engine's --parallel")
    parser.add_argument("--queue-depth", type=int)
    parser.add_argument("--queue-timeout-ms", type=int)
    parser.add_argument("--tokens-file")
    parser.add_argument(
        "--no-auth",
        action="store_true",
        default=None,
        help="disable bearer auth; refused unless bound to loopback",
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    overrides = {k: v for k, v in vars(args).items() if v is not None}
    if overrides:
        settings = replace(settings, **overrides)

    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
