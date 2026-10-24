"""Command line: ``llm-gateway serve`` and ``llm-gateway create-key``."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from llm_gateway import __version__
from llm_gateway.config import Settings


def cmd_serve(args: argparse.Namespace, settings: Settings) -> int:
    import uvicorn

    uvicorn.run(
        "llm_gateway.asgi:app",
        host=args.host or settings.host,
        port=args.port or settings.port,
        log_level=settings.log_level.lower(),
        log_config=None,  # keep uvicorn on the gateway's JSON handler
        proxy_headers=True,
    )
    return 0


def cmd_create_key(args: argparse.Namespace, settings: Settings) -> int:
    from llm_gateway.db import Database
    from llm_gateway.keys import KeyCreate, KeyService

    async def run() -> dict:
        db = Database(settings.database_url)
        await db.create_all()
        try:
            created = await KeyService(db).create(
                KeyCreate(
                    name=args.name,
                    budget_usd=args.budget,
                    budget_period=args.period,
                    rpm_limit=args.rpm,
                    tpm_limit=args.tpm,
                    allowed_models=args.models or [],
                )
            )
            return created.model_dump(mode="json")
        finally:
            await db.dispose()

    print(json.dumps(asyncio.run(run()), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-gateway", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the HTTP server")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.set_defaults(func=cmd_serve)

    key = sub.add_parser("create-key", help="create a virtual API key directly in the database")
    key.add_argument("--name", required=True)
    key.add_argument("--budget", type=float, default=None, help="budget in USD")
    key.add_argument("--period", choices=("total", "monthly"), default="total")
    key.add_argument("--rpm", type=int, default=None)
    key.add_argument("--tpm", type=int, default=None)
    key.add_argument("--models", nargs="*", help="allowed model aliases (globs)")
    key.set_defaults(func=cmd_create_key)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args, Settings()))


if __name__ == "__main__":
    sys.exit(main())
