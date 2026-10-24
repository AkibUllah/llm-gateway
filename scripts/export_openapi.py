"""Export the OpenAPI document without a database, Redis or provider keys.

    python scripts/export_openapi.py [output path]      # default: openapi.json

The GitHub Pages workflow uses this to publish the static API reference.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from llm_gateway.app import create_app
from llm_gateway.config import Settings


def main(argv: list[str]) -> int:
    out = Path(argv[1]) if len(argv) > 1 else Path("openapi.json")
    settings = Settings(
        _env_file=None, database_url="sqlite+aiosqlite:///:memory:", log_level="WARNING"
    )
    spec = create_app(settings, configure_logs=False).openapi()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(spec['paths'])} paths, openapi {spec['openapi']})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
