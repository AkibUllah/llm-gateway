"""Send a realistic mix of traffic through a running gateway so the dashboard has data.

    python scripts/demo_traffic.py --url http://localhost:8080 \
        --key sk-gw-demo-local-key-0001 --admin-key change-me-admin --requests 60

With --admin-key it also creates two extra virtual keys ("mobile-app" with a small budget that
crosses its alert thresholds, and "analytics-batch" limited to embeddings).
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter

import httpx

PROMPTS = [
    "How long does a bKash settlement take?",
    "Summarise today's failed Nagad payments in one line.",
    "Draft an SMS telling a Mirpur customer their parcel is out for delivery.",
    "Translate 'your order has shipped' to Bangla.",
    "ping",
    "What is the refund policy for orders above 5,000 taka?",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--key", required=True, help="virtual key for chat traffic")
    parser.add_argument("--admin-key", default=None)
    parser.add_argument("--requests", type=int, default=60)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    outcomes: Counter[str] = Counter()
    with httpx.Client(base_url=args.url, timeout=30) as client:
        keys = [args.key]
        embed_key = args.key
        if args.admin_key:
            admin = {"Authorization": f"Bearer {args.admin_key}"}
            mobile = client.post(
                "/admin/keys",
                headers=admin,
                json={
                    "name": "mobile-app",
                    "budget_usd": 0.0004,
                    "rpm_limit": 60,
                    "allowed_models": ["demo-*", "claude-haiku"],
                    "metadata": {"team": "mobile"},
                },
            ).json()
            batch = client.post(
                "/admin/keys",
                headers=admin,
                json={
                    "name": "analytics-batch",
                    "allowed_models": ["demo-embed", "text-embedding-*"],
                    "tpm_limit": 50000,
                    "metadata": {"team": "data"},
                },
            ).json()
            keys.append(mobile["key"])
            embed_key = batch["key"]

        for i in range(args.requests):
            key = rng.choice(keys)
            headers = {"Authorization": f"Bearer {key}"}
            roll = rng.random()
            if roll < 0.15:
                response = client.post(
                    "/v1/embeddings",
                    headers={"Authorization": f"Bearer {embed_key}"},
                    json={"model": "demo-embed", "input": [rng.choice(PROMPTS)]},
                )
                label = "embeddings"
            else:
                model = "claude-haiku" if roll > 0.85 else "demo-chat"
                body = {
                    "model": model,
                    "temperature": 0 if rng.random() < 0.6 else 0.7,
                    "stream": rng.random() < 0.3,
                    "messages": [{"role": "user", "content": rng.choice(PROMPTS)}],
                }
                response = client.post("/v1/chat/completions", headers=headers, json=body)
                label = f"{model}{' stream' if body['stream'] else ''}"
            outcomes[f"{label} -> {response.status_code}"] += 1
            if i % 10 == 9:
                print(f"{i + 1} requests sent")

    print("\noutcomes:")
    for name, count in sorted(outcomes.items()):
        print(f"  {name:32} {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
