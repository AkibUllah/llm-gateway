#!/usr/bin/env bash
# Codespaces start hook: gateway + PostgreSQL + Redis via compose, then demo traffic for the dashboard.
set -euo pipefail
cd "$(dirname "$0")/.."

export ADMIN_API_KEY="${ADMIN_API_KEY:-change-me-admin}"
export DEMO_API_KEY="${DEMO_API_KEY:-sk-gw-demo-local-key-0001}"

docker compose up -d --build

echo "waiting for the gateway on :8080 ..."
for _ in $(seq 1 90); do
  if curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

python scripts/demo_traffic.py --url http://127.0.0.1:8080 \
  --key "$DEMO_API_KEY" --admin-key "$ADMIN_API_KEY" --requests 80 >/dev/null || true

cat <<MSG

LLM Gateway is up (PostgreSQL + Redis, fake providers; add OPENAI_API_KEY / ANTHROPIC_API_KEY
as Codespaces secrets to route to real models).
  Dashboard:  forwarded port 8080, path /dashboard   (admin key: $ADMIN_API_KEY)
  API docs:   forwarded port 8080, path /docs
  Try:        curl http://127.0.0.1:8080/v1/chat/completions -H "Authorization: Bearer $DEMO_API_KEY" \\
                -H 'content-type: application/json' \\
                -d '{"model":"demo-chat","temperature":0,"messages":[{"role":"user","content":"ping"}]}'
MSG
