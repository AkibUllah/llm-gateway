<!-- topics: llm, llm-gateway, openai-api, anthropic, ollama, fastapi, python, rate-limiting, circuit-breaker, observability -->
# LLM Gateway

An OpenAI-compatible gateway in front of OpenAI, Anthropic and Ollama, with virtual API keys, per-key budgets and rate limits, weighted routing with retries, fallbacks and circuit breakers, response caching, and usage and cost reporting.

[![CI](https://github.com/AkibUllah/llm-gateway/actions/workflows/ci.yml/badge.svg)](https://github.com/AkibUllah/llm-gateway/actions/workflows/ci.yml)
[![CodeQL](https://github.com/AkibUllah/llm-gateway/actions/workflows/codeql.yml/badge.svg)](https://github.com/AkibUllah/llm-gateway/actions/workflows/codeql.yml)
[![Release](https://img.shields.io/github/v/release/AkibUllah/llm-gateway)](https://github.com/AkibUllah/llm-gateway/releases)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688)
![License](https://img.shields.io/badge/license-MIT-green)

## Try it live

| | |
|---|---|
| API reference | https://akibullah.github.io/llm-gateway/ — Swagger UI built from the exported OpenAPI document (static; no backend behind it) |
| Run in your browser | [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/AkibUllah/llm-gateway?quickstart=1) — starts the gateway with PostgreSQL and Redis via `docker compose`, sends 80 demo requests, then open forwarded port 8080 at `/dashboard` (admin key `change-me-admin`) or `/docs`. Virtual key for requests: `sk-gw-demo-local-key-0001`. |
| One-click deploy | [![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/AkibUllah/llm-gateway) — `render.yaml` provisions the gateway, a free PostgreSQL and a free Key Value store; `ADMIN_API_KEY` and a `DEMO_API_KEY` (with a $5 budget) are generated, add `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` to route to real models. |
| Local | `docker compose up` (see Quick Start) |

Everything runs offline: without provider keys, the `demo-*` aliases and every fallback chain end at deterministic fake providers.

## Screenshots

![Usage dashboard](docs/dashboard.png)

*The built-in dashboard at `/dashboard` after a run of `scripts/demo_traffic.py` (spend, requests per alias, keys, provider breaker state).*

![curl session](docs/images/curl-session.png)

*A real curl session against a local gateway on the offline config: a retried call on the flaky fake upstream, the cache hit that follows, a `claude-haiku` request served by its fallback chain, a `model_not_found` error and the resulting usage report.*

![API reference](docs/images/api-reference.png)

*The OpenAI-compatible and admin endpoints in the [live API reference](https://akibullah.github.io/llm-gateway/).*

## Why this exists

Once more than one team calls LLM APIs, the same problems show up as with payment gateways: shared secrets in every service, nobody knows who spent what, one provider's outage or 429 storm takes the product down, and switching vendors means touching every client. Having built payment routing with retries and settlement reports for bKash and card processors, I wanted the same discipline here: one OpenAI-shaped endpoint, keys that can be scoped and capped, routing that degrades gracefully, and numbers finance can read.

## Features

- OpenAI-compatible `POST /v1/chat/completions` (including SSE streaming and `stream_options.include_usage`), `POST /v1/embeddings` and `GET /v1/models`; existing OpenAI SDK clients work by changing `base_url`
- Provider adapters: OpenAI-compatible HTTP, Anthropic Messages (full request/response translation including tools, images, system prompts and stream events), Ollama native API (NDJSON streaming), and a deterministic `FakeProvider` with configurable latency and failures
- Model aliases with weighted load balancing, ordered fallback chains, retries with exponential backoff and jitter (`Retry-After` honoured), per-request timeout, and a circuit breaker per provider with half-open probes
- Virtual keys (`sk-gw-...`), stored as SHA-256 hashes: per-key budget (total or monthly), requests-per-minute and tokens-per-minute limits, model allow-list (globs), expiry, revocation
- Spend tracking from a YAML price table, budget alerts at 50/80/100% (stored, logged, optional webhook), `402 insufficient_quota` once the budget is gone
- Exact-match response cache keyed on a normalised request hash (whitespace, Unicode forms, optional case folding), memory or Redis, TTL, replayed as SSE for streaming clients
- Token counting with `tiktoken` when installed, otherwise a script-aware heuristic; provider-reported usage always wins for billing
- Admin API for keys, usage and cost reports (by key, model, provider or day), alerts, provider health and cache control, plus a small dashboard at `/dashboard`
- Structured JSON logs with request ids and PII redaction (emails, Bangladeshi and international phone numbers, Luhn-valid card numbers, API keys), Prometheus `/metrics`

## Architecture

```mermaid
flowchart LR
    C[Clients<br/>OpenAI SDKs, curl, services] -->|Bearer sk-gw-...| API[FastAPI<br/>/v1/chat/completions<br/>/v1/embeddings · /v1/models]
    API --> K[KeyService<br/>hash lookup · expiry<br/>allow-list · budget]
    API --> RL[KeyRateLimiter<br/>RPM / TPM windows]
    API --> CA{ResponseCache<br/>normalised hash}
    CA -- hit --> API
    CA -- miss --> R[Router<br/>aliases · weights<br/>retries · fallbacks]
    R --> B1[[breaker]] --> P1[OpenAIProvider]
    R --> B2[[breaker]] --> P2[AnthropicProvider<br/>Messages translation]
    R --> B3[[breaker]] --> P3[OllamaProvider]
    R --> B4[[breaker]] --> P4[FakeProvider]
    API --> U[(PostgreSQL / SQLite<br/>keys · usage · alerts)]
    RL --- RD[(Redis<br/>counters · cache)]
    CA --- RD
    API --> M[/metrics · JSON logs/]
    ADM[Admin API + dashboard] --> U
```

Key decisions:

- **OpenAI's schema is the canonical format.** Every adapter translates to and from it, so the routing, caching and accounting code never branches on provider. Unknown request fields are kept and passed through to OpenAI-compatible upstreams.
- **Retry the provider, then fall back; never retry a bad request.** 429/5xx/timeouts are retried on the same deployment with backoff; 401/403/404 skip straight to the next deployment (the provider is misconfigured, not the request); 400/422 are returned to the client immediately because they would fail everywhere.
- **Streams fall back only before the first byte.** The router awaits the first chunk inside the retry loop. After that the client is already receiving data, so a mid-stream failure is reported in-band as an OpenAI-style error event followed by `[DONE]`, and still recorded in usage.
- **Accounting never fails a request.** Usage rows, spend updates (a single atomic `UPDATE ... RETURNING`), budget alerts and metrics run after the response is built and are guarded; streamed responses are priced when the last chunk has been sent.
- **Rejected requests are recorded too.** Budget, rate-limit and allow-list rejections land in `usage_records` with their status and error code, so the dashboard's error rate tells the truth.

### Anthropic stream mapping

| Anthropic event | OpenAI chunk |
| --- | --- |
| `message_start` | first chunk, `delta.role = "assistant"`; input tokens remembered |
| `content_block_start` (`tool_use`) | `delta.tool_calls[i]` with `id`, `type`, `function.name` |
| `content_block_delta` `text_delta` | `delta.content` |
| `content_block_delta` `input_json_delta` | `delta.tool_calls[i].function.arguments` (partial JSON) |
| `message_delta` | `finish_reason` (`end_turn`/`stop_sequence` → `stop`, `max_tokens` → `length`, `tool_use` → `tool_calls`); output tokens |
| `message_stop` | final chunk with `usage`, no choices (forwarded only when `include_usage` is set) |
| `error` | `overloaded_error` / `rate_limit_error` are retryable before the first chunk; in-band error event after |

## Tech Stack

| Area | Choice |
| --- | --- |
| API | FastAPI, uvicorn, pydantic v2 |
| Upstream HTTP | httpx (no provider SDKs) |
| Storage | SQLAlchemy 2.0 async: PostgreSQL via asyncpg, SQLite via aiosqlite (WAL) |
| Shared state | Redis (optional): cache and rate-limit counters across replicas |
| Observability | prometheus-client, JSON logs with PII redaction |
| Dashboard | Static HTML + Chart.js served by the gateway |
| Tests | pytest, anyio, httpx `ASGITransport` and `MockTransport`, fakeredis |
| Tooling | ruff, Docker, docker compose, GitHub Actions, GitHub Pages, Render blueprint, Codespaces |

## Quick Start

### Docker compose (gateway + PostgreSQL + Redis)

```bash
git clone https://github.com/AkibUllah/llm-gateway.git
cd llm-gateway
docker compose up --build -d
python3 scripts/demo_traffic.py --key sk-gw-demo-local-key-0001 --admin-key change-me-admin   # needs httpx
open http://localhost:8080/dashboard     # admin key: change-me-admin
```

Pass real provider keys through the environment (`OPENAI_API_KEY=... docker compose up`) and the `gpt-4o-mini`, `claude-haiku`, `claude-sonnet` and `text-embedding-3-small` aliases start using them.

### Local (SQLite, in-memory cache)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
make run            # http://localhost:8080 ; ADMIN_API_KEY=change-me-admin, DEMO_API_KEY=sk-gw-demo-local-key-0001
make traffic        # in another terminal, fills the dashboard
```

## Usage

Any OpenAI client works; point it at the gateway and use a virtual key:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-gw-demo-local-key-0001")
reply = client.chat.completions.create(
    model="claude-haiku",            # an alias from config/gateway.yaml
    temperature=0,                   # temperature 0 responses are cacheable
    messages=[{"role": "user", "content": "How long does a bKash settlement take?"}],
)
print(reply.choices[0].message.content)
```

```console
$ curl -si http://localhost:8080/v1/chat/completions \
    -H "Authorization: Bearer sk-gw-demo-local-key-0001" -H "content-type: application/json" \
    -d '{"model":"demo-chat","temperature":0,"messages":[{"role":"user","content":"How does bKash settle?"}]}'
HTTP/1.1 200 OK
x-gateway-cache: miss
x-gateway-provider: fake-primary
x-gateway-model: fake-large
x-gateway-attempts: 1
x-gateway-fallback: false
x-gateway-cost-usd: 0.00003500

{"id":"chatcmpl-fb5a273ce777475aa448848d","object":"chat.completion","model":"fake-large",
 "choices":[{"index":0,"message":{"role":"assistant","content":"bKash settlements arrive T+1; check the merchant portal for the batch file."},"finish_reason":"stop"}],
 "usage":{"prompt_tokens":13,"completion_tokens":19,"total_tokens":32}}
```

Sending the same request again returns `x-gateway-cache: hit` and costs nothing; `Cache-Control: no-cache` or `x-gateway-cache: no-store` bypasses the cache.

Create a scoped key for a team:

```bash
curl -s http://localhost:8080/admin/keys -H "Authorization: Bearer change-me-admin" \
  -H "content-type: application/json" \
  -d '{"name":"mobile-app","budget_usd":25,"budget_period":"monthly","rpm_limit":120,
       "tpm_limit":200000,"allowed_models":["claude-haiku","demo-*"],"metadata":{"team":"mobile"}}'
# -> {"id":"key_...","key":"sk-gw-...", ...}   the secret is shown once; only its hash is stored
```

### Routing configuration

```yaml
models:
  claude-haiku:
    targets:
      - { provider: anthropic, model: claude-haiku-4-5 }
    fallbacks: [gpt-4o-mini, demo-chat]     # tried in order once every target has failed
  demo-chat:
    targets:                                 # weighted 3:1, then the other target, then fallbacks
      - { provider: fake-primary, model: fake-large, weight: 3 }
      - { provider: fake-flaky, model: fake-large, weight: 1 }
    fallbacks: [demo-backup]
routing:
  max_retries: 2
  backoff_base_ms: 100
  request_timeout_s: 60
  breaker: { failure_threshold: 5, recovery_s: 30 }
```

A provider whose `api_key_env` or `base_url_env` is unset is skipped at startup (shown as disabled in `/admin/providers`), and its aliases go straight to their fallbacks. Prices live in `config/pricing.yaml` (USD per 1M tokens, globs allowed).

## API

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| POST | `/v1/chat/completions` | virtual key | Chat completion; `stream: true` for SSE |
| POST | `/v1/embeddings` | virtual key | Embeddings |
| GET | `/v1/models` | virtual key | Aliases this key may use |
| POST | `/admin/keys` | admin | Create a key (secret returned once) |
| GET | `/admin/keys`, `/admin/keys/{id}` | admin | List / inspect keys, spend and remaining budget |
| PATCH | `/admin/keys/{id}` | admin | Change limits, budget, allow-list, status; `reset_spend` |
| DELETE | `/admin/keys/{id}` | admin | Revoke |
| GET | `/admin/usage?group_by=key\|model\|provider\|day&since=&until=&key_id=` | admin | Requests, errors, cache hits, fallbacks, tokens, cost, latency |
| GET | `/admin/alerts` | admin | Budget threshold alerts |
| GET | `/admin/providers` | admin | Availability and circuit-breaker state per provider |
| POST | `/admin/cache/clear` | admin | Drop cached responses |
| GET | `/health` | – | Database, Redis, available providers and aliases |
| GET | `/metrics` | – | Prometheus metrics (`llmgw_requests_total`, `llmgw_cost_usd_total`, `llmgw_circuit_state`, ...) |
| GET | `/dashboard` | admin key in the page | Usage dashboard |

Errors use OpenAI's shape, `{"error": {"message", "type", "param", "code"}}`: `401 invalid_api_key`, `402 budget_exceeded`, `403 model_not_allowed`, `404 model_not_found`, `429 requests_per_minute` / `tokens_per_minute` / `upstream_rate_limited` (with `Retry-After`), `502 upstream_error`, `503 circuit_open`.

### Demo traffic

`scripts/demo_traffic.py` against the offline configuration (real run, 80 requests, seed 3; the `mobile-app` key has a $0.0004 budget on purpose):

```
outcomes:
  claude-haiku -> 200              8
  claude-haiku -> 402              1
  claude-haiku stream -> 200       4
  claude-haiku stream -> 402       2
  demo-chat -> 200                 38
  demo-chat -> 402                 12
  demo-chat stream -> 200          8
  demo-chat stream -> 402          3
  embeddings -> 200                4
```

From `/metrics` and `/admin/usage` after that run: `llmgw_upstream_retries_total{provider="fake-flaky"} 7` (the flaky upstream fails one call in three and every failure was retried successfully), `llmgw_fallbacks_total{alias="claude-haiku"} 8` (Anthropic is not configured offline, so routed `claude-haiku` calls went down the fallback chain; the other 4 were cache hits), 26 cache hits out of 80 requests, and 18 `402`s once `mobile-app` crossed its 50%, 80% and 100% alerts. The dashboard screenshot under [Screenshots](#screenshots) was taken after this run.

## Project Structure

```
llm-gateway/
├── src/llm_gateway/
│   ├── app.py                # application factory, lifespan wiring, /health, /metrics, /dashboard
│   ├── gateway.py            # preflight -> cache -> router -> accounting; SSE assembly
│   ├── api/                  # OpenAI-compatible and admin routes, auth dependencies
│   ├── routing/              # Router (aliases, weights, retries, fallbacks) and CircuitBreaker
│   ├── providers/            # openai.py, anthropic.py, ollama.py, fake.py, shared HTTP/SSE plumbing
│   ├── keys.py · usage.py    # virtual keys, budgets, alerts; usage reports
│   ├── cache.py · ratelimit.py
│   ├── tokens.py · pricing.py
│   ├── logs.py · metrics.py  # JSON logs with PII redaction, Prometheus registry
│   ├── db.py · schemas.py · config.py · errors.py · cli.py
│   └── static/dashboard.html
├── config/gateway.yaml · config/pricing.yaml
├── scripts/export_openapi.py · scripts/demo_traffic.py
├── tests/                    # 85 tests
├── site/index.html           # GitHub Pages API reference
├── Dockerfile · docker-compose.yml · render.yaml · .devcontainer/
└── Makefile · pyproject.toml
```

## Testing

```bash
make test     # 85 tests, ~16 s, no network, no API keys
make lint
```

- **Anthropic translation** (23): system/developer extraction, turn merging, tool calls and results, images, sampling parameters, tool schemas and `tool_choice`, response and stop-reason mapping, cache-token usage, the full stream event mapping for text and tool use, error events, and the adapter over `httpx.MockTransport` (headers, SSE parsing, 429 with `Retry-After`, 400 vs 401 classification)
- **Other adapters**: OpenAI pass-through (unknown fields, forced `include_usage`, SSE `[DONE]`, error and timeout mapping), Ollama request/response/NDJSON stream/embeddings, fake provider determinism, scripting, truncation, tool calls and failure injection
- **Routing**: weight distribution, exponential backoff timings, fallback order, auth errors skip retries, bad requests fail fast, all-429 becomes 429, breaker open/half-open/recover, timeouts, `Retry-After`, unknown and unavailable models, streaming fallback before the first chunk, fallback cycles
- **Components**: cache-key normalisation, cacheability, memory LRU/TTL, Redis cache, RPM/TPM limits on memory and Redis, token heuristic, price table, PII redaction patterns, JSON log format, URL normalisation
- **API end to end** (httpx ASGI transport): auth errors, routing headers, cache hit/bypass, SSE with and without usage, cache replay as SSE, embeddings, OpenAI-style validation errors, key lifecycle with allow-lists and revocation, expiry, budgets with all three alerts and `402`, per-key rate limits, fallback when a provider is unconfigured, usage reports, Anthropic streaming through the whole stack, mid-stream failure reported in-band, Redis-backed deployment

CI also runs the gateway against real PostgreSQL and Redis service containers and drives it with `demo_traffic.py`.

## Roadmap

- [ ] Alembic migrations (tables are created on startup today) (#1)
- [ ] Semantic cache option backed by an embeddings alias and pgvector, behind the same cache protocol (#2)
- [ ] Per-key provider credentials (bring-your-own-key) encrypted at rest (#3)
- [ ] Latency-aware routing that shifts weight away from slow deployments using the recorded p95 (#4)
- [ ] OpenAI Responses API and image endpoints (#5)

## License

MIT, see [LICENSE](LICENSE).
