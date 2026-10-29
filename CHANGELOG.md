# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-10-29

### Added

- Latency-aware routing (`routing.latency_aware`, off by default). Each deployment keeps a rolling window of latencies, and a slow deployment's weight is scaled by `fastest p95 / its p95`, within a noise tolerance and never below a floor. Old samples expire, so a deployment that speeds up wins its share back (#4).
- `GET /admin/providers` lists each deployment of a provider with its configured weight, effective weight, p95 latency and sample count.

- CodeQL analysis for Python and GitHub Actions on pushes, pull requests and a weekly schedule.
- Dependabot configuration for pip, Docker and GitHub Actions with grouped minor and patch updates.
- Screenshots in the README: a curl session against a local gateway and the API reference.
- Roadmap items tracked as GitHub issues under the v0.2.0 milestone.

### Changed

- CI workflow runs with a read-only `GITHUB_TOKEN`.

### Security

- The startup log names the demo key by its id instead of printing the first characters of the secret.

## [0.1.0] - 2026-10-25

First public release.

### Added

- OpenAI-compatible `POST /v1/chat/completions` (with SSE streaming and `stream_options.include_usage`), `POST /v1/embeddings` and `GET /v1/models`.
- Provider adapters for OpenAI-compatible HTTP, Anthropic Messages (request, response and stream translation including tools and images), Ollama (NDJSON streaming) and a deterministic fake provider.
- Router with model aliases, weighted load balancing, ordered fallback chains, retries with exponential backoff and jitter, `Retry-After` handling, per-request timeouts and a circuit breaker per provider.
- Virtual keys stored as SHA-256 hashes with budgets (total or monthly), RPM/TPM limits, model allow-lists, expiry and revocation.
- Spend tracking from a YAML price table, budget alerts at 50/80/100% with optional webhook, and `402` once a budget is exhausted.
- Exact-match response cache (memory or Redis) keyed on a normalised request hash, replayed as SSE for streaming clients.
- Admin API for keys, usage and cost reports, alerts, provider health and cache control, plus a dashboard at `/dashboard`.
- JSON logs with request ids and PII redaction, Prometheus `/metrics`.
- Dockerfile, compose stack with PostgreSQL and Redis, Render blueprint, Codespaces config and a GitHub Pages API reference.
- CI on Python 3.10 and 3.11 with an integration run against PostgreSQL and Redis.

[Unreleased]: https://github.com/AkibUllah/llm-gateway/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/AkibUllah/llm-gateway/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/AkibUllah/llm-gateway/releases/tag/v0.1.0
