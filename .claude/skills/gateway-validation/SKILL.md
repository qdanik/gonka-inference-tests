---
name: gateway-validation
description: Use when the user wants to validate the Gonka devshard gateway's request-handling against a live server — chat-completion param clamping/rejection, the max_tokens:0 hang, or any "problem inference" that the gateway should clamp, reject, or normalize before vLLM. Covers the e2e/gateway harness, the inferences/gateway corpus, .env config, per-model fan-out, and the find→root-cause→fix→redeploy→re-verify loop.
---

# Gateway request-validation testing for Gonka

## When to invoke

The user wants to verify the **gateway** (not raw vLLM) on a server:
- chat-completion params are clamped / rejected / normalized before forwarding
- a request misbehaves (opaque 400, empty 200, hang) and should be captured as a regression
- confirm a gateway change end-to-end across every served model

Trigger phrases: "check the gateway", "validate chat params", "why is a request hanging or empty", "add a problem inference", "run the gateway tests".

NOT for raw-vLLM PoC/cross-validation — that is the `cross-gpu-validation` skill (`python -m e2e ...`).

## Harness location

`python -m e2e.gateway run` — package at `e2e/gateway/`. Runs **locally**; reaches the gateway (server loopback, e.g. `http://127.0.0.1:18080`) through an SSH forward tunnel it opens itself.

```
e2e/gateway/
├── inference.py   # send one chat-completions request to the gateway, decode status/error/content
├── cases.py       # load the JSON corpus + schema
├── runner.py      # tunnel → discover served models → fan out cases → assert → artifacts
├── config.py      # GatewayTarget (ssh, gateway-url, models)
└── __main__.py    # CLI + .env auto-load
inferences/gateway/*.json   # the problem-inference corpus (one file per case)
```

## Config & secrets (THIS REPO IS PUBLIC)

Never commit secrets. Config is read from `.env` at the repo root (auto-loaded — no `source` needed), or from flags / real env vars (which win over `.env`):

```
GONKA_GATEWAY_ADMIN_KEY=...      # admin Bearer key
GONKA_SSH_HOST=user@host
GONKA_SSH_PORT=22
GONKA_GATEWAY_URL=http://127.0.0.1:18080
```

`.env` is gitignored; `.env.example` is the committed template. The admin key is never logged or written into artifacts. A model id under `expect.per_model` is fine (it pins behavior, not infrastructure); a host/key/IP is not.

## Run

```bash
# fill .env from .env.example, then:
python -m e2e.gateway run                       # all cases, all served models
python -m e2e.gateway run --cases top_p_zero_rejected,max_tokens_zero_rejected
python -m e2e.gateway run --models Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
python -m e2e.gateway run --pr 1316 --cases n_zero_clamped_to_one,n_above_max_clamped
```

- `--cases a,b` — run only those case names (targeted runs); empty = all.
- `--pr <id>` — tag the run as a PR's; results go to `artifacts/<date>/gateway-pr-<id>/`. Fixtures stay flat in `inferences/gateway/`; `--pr` only changes the output dir, so pick the PR's cases with `--cases`.

Exit code is non-zero if any case fails. Results: `artifacts/<date>/gateway-<pr-<id>|chat-params>/summary.json`. After a clean run, write a `README.md` next to it summarizing the validation (models, tallies, anything found/fixed).

## The corpus (inferences/gateway/)

One JSON per case; the runner picks up new files automatically. Schema:

```json
{
  "name": "max_tokens_zero_rejected",
  "description": "why it matters",
  "request": { "messages": [...], "max_tokens": 0 },
  "expect": {
    "outcome": "reject",            // "reject" | "clamp" | "accept"
    "status": 400,
    "message_contains": "max_tokens",
    "max_latency_s": 30,            // reject hang-guard: a slow reject fails
    "per_model": { "moonshotai/Kimi-K2.6": { "outcome": "clamp", "status": 200 } }
  }
}
```

- **Every case runs against every served model.** `model` is injected per route (never in the fixture).
- A model that diverges (e.g. Kimi floors `max_tokens:0` instead of rejecting) gets an `expect.per_model` override — do NOT tag cases kimi/non-kimi.
- Assertions: `reject` → status + `message_contains` in `error.message` (+ latency under `max_latency_s`); `clamp`/`accept` → 200 with non-empty content.

To add a problem inference: drop a new fixture in `inferences/gateway/`, set `expect`, run.

## The validation loop (find → fix → re-verify)

A failing case is the harness working. When a case goes red:

1. **Run** → note the failure mode (e.g. "200 but empty content", "want 400 got 200", "rejected but slow").
2. **Probe** the gateway directly (ssh+curl, vary `seed` to dodge cache) to characterize: status, `finish_reason`, `usage.completion_tokens`, content. Compare the suspect param against a known-good control request.
3. **Root-cause in the gateway code** (`devshard/cmd/devshardctl/request_filters*.go`): trace the param through PreValidation → Limits (`applyOutputTokenLimits`) → PostLimits.
4. **Fix + Go test** (TDD), `go test ./cmd/devshardctl/...`.
5. **Rebuild + redeploy** the binary to the server (needs root on the box).
6. **Re-run** `python -m e2e.gateway run` → expect green. Update the artifacts README.

## Gotchas

- The gateway is loopback-only on the server → always via the SSH tunnel; `--gateway-url` is its address **as seen on the server**, separate from `--ssh-host`.
- Validation happens **before** forwarding, so non-streaming requests exercise it fully and give deterministic status/body. The `max_latency_s` guard distinguishes a fast reject from a request that hangs until the upstream deadline (e.g. a zero-budget request that produces no tokens).
- A `clamp` result only proves the request **succeeded** (200 + content) — the response does not echo the clamped value; the exact clamped number is locked by the gateway's Go unit tests, not here.
- Re-running burns real escrow balance on the clamp/accept cases (they run actual inference). Use `--cases` to target a subset while iterating.
- After redeploy, confirm the served set with the run's `served=[...]` line; a missing model means its escrow isn't active.
