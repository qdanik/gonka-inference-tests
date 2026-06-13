# e2e/gateway — gateway request-validation harness

Verifies how the Gonka **devshard gateway** handles chat-completion parameters —
clamping, rejecting, or normalizing out-of-range and wrong-typed values before
they reach vLLM.

This is distinct from `e2e infer`, which drives a raw vLLM server. Here every
request goes to the gateway's `/v1/chat/completions` on the server loopback,
through an SSH forward tunnel.

## Run

Copy `.env.example` to `.env` (gitignored), fill it in, then:

```bash
python -m e2e.gateway run
```

`.env` is auto-loaded from the repo root. Flags and real env vars override it:

```bash
GONKA_GATEWAY_ADMIN_KEY=<admin-key> python -m e2e.gateway run \
  --ssh-host <user>@<host> --ssh-port <port> --gateway-url http://127.0.0.1:18080
```

- `--ssh-host` / `--ssh-port` — how to reach the box.
- `--gateway-url` — where the gateway listens **on that box** (server-side loopback).
- `--models` — comma-separated model ids to test; empty = every served model.
- `--names` — comma-separated case names; empty = all.

Exit code is non-zero if any case fails. Results go to
`artifacts/<date>/gateway-chat-params/summary.json`.

## Behavior tested

- **reject** → HTTP 400 whose `error.message` names the field; for `max_tokens:0`
  it must also be *fast* (no 0-byte hang).
- **clamp / accept** → HTTP 200 with non-empty content.

Every case runs against every served model, so a rule is verified on the whole
fleet. A case that diverges on one model (e.g. Kimi floors `max_tokens:0`)
overrides that model's expectation via `expect.per_model` in its fixture.

## Secrets policy (this repo is public)

Nothing secret is committed. The admin key comes from `$GONKA_GATEWAY_ADMIN_KEY`
or `--admin-key`; the SSH host/port and gateway URL are runtime flags. None are
logged or written into artifacts.

## Adding a case

Drop a new JSON file in [`inferences/gateway/`](../../inferences/gateway/) — the
runner picks it up automatically. See that directory's README for the schema.
