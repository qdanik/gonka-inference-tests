# Throughput benchmarking — `python -m e2e.gateway bench`

`run` asks whether the gateway handles a parameter correctly. `load` asks whether it stays up under concurrency. **`bench` asks how many tokens per second the network actually delivers**, which needs the request shaped deliberately and the load generated in the right place.

## What makes a token-rate number honest

**Serving has two phases with opposite bottlenecks.** Prefill reads the whole prompt in one compute-bound pass; decode emits one token at a time and is bound by memory bandwidth. A figure reported as "tokens/s" is almost always decode throughput, so it is governed by the OUTPUT length — a benchmark with a huge prompt and a short answer measures something else. Prefill is roughly 350× cheaper per token, so **input tokens/s is an admission probe, not a speed**.

Hence three profiles, each isolating one regime:

| profile | prompt | output | isolates |
|---|---|---|---|
| `decode` | 400 | 4,096 | the headline tokens/s figure |
| `prefill` | 100,000 | 64 | how fast the network ingests context |
| `balanced` | 4,096 | 16,384 | the whole pipeline at realistic proportions |

Override either side with `--prompt-tokens` / `--output-tokens` — that is how a sweep at a series of prompt lengths is run against one profile shape.

**Every request must do identical work**, or the rate varies with how talkative the model felt. `min_tokens = max_tokens` forces exactly the requested output on every request.

**The context ceiling is shared.** prompt + completion must fit `--context-limit` (240,000 on the Kimi route). There is no separate output cap. The harness refuses a configuration that would exceed it rather than letting the gateway clamp silently and invalidate the measurement.

**The tunnel must be out of the measurement path.** Measured on this setup, an SSH forward tunnel carries 1,000 concurrent connections without errors but inflates median latency from 0.7 s to 8.0 s and peaks around 200 connections in flight. Always pass `--on-server` for a rate measurement: the load generator is uploaded to the gateway box and runs against the gateway's loopback, and only per-request records come home. Analysis is the same `build_report` either way, so the two modes stay comparable.

## Prompts are real books, not filler

Earlier runs synthesized prompts from word lists — grammatical sentences carrying no meaning. They hit the token count exactly and busted the prefix cache, but the model had nothing to read, and it showed: answers degenerated into runs of `(.) (.) (.)`.

Prompts now come from a pool of **whole public-domain books**, one per request, read from the first page:

```bash
python -m scripts.build_corpus --count 128 --out corpus/documents.json
python -m e2e.gateway bench ... --corpus corpus/documents.json
```

- Books shorter than `--min-chars` (default 560,000 — enough to fill a 100k-token prompt alone) are rejected rather than repeated.
- Gutenberg licence headers are stripped. Left in, they would give every prompt the same opening — a shared prefix is exactly what a prefix cache exploits.
- Request *i* gets `documents[i % N]`. Once the pool is exhausted, later passes step **deeper into the same books** rather than resending a prompt the cache already holds. A soak needs this: 128 books cannot cover a 30-minute run, and repeated prompts would measure the cache instead of the fleet.
- The collector prints the number of distinct windows the corpus yields at the current prompt size, and warns when the request count exceeds it.
- The corpus is uploaded once and cached on the box under a content-derived name (`/tmp/gonka-corpus-<stem>-<size>.json`); later runs skip the upload.

**Public-domain only.** These prompts go to a third-party inference network — nothing private or proprietary belongs in the pool.

## Soak mode

`--duration-hours H` holds `--concurrency` requests in flight for the whole window, replacing each as it completes, instead of firing a fixed burst that drains.

```bash
python -m e2e.gateway bench --profile balanced --model MiniMaxAI/MiniMax-M2.7 \
  --prompt-tokens 30000 --output-tokens 1024 --concurrency 64 \
  --duration-hours 0.5 --on-server --timeout 1800 \
  --corpus corpus/documents.json --out-dir artifacts/<date>/soak-...
```

Records stream to a progress file on the box as they complete, so an aborted soak leaves the work that did finish. Holding everything in memory until the end once cost a full hour of a run that had to be stopped.

## Saving what was asked and answered

| flag | effect |
|---|---|
| `--save-content` | write `responses.jsonl` (complete response bodies) **and** `requests.jsonl` |
| `--no-save-requests` | with `--save-content`, keep the responses but drop the request bodies |
| `--logprobs --top-logprobs N` | ask for per-token logprobs; responses grow by roughly 30× |

Sizes are not small. At 100k in / 4k out with logprobs: `responses.jsonl` ≈ 1.7 MB per answer, `requests.jsonl` ≈ 0.5 MB per request. A prompt is fully regenerable from the corpus given `document.id` and `offset`, which is what `--no-save-requests` is for.

## Artifacts

```
artifacts/<date>/<run>/
├── summary.json        report + every per-request record (status, latency, tokens,
│                       response_id, system_fingerprint, finish_reason, attempts)
├── requests.jsonl      one line per request, keyed on `index`:
│                       {index, seed, document: {id, title, offset}, request: {...}}
├── responses.jsonl     one line per answer, same `index`: {index, response: {...}}
├── devshards.json      per-devshard aggregates
├── prompt.txt          one sample prompt
├── inferences/         written by scripts/split_responses.py — one readable JSON
│                       per inference, named `<index>-<devshard>.json`
└── README.md           mandatory; see the gateway-validation skill
```

`requests.jsonl` and `responses.jsonl` are deliberately two files keyed on `index`: an answer is worth reading on its own, and a half-megabyte prompt in the same record buries it.

## Companion scripts

| script | what it is for |
|---|---|
| `python -m scripts.build_corpus` | download the book pool (`--count`, `--min-chars`, `--max-chars`) |
| `python -m scripts.recover_run` | rebuild artifacts from a run still on the box (see below) |
| `python -m scripts.split_responses <dir>` | one readable JSON per inference; `--with-logprobs`, `--no-prompt` |
| `python -m scripts.compare_answers <day-dir>` | score answers on quote fidelity, structure, degeneracy |

### Recovering a run

The collector runs detached (`nohup setsid`) so it survives a dead poller, an SSH drop, or a closed laptop — but it does no analysis, so when the poller gives up the run is finished on the box and invisible here.

```bash
python -m scripts.recover_run --seed-base 178645005500000 \
  --model MiniMaxAI/MiniMax-M2.7 --prompt-tokens 100000 --output-tokens 4096 \
  --concurrency 34 --out-dir artifacts/<date>/<run>
```

`--from-progress` reads the streamed records instead of the result file, for a run that has not finished. The seed base names the run's files on the box: `/tmp/gonka-bench-<seed>.{jsonl,result}`. Raw results are left on the box on purpose; only the script, config and log are cleaned up.

### Scoring answers

`compare_answers` exists because "the tokens arrived fast" is not "the tokens were worth arriving". It works when several runs were sent **the same prompts in the same order**, which the corpus guarantees, so answers can be compared request-for-request.

| metric | meaning |
|---|---|
| `quote_exact` | quoted spans (≥40 chars) found verbatim in the prompt; elided quotes are split on the ellipsis and each fragment checked |
| `quote_anchored` | only the opening eight words found — separates "started in the book and drifted" from "invented" |
| `sections` | how many of the requested parts the answer contains |
| `answer_share` | answer over total output, counting thinking wherever it landed |
| `distinct_trigrams` | unique trigrams over total; degenerate loops crash this |
| `grounded_names` | capitalised words in the answer that occur in the prompt |

`quote_exact` around 0.5 does **not** mean half the quotes were invented — misses include ellipsis elisions and modernised punctuation. It is calibrated for comparing runs on identical prompts, not as an absolute hallucination rate.

## Gotchas — every one of these cost a run

| symptom | cause | fix |
|---|---|---|
| Run dies before sending anything: `calibration failed — 503` | the tokenizer-calibration probe was the only request without a retry policy | it now retries like every other request |
| A finished run is lost: `could not fetch result: Operation timed out` | one dropped `scp` after the burst completed | fetch retries four times; the raw result stays on the box; recover with `recover_run` |
| Poller abandons a run at 99/100 | its deadline assumed the burst was one wave; at 34 concurrent, 100 requests is three waves | the poller now gives up on being **stuck** (no completion for longer than one request could take), not on being slow |
| A run is reported green but the results are stale | `/tmp/gonka-bench-*` from an earlier run was read | check the seed base in the output; it encodes launch time |
| Throughput looks impossibly good in a soak | prompts repeated and the prefix cache served them | use `--corpus`; the collector warns when requests exceed distinct windows |
| Soak spins at tens of requests per second | a **non-retryable** error (400) returns in 0.01 s and the worker immediately takes the next index | check `status_counts` before trusting a soak; 400 is not shed, it is rejection |
| `unsupported model "X"; supported models: Y` on a model `/v1/models` advertises | only some devshards serve that model; the router fans out across all of them | 97% of requests were rejected this way on Kimi (2026-08-12) — the rejection carries no devshard id |
| Actual `prompt_tokens` overshoots the target by 30%+ | calibration measures chars/token on a fixed filler sentence; prose tokenises differently, and the gap is model-dependent (7% on MiniMax, 37% on Kimi) | read the achieved `prompt_tokens` from the report; do not assume the target |
| `logprobs.content[].token` holds text, not a token id | depends on the server build — vLLM 0.23.0 returns ids, 0.26.0 returns detokenized text. `bytes` is correct in both | read `bytes`, or send `return_token_ids: true` as the cross-GPU harness does |
| `thinking_token_budget: 0` is ignored | every participant generated hidden reasoning anyway, charged against the same output ceiling — 22–31% of the budget | measure `answer_share`; a run that thinks more delivers less answer at a fixed cap |
| `content` is empty but the model clearly answered | MiniMax puts the answer in `reasoning` on some shards and in `content` with an inline `<think>` block on others, in the same run | read both fields; the harness records `content_chars` and `reasoning_chars` separately |
| Post-run "could not re-check the served models" | the SSH tunnel closed before the sanity check ran | harmless; the burst itself used no tunnel |

## Devshards are not hosts

Every response carries an id like `devshard-48087-4203`, and the leading part looks like a machine label. It is not one. Across four runs on 2026-08-11, each taken while a different participant was the only one switched on at the gateway, the shard sets overlapped heavily and the last run introduced **no** shard that had not already served an earlier one. The same shard number returned different `system_fingerprint` values in different runs.

Attribution to a participant needs the gateway to report it. Until then, a run's directory name records **which host was being switched at the gateway**, not which machine produced the tokens — say so in the README.

`system_fingerprint` is the one field that did separate the runs cleanly (`vllm-0.23.0-ecc46a1e`, `vllm-0.26.0-01ba24ef`, and one participant reporting none at all). A participant that reports no fingerprint cannot be checked for a silent build change.
