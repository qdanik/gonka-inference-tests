# Are `98tsj8s` and `qsy8ts3e` serving honest inference? — 2026-08-11

The two participants under suspicion behaved strangely on throughput: `qsy8ts3e` ran 2.9× slower than `cxmn4rxv` on identical prompts, and `98tsj8s` refused a quarter of all requests. The question is whether their **answers** are honest — the same model, actually reading the prompt — or whether the strangeness reaches into the inference itself.

Evidence below is from the four runs of 2026-08-11, which sent the same 100 books in the same order, with `logprobs: true, top_logprobs: 5` recorded in full.

## Verdict

**Their answers are sound. Their service is not.** Nothing in the responses suggests a substituted, quantised, or shortcut model. Every anomaly found is infrastructural — build version, capacity, and how much of the paid token budget goes to hidden reasoning.

## What was checked, and what it showed

### 1. Same model? — yes

| run | `model` returned | mean logprob of chosen token | tokens at logprob 0 |
|---|---|---|---|
| cxmn4rxv (honest) | MiniMaxAI/MiniMax-M2.7 | −0.838 | 55.0% |
| ulzldum (honest) | MiniMaxAI/MiniMax-M2.7 | −0.875 | 55.0% |
| **qsy8ts3e** | MiniMaxAI/MiniMax-M2.7 | **−0.836** | 57.2% |
| **98tsj8s** | MiniMaxAI/MiniMax-M2.7 | **−0.959** | 53.3% |

A different model — or the same model quantised harder — shifts the confidence profile. These four are indistinguishable: the suspect `qsy8ts3e` is *closer* to the honest `cxmn4rxv` (−0.836 vs −0.838) than the two honest ones are to each other.

### 2. Same tokenizer? — yes

The `bytes` field of each logprob entry decodes identically across runs: token id `9210` is `" wants"` and `355` is `" is"` on every participant. All four opened their answer with the same two tokens on the same prompt.

### 3. Answering the book they were sent? — yes

| run | answers naming the book they received |
|---|---|
| cxmn4rxv | 100/100 |
| qsy8ts3e | **100/100** |
| ulzldum | 97/100 |
| 98tsj8s | **70/75 (93%)** |

### 4. Do their answers agree with the honest ones? — yes

Vocabulary overlap between two answers to the **same** book, averaged over the 75 books all four answered:

| pair | overlap |
|---|---|
| cxmn4rxv ↔ qsy8ts3e | 0.195 |
| cxmn4rxv ↔ ulzldum (both honest) | 0.188 |
| qsy8ts3e ↔ ulzldum | 0.182 |
| 98tsj8s ↔ cxmn4rxv | 0.171 |
| 98tsj8s ↔ ulzldum | 0.161 |
| **two different books, same run — baseline** | **0.090** |

Every pair sits at roughly twice the different-book baseline, and the suspects are inside the honest pair's range. The single highest agreement in the whole matrix is between an honest participant and a suspect. A participant serving something else would sit near the baseline; none does.

### 5. Grounded quotation? — the same as the honest ones

| run | quotes verbatim in the prompt | anchored in the prompt | grounded names |
|---|---|---|---|
| cxmn4rxv | 0.510 | 0.663 | 0.974 |
| **qsy8ts3e** | 0.454 | 0.598 | **0.976** |
| ulzldum | 0.467 | 0.647 | 0.972 |
| **98tsj8s** | 0.467 | 0.604 | 0.949 |

`98tsj8s` quotes the book *more* faithfully than the honest `ulzldum`. The gaps here are within the noise established in [README.md](README.md).

## What is genuinely wrong with them

### `98tsj8s`: two different builds, and a quarter of requests refused

It answered from **two different vLLM builds inside a single run** — `vllm-0.23.0-dbb57c36` (39 answers) and `vllm-0.23.0-ecc46a1e` (36). Quality is identical between them:

| build | n | quote_exact | answer_share | answer chars |
|---|---|---|---|---|
| dbb57c36 | 39 | 0.464 | 0.689 | 13,631 |
| ecc46a1e | 36 | 0.471 | 0.691 | 14,968 |

So the split is not a quality problem, but a single participant should not be presenting two builds. On top of that, 25 of 100 requests were refused outright with `all hosts are at capacity` after five attempts, and 147 responses were shed along the way.

### `qsy8ts3e`: slow, and on an old build

2.9× slower decode than `cxmn4rxv` on byte-identical prompts (4.8 vs 13.9 tok/s), uniformly across all eight shards that served it, with 47 × 502 shed on retries. Its answers are fine; they arrive at a third of the rate.

### `cxmn4rxv` — and only it — returns text where the token id belongs

`logprobs.content[].token` is expected to carry the numeric token id. Three of the four participants do exactly that. `cxmn4rxv` does not:

| run | vLLM build | `token` field | non-numeric tokens |
|---|---|---|---|
| **cxmn4rxv** | **0.26.0**-01ba24ef | **text** — `"The"`, `" user"`, `" wants"` | **404,767 / 409,600 (98.8%)**, in **all 100** answers, from position 0 |
| qsy8ts3e | 0.23.0-ecc46a1e | ids — `"758"`, `"3100"`, `"9210"` | 0 / 409,600 |
| 98tsj8s | 0.23.0-ecc46a1e / dbb57c36 | ids | 0 / 307,200 |
| ulzldum | *no fingerprint reported* | ids | 0 / 409,600 |

Same for the alternatives: 2,042,060 of 2,048,000 `top_logprobs` entries carry text on `cxmn4rxv`, none anywhere else. The 1.2% of its tokens that pass a numeric test are digits the model actually wrote — `"1"`, `"2"`, `"3"` from its numbered sections — not ids.

**It follows the build, not the machine.** Six devshards served both runs, and each returned ids under 0.23.0 and text under 0.26.0:

| devshard | in the cxmn4rxv run | in the qsy8ts3e run |
|---|---|---|
| devshard-48044 | 0.26.0 → text `'The'` | 0.23.0 → ids |
| devshard-48050 | 0.26.0 → text `'The'` | 0.23.0 → ids |
| devshard-48054 | 0.26.0 → text `'The'` | 0.23.0 → ids |

So the deviation belongs to vLLM 0.26.0, and a client parsing `token` as an integer breaks on every response from it. The `bytes` field is correct in both formats, so a client reading `bytes` is unaffected.

Separately, `ulzldum` returns **no `system_fingerprint` at all**, so its build cannot be identified from a response — a participant that can hide its build can change it silently.

### All four: `thinking_token_budget: 0` is ignored

Every request carried `thinking_token_budget: 0`. Every participant produced hidden reasoning anyway, and it is charged against the same 4,096-token ceiling as the answer:

| run | share of output spent thinking | ≈ tokens per answer | answers where thinking took over half |
|---|---|---|---|
| cxmn4rxv | 21.9% | ~900 | 11 / 100 |
| qsy8ts3e | 24.8% | ~1,020 | 12 / 100 |
| ulzldum | 25.7% | ~1,050 | 19 / 100 |
| **98tsj8s** | **31.0%** | **~1,270** | **22 / 75** |

This is the one finding that costs the caller money on every request, and it is worst on `98tsj8s` — which is also why its answers stop around section 2 of 5 while the others reach section 4. The parameter is being accepted and disregarded across the whole fleet.

## What this does NOT establish

- **That the runs map to the participants named.** The four runs' devshard sets overlap heavily; what separates them cleanly is the build fingerprint, not the shard id. The mapping from run to participant rests on which host was switched on at the gateway at the time, not on anything in the responses.
- **That `98tsj8s` and `qsy8ts3e` are honest in general.** This is one afternoon, 175 answers, one model, one task shape. It rules out the crude failures — wrong model, ignored prompt, fabricated output — not a subtle one.
- **That the thinking is wasteful rather than useful.** Answer quality does not correlate with thinking share on the quote metrics (r = +0.018); it correlates only with how much of the structure gets finished. Whether the reasoning improves the answer was not tested.
- **A cause for the capacity refusals or the slowness.** Both are visible only as symptoms from the client side.

## How to settle the remaining doubt

The response body cannot prove which machine ran the tokens. Two things would:

1. **Have the gateway report the participant** that served each request, alongside the devshard id. Every attribution question in this series traces back to its absence.
2. **Require `system_fingerprint`** from every participant, and reject or flag those that omit it — `ulzldum` currently reports none, so its build is unknown, and a participant that can hide its build can change it silently.

## Sources

- `scores.json` — 375 per-answer scores
- `../host-*/responses.jsonl` — full bodies with logprobs
- `../host-*/inferences/` — one readable JSON per inference
- [README.md](README.md) — the quality metrics, their confidence intervals, and their limits
