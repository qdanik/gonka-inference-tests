# MiniMax-M2.7 Validation Similarity Run — 2026-06-01

Cross-node similarity stress test of the devshard inference validator against
**`MiniMaxAI/MiniMax-M2.7`** on the live decentralized fleet.

## Method

- **37 prompts** (the diverse sweep + word-salad / degenerate edge cases).
- For each prompt: **1 reference inference** (no `enforced_tokens`) → captures the
  forced token sequence + reference logits.
- Then **50 validations** of that inference, concurrency 5, distinct seeds. Each
  validation replays the *same* `enforced_tokens`, so the output tokens are
  identical and the similarity reflects **only logit divergence** between the
  executor and whichever participant served the validation.
- Similarity is computed exactly like the on-chain validator
  (`CompareLogits` → `positionDistance`). Validation threshold = **0.90**.

Each request is logged as `{address, similarity, request, response}`:
- `inference.json` — the reference inference (similarity = `null`).
- `validation-1.json` … `validation-50.json` — one per validation.
- `address` = the participant (`participant_key`) that executed that request,
  resolved via `/devshard/<escrow>/v1/debug/perf`.

## TL;DR — what broke

`similarity_below` (validation similarity < 0.90) **reproduces on degenerate /
word-salad / repetitive-CJK prompts**, fleet-wide. It is **not** a single bad
participant and **not** a metric padding artifact — it is **genuine cross-node
logit divergence**: heterogeneous Blackwell hardware (B200 / B300 / RTX PRO 6000)
computes materially different logit distributions for the *same* forced context
on numerically-sensitive content. On "easy" English prompts (e.g. `sys_math_en`)
agreement is perfect (0% top-1 mismatch, similarity ~0.99).

Evidence: on `user_only_chinese_salad`, the executor and validator disagree on
the top-1 token in **17% of *confident* positions** and **53% of flat positions**;
on `sys_math_en` the disagreement is **0%** everywhere.

## Test cases that found the problem (failing: min similarity < 0.92)

Ranked worst-first. Each row links to the test directory and the single
lowest-similarity validation file in it.

| Test case | min sim | mean sim | worst validation | served by (worst) |
|---|---|---|---|---|
| [user_only_word_salad](./user_only_word_salad) | **0.8616** | 0.9788 | [validation-13.json](./user_only_word_salad/validation-13.json) | `gonka168rtjf…` |
| [user_only_summarize_garbled](./user_only_summarize_garbled) | **0.8638** | 0.9804 | [validation-43.json](./user_only_summarize_garbled/validation-43.json) | `gonka168rtjf…` |
| [user_only_chinese_salad](./user_only_chinese_salad) | **0.8735** | **0.8852** | [validation-28.json](./user_only_chinese_salad/validation-28.json) | `gonka1d694r00…` |
| [user_only_prose_poem_long](./user_only_prose_poem_long) | **0.8884** | 0.9512 | [validation-35.json](./user_only_prose_poem_long/validation-35.json) | `gonka1d694r00…` |
| [user_only_no_question](./user_only_no_question) | **0.8931** | 0.9604 | [validation-3.json](./user_only_no_question/validation-3.json) | `gonka1d694r00…` |
| [user_only_very_long_text](./user_only_very_long_text) | 0.9193 | 0.9237 | [validation-1.json](./user_only_very_long_text/validation-1.json) | `gonka1tja3g2da…` |
| [long_input_max_1024](./long_input_max_1024) | 0.9196 | 0.9701 | [validation-45.json](./long_input_max_1024/validation-45.json) | `gonka1d694r00…` |
| [translate_chain](./translate_chain) | 0.9222 | 0.9594 | [validation-12.json](./translate_chain/validation-12.json) | `gonka168rtjf…` |

**`user_only_chinese_salad` is the strongest reproducer**: mean **0.8852** with
**every** participant landing ~0.87–0.91 (systematic, not an outlier). See the
per-participant breakdown below.

### `user_only_chinese_salad` — per-participant (all below 0.90)

| Participant | n | min | mean |
|---|---|---|---|
| `gonka1d694r00czmq75txghwjcuk07lxvc8d4ekgsha0` | 15 | 0.8735 | 0.8758 |
| `gonka168rtjfkszuhcggg4dfyse4yh7xn9zwfglnkns2` | 11 | 0.8750 | 0.9098 |
| `gonka1hwvel7n3zuk6wruefuzc356l9myske9stckwnz` | 8 | 0.8775 | 0.8803 |
| `gonka1tja3g2da45efhe2p83gk3whtussmgmtsdlgprt` | 15 | 0.8776 | 0.8795 |
| `gonka1ym3np7guxart483yfdxnlztuazx22cjt0e4a2p` | 1 | 0.8787 | 0.8787 |

## Passing test cases (sanity)

Clean, high-similarity (English / structured / deterministic). A few examples:

| Test case | min sim | mean sim |
|---|---|---|
| [structured_json](./structured_json) | 0.9884 | 0.9887 |
| [sys_math_en](./sys_math_en) | 0.9864 | 0.9935 |
| [sys_math_cn](./sys_math_cn) | 0.9817 | 0.9926 |
| [multi_turn_cn](./multi_turn_cn) | 0.9787 | 0.9955 |
| [code_review](./code_review) | 0.9725 | 0.9738 |

All 37 tests completed with **50/50 successful validations** (no request errors).

## Worst participants across the whole run

Aggregated over all 37 prompts × 50 validations. `n` = validations served,
`<0.9` = count of sub-threshold validations. **The sub-threshold counts track
volume and prompt mix, not a single faulty host** — every participant is healthy
on easy content and dips only on the degenerate-content prompts above.

| Participant | n | min | mean | `<0.9` |
|---|---|---|---|---|
| `gonka1d694r00czmq75txghwjcuk07lxvc8d4ekgsha0` | 658 | 0.8735 | 0.9735 | 19 |
| `gonka1tja3g2da45efhe2p83gk3whtussmgmtsdlgprt` | 435 | 0.8776 | 0.9653 | 15 |
| `gonka168rtjfkszuhcggg4dfyse4yh7xn9zwfglnkns2` | 455 | 0.8616 | 0.9730 | 10 |
| `gonka1hwvel7n3zuk6wruefuzc356l9myske9stckwnz` | 198 | 0.8775 | 0.9681 | 8 |
| `gonka1ym3np7guxart483yfdxnlztuazx22cjt0e4a2p` | 72 | 0.8787 | 0.9624 | 1 |
| `gonka1duuaqdx06sx8v2dzggltwwmqyuw8lvjkjq7xll` | 14 | 0.9450 | 0.9753 | 0 |
| `gonka1fc9tzt83dgrqswlgay4668cuqjrk7zsqks2vm2` | 1 | 0.9214 | 0.9214 | 0 |
| `gonka1zktn8j65wlys8a8e38hqhf4y3x6m4x04zskkrx` | 1 | 0.9570 | 0.9570 | 0 |
| `gonka10mmdjau4dnj8krs7sh7t7635ttnmq9u3vqgz09` | 1 | 0.9781 | 0.9781 | 0 |

> `unknown` (n=15) = requests whose participant rolled out of the
> `/v1/debug/perf` window before resolution; not attributable.

## Notes for developers

- To reproduce: replay any `inference.json` request, then re-send it with the
  `enforced_tokens` from that response (as in the `validation-*.json` requests)
  and compare logits against the inference.
