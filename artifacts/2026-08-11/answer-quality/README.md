# Answer quality across the four runs of 2026-08-11

375 answers scored on what can be checked mechanically. The four runs were sent the **same 100 books in the same order**, so every answer has three counterparts generated from a byte-identical prompt — the comparison is paired per book, which removes "some books are harder" from the difference.

```bash
python -m scripts.compare_answers artifacts/2026-08-11 --out scores.json
```

## Metrics

| metric | what it measures | why it is checkable |
|---|---|---|
| **quote_exact** | share of quoted spans (≥40 chars) found verbatim in the prompt that was sent | the book was in the prompt; a fabricated quote is not in it. Elided quotes are split on their ellipsis and each fragment checked |
| **quote_anchored** | share whose **opening eight words** are in the prompt | separates "started in the book and drifted" from "invented outright" |
| **sections** | how many of the five requested parts the answer contains | the task named them: scenes, characters, four quotes, themes, unresolved |
| **answer_share** | answer text over total output, counting thinking wherever it landed — inline `<think>` **or** the `reasoning` field | output was capped at 4,096 tokens; thinking and answering compete for it |
| **distinct_trigrams** | unique trigrams over total | degenerate repetition loops crash this |
| **grounded_names** | capitalised words in the answer that occur in the prompt | catches characters imported from the model's memory of the book |

## Result

| run | answers | quote_exact | quote_anchored | quotes | sections | answer_share | distinct | grounded names |
|---|---|---|---|---|---|---|---|---|
| **cxmn4rxv** 13:10 | 100 | **0.510** | **0.663** | 10 | **4** | **0.781** | 0.909 | 0.974 |
| qsy8ts3e 13:51 | 100 | 0.454 | 0.598 | 9 | 4 | 0.752 | **0.932** | **0.976** |
| ulzldum 16:29 | 100 | 0.467 | 0.647 | 8 | 4 | 0.743 | 0.902 | 0.972 |
| **98tsj8s** 15:33 | 75 | 0.467 | 0.604 | 6 | **2** | **0.690** | 0.891 | 0.949 |

**Best: `cxmn4rxv`.** Leads on quote fidelity in both strengths, on structure, and on how much of the budget reached the answer.

**Worst: `98tsj8s`.** Median of 2 of 5 sections against 4 elsewhere, the fewest quotes, and the largest share of the budget spent thinking.

## Are the gaps real?

Paired per book, 95% confidence on the mean difference:

| comparison | difference | 95% CI | verdict |
|---|---|---|---|
| quote_exact, cxmn4rxv − qsy8ts3e | +0.076 | [+0.000, +0.152] | **barely** significant — the lower bound touches zero |
| quote_anchored, cxmn4rxv − qsy8ts3e | +0.128 | [+0.050, +0.206] | significant |
| sections, cxmn4rxv − 98tsj8s | +0.680 | [+0.122, +1.238] | significant |
| answer_share, cxmn4rxv − 98tsj8s | +0.093 | [+0.022, +0.164] | significant |
| distinct_trigrams, qsy8ts3e − 98tsj8s | +0.038 | [−0.015, +0.091] | **not** significant |

The effects are small and the per-book spread is large — on quote_exact the best run beats the worst on 42 of 67 books, not on all of them. Treat the ranking as a weak ordering, not a verdict.

## Why the worst run is worst: the budget, not the reading

Output was pinned at exactly 4,096 tokens (`min_tokens = max_tokens`) and **every answer stopped at `length`**. Thinking and answering therefore draw on one fixed budget, and whatever goes to `<think>` is text the answer never gets.

| run | thinking share | answers where thinking took over half the budget |
|---|---|---|
| cxmn4rxv | 21.9% | 11 / 100 |
| qsy8ts3e | 24.8% | 12 / 100 |
| ulzldum | 25.7% | 19 / 100 |
| **98tsj8s** | **31.0%** | **22 / 75** |

Across all 375 answers, `answer_share` correlates with `sections` at **r = +0.235** and answer length with sections at **r = +0.311**: shorter answers are cut earlier in their five-part structure. `98tsj8s` spent ~370 more tokens per answer on thinking than `cxmn4rxv`, and its answers stop around section 2.

Note what this does *not* explain: `answer_share` and `quote_exact` correlate at **r = +0.018** — essentially zero. Quote fidelity is a separate axis; budget pressure does not make the model misquote.

## Speed does not buy or cost quality

The `ulzldum` run split into 70 fast requests (~80 s) and 30 slow ones (~490 s). On identical prompts:

| metric | fast | slow | difference, 95% CI |
|---|---|---|---|
| quote_exact | 0.470 | 0.460 | +0.010 [−0.112, +0.133] — no difference |
| answer_share | 0.744 | 0.741 | +0.002 [−0.096, +0.100] — no difference |
| sections | 2.81 | 3.67 | −0.852 [−1.620, −0.084] — slow answers were *more* complete |
| distinct_trigrams | 0.881 | 0.953 | −0.072 [−0.139, −0.006] — slow answers repeated *less* |

Requests served 6× faster produced answers no worse on grounding, and if anything slightly worse on structure. The same check on `cxmn4rxv` (13 fast, 87 slow) found no significant difference on anything.

## Per devshard, pooled across runs

Suggestive only — most shards have 10–25 answers, and the confidence intervals at that size overlap heavily.

| devshard | answers | quote_exact | sections | answer_share |
|---|---|---|---|---|
| devshard-48058 | 10 | 0.634 | 4 | 0.816 |
| devshard-48049 | 15 | 0.615 | 4 | 0.664 |
| devshard-48054 | 23 | 0.582 | 4 | 0.763 |
| devshard-47927 | 19 | 0.563 | 2 | 0.770 |
| devshard-48052 | 14 | 0.526 | 4 | 0.689 |
| devshard-48056 | 13 | 0.525 | 4 | 0.812 |
| devshard-48045 | 19 | 0.510 | 4 | 0.713 |
| devshard-48044 | 42 | 0.507 | 4 | 0.792 |
| devshard-48055 | 23 | 0.481 | 4 | 0.685 |
| devshard-48023 | 20 | 0.459 | 3 | 0.757 |
| devshard-48046 | 21 | 0.444 | 4 | 0.799 |
| devshard-48086 | 19 | 0.438 | 2 | 0.731 |
| devshard-48048 | 22 | 0.428 | 2 | 0.713 |
| devshard-48050 | 55 | 0.422 | 3 | 0.739 |
| devshard-48087 | 19 | 0.401 | 5 | 0.738 |
| devshard-48053 | 27 | 0.400 | 3 | 0.746 |
| devshard-48051 | 14 | 0.298 | 4 | 0.712 |

## What this establishes

- **The four runs differ little in content.** Every gap is under 0.13 on a 0–1 scale, one is not significant at all, and the largest — structure completeness — is explained by budget spent on thinking rather than by reading quality.
- **`cxmn4rxv` is the best of the four and `98tsj8s` the worst**, on a weak but consistent ordering across four of six metrics.
- **Thinking eats the answer.** With `min_tokens = max_tokens` and every answer truncated at `length`, a run that thinks 9 percentage points more delivers visibly less structure. If completeness matters more than a fixed token count, the cap needs raising or the thinking budget enforcing.
- **Fast serving is not degraded serving.** A 6× decode difference within one run produced no measurable loss of grounding.

## What this does NOT establish

- **Anything about the hosts.** The four runs' devshard sets overlap heavily (see the individual run READMEs); the names are the host being switched at the gateway, not the machine that served the tokens.
- **An absolute hallucination rate.** `quote_exact` of ~0.5 does **not** mean half the quotes were invented. Misses include ellipsis elisions, modernised punctuation, and quotes that begin in the book and drift — `quote_anchored` is 0.60–0.66 on the same answers. The metric is sound for comparing runs on identical prompts; it is not calibrated as an absolute measure.
- **That any single devshard is better or worse.** The per-shard table is underpowered at 10–25 answers each.
- **Anything about correctness of the analysis itself.** These are surface metrics. No answer was read end to end and graded against the book.

## A correction

The first version of this scoring put quote fidelity at 12% and would have reported rampant fabrication. The extraction was wrong: the minimum-length filter was inside the regex, so a short quote failed to match and its closing mark was paired with the **opening** mark of the next quote — capturing the model's own narration between two quotations and scoring it as an invented quote. Length is now filtered after extraction. The lesson is in the script as a comment, and the paired checks in `scripts/compare_answers.py` cover it.

## Artifacts

- `scores.json` — all 375 per-answer scores, keyed by run and index
- `../host-*/inferences/` — the answers themselves, one readable JSON per inference
