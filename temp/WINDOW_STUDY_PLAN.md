# Window-only study plan: develop on rag26, confirm on ragtime26

**Status: DRAFT, written before any window-only results exist (2026-10-09). Edit the
criteria now, not after looking at the results.**

## Why

The original Jev study (FINDINGS sections 4–13) used restricted topics: rag2026-10, -100,
-107 and -50, and ragtime 2041, 2046 and 2050. It also read statistics from all topics.
Under the data policy (CLAUDE.md), development may only use the **permitted window**: the
first 10 topics of each topics file (rag26 rag2026-0…9, ragtime26 2000…2009). This plan
redoes the method choices on that window only, as follows.

- **Development set:** the rag26 window. Every choice below is decided here.
- **Confirmation set:** the ragtime26 window. It is run and read **once**, after the rag26
  decisions are recorded at the end of this file. Ragtime results do not feed back into
  the choices. A failed confirmation is reported, not tuned away.

## What runs (budget-capped: total spend ≤ the key's remaining $57.21)

Conservative estimates (the guard uses these): **≈ $28 rag26 + $12 ragtime26 = $40**.
Expected spend is lower (≈ $31) because of cache hits.

- **rag26:** `bash temp/run_window_suite.sh rag26`, then `bash temp/run_proportionate_test.sh rag26`.
- **ragtime26, later:** the same commands with `ragtime26`.
- **Budget guard:** each paid step is skipped unless the key's remaining balance covers its
  estimate plus a $10 floor.

| Experiment | Topics | Design | Est. rag26 / ragtime26 | Decision |
| --- | --- | --- | --- | --- |
| Broad vs narrow (better and proportionate mirror pairs) | all 10 | one orientation; one narrow sub-question per topic | $7.2 / $2.6 | 1 |
| Identical summaries (Choice; proportionate Noul a and b) | all 10 | each run's report against itself | $0.4 / $0.3 | 2 |
| Slot bias, ties, spread, length, Gemini agreement **of the submitted form** | all 10 | from the full runs' window comparisons (both orientations) | **$0 / $0** | 3 |
| Padding with the submitted form (proportionate mirror) | all 10 | 16 targets × 8 opponents, both orientations; off-topic source = window topic i+5 | $2.0 / $2.0 | 4 |
| Determinism of the proportionate mirror | 1 (rag2026-3 / 2003) | 2 passes, cache bypassed | $0.8 / $0.3 | 5 |
| Choice pilot | all 10 | both orientations | $5.0 / $2.0 | 5 |
| Prompt variants (includes Choice) | all 10 | one orientation; also chooses padding targets | $3.5 / $1.2 | context |
| Gemini baseline | rag26 all 10 (0–6 cached, 7–9 new); ragtime 2000–2002 (existing run) | one orientation | $6.5 / $0 | agreement |
| Gemini padding | first 3 | as padding | $2.5 / $3.5 | comparison |

**Dropped to fit the budget** (no decision depends on them): the Choice and Noul instruction
checks (instruction-following already replicated on both datasets), the Noul "better" pilot,
Choice determinism, and Choice-only padding. The `appropriate` prompt is removed
altogether.

Statistics are reported per topic and pooled. Uncertainty comes from bootstrapping over
**topics**, not comparisons (comparisons share runs).

## Decisions (on rag26 window) and criteria

The current submission (mirrored proportionate Noul, both orientations) is the **default**.
It changes only if a criterion below fails on the rag26 window.

1. **Score proportional to the need (proportionate vs better wording).** *Data: broad vs narrow test.*
   - Keep proportionate if its broad→narrow shift relative to `better` (the
     difference-in-differences) is negative on **≥ 7 of 10** topics, and the pooled value's
     topic-bootstrap 95% CI excludes 0.
   - Otherwise switch to the `better` mirror pair.
2. **Ties.**
   - The chosen form must give identical summaries a mean P(A better) within
     **[0.40, 0.60]**.
   - If the mirror fails this, report it. No ad hoc correction.
3. **Orientation.** *Data: the full run's window comparisons of the submitted form ($0).*
   - Keep both orientations unless P(slot A wins) on close pairs (orientation-averaged P
     in 0.4–0.6) has a topic-bootstrap 95% CI that includes 0.50 **and** identical
     summaries are within [0.45, 0.55].
4. **Content, not length (padding).** For the chosen form:
   - Each uninformative variant (repeat, generic, off-topic) must have pooled ΔP < 0, with a
     topic-bootstrap CI excluding 0.
   - pad_relevant must have ΔP > 0.
   - A failure is reported as a known weakness; it doesn't trigger a change of form.
5. **Choice vs Noul.** *Data: determinism probe, and the Choice pilot against the submitted form on the window.*
   - Not reopened unless the mirror flips more than **2%** of winners between identical
     requests (determinism), or ranks runs with Spearman < 0.95 against Choice on the
     window.

If any decision changes, the full runs are redone: about $95 at mirrored-Noul cost.
That is **outside this budget** and needs a separate go-ahead.

## Confirmation (ragtime26 window, read once)

- For each criterion 1–5, record whether ragtime26 points the same way as rag26.
- Report confirmations and failures as they are.
- No further method changes are made because of ragtime26.

## Decisions recorded (fill in after the rag26 window results, BEFORE running/reading ragtime26)

Recorded 2026-10-09, after the rag26 window results and BEFORE running or reading
anything on ragtime26. rag26 spend: $25.27 (key $55.62 → $30.35).
Outputs: `temp/win_rag26_*.txt` and `temp/jev_probes_win_rag26/`.

**All five criteria pass. The submitted form (mirrored proportionate Noul, both
orientations) is kept, and the full runs are not redone.**

- **1 (proportionate vs better): PASS.**
  - The difference-in-differences is negative on **10 of 10** topics (−0.015 to −0.053).
  - Pooled: −0.0317. Every per-topic value is negative, so the topic-bootstrap 95% CI
    excludes 0.
  - Under the narrowed request, proportionate's shift toward the shorter answer is
    −0.064 against −0.032 for better.
  - On broad requests the two wordings differ on 0.96% of winners.
- **2 (ties): PASS.**
  - Identical summaries give a mirror score of (0.177 + 1 − 0.108) / 2 = **0.535**,
    within [0.40, 0.60].
  - All 828 are flagged as ties: P(yes) < 0.5 for both questions.
  - Choice, for comparison: 0.792.
- **3 (orientation): keep both.**
  - P(slot A wins) on close pairs is **0.553**, topic-bootstrap 95% CI
    [0.547, 0.559], which excludes 0.50.
  - The bias is present on all 10 topics (0.537 to 0.572).
- **4 (padding): PASS.**
  - ΔP for each uninformative variant:

    | Variant | ΔP | 95% CI |
    | --- | --- | --- |
    | repeat | −0.048 | [−0.053, −0.044] |
    | generic | −0.095 | [−0.102, −0.088] |
    | off-topic | −0.196 | [−0.209, −0.183] |

  - pad_relevant: +0.179, with ΔP > 0 for 160 of 160 targets.
  - Truncation: −0.081.
  - Every topic shows the same signs.
  - Gemini on the first 3 topics points the same way (relevant +0.104, off-topic
    −0.206).
- **5 (Choice vs Noul): not reopened.**
  - Determinism: the mirror flips 0.155% of winners between identical requests
    (threshold 2%).
  - Choice pilot vs the submitted mirror on the window: board Spearman **0.998**
    (threshold 0.95), Kendall 0.975; per-topic median 0.997.

Context (no decision depends on it): on the 10 window topics the submitted form agrees
with Gemini at board Spearman 0.974 (Kendall 0.866, per-topic median 0.952).
Split-half reliability is 0.974.

## Confirmation recorded (ragtime26 window, read once, 2026-10-13)

- ragtime26 spend: $6.82 (key $30.34 → $23.52). The run was interrupted by a terminal
  restart during the Choice pilot and resumed from cache. No call was paid twice.
- Outputs: `temp/win_ragtime26_*.txt` and `temp/jev_probes_win_ragtime26/`.

**All five decisions are confirmed. No method change.**

- **1: CONFIRMED.**
  - The difference-in-differences is negative on **10 of 10** topics (−0.008 to −0.027),
    with every per-topic value below 0.
  - Pooled: −0.016, about half rag26's effect.
- **2: CONFIRMED.**
  - Identical summaries: mirror (0.184 + 1 − 0.106) / 2 = **0.539**.
  - All 490 have P(yes) < 0.5 for both questions.
  - Choice, for comparison: 0.789.
- **3: CONFIRMED (keep both orientations).**
  - Close pairs: P(slot A wins) = **0.523**, topic-bootstrap CI [0.514, 0.532], which
    excludes 0.50.
  - The bias is smaller than on rag26 (0.553).
- **4: CONFIRMED.**
  - ΔP for each variant:

    | Variant | ΔP | 95% CI |
    | --- | --- | --- |
    | repeat | −0.074 | [−0.080, −0.069] |
    | generic | −0.135 | [−0.144, −0.125] |
    | off-topic | −0.258 | [−0.271, −0.245] |
    | relevant | **+0.098** | [+0.088, +0.107] |
    | truncate | −0.078 | |

  - Relevant padding is positive for 155 of 160 targets.
  - Every topic shows the same signs.
  - Gemini (first 3 topics), for comparison only: it again **penalises** relevant padding
    (−0.117), as in the original ragtime26 study.
- **5: CONFIRMED.**
  - The mirror flips **0.684%** of winners between identical requests (threshold 2%).
  - Choice vs the submitted mirror: board Spearman **0.998**, Kendall 0.980; per-topic
    median 0.992 (minimum 0.969).

Context: on the window, the submitted form's split-half reliability is 0.948. The
length relationship is again reversed relative to rag26: P(longer wins) 0.486,
per-topic Spearman(win rate, words) median −0.090. Agreement with Gemini on its 3
topics: Spearman 0.939.

