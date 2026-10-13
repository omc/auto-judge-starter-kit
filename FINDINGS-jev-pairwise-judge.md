# Pairwise Summary Judging with Jev and Gemini on TREC RAG 2026 — Findings

**Status:** working notes for the team and raw material for a TREC paper. Revised
2026-10-13.

**Data handling.** Every per-comparison result in this document comes from the
**permitted window**: the first 10 topics of each topics file, in file order.

- rag26: `rag2026-0` … `rag2026-9`.
- ragtime26: `2000` … `2009`.
- kiddie (synthetic, unrestricted), for smoke tests.

The TREC AutoJudge data policy forbids inspecting, aggregating or tuning on any other
topic. Over all topics we read only the scored leaderboards (`.eval.txt`), for
completeness checks and for reporting the leaderboard (Section 13). We did not use them
to make decisions.

- **Earlier version withdrawn.** An earlier version of this document (2026-09-10 to
  2026-10-09) drew pilots, probes and statistics from topics outside the window. It has
  been withdrawn in full.
- **Studies redone.** Every study it reported was either redone on the window under a
  plan written before any results existed (Section 4), or is marked here as not re-run.
- **Sources.** Section 16 gives the command and output file behind each table.

**Summary.** We built a pairwise "tournament" judge for TREC 2026 AutoJudge, using
TypeSafe's **Jev** decision model (`typesafe/jev-1.13`, served via OpenRouter's Decisions
API). Jev returns a *probability* that summary A is better rather than a binary choice.
We compared it with our earlier binary pairwise judge built on **Gemini**
(`google/gemini-3.5-flash-lite`). We developed on the rag26 window and confirmed once
on the ragtime26 window.

1. **Jev and Gemini rank runs alike.** Board Spearman is 0.976 on the 10 rag26 window
   topics and 0.937 on the 3 ragtime26 topics Gemini judged. Jev costs about 6× less per
   comparison.
2. **Probabilities don't change the ranking.**
   - Choice probabilities are often exactly 0 or 1: 50% on rag26 and 19% on ragtime26.
   - Probability-weighted and binary win rates rank runs alike (ρ ≥ 0.998).
3. **Choice has a strong first-slot bias on ties.**
   - Identical summaries get P(A) = 0.79 on both datasets.
   - On close real pairs the bias is weaker and depends on the dataset: slot A wins
     0.540 on rag26, but slot **B** is favoured on ragtime26 (0.465).
   - Judging both orientations and averaging removes it.
4. **A mirrored yes/no (Noul) pair handles ties correctly.**
   - Asking "is A better than B?" and "is B better than A?" in one request and
     averaging gives 0.535 / 0.539 on identical summaries.
   - It never saturates.
   - It ranks runs like Choice (ρ = 0.998 on both datasets).
   - It is near-deterministic: 0.16% / 0.68% of winners flip between identical
     requests.
5. **It rewards content, not length, in a controlled padding test.**
   - Repeated, generic and off-topic padding lower P(win) on every topic of both
     datasets.
   - Relevant padding raises it.
   - Gemini also penalises uninformative padding, but on ragtime26 it penalises added
     relevant content too.
6. **The natural length relationship is dataset-specific.** On rag26 longer reports
   win (P = 0.70 for Choice); on ragtime26 they don't (0.49).
7. **A depth-proportionate prompt adapts to the request.**
   - An explicit rule ("depth proportionate to what the query asks for: thorough when
     broad, brief and direct when narrow") judges like the plain question on the
     original broad requests.
   - When the request is narrowed to one sub-question, it moves further toward shorter
     answers than the plain question does, on **10 of 10 topics of each dataset**.
8. **The submitted form passed all five criteria written before the window study, on
   both datasets (Section 12).** The submitted form is mirrored proportionate Noul,
   both orientations.
9. **Full runs (Section 13).** All topics of both datasets: 971,304 comparisons, all
   valid, $93.84.
10. **A bag-of-decisions graded judge (Section 14)** costs $1.31 for both full datasets.
    It agrees with the pairwise judge at ρ = 0.990 on the rag26 window, and it passes
    the padding test.

**Starting hypothesis.** We began by suspecting that the Gemini judge was mainly
rewarding length. The controlled tests don't support that: both judges penalise
uninformative length, and on ragtime26 neither favours longer reports. Subtle on-topic
verbosity is still untested (Section 15.4).

---

## Contents

1. [Background and motivation](#1-background-and-motivation)
2. [Data](#2-data)
3. [Judges and method](#3-judges-and-method)
4. [Study design: development on the permitted window](#4-study-design-development-on-the-permitted-window)
5. [Choice pilot: cost, distribution, position, agreement](#5-choice-pilot-cost-distribution-position-agreement)
6. [Length dependence in natural comparisons](#6-length-dependence-in-natural-comparisons)
7. [Prompt variants](#7-prompt-variants)
8. [Identical summaries and determinism](#8-identical-summaries-and-determinism)
9. [Padding test](#9-padding-test)
10. [Mirrored Noul formulation](#10-mirrored-noul-formulation)
11. [Depth-proportionate prompt: broad vs narrow](#11-depth-proportionate-prompt-broad-vs-narrow)
12. [Pre-registered decisions and confirmation](#12-pre-registered-decisions-and-confirmation)
13. [Full runs: mirrored proportionate Noul on all topics](#13-full-runs-mirrored-proportionate-noul-on-all-topics)
14. [Bag-of-decisions graded judge (BOD)](#14-bag-of-decisions-graded-judge-bod)
15. [Discussion, threats to validity, open questions](#15-discussion)
16. [Reproduction](#16-reproduction)
17. [Appendix: prompts and questions verbatim](#17-appendix-prompts-and-questions-verbatim)

---

## 1. Background and motivation

The task is to rank TREC 2026 RAG runs **without human assessments**. Our approach is a
pairwise tournament:

- For each topic, every run's summary is compared against every other team's summary.
- The outcomes are aggregated into a per-run win rate.

The first version (September 2026) used Gemini 3.5 Flash-Lite, with a prompt that asks
for a single `A`/`B` token. The team suspected that **summary length**, rather than
quality, was driving its decisions.

Jev was attractive for two reasons:

- **Probabilistic output.** Jev answers a typed Choice question with a probability per
  option, giving a graded outcome per comparison instead of 0/1. The hope was that close
  pairs would contribute partial credit rather than coin-flip noise.
- **Cost.** Jev bills input tokens only, which makes a full tournament much cheaper.

Research questions:

- **RQ1** Do Jev and Gemini agree on run rankings?
- **RQ2** Do Jev's probabilities change the ranking relative to binary outcomes?
- **RQ3** Is either judge biased toward longer summaries independent of content?
- **RQ4** Does rewording the question change outcomes?
- **RQ5** How reliable are Jev's probabilities (determinism, position bias, calibration
  on ties)?
- **RQ6** Can the question make the judge reward depth proportionate to the need, rather
  than length in either direction?

---

## 2. Data

TREC 2026 AutoJudge test data:

| Property | rag26 (`data/rag26/runs/generation/`) | ragtime26 (`data/ragtime26/runs/repgen/`) |
| --- | --- | --- |
| Runs / teams / topics | 83 / 25 / 119 | 49 / 10 / 103 |
| Reports (non-empty) | 9,875 (9,836) | 5,047 (5,041) |
| Cross-team pairs per topic, single / ordered | about 3,200 / 6,400 | 1,024 / 2,048 |
| Full tournament, ordered (unique calls) | 760,886 (703,536) | 210,418 (196,958) |
| Topic text | `title` only: a long, multi-part question | short `title` (a label) + `problem_statement` + `background` |
| **Permitted window** | rag2026-0 … rag2026-9 | 2000 … 2009 |
| Window reports: count, median words (range) | 830, 738 (0–1,024) | 490, 562 (30–1,016) |
| Assessed topics (ground truth) | **0** | **0** |

- **Topics are broad, multi-part requests.** For example, rag2026-0 asks a nursing DEI
  council what to prioritise across five named areas, how to think about tradeoffs,
  which outcomes to measure, and which mistakes to avoid.
- **Report lengths** appear capped near 1,000 words on rag26. ragtime requests carry a
  5,000-character limit.
- **ragtime topic text.** On ragtime the title is just a label, for example
  "Anti-vaping Legislation"; the request is in `problem_statement` and `background`.
  - Jev receives these fields in its `state`.
  - Gemini has them appended to `<%=topic_query%>` (`topic_fields` / `topic_query_text`
    in `judges/bonsai_judge/pairwise.py`).
  - rag26 requests and cache keys are unchanged.
- **Comparisons are cross-team only.** A team's runs are never compared with each other.

**No ground truth.** Without assessments we cannot measure judge *accuracy*. Every
experiment instead measures agreement, sensitivity to controlled manipulations, or
internal consistency.

**The window is not a random sample.** It is fixed by the topics files' order. Results
may not transfer to the other topics (Section 15.3).

---

## 3. Judges and method

### 3.1 Gemini binary judge (baseline)

- **Model:** `google/gemini-3.5-flash-lite` via OpenRouter, temperature 0, `max_tokens`
  8.
- **Prompt:** `judges/bonsai_judge/prompts/pairwise_summary_1.md` (verbatim in
  Section 17.1). It contains the query and both summaries and asks for exactly one
  character, `A` or `B`.
- **Parsing:** only a bare `A`/`B` is accepted, after stripping non-letters. On the
  window, 32,070 of 32,070 rag26 and 3,072 of 3,072 ragtime26 comparisons parsed.
- **Orientation:** single direction. Each unordered pair is judged once, with the A/B
  orientation chosen by a deterministic hash of `(topic, run_x, run_y)`.
- **Coverage on the window:**
  - rag26: all 10 topics (workflow `workflow.pairwise.yml`, variant `dev`).
  - ragtime26: topics 2000–2002 only.
- **Cost:** about $0.00059 per comparison. The 9,281 new rag26 window calls cost $5.46.
- **Full run:** a September 2026 full Gemini run stopped at a spend limit after 78 of
  119 rag26 topics. It is not analysed here beyond the window topics it shares.

### 3.2 Jev probabilistic judge

- **Model and endpoint:** `typesafe/jev-1.13` through OpenRouter's Decisions API
  (`POST /api/alpha/decisions`). Every response reported the snapshot
  `typesafe/jev-1.13-20260917`.
- **Request shape:** Jev is not a chat model. Each request carries a `state` object
  (`{query, [problem_statement], [background], summary_a, summary_b}`) and one or more
  typed **questions**. The request shape is in Section 17.2 and the questions in 17.3.
  - **Choice** questions have criteria `A` and `B`, and return `probabilities: {A, B}`
    (2 decimals; 101 distinct values seen).
  - **Noul** (yes/no) questions return P(yes).
- **Several questions per request.** Questions are answered independently and the input
  state is billed once, so extra questions add only their own text to the cost. We use
  this to compare prompts at almost no extra cost.
- **Cost:** input tokens only.
  - On the window Choice pilot: $0.0000973 per call on rag26 (mean 2,317 input tokens)
    and $0.0000835 on ragtime26 (1,989 tokens).
- **Implementation:** `judges/bonsai_judge/pairwise_jev.py` (`BonsaiJevPairwiseJudge`),
  `workflow.pairwise_jev.yml`.
  - Development variants carry `dev_topics`, and the judge refuses to judge anything
    else.
  - `tests/test_dev_window.py` checks this, and checks that the study code never names
    a restricted topic.
- **Caching:** responses are cached on disk under
  `sha256(endpoint, model, state, questions)`.
  - Only well-formed answers are cached; failures are retried on rerun.
  - Identical payloads are asked once.

**Scoring.** In a comparison of runs a (slot A) and b (slot B), run a earns P(A better)
and run b earns 1 − P(A better).

- `PAIRWISE_WINRATE`: expected wins / games, per topic. The leaderboard is the mean over
  topics.
- `PAIRWISE_EXP_WINS`, `PAIRWISE_GAMES`.
- `PAIRWISE_CONFIDENCE`: mean |2P − 1|. Diagnostic only.
- `PAIRWISE_TIE_RATE`: Section 10.
- Empty reports score 0 without a call.

### 3.3 Abandoned variant: Jev Router

We first tried OpenRouter's `typesafe/jev-router`. It uses Jev to pick a *chat model and
reasoning effort* per request; it is not a probabilistic judge.

- In a 12-comparison kiddie smoke test it routed to two different chat models.
- Reasoning tokens were about 92% of the cost for a one-letter answer.
- We dropped it: the "judge" becomes a mixture of models that changes per request, and
  it returns no probabilities.

---

## 4. Study design: development on the permitted window

**Plan written before any results** (`temp/WINDOW_STUDY_PLAN.md`):

- **Development set:** the rag26 window. Every method decision is made here.
- **Confirmation set:** the ragtime26 window, run and read **once**, after the rag26
  decisions were recorded. A failed confirmation would be reported, not tuned away.
- **Default:** the form already used for the full runs (mirrored proportionate Noul,
  both orientations). It changes only if a criterion below fails on rag26.

| # | Decision | Criterion (written beforehand) |
| --- | --- | --- |
| 1 | Proportionate vs better wording | Proportionate's broad→narrow shift toward shorter answers, relative to `better` (difference-in-differences), is negative on ≥ 7 of 10 topics, and the pooled value's topic-bootstrap 95% CI excludes 0 |
| 2 | Ties | Identical summaries score within [0.40, 0.60] |
| 3 | Orientation | Keep both orientations unless P(slot A wins) on close pairs has a topic-bootstrap CI that includes 0.50 **and** identical summaries score within [0.45, 0.55] |
| 4 | Padding | Each uninformative padding variant has pooled ΔP < 0, with a CI excluding 0; relevant padding has ΔP > 0 |
| 5 | Choice vs Noul | Reopened only if the mirror flips > 2% of winners between identical requests, or ranks runs at Spearman < 0.95 against Choice |

**What was run.** Everything on all 10 window topics, except:

- Determinism: one topic.
- Gemini padding: the first 3 topics.
- ragtime26 Gemini: the existing 2000–2002 run.

**Statistics.** Results are given per topic and pooled. Uncertainty comes from
bootstrapping over **topics**, or from the sign pattern across topics. CIs over
comparisons are reported only where marked, because comparisons share runs and such
CIs are too narrow.

**Not re-run** (no decision depended on them; dropped to fit the budget):

- Instruction-following probes: negation, label swap, "which is shorter".
- A Noul `better` pilot in both orientations.
- Choice determinism.
- Choice-only padding.
- The "appropriate" (conciseness) prompt, removed altogether.
- Bradley–Terry analyses.

Earlier results on these came partly from restricted topics and are withdrawn. Section
15.4 lists what remains open as a result.

**Budget and spend.** The OpenRouter key had $57.21 left. A guard skipped any step that
would take the balance below its estimate plus $10.

| Step | rag26 | ragtime26 |
| --- | --- | --- |
| Prompt variants (Section 7) | $2.90 | $0.67 |
| Choice pilot, both orientations (Section 5) | $5.22 | $1.42 |
| Identical summaries (Section 8) | $0.24 | $0.11 |
| Determinism, mirror (Section 8) | $0.71 | $0.18 |
| Padding, mirror (Section 9) | $1.85 | $1.52 |
| Gemini pairwise on the window (Sections 5–7) | $5.46 | — (existing run) |
| Gemini padding, 3 topics (Section 9) | $1.91 | $0.96 |
| Broad vs narrow, 2 framings (Section 11) | $6.98 | $1.96 |
| **Total** | **$25.27** | **$6.82** |

Costs are the change in the key's balance, so cache hits cost nothing. A terminal
restart interrupted the ragtime26 Choice pilot; it resumed from cache with no call paid
twice.

---

## 5. Choice pilot: cost, distribution, position, agreement

**Setup.**

- Question `better_summary` (verbatim in 17.3).
- All 10 window topics, all runs.
- **Both orientations**: every cross-team pair is judged as (x, y) and as (y, x).
  - rag26: 64,140 comparisons, 59,332 unique calls, 52 minutes.
  - ragtime26: 20,480 comparisons, 18,920 unique calls.
- All comparisons valid. Four rag26 calls failed during a network outage and were filled
  on rerun.

### 5.1 Agreement with Gemini (RQ1) and stability

| Metric | rag26 (10 topics) | ragtime26 |
| --- | --- | --- |
| Board Spearman / Kendall vs Gemini | **0.976** / 0.869 | **0.937** / 0.811 (3 topics) |
| Per-topic Spearman vs Gemini, median | 0.957 | 0.925 |
| Split-half (5 vs 5 topics, 200 splits), Spearman | 0.975 [0.959, 0.983] | 0.941 [0.908, 0.967] |
| Top 5 | carmen, lars, edith, ariel, hana | alyssa, marie, dina, lori, alice |

Jev and Gemini rank runs alike. ragtime26 is noisier, with fewer runs, closer
competition and more varied topics.

### 5.2 Distribution of Choice probabilities (RQ2)

| Metric | rag26 | ragtime26 |
| --- | --- | --- |
| Exactly 0 or 1 | **49.8%** | 18.5% |
| Decisive (P < 0.25 or > 0.75) | 88.5% | 79.5% |
| P in [0.4, 0.6] | 4.4% | 8.0% |
| Mean P(A) | 0.506 | 0.483 |
| Probability-weighted vs binary ranking (Spearman) | **0.9992** | **0.9987** |

- **Probabilities don't change the ranking.** Choice is saturated on rag26 and less so
  on ragtime26.
- **Accuracy is a separate question.** Without ground truth we can't tell whether the
  probabilities improve accuracy on close pairs.

### 5.3 Position effects (RQ5)

Pairs are grouped by their orientation-averaged P(x better). A good judge gives
P(slot A wins) = 0.5 in every group.

| Closeness (orientation-averaged P) | rag26 pairs | rag26 P(slot A wins) / winner flips | ragtime26 pairs | ragtime26 P(slot A wins) / winner flips |
| --- | --- | --- | --- | --- |
| 0.00–0.05 | 12,604 | 0.500 / 0.00 | 2,261 | 0.497 / 0.00 |
| 0.05–0.25 | 2,993 | 0.511 / 0.00 | 1,446 | 0.478 / 0.00 |
| 0.25–0.40 | 1,188 | 0.526 / 0.16 | 663 | 0.458 / 0.09 |
| **0.40–0.60** | 1,406 | **0.540 / 0.71** | 797 | **0.465 / 0.64** |
| 0.60–0.75 | 1,176 | 0.530 / 0.17 | 679 | 0.460 / 0.08 |
| 0.75–0.95 | 2,680 | 0.514 / 0.00 | 1,723 | 0.477 / 0.00 |
| 0.95–1.00 | 10,023 | 0.500 / 0.00 | 2,671 | 0.496 / 0.00 |
| **All pairs** | 32,070 | 0.506 / 0.043 | 10,240 | 0.483 / 0.061 |

- **The position effect is concentrated in close pairs,** where the winner flips with
  the orientation in 64–71% of cases. Those are exactly the pairs where probabilities
  should add information.
- **Its direction depends on the dataset.** Slot A is favoured on rag26 (0.540), slot
  **B** on ragtime26 (0.465).
- **On ties, Choice favours slot A strongly on both datasets** (P(A) ≈ 0.79,
  Section 8). So the close-pair behaviour is not simply the tie bias.
- **Judging both orientations and averaging cancels the effect in the win rates.** A
  single-orientation run would bias close pairs, in a direction that can't be predicted
  from one dataset.

---

## 6. Length dependence in natural comparisons

Same Choice pilot, orientation-averaged. "Longer" compares report word counts.

| Metric | rag26 | ragtime26 |
| --- | --- | --- |
| P(longer wins), all pairs | **0.700** | **0.488** |
| … length ratio < 1.25× / 1.25–2× / ≥ 2× (single orientation, Section 7) | 0.530 / 0.632 / 0.865 | 0.439 / 0.502 / 0.519 |
| Per-topic Spearman(win rate, words): median [min, max] | **+0.541** [0.475, 0.664] | **−0.102** [−0.377, 0.069] |

- **rag26 rewards longer reports, mostly in lopsided pairs.** P = 0.865 when one report
  is at least twice as long.
- **ragtime26 does not.** Within topics, longer reports tend to rank slightly *lower*.
- **The effect belongs to the dataset, not the judge.** All four Choice wordings
  (Section 7) and the mirrored Noul form (Section 10) show the same pattern. On rag26's multi-part topics, a
  short report can't cover all the facets.
- **Natural correlations can't separate length from content.** The padding test
  (Section 9) does that.

---

## 7. Prompt variants

**Setup.**

- Four Choice questions **in one request**, so every comparison has all four answers
  from identical input. The questions are verbatim in 17.3:
  - `better_summary` (control)
  - `length_neutral`: judge substance not length; prefer the concise answer when equal.
  - `needs_coverage`: credit only content that answers what the user asked.
  - `precision`: judge the *proportion* of relevant, specific statements.
- All 10 window topics, single orientation (hashed).
- rag26: 32,070 comparisons; ragtime26: 10,240.

**Length dependence (RQ3) by question.**

| Question | rag26 P(longer) | rag26 ρ(win rate, words) | ragtime26 P(longer) | ragtime26 ρ |
| --- | --- | --- | --- | --- |
| better_summary | 0.701 | +0.545 | 0.489 | −0.086 |
| length_neutral | 0.669 | +0.460 | 0.450 | −0.199 |
| needs_coverage | 0.701 | +0.540 | 0.499 | −0.044 |
| precision | 0.673 | +0.474 | 0.439 | −0.232 |

ρ is the mean over topics of Spearman(run win rate, words).

**Agreement with the control and with Gemini (RQ4).**

| Question | Ranking vs control (per-topic ρ, mean): rag26 / ragtime26 | Pair decisions agreeing with control: rag26 / ragtime26 | Board ρ vs Gemini: rag26 / ragtime26 |
| --- | --- | --- | --- |
| better_summary | — | — | 0.976 / 0.935 |
| length_neutral | 0.984 / 0.957 | 95.4% / 93.4% | **0.987** / **0.955** |
| needs_coverage | 0.992 / 0.944 | 96.7% / 91.6% | 0.977 / 0.950 |
| precision | 0.982 / 0.938 | 95.1% / 91.0% | 0.977 / 0.914 |

**Test–retest.** The control asked alongside three other questions was compared with
the same oriented comparison in the single-question Choice pilot.

- rag26: mean |ΔP| 0.0050, 72.2% identical, winner agrees 99.5%.
- ragtime26: mean |ΔP| 0.0095, 50.8% identical, winner agrees 99.0%.

Asking extra questions in the same request barely changes an answer.

**Findings.**

1. **Rewording changes 3–9% of pair decisions but hardly changes the ranking.**
   Per-topic ρ against the control is ≥ 0.98 on rag26 and ≥ 0.94 on ragtime26.
2. **`length_neutral` and `precision` lower the length correlation on both datasets**
   (rag26 0.545 → 0.460 / 0.474), and `length_neutral` agrees with Gemini slightly more.
   That does not validate them: neither judge has been checked against human judgments.
3. **`needs_coverage` behaves like the control on length,** and gives the most
   saturated answers (58.9% exactly 0 or 1 on rag26).

---

## 8. Identical summaries and determinism

### 8.1 Identical summaries (RQ5)

Each run's own report against itself: 828 runs × topics on rag26, 490 on ragtime26.

| Form | rag26 | ragtime26 |
| --- | --- | --- |
| Choice `better_summary`: mean P(A) | **0.792** (per topic 0.748–0.822; A chosen 828/828) | **0.789** (A chosen 490/490) |
| Noul `noul_a_proportionate`: mean P(yes) | 0.177 | 0.184 |
| Noul `noul_b_proportionate`: mean P(yes) | 0.108 | 0.106 |
| **Mirror** (yA + 1 − yB) / 2 | **0.535** | **0.539** |
| Mirror tie flag ("no" to both) | 828/828 | 490/490 |

- **Choice has a strong first-slot bias when the two summaries are identical.**
  P(A) ≈ 0.79, never near 0.5.
- **The Noul questions answer "no" to "is X better than Y?" for identical summaries.**
  Their mirror average is close to 0.5, and every identical pair is flagged as a tie.
- **A small residual lean remains:** "is A better?" gets a higher P(yes) than "is B
  better?".

### 8.2 Determinism of the mirror

Each comparison of one topic (rag2026-3 / 2003) is asked twice with the cache bypassed.

| Metric | rag26 (3,223 comparisons) | ragtime26 (1,024) |
| --- | --- | --- |
| Identical mirror score | 39.1% | 24.0% |
| Mean / p95 / max \|ΔP\| | 0.0049 / 0.015 / 0.050 | 0.0067 / 0.020 / 0.040 |
| **Winner flips** | **5 (0.155%)** | **7 (0.684%)** |
| Tie flag pass 1 / pass 2 / changed | 43 / 38 / 17 | 17 / 15 / 6 |

- **Jev is near-deterministic.** Repeated answers differ by about 0.005 on average.
  Winners flip only for pairs already near 0.5.
- **The per-pair tie flag is unstable** (17 of 43 flags change on rag26), so it should
  be used in aggregate only.
- **Choice determinism was not re-run on the window** (Section 4).

---

## 9. Padding test

**Question.** Does the judge reward length itself, or content? Natural correlations
(Section 6) can't tell, so we manipulated reports directly.

### 9.1 Design

- **Targets.** On each topic, 16 mid-ranked runs, ranked by the prompt-variant control
  win rate (0.25 ≤ win rate ≤ 0.75).
- **Opponents.** 8 fixed cross-team opponents spread over the whole ranking.
- **Variants of each target** (padding sentences interleaved at seeded random positions,
  not appended):

| Variant | Change | Information |
| --- | --- | --- |
| original | unchanged | — |
| pad_repeat | +≈50% words: the report's own sentences duplicated | none |
| pad_generic | +≈50% words: topic-agnostic filler (Section 17.4) | none |
| pad_offtopic | +≈50% words: sentences from another window topic (topic i+5) | irrelevant |
| pad_relevant | +≈50% words: unseen sentences from the topic's strongest run of another team | positive control |
| truncate | last third of sentences removed | less, and shorter |

- **Judging.** Each target variant against each opponent, in **both orientations**.
- **Measure.** ΔP = the mean change in orientation-averaged P(target beats opponent),
  relative to the original, bootstrapped over targets.
- **Judges.**
  - Jev in the **submitted form** (mirrored proportionate Noul): all 10 topics,
    160 targets, 15,360 calls.
  - Gemini: the first 3 topics, 48 targets, 4,608 calls.

### 9.2 Results: submitted form (mirrored proportionate Noul), all 10 topics

| Variant | Words × | rag26 ΔP [95% CI] | rag26 targets ΔP > 0 | ragtime26 ΔP [95% CI] | ragtime26 targets ΔP > 0 |
| --- | --- | --- | --- | --- | --- |
| pad_repeat | 1.53 | −0.048 [−0.053, −0.044] | 1/160 | −0.074 [−0.080, −0.069] | 1/160 |
| pad_generic | 1.51 | −0.095 [−0.102, −0.088] | 0/160 | −0.135 [−0.144, −0.125] | 1/160 |
| pad_offtopic | 1.53 | −0.196 [−0.209, −0.183] | 0/160 | −0.258 [−0.271, −0.245] | 0/160 |
| **pad_relevant** | 1.53 | **+0.179** [+0.170, +0.188] | 160/160 | **+0.098** [+0.088, +0.107] | 155/160 |
| truncate | 0.67 | −0.081 [−0.088, −0.072] | 2/160 | −0.078 [−0.088, −0.068] | 16/160 |

Baseline P(win) for the originals: 0.333 on rag26, 0.422 on ragtime26.

**Every topic of both datasets shows the same signs:**

- the three uninformative variants are negative;
- pad_relevant is positive;
- truncation is negative.

Other observations:

- Near-tie flags: 124 of 15,360 on rag26 and 432 on ragtime26.
- Mean P(slot A wins): 0.526 and 0.514.

### 9.3 Gemini vs Jev on the same 3 topics

| Variant | rag26: Gemini ΔP [95% CI] | rag26: Jev mirror ΔP | ragtime26: Gemini ΔP [95% CI] | ragtime26: Jev mirror ΔP |
| --- | --- | --- | --- | --- |
| pad_repeat | −0.132 [−0.160, −0.104] | −0.040 | −0.203 [−0.249, −0.160] | −0.091 |
| pad_generic | −0.104 [−0.130, −0.079] | −0.075 | −0.186 [−0.224, −0.150] | −0.163 |
| pad_offtopic | −0.206 [−0.250, −0.165] | −0.189 | −0.309 [−0.368, −0.250] | −0.265 |
| pad_relevant | **+0.104** [+0.070, +0.138] (36/48 > 0) | +0.182 (48/48) | **−0.117** [−0.151, −0.085] (3/48 > 0) | +0.070 (46/48) |
| truncate | −0.078 [−0.107, −0.049] | −0.077 | −0.152 [−0.195, −0.113] | −0.105 |

Gemini's mean P(slot A wins) is 0.479 on rag26 and 0.466 on ragtime26.

### 9.4 Findings (RQ3)

1. **Neither judge rewards crude, uninformative length.** All three padding types lower
   P(win), for every target-averaged topic, on both datasets.
2. **Jev credits added relevant content on both datasets.** It does so on every one of
   the 10 topics, and for 315 of 320 targets.
3. **Gemini penalises any added text on ragtime26, including relevant text.** The
   splice may read as disruption to Gemini, or Gemini may not credit coverage on
   ragtime's shorter reports; this test can't tell which (Section 15.4).
4. **Gemini penalises repetition more than Jev does.** Jev mostly ignores a repeated
   sentence, and penalises generic and off-topic text strongly.

**Not tested:** subtle, on-topic verbosity, meaning restatement and hedging without new
facts (Section 15.4).

---

## 10. Mirrored Noul formulation

**Motivation.** Choice's tie bias (Section 8.1) means a Choice probability is not a
calibrated P(A better). The Noul primitive answers yes/no questions independently, so
the pair "is A better than B?" / "is B better than A?" can be asked in one request.

**Scoring.** With yA = P(yes | "is A better?") and yB = P(yes | "is B better?"):

- mirror = (yA + 1 − yB) / 2.
- "No" to both (yA, yB < 0.5) flags a **near-tie**.
- Identical text is scored 0.5 without a call.

`PAIRWISE_TIE_RATE` reports flagged ties. The submitted form uses the proportionate pair
(Section 11); its data on the window comes from the full runs (Section 13), which judged
every pair in both orientations.

| Metric (window, both orientations) | rag26 Choice | rag26 mirror | ragtime26 Choice | ragtime26 mirror |
| --- | --- | --- | --- | --- |
| Exactly 0 or 1 | 49.8% | **0.0%** | 18.5% | **0.0%** |
| Decisive (< 0.25 or > 0.75) | 88.5% | 77.2% | 79.5% | 58.7% |
| In [0.4, 0.6] | 4.4% | 8.1% | 8.0% | 13.4% |
| Distinct values | 101 | 406 | 101 | 373 |
| Identical summaries (Section 8.1) | 0.792 | **0.535** | 0.789 | **0.539** |
| P(slot A wins), close pairs | 0.540 | 0.553 | 0.465 | 0.523 |
| Winner flips on swap, all / close pairs | 4.3% / 71% | 4.8% / 57% | 6.1% / 64% | 5.0% / 37% |
| Probability-weighted vs binary ranking | 0.9992 | 0.9982 | 0.9987 | 0.9978 |
| Board ρ vs Gemini | 0.976 | 0.974 | 0.937 | 0.939 |
| P(longer wins) | 0.700 | 0.675 | 0.488 | 0.486 |
| Flagged near-ties / both "yes" | — | 0.84% / 1.95% | — | 2.80% / 1.65% |

**Ranking, Choice vs mirror** (board over the window, both orientations):

- rag26: ρ **0.998**, Kendall 0.975 (per-topic median 0.997).
- ragtime26: ρ **0.998**, Kendall 0.980 (per-topic median 0.992, min 0.969).

**Findings.**

1. **The mirror is correct on ties and never saturates.** Its probabilities are graded
   (about 400 distinct values against Choice's 101).
2. **It ranks runs like Choice** (ρ = 0.998 on both datasets).
3. **It does not remove close-pair slot effects.**
   - It keeps a slot-A lean on both datasets: 0.553 on rag26, 0.523 on ragtime26.
   - On ragtime26 Choice leans to slot B.
   - So **both orientations are still needed** (decision 3, Section 12).
   - With both orientations averaged, fewer close pairs flip with the mirror than with
     Choice on both datasets.
4. **It is near-deterministic** (0.16% / 0.68% winner flips, Section 8.2).

---

## 11. Depth-proportionate prompt: broad vs narrow

**Motivation.** Some needs require long answers; others need a short, direct one. A
good judge should reward depth that is *proportionate* to the need, rather than always
preferring longer or shorter answers.

**Questions** (verbatim in 17.3):

- **`noul_*_better`**: "Does summary_a satisfy the information need … better than
  summary_b does, considering relevance, completeness, and how well its claims are
  grounded?"
- **`noul_*_proportionate`**: the same, plus "…**depth proportionate to what the query
  asks for (thorough when the query is broad or multi-part, brief and direct when it is
  narrow)**…".

**Design: a controlled change to the request.**

- Neither dataset has genuinely narrow needs.
- The *same* report pairs are judged under two framings:
  - **Broad:** the original topic.
  - **Narrow:** one sub-question written from that topic's own text, with no problem
    statement or background. All 20 are listed in Section 17.6.
- Both mirrored pairs (4 questions) are asked in each request, single orientation
  (hashed, identical across framings), on all 10 window topics.
- **Measure:** the change in P(longer wins) from broad to narrow. The key number is the
  **difference-in-differences (DiD) against `better`** on the same comparisons.
  Negative means proportionate moves toward shorter answers *more* than `better` does
  when the request is narrowed, which is the intended behaviour.

| Topic | rag26: better shift | rag26: proportionate shift | **rag26 DiD** | ragtime26 topic | ragtime26: better shift | ragtime26: proportionate shift | **ragtime26 DiD** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rag2026-0 | −0.008 | −0.024 | **−0.015** | 2000 | −0.056 | −0.064 | **−0.008** |
| rag2026-1 | −0.043 | −0.076 | **−0.033** | 2001 | −0.009 | −0.036 | **−0.027** |
| rag2026-2 | −0.054 | −0.090 | **−0.036** | 2002 | −0.034 | −0.049 | **−0.015** |
| rag2026-3 | −0.013 | −0.066 | **−0.053** | 2003 | −0.001 | −0.017 | **−0.016** |
| rag2026-4 | −0.006 | −0.023 | **−0.017** | 2004 | +0.001 | −0.010 | **−0.011** |
| rag2026-5 | −0.020 | −0.044 | **−0.023** | 2005 | −0.017 | −0.042 | **−0.025** |
| rag2026-6 | −0.058 | −0.093 | **−0.035** | 2006 | −0.007 | −0.022 | **−0.015** |
| rag2026-7 | −0.037 | −0.068 | **−0.030** | 2007 | −0.013 | −0.022 | **−0.010** |
| rag2026-8 | −0.047 | −0.088 | **−0.041** | 2008 | +0.026 | +0.015 | **−0.011** |
| rag2026-9 | −0.034 | −0.068 | **−0.034** | 2009 | +0.011 | −0.010 | **−0.020** |
| **Pooled** | 0.672 → 0.640 (−0.032) | 0.676 → 0.612 (−0.064) | **−0.032** | | 0.486 → 0.477 (−0.010) | 0.487 → 0.461 (−0.026) | **−0.016** |

| Pooled metric | rag26: better / proportionate | ragtime26: better / proportionate |
| --- | --- | --- |
| ρ(win rate, words), broad → narrow | 0.557 → 0.499 / **0.563 → 0.420** | −0.086 → −0.116 / **−0.079 → −0.180** |
| Run ranking, broad vs narrow | 0.944 / 0.903 | 0.889 / 0.873 |
| Tie flag, broad → narrow | 0.7 → 2.0% / 0.8 → 3.8% | 2.8 → 1.7% / 2.7 → 2.9% |
| Winner differs from `better`: broad / narrow | — / 0.96%, 3.57% | — / 1.53%, 3.21% |

**Findings.**

1. **Proportionate adapts to the request.** The DiD is negative on **10 of 10** topics
   of rag26 (−0.015 to −0.053) and **10 of 10** of ragtime26 (−0.008 to −0.027). A
   bootstrap over topics therefore excludes 0 on both datasets.
2. **On broad requests it behaves like `better`.** The winner differs in 1–1.5% of
   comparisons, and pooled P(longer) is within 0.005.
3. **`better` also responds to a narrower request.** The explicit proportionality
   wording roughly doubles (rag26) or nearly triples (ragtime26) that response.
4. **The effect is modest:** 1–5 points of P(longer). It is smaller on ragtime26, whose
   reports are shorter to begin with.

**Caveat.** The narrow requests are **synthetic**: the reports were written for the
broad request. This shows the judge responds to the request, not how it performs on
real short-answer needs.

---

## 12. Pre-registered decisions and confirmation

The decisions were recorded in `temp/WINDOW_STUDY_PLAN.md` before anything was run or
read on ragtime26.

| # | Criterion | rag26 (decision) | ragtime26 (confirmation) |
| --- | --- | --- | --- |
| 1 | Proportionate DiD < 0 on ≥ 7/10 topics, CI excludes 0 | **Pass**: 10/10, pooled −0.032 | **Confirmed**: 10/10, pooled −0.016 |
| 2 | Identical summaries within [0.40, 0.60] | **Pass**: 0.535 | **Confirmed**: 0.539 |
| 3 | Drop one orientation only if close-pair slot bias is non-significant and ties are within [0.45, 0.55] | **Keep both**: 0.553 [0.547, 0.559] | **Confirmed**: 0.523 [0.514, 0.532] |
| 4 | Uninformative padding ΔP < 0 (CI excludes 0); relevant ΔP > 0 | **Pass** (Section 9.2) | **Confirmed** |
| 5 | Reopen Choice only if mirror flips > 2% or ρ vs Choice < 0.95 | **Not reopened**: 0.155%, ρ 0.998 | **Confirmed**: 0.684%, ρ 0.998 |

The submitted form (mirrored proportionate Noul, both orientations) is kept. The full
runs did not need redoing.

The decision-3 CIs are topic-bootstrap 95% intervals on P(slot A wins) for close pairs
(orientation-averaged P in 0.4–0.6) in the full runs' window comparisons. The bias is
present on every topic: 0.537–0.572 on rag26 and 0.503–0.549 on ragtime26.

---

## 13. Full runs: mirrored proportionate Noul on all topics

**Setup.**

- Workflow variant `full_noul_proportionate`:
  - questions `noul_a_proportionate` + `noul_b_proportionate`;
  - mirror average as the score;
  - **both orientations**;
  - every topic.
- ragtime26 ran first, then rag26 (`temp/run_full_proportionate.sh`, under nohup).
- Snapshot served throughout: `typesafe/jev-1.13-20260917`.

| | ragtime26 | rag26 |
| --- | --- | --- |
| Topics / runs / teams | 103 / 49 / 10 | 119 / 83 / 25 |
| Comparisons (ordered) | 210,418, **all valid**, 0 missing | 760,886, **all valid**, 0 missing |
| Unique calls | 196,958 | 703,536 |
| Cost | **$18.95** ($0.0000962/call) | **$74.89** ($0.0001065/call) |
| Wall-clock | 3.0 h (about 18 calls/s) | 11.1 h (about 18 calls/s) |

**Scoring and submission.**

- The leaderboard is the **mean over topics of each run's per-topic mirror win rate**;
  every topic weighs equally.
- Empty reports get explicit zero rows, so that every run has every topic.
- Submission leaderboards:
  `submissions/bonsai-jev-pairwise-noul-mirror-proportionate-20261009/{rag26,ragtime26}/`.
  Both pass TIRA's dry-run validation.

### 13.1 Leaderboards (scored outputs, all topics)

The rows below are read from the submitted `.eval.txt` "all" rows. They are reported
as results; nothing was tuned on them.

**rag26 (top 12 of 83).**

| Rank | Run | Team | Win rate |
| --- | --- | --- | --- |
| 1 | carmen | T569 | 0.8611 |
| 2 | lars | T569 | 0.8564 |
| 3 | edith | T569 | 0.8456 |
| 4 | ariel | T569 | 0.8453 |
| 5 | hana | T300 | 0.8429 |
| 6 | xavi | T300 | 0.8397 |
| 7 | luca | T300 | 0.8295 |
| 8 | yara | T649 | 0.8216 |
| 9 | todd | T736 | 0.8047 |
| 10 | eve | T341 | 0.8029 |
| 11 | rita | T736 | 0.8024 |
| 12 | eden | T736 | 0.8017 |

- Bottom: carlo 0.084, annie 0.048, paula 0.044.
- Teams by best run: T569 0.861, T300 0.843, T649 0.822, T736 0.805, T341 0.803,
  T267 0.794, T410 0.744, T086 0.720, T448 0.707, T149 0.697.

**ragtime26 (top 12 of 49).**

| Rank | Run | Team | Win rate |
| --- | --- | --- | --- |
| 1 | marie | T958 | 0.8093 |
| 2 | alice | T958 | 0.8083 |
| 3 | lori | T958 | 0.7956 |
| 4 | alyssa | T958 | 0.7950 |
| 5 | briar | T289 | 0.7127 |
| 6 | kent | T289 | 0.7112 |
| 7 | bree | T706 | 0.7065 |
| 8 | dina | T846 | 0.7035 |
| 9 | diana | T706 | 0.7021 |
| 10 | susan | T706 | 0.6904 |
| 11 | joyce | T706 | 0.6882 |
| 12 | kurt | T289 | 0.6770 |

- Bottom: hans 0.171, faye 0.110, adair 0.091.
- Teams by best run: T958 0.809, T289 0.713, T706 0.707, T846 0.704, T673 0.587,
  T149 0.575, T768 0.545, T986 0.522, T131 0.497, T047 0.470.

### 13.2 Diagnostics on the window

Computed from the full runs' comparisons on the 10 window topics of each dataset; the
details are in Sections 5, 6, 10 and 12.

| Metric | rag26 | ragtime26 |
| --- | --- | --- |
| Comparisons on the window | 64,140 | 20,480 |
| Split-half reliability (5 vs 5 topics) | 0.974 [0.960, 0.982] | 0.948 [0.915, 0.970] |
| Board ρ / Kendall vs Gemini | 0.974 / 0.866 (10 topics) | 0.939 / 0.818 (3 topics) |
| P(longer wins); per-topic ρ(win rate, words), median | 0.675; +0.558 | 0.486; −0.090 |
| P(slot A wins): all pairs / close pairs | 0.520 / 0.553 | 0.511 / 0.523 |
| Winner flips on swap | 4.8% | 5.0% |
| Decisive / in [0.4, 0.6] | 77.2% / 8.1% | 58.7% / 13.4% |
| Flagged near-ties / both "yes" | 0.84% / 1.95% | 2.80% / 1.65% |

**Findings.**

- **The full-run form behaves on the window as the development experiments predicted.**
- **The length relationship is dataset-specific.**
- **Judging both orientations was necessary** to cancel the close-pair lean.

**Caveats.** There are no human judgments yet, and the Gemini comparison covers 10 + 3
topics.

**Outputs** (excluded from git):

- `output-ragtime26-jev-full-prop/` and `output-pairwise-jev-full-prop/`.
- Window diagnostics: `temp/jev_full_analysis.py` and `temp/jev_win_report.py`; results
  in `temp/win_{rag26,ragtime26}_full_window_diag.txt` and `temp/win_*_report.txt`.

---

## 14. Bag-of-decisions graded judge (BOD)

**Data handling.** As in the pairwise study (Section 4), every development decision for
this judge was made only on the **permitted window**: the first 10 topics of each topics file
(rag26 `rag2026-0…9`, ragtime26 `2000…2009`), plus kiddie. Over all topics we
read only the scored leaderboards (`.eval.txt`), for completeness and reporting
(Sections 14.6–14.7).

**Idea.** The method adapts the "bag of decisions" reranker [1]. There, an LLM writes 10
yes/no criteria per query, Jev answers all of them about a document in one request, and
the score is the sum of P(yes). We change three things:

- **Number of questions:** it varies by topic.
- **Scale:** each question is graded 0–3, not yes/no.
- **Scoring:** the per-question grades are stored and the leaderboard is tallied from
  them afterwards.

Every answer is scored on its own against the topic's rubric, so a topic needs O(runs)
calls instead of O(runs²) as in the tournament.

[1] https://softwaredoug.com/blog/2026/10/08/bag-of-decisions. The post does not name the
question-generation model or give costs. Its only evaluation is an NDCG gain over BM25 on
two e-commerce datasets.

### 14.1 Design

The code is `judges/bonsai_judge/bod.py`, run with `workflow.bod.yml`.

1. **Rubric (nugget creator).**
   - Gemini 2.5 Flash Lite (temperature 0) writes 5–15 "Does the answer …?" questions per
     topic, sized to how many distinct parts the information need has. The prompt is
     verbatim in Section 17.5.
   - It sees only the topic (query, problem statement, background), never any response
     (`nugget_depends_on_responses: false`). The rubric is therefore independent of the
     runs, and readable for every topic under the data policy.
   - Near-duplicates are removed (token Jaccard ≥ 0.8).
   - The rubric is saved as the run's nugget banks and reused unchanged by every variant.
2. **Grading.**
   - One Jev Decisions request per (run, topic). The state is the topic fields plus the
     answer.
   - Each rubric question is a **4-option Choice with anchored grades**:
     - 0: not addressed
     - 1: mentioned without substance
     - 2: partially answered
     - 3: fully answered with specific, accurate detail
   - The same request carries one Noul **padding** question: "does the answer contain
     substantial off-topic, repeated or generic filler?"
   - Jev bills input tokens, which are dominated by the state, so the number of questions
     barely changes cost.
3. **Scoring.**
   - Each question's score is the **expected grade** Σ_g P(g)·g / 3, using the
     probabilities rather than the most likely grade.
   - Grades are written to `<filebase>.bod/grades.jsonl`, one row per question.
   - The leaderboard is tallied from those rows with no further calls.

| Measure | Definition |
| --- | --- |
| `BOD_MEAN` (primary) | mean expected grade over the topic's questions, /3; the overall score is the mean over topics, so topics with more questions don't weigh more |
| `BOD_COVERAGE` | fraction of questions with P(grade ≥ 2) > 0.5 |
| `BOD_FULL` | fraction of questions with P(grade = 3) > 0.5 |
| `BOD_PADDING` | P(yes) on the padding question; diagnostic only, not part of `BOD_MEAN` |
| `BOD_QUESTIONS` | number of rubric questions |

Empty reports score 0 without a call, as in the other judges.

### 14.2 Smoke test and kiddie

**Does Jev accept a graded Choice?** Yes. We ran the kiddie topic `leaf` with 5
hand-written questions (`temp/bod_smoke.py`, 4 calls). A Choice with labels `"0"…"3"`
returns `probabilities` over all four labels, plus `choice` and `confidence`.

| Case | BOD_MEAN | Padding P(yes) |
| --- | --- | --- |
| On-topic answer (run1/leaf) | 0.643 | 0.06 |
| Off-topic answer (run1/cloud, graded on the leaf rubric) | 0.000 (every question P(0) = 1.00) | 0.98 |
| Same as the first row, grades listed 3→0 | 0.642 (per question \|Δ\| ≤ 0.03) | 0.06 |
| Shorter answer (run2/leaf) | 0.261 | 0.09 |

The grades match the texts. For example, run2 partly explains the colour change
(E[g] 2.08), mentions evergreens (1.78), and says nothing about whether the leaves are
dying, whether the tree is fine, or the cold (≈ 0 each).

**First rubric prompt.** The first version told the generator to include "constraints or
context stated in the need (audience, situation, goals)". On kiddie that produced
persona questions, for example "Does the answer acknowledge the child's interest in
sweet foods and insects?" and "…maintain a gentle and reassuring tone…".

We tightened the prompt: background may only decide *what content* is needed, and no
questions about acknowledging interests, tone or reading level. Afterwards:

- The kiddie rubrics are all content questions (7–9 per topic instead of 8–11).
- The kiddie ranking is unchanged.
- This prompt was settled on kiddie only, before any rag26 or ragtime26 data.

**End to end on kiddie.** 20 answers, 20/20 graded, $0.0015. Against the synthetic
kiddie truth, Kendall τ = 1.0 for `BOD_MEAN` and `BOD_COVERAGE`, and −1.0 for
`BOD_PADDING` (run4 is an off-topic hockey text, padding 0.98). Kiddie truth is
synthetic, so this checks the pipeline, not quality.

### 14.3 rag26 window pilot

- Variant `pilot`: rubrics are generated for all 119 topics, from topic text only (the
  runner requires a rubric for every topic). Answers are judged on the 10 window topics
  only.
- 83 runs × 10 topics = 828 answers, 797 new calls, **$0.09**, 8 seconds.
- Analysis: `temp/bod_pilot_analysis.py`, which reads per-answer data on window topics
  only.

| | rag26 window |
| --- | --- |
| Questions per topic | 5–13 (mean 9.0) |
| BOD_MEAN p10 / p50 / p90 | 0.420 / 0.756 / 0.983 |
| Most likely grade 0 / 1 / 2 / 3 | 8.2% / 9.2% / 40.3% / 42.3% |
| Max grade probability > 0.9 | 39.8% of questions |
| \|E[g] − most likely grade\| > 0.25 | 26.9% of questions |
| Split-half (5 vs 5 topics, 200 splits), Spearman | 0.969 [0.956, 0.979] |
| Per-topic Spearman(BOD_MEAN, words), median | 0.518 [min 0.396, max 0.724] |
| Same for the Jev pairwise (mirror) full run | 0.558 |
| Same for Gemini (7 window topics) | 0.451 |
| Mean padding P(yes); Spearman(padding, words) median | 0.426; −0.182 |
| Agreement with Jev pairwise (mirror), board Spearman / Kendall | **0.990** / 0.919 (per-topic median 0.968) |
| Agreement with Gemini (7 window topics), board Spearman / Kendall | 0.972 / 0.859 (per-topic median 0.938) |

- **Agreement:** on the window, BOD ranks runs almost exactly like the submitted
  pairwise judge, at about 1/1000 of its cost per topic.
- **Probabilities:** the expected grade differs meaningfully from the most likely grade
  on about a quarter of questions, so the probabilities carry information. That is
  unlike the pairwise Choice judge, where they barely changed the ranking (Section 5.2).
- **Ceiling:** grade 3 is easy to reach on rag26's title-only topics, which compresses
  the top of the board.

### 14.4 Padding test (rag26 window)

`temp/bod_padding.py` reuses the Section 9 variants: the same generic pool, off-topic
sentences from window topic i+5, relevant sentences from the topic's top BOD run of
another team, and seeded interleaving. Each variant is graded on its own against the
topic's rubric.

- Targets: 16 interquartile runs per topic × 10 topics = 160 targets × 6 variants.
- 799 new calls, $0.11.
- The pass criteria were written into the script before the run:
  - `BOD_MEAN` should not increase for repeated, generic or off-topic padding
    (CI upper bound ≤ +0.01).
  - It should increase for relevant padding and decrease for truncation.
  - `BOD_PADDING` should increase for the three uninformative variants.

| Variant | Words × | ΔBOD_MEAN | 95% CI | ΔPADDING | 95% CI |
| --- | --- | --- | --- | --- | --- |
| pad_repeat | 1.52 | −0.023 | [−0.025, −0.020] | +0.510 | [+0.474, +0.544] |
| pad_generic | 1.51 | −0.032 | [−0.035, −0.028] | +0.537 | [+0.499, +0.574] |
| pad_offtopic | 1.53 | −0.043 | [−0.047, −0.039] | +0.574 | [+0.535, +0.612] |
| pad_relevant | 1.54 | **+0.132** | [+0.121, +0.144] | +0.038 | [+0.021, +0.055] |
| truncate | 0.67 | **−0.103** | [−0.114, −0.092] | −0.044 | [−0.061, −0.027] |

(Baseline: BOD_MEAN 0.761, padding 0.398. CIs are bootstraps over targets.)

**Every criterion passes, on each of the 10 topics individually.**

- The rubric score does not reward length: uninformative padding lowers it slightly, and
  only new relevant content raises it.
- The padding question strongly detects synthetic filler.
- Its baseline on real answers (≈ 0.40) is unexplained, which is why it stays out of
  `BOD_MEAN`.

### 14.5 ragtime26 window pilot

- 49 runs × 10 topics = 490 answers, 470 new calls, **$0.05**.
- **Rubrics:** 6–10 questions per topic, and the background is used as intended. For
  example, the VA-practitioner ketamine topic (2003) gets questions on protocols,
  supervision, contraindications and patient selection, and none about acknowledging the
  user.
- **Overlap:** some redundancy remains, for example separate questions on "life
  satisfaction" and "happiness" for 2006.

| | ragtime26 window |
| --- | --- |
| BOD_MEAN p10 / p50 / p90 | 0.514 / 0.710 / 0.894 |
| Most likely grade 0 / 1 / 2 / 3 | 8.4% / 7.8% / 46.4% / 37.5% |
| \|E[g] − most likely grade\| > 0.25 | 37.7% of questions |
| Split-half (5 vs 5), Spearman | 0.854 [0.789, 0.914] |
| Per-topic Spearman(BOD_MEAN, words), median | **−0.120** [min −0.439, max 0.306] |
| Per-topic Spearman(PADDING, words), median | +0.491 |

- **Length:** as with the pairwise judges (Section 6), the length relationship is
  dataset-specific. On ragtime26 longer answers do not score higher, and they are more
  often flagged as padded.
- **Stability:** split-half reliability is lower than on rag26, with fewer runs and more
  heterogeneous topics.
- **Not computed:** agreement with the pairwise judges on the ragtime26 window. That
  window is the confirmation set of the window-only pairwise re-study
  (`temp/WINDOW_STUDY_PLAN.md`), to be read once after the rag26 decisions are recorded.

### 14.6 Full runs

Variant `full`, run with `temp/run_bod_full.sh`; snapshot `typesafe/jev-1.13-20260917`.

| | ragtime26 | rag26 |
| --- | --- | --- |
| Topics / runs | 103 / 49 | 119 / 83 |
| Rubric questions per topic | 5–11 (mean 7.7) | 5–15 (mean 8.4) |
| Answers graded | **5,041 / 5,041** | **9,838 / 9,838** |
| New Jev calls (the rest cached from the pilots) | 4,398 | 8,670 |
| Wall-clock | 3.0 min | 8.5 min |

- **Cost:** the key's usage rose by **$1.31** for both runs together, including rubric
  generation. The per-answer cost is ≈ $0.0001, the same as one pairwise call, but
  there are O(runs) answers per topic instead of O(runs²) comparisons.
- **Completeness:** both leaderboards have every run × topic × measure.
- **TIRA:** both pass `tira-cli upload --dry-run`. They are in
  `submissions/bonsai-bod-jev-graded-20261009/{rag26,ragtime26}/`.

### 14.7 Full leaderboards vs the pairwise submission (reporting only)

This compares the two submitted leaderboards' "all" rows. It was produced for
information, after both submissions were fixed, and was not used to choose or rank
submissions.

| | rag26 | ragtime26 |
| --- | --- | --- |
| Spearman / Kendall (BOD_MEAN vs PAIRWISE_WINRATE) | 0.991 / 0.929 | 0.943 / 0.827 |
| Median \|rank change\| | 1 | 2 |
| Runs moving ≥ 5 / ≥ 10 places | 13 / 2 | 10 / 2 |
| Largest move | 11 | 21 |
| Shared top-10 / top-5 | 8 / 4 | 8 / 4 |
| Same #1 | no | no |

- **rag26:** the full-run agreement matches the window pilot (0.990).
- **ragtime26:** the judges diverge more. One candidate explanation is the
  length-dependence difference in 14.5. That is a hypothesis, untested, and without
  assessments neither leaderboard can be called more accurate.

### 14.8 Findings and caveats

1. **Jev supports graded rubric judging directly.** A 4-option anchored Choice gives
   usable, order-invariant grade distributions.
2. **It is cheap.** Both full datasets cost $1.31, against $93.84 for the mirrored
   pairwise tournament (Section 13).
3. **It agrees with the pairwise judge on rag26** (ρ ≈ 0.99) **but not as closely on
   ragtime26** (ρ 0.94). Two near-identical submissions add little independent
   information on rag26.
4. **It does not reward uninformative length**, and it rewards added relevant content
   (14.4).
5. **Caveats:**
   - The rubric is generated from the topic alone, so it cannot credit useful content
     it did not anticipate.
   - Rubrics vary in granularity: some needs are split into near-duplicate questions.
   - Grades saturate near 3 on rag26, compressing the top of the board.
   - The padding diagnostic's baseline on real answers is high and unexplained.
   - The rubric prompt and grade anchors were fixed on kiddie and not tuned further.
   - All evaluation of quality awaits the official assessments.

---

## 15. Discussion

### 15.1 Answers to the research questions

All answers below come from the permitted window only.

| RQ | Answer |
| --- | --- |
| RQ1 Jev vs Gemini agreement | **High.** Board Spearman 0.976 on the 10 rag26 window topics and 0.937 on ragtime26's 3 (Choice); 0.974 / 0.939 for the submitted mirror form. |
| RQ2 Do probabilities matter? | **Not for ranking.** Probability-weighted and binary win rates have ρ ≥ 0.998 for both Choice (19–50% saturated) and the mirror (never saturated). Without ground truth we can't say whether probabilities improve accuracy. |
| RQ3 Length bias | **Not for crude padding.** Both judges penalise repeated, generic and off-topic additions on every topic (Section 9). The natural length correlation is dataset-specific: about +0.54 on rag26 and about −0.10 on ragtime26. Subtle on-topic verbosity is untested. |
| RQ4 Prompt sensitivity | **Wording matters little for ranking.** Rewording changes 3–9% of pair decisions; ranking ρ is ≥ 0.98 (rag26) and ≥ 0.94 (ragtime26) against the control. Instruction-following probes (negation, label swap) were not re-run on the window, so instruction following is not established here. |
| RQ5 Reliability | **Mixed.** The mirror is near-deterministic (0.16% / 0.68% winner flips). Choice is strongly slot-A biased on ties (P(A) ≈ 0.79); the mirror is not (0.535 / 0.539). Close-pair position effects remain in both forms, with a dataset-dependent direction for Choice, so both orientations are needed. Split-half reliability on the window is 0.94–0.98. |
| RQ6 Depth proportionality | **Yes, modestly.** The proportionate wording matches `better` on broad requests, and moves 1–5 points further toward shorter answers when the request is narrowed, on 20 of 20 topics (Section 11). This is shown with synthetic narrow requests only. |

### 15.2 Recommended protocol for a Jev pairwise run

1. **Judge both orientations and average.**
   - Full tournament: 760,886 (rag26) and 210,418 (ragtime26) ordered comparisons.
   - That is 703,536 and 196,958 unique calls.
2. **Use the mirrored proportionate Noul pair.**
   - It is correct on ties and never saturated.
   - It ranks like Choice.
   - It responds to how narrow the request is.
3. **Leaderboard:** the mean over topics of per-topic win rates.
   - Report `PAIRWISE_TIE_RATE` as a diagnostic.
   - Use the tie flag in aggregate only, because per pair it is unstable.
4. **Pin `typesafe/jev-1.13` and record the served snapshot.** Cache everything;
   reproducibility comes from the cache.
5. **Cost.** At about $0.0001 per call, a full run costs about $75 (rag26) and $19
   (ragtime26), measured. Check the key's spend limit first, and budget about 10%
   headroom.
6. **Develop only on permitted topics.**
   - Give development variants `dev_topics`.
   - Read per-comparison outputs through a window filter.
   - Write decision criteria before looking at results.

### 15.3 Threats to validity

- **No ground truth.** All conclusions are about agreement, consistency and controlled
  sensitivity, not accuracy. Two judges agreeing can share a bias.
- **The window is small and not random.** It is 10 topics per dataset, fixed by file
  order.
  - rag26's window topics are the dataset's first ten requests.
  - Effects with a dataset-dependent sign (length, close-pair slot bias) may also vary
    across topics beyond the window.
- **Gemini comparisons are thin on ragtime26:** 3 topics, single orientation, binary.
- **Synthetic narrow requests (Section 11).** The reports were written for the broad
  request.
- **Padding limits.**
  - Targets are mid-ranked runs only.
  - Verbatim repeats, a 20-sentence generic pool and another topic's sentences may be
    easier to detect than real verbosity.
  - pad_relevant adds both information and text from the strongest run. Its direction
    is clear; its size does not isolate "more information".
- **Model drift.** Jev 1.13 (snapshot 20260917) and the Gemini slug may change
  server-side. Results are reproducible from the cache, not necessarily from fresh calls
  later.
- **Uncalibrated probabilities.** Choice's P is not a win probability (tie bias,
  Section 8.1).
- **Withdrawn earlier study.** Parts of the method were first chosen in an earlier study
  that used restricted topics.
  - Those choices: the proportionate wording, mirrored Noul, both orientations.
  - The window study re-tested each of them against criteria written beforehand.
  - Each passed on rag26 and was confirmed on ragtime26.
  - The judges' development history is still not independent of that earlier work.

### 15.4 Open questions and next experiments

1. **On-topic verbosity padding.** Have an LLM rewrite each target to about 1.5× its
   length with no new facts (restatement, hedging, background), then judge with Jev and
   Gemini. This is the decisive test for RQ3.
2. **Splice-free relevant padding.** Have an LLM integrate missing facts smoothly. This
   would separate "Gemini penalises disruption" from "Gemini doesn't credit coverage" on
   ragtime26 (Section 9.4).
3. **Instruction-following probes on the window.** Negation, label swap and "which is
   shorter", for Choice and Noul (not re-run; Section 4).
4. **Close-pair position effects.** What makes Choice favour slot A on rag26 and slot B
   on ragtime26?
5. **Ground truth.** Once assessments are released, measure accuracy and test whether
   probabilities help on close pairs.
6. **Length-controlled scoring.** Fit BT with a log-length-ratio covariate and report
   β per judge, using existing window data at no API cost.

### 15.5 Cost summary

| Item | Cost |
| --- | --- |
| Window study, rag26 (Section 4) | $25.27 |
| Window study, ragtime26 (Section 4) | $6.82 |
| Full runs, ragtime26 + rag26 (Section 13) | $93.84 |
| Bag-of-decisions judge: smoke test, pilots, padding test, full runs (Section 14) | $1.57 |
| Earlier study, withdrawn (Jev pilots and probes, about $19.49) | — |
| **Total for the results reported here** | **≈ $127.50** |

The partial Gemini tournament from September 2026 (78/119 rag26 topics) cost about
$149 and is not counted.

---

## 16. Reproduction

**Environment.** `.env` provides:

- `OPENAI_BASE_URL=https://openrouter.ai/api/v1`
- `OPENAI_API_KEY` (OpenRouter)
- `OPENAI_MODEL` (Gemini slug)
- `CACHE_DIR=./cache`

Load it with `set -a; source ./.env; set +a`.

**Dataset switch.** `JEV_DATASET=rag26|ragtime26` selects the dataset for every study
script (`temp/jev_dataset.py`). That module defines the window and refuses any
configured topic outside it.

**Window study.**

| Section | Command | Output |
| --- | --- | --- |
| 4–10 | `bash temp/run_window_suite.sh <ds>`: prompt variants, Choice pilot, identical summaries, determinism, mirror padding, Gemini window run (rag26), Gemini padding; budget-guarded | `output-win-<ds>-jev{,-prompts}/`, `output-win-rag26-gemini/`, `temp/jev_probes_win_<ds>/`; logs `temp/win_<ds>_*.log` |
| 11 | `bash temp/run_proportionate_test.sh <ds>`, then `JEV_DATASET=<ds> python temp/jev_proportionate_compare.py <broad comparisons> <narrow comparisons>` | `output-win-<ds>-prop-{broad,narrow}/`; `temp/win_<ds>_proportionate.txt` |
| 5, 6, 10, 13.2 | `JEV_DATASET=<ds> python temp/jev_win_report.py <comparisons.jsonl> <score> [gemini comparisons]` | `temp/win_<ds>_report.txt` |
| 7 | `JEV_DATASET=<ds> python temp/jev_prompt_compare.py <prompts comparisons> <Choice pilot comparisons> <gemini comparisons>` | `temp/win_<ds>_prompt_compare.txt` |
| 8 | `JEV_DATASET=<ds> python temp/jev_probes.py report --question better_summary\|noul_a_proportionate\|noul_b_proportionate\|mirror` | `temp/win_<ds>_probes.txt` |
| 9 | `JEV_DATASET=<ds> python temp/jev_padding.py report --judge noul\|gemini [--topics …]` | `temp/win_<ds>_padding_{noul,gemini}.txt` |
| 12 | Decisions and confirmation recorded in `temp/WINDOW_STUDY_PLAN.md`. Decision 3 CIs and diagnostics: `JEV_DATASET=<ds> python temp/jev_full_analysis.py <full-run out-dir> <gemini comparisons>` | `temp/win_<ds>_full_window_diag.txt`, `temp/win_<ds>_choice_vs_mirror.txt` |

`<ds>` is `rag26` or `ragtime26`. Gemini comparisons:

- rag26: `output-win-rag26-gemini/bonsai_pairwise.pairwise/comparisons.jsonl`.
- ragtime26: `output-ragtime26-gemini/bonsai_pairwise.pairwise/comparisons.jsonl`.

**Full runs (Section 13).**

- Run: `bash temp/run_full_proportionate.sh` (`--variant full_noul_proportionate`;
  resumable).
- Outputs: `output-ragtime26-jev-full-prop/` and `output-pairwise-jev-full-prop/`
  (git-ignored).
- Submission copies are in `submissions/`.

**Bag-of-decisions (Section 14).**

| Step | Command |
| --- | --- |
| Smoke test | `python temp/bod_smoke.py` |
| Kiddie | `auto-judge run --workflow judges/bonsai_judge/workflow.bod.yml --variant kiddie --rag-responses data/kiddie/runs/repgen/ --rag-topics data/kiddie/topics/kiddie-topics.jsonl --out-dir ./output-kiddie-bod2/` |
| Pilots | the same workflow with `--variant pilot`, then `JEV_DATASET=<ds> python temp/bod_pilot_analysis.py output-win-<ds>-bod` |
| Padding | `JEV_DATASET=rag26 python temp/bod_padding.py run`, then `report` |
| Full runs | `bash temp/run_bod_full.sh` |

**Caching.** All Jev and Gemini calls are in `cache/minima_llm.db`. The determinism probe
deliberately bypasses the cache; its raw responses are in the JSONL files under
`temp/jev_probes_win_<ds>/`.

---

## 17. Appendix: prompts and questions verbatim

### 17.1 Gemini prompt (`judges/bonsai_judge/prompts/pairwise_summary_1.md`)

```markdown
# Pairwise Summary Comparison

You are a capable and knowledgeable subject matter expert, you need to select the better of two RAG summaries (A or B) that satisfy the information needs given in a query.

The query used to find the relevant documents and generate the summary is "<%=topic_query%>". Choose between the two presented summaries which satisfies the information needs better.

## Summary A

<%=summary_a%>

## Summary B

<%=summary_b%>

## Your decision

Judge which summary better satisfies the information need in the query, considering relevance, completeness, and how well the claims are grounded. Respond with exactly one character and nothing else: `A` if Summary A is better, or `B` if Summary B is better. No explanation, punctuation, or whitespace.
```

### 17.2 Jev request shape

```json
{
  "model": "typesafe/jev-1.13",
  "state": {
    "query": "<topic title>",
    "summary_a": "<text>",
    "summary_b": "<text>"
  },
  "questions": {
    "better_summary": {
      "type": "choice",
      "instructions": "...",
      "criteria": { "A": "...", "B": "..." }
    }
  }
}
```

Response (per question): `{"type": "choice", "choice": "A", "confidence": 0.84, "probabilities": {"A": 0.92, "B": 0.08}}`.

### 17.3 Jev questions (`judges/bonsai_judge/prompts/jev_questions.yml`)

The probe questions (`shorter`, `worse_summary`, `swapped_labels`, `noul_a_worse`,
`noul_a_shorter`) were used only in instruction-following probes. Those probes have
not been re-run on the permitted window (Section 4).

**better_summary** (control)

- _instructions:_ You are a subject matter expert. Two RAG summaries (summary_a and summary_b) were generated for the query. Which summary better satisfies the information need of the query, considering relevance, completeness, and how well its claims are grounded?
- _A:_ summary_a better satisfies the information need of the query.
- _B:_ summary_b better satisfies the information need of the query.

**length_neutral**

- _instructions:_ Two answers (summary_a and summary_b) were written for the user's query. Judge quality, not quantity: a longer or more detailed answer is NOT better unless the extra content gives the user something they need. Content that is generic, repeated, off-topic, or not specific to the user's situation earns no credit and makes an answer worse by burying the useful parts. If both answers serve the user equally well, prefer the more concise one. Which answer is better for this user?
- _A:_ summary_a serves the user better, judged on substance rather than length.
- _B:_ summary_b serves the user better, judged on substance rather than length.

**needs_coverage**

- _instructions:_ The query asks for several specific things (explicit questions, requested comparisons, priorities, tradeoffs, measures, warnings). Consider each thing the user asked for and whether each answer (summary_a, summary_b) addresses it directly with specific, actionable content tailored to the user's situation. Only content that answers something the user asked counts; general background and filler count for nothing however much there is. Which answer directly addresses more of what the user asked, and addresses it better?
- _A:_ summary_a directly answers more of the user's specific requests, and answers them better.
- _B:_ summary_b directly answers more of the user's specific requests, and answers them better.

**precision**

- _instructions:_ Compare the two answers (summary_a and summary_b) statement by statement. For each answer, consider what proportion of its statements are on-topic for the query, specific rather than generic, non-redundant, and plausible as accurate, well-supported claims. Judge the proportion, not the total number of good statements: an answer cannot win by being longer. Which answer has the higher proportion of relevant, specific, trustworthy content?
- _A:_ summary_a has the higher proportion of relevant, specific, trustworthy statements.
- _B:_ summary_b has the higher proportion of relevant, specific, trustworthy statements.

**shorter** (probe)

- _instructions:_ Ignore quality entirely. Which of the two answers (summary_a or summary_b) is shorter, i.e. contains fewer words?
- _A:_ summary_a is shorter (fewer words) than summary_b.
- _B:_ summary_b is shorter (fewer words) than summary_a.

**worse_summary** (probe)

- _instructions:_ You are a subject matter expert. Two RAG summaries (summary_a and summary_b) were generated for the query. Which summary WORSE satisfies the information need of the query, considering relevance, completeness, and how well its claims are grounded?
- _A:_ summary_a worse satisfies the information need of the query.
- _B:_ summary_b worse satisfies the information need of the query.

**swapped_labels** (probe)

- _instructions:_ identical to better_summary.
- _A:_ summary_b better satisfies the information need of the query.
- _B:_ summary_a better satisfies the information need of the query.

**noul_a_better** (Noul, `x_polarity: a`; P(A better) = P(yes))

- _instructions:_ You are a subject matter expert. Two RAG summaries (summary_a and summary_b) were generated for the query. Does summary_a satisfy the information need of the query better than summary_b does, considering relevance, completeness, and how well its claims are grounded?
- _true:_ summary_a satisfies the information need of the query better than summary_b.
- _false:_ summary_a does not satisfy the information need of the query better than summary_b.

**noul_b_better** (Noul mirror, `x_polarity: b`; P(A better) = 1 − P(yes))

- _instructions:_ You are a subject matter expert. Two RAG summaries (summary_a and summary_b) were generated for the query. Does summary_b satisfy the information need of the query better than summary_a does, considering relevance, completeness, and how well its claims are grounded?
- _true:_ summary_b satisfies the information need of the query better than summary_a.
- _false:_ summary_b does not satisfy the information need of the query better than summary_a.

**noul_a_worse** (Noul probe, negation of noul_a_better)

- _instructions:_ You are a subject matter expert. Two RAG summaries (summary_a and summary_b) were generated for the query. Does summary_a satisfy the information need of the query worse than summary_b does, considering relevance, completeness, and how well its claims are grounded?
- _true:_ summary_a satisfies the information need of the query worse than summary_b.
- _false:_ summary_a does not satisfy the information need of the query worse than summary_b.

**noul_a_proportionate** / **noul_b_proportionate** (Sections 11 and 13; `x_polarity` a / b; used for the full runs)

- _instructions:_ You are a subject matter expert. Two RAG summaries (summary_a and summary_b) were generated for the query. Does summary_a [summary_b] satisfy the information need of the query better than summary_b [summary_a] does, considering relevance, completeness, depth proportionate to what the query asks for (thorough when the query is broad or multi-part, brief and direct when it is narrow), and how well its claims are grounded?
- _true:_ summary_a [summary_b] satisfies the information need of the query better than summary_b [summary_a], at a depth proportionate to what the query asks for.
- _false:_ summary_a [summary_b] does not satisfy the information need of the query better than summary_b [summary_a], at a depth proportionate to what the query asks for.

**noul_a_shorter** (Noul probe, objective)

- _instructions:_ Ignore quality entirely. Is summary_a shorter than summary_b, i.e. does it contain fewer words?
- _true:_ summary_a contains fewer words than summary_b.
- _false:_ summary_a does not contain fewer words than summary_b.

`x_*` keys are local metadata and are stripped before the request is sent. The Noul
response shape is `{"type": "noul", "noul": 0.96}`.

### 17.4 Generic filler pool (padding test)

20 topic-agnostic sentences, for example: "It is important to consider multiple perspectives
when approaching this question."; "Every situation is different, so what works in one
context may not work in another."; "Ultimately, the right choice depends on individual
circumstances and goals." The full list is `GENERIC` in `temp/jev_padding.py`.

### 17.5 BOD rubric prompt and grade question (`judges/bonsai_judge/prompts/bod_questions.md`, `bod.py`)

Rubric generation (Gemini 2.5 Flash Lite, temperature 0). `<%=topic_query%>` is the query,
problem statement and background:

````markdown
You are designing a grading rubric for answers to a user's information need.

<information_need>
<%=topic_query%>
</information_need>

Write the questions a careful expert would use to grade how well an answer meets this information need. Each question checks one distinct part of what the user needs to know or do.

Rules:
- Write between <%=min_questions%> and <%=max_questions%> questions. Use as many as the need has genuinely distinct parts: a narrow, single-fact need gets few questions; a broad, multi-part need gets more. Do not pad the list to reach a number.
- Each question must be gradable on a scale from "not addressed" to "fully answered with specific detail", so phrase it as "Does the answer ...?" about content, e.g. "Does the answer explain why ...?" or "Does the answer give ...?".
- Every question must check substantive content: a fact, explanation, step, or piece of advice the user asked for or clearly needs to act on what they asked.
- Use any background or situation described only to decide WHAT content the user needs (for example, a safety concern makes safety advice necessary). Do not write questions that merely check whether the answer acknowledges, mentions, or relates to the user's interests, hobbies, or feelings.
- No two questions may check the same thing. Do not ask about tone, reading level, style, length, formatting, or citations.
- Do not mention any particular source, product, or answer.

Return ONLY a JSON array of strings, one per question, with no other text.
````

Each rubric question is sent to Jev as:

```json
{"type": "choice",
 "instructions": "Grade the answer on this question: <rubric question>",
 "criteria": {
   "0": "The answer does not address this question at all.",
   "1": "The answer mentions this topic but gives no real substance or explanation.",
   "2": "The answer partially answers this question; important parts are missing or vague.",
   "3": "The answer fully answers this question with specific, accurate detail."}}
```

The padding question is a Noul: "Does the answer contain substantial content that is
off-topic, repeated, or generic filler that does not help with the user's question?"

### 17.6 Narrow requests for the broad vs narrow test (Section 11)

Each was written from the topic's own text, and sent as the only topic field (`query`).
Files: `temp/narrow_topics_{rag26,ragtime26}.jsonl`.

| Dataset | Topic | Narrow request |
| --- | --- | --- |
| rag26 | rag2026-0 | What outcomes should a hospital nursing DEI plan measure to show it improves recruitment and retention of nurses of color? |
| rag26 | rag2026-1 | My 72-year-old mother was widowed six weeks ago and has chest tightness. What warning signs should we take seriously? |
| rag26 | rag2026-2 | Would treating health care as a right in U.S. policy require a single-payer system? |
| rag26 | rag2026-3 | Should an 18-year-old investing $10,000 for the long term put it in a low-cost broad-market index fund? |
| rag26 | rag2026-4 | When is looking back at nostalgic memories psychologically useful rather than harmful? |
| rag26 | rag2026-5 | Is shipping straight from China to each customer under the $800 de minimis rule a smart move for a small US online shop? |
| rag26 | rag2026-6 | How did work culture change after the COVID-19 pandemic? |
| rag26 | rag2026-7 | Is it legal for a public pension fund to trade on congressional trading disclosures? |
| rag26 | rag2026-8 | Did the 2020 Arizona election audit find evidence that changed the official result? |
| rag26 | rag2026-9 | Does Russia's role in uranium enrichment create a fuel-supply risk for U.S. nuclear utilities? |
| ragtime26 | 2000 | Have anti-vaping laws measurably reduced vaping among teenagers? |
| ragtime26 | 2001 | Is Nordic walking better exercise than normal walking? |
| ragtime26 | 2002 | What is the Chinese government's attitude toward the "lying flat" movement? |
| ragtime26 | 2003 | Is ketamine an effective treatment for PTSD? |
| ragtime26 | 2004 | Is felinotherapy (cat-assisted therapy) beneficial for autistic children? |
| ragtime26 | 2005 | Is ketamine an approved medical treatment for depression? |
| ragtime26 | 2006 | Do night owls report lower life satisfaction than morning people? |
| ragtime26 | 2007 | Do children educated in Montessori schools do better later in life? |
| ragtime26 | 2008 | How was the Zaporizhzhia nuclear power plant affected by the Russian invasion of Ukraine? |
| ragtime26 | 2009 | Does radiation from X-rays and CT scans cause cancer or alter DNA? |
