# Pairwise Summary Judging with Jev and Gemini on TREC RAG 2026 — Findings

**Status:** working notes for the team and raw material for a TREC paper. Pilot-scale
results (1–3 topics per experiment). No full Jev run yet. All numbers here come from
runs in this repository between 2026-09-10 and 2026-10-08. Section 12 gives the
command and output file behind each table.

**One-paragraph summary.** We built a pairwise "tournament" judge for TREC RAG 2026
generation runs using TypeSafe's **Jev** decision model (`typesafe/jev-1.13`, served via
OpenRouter's Decisions API). Jev returns a _probability_ that summary A is better rather
than a binary choice. We compared it against our earlier binary pairwise judge built on
**Gemini** (`google/gemini-3.5-flash-lite`). On a pilot topic the two judges produce
nearly the same run ranking (Spearman 0.973). Jev costs about 6× less per comparison.
Jev's probabilities are close to deterministic, and it follows question wording (negated
and label-swapped questions invert its answers ≥99.6% of the time on decisive pairs). It
has four properties that matter for the method:

1. **Its probabilities are mostly saturated.** 56% are exactly 0 or 1, and on 1 topic,
   probability-weighted and binary win rates rank runs almost identically (ρ = 0.999).
2. **It has a strong first-slot (A) bias when uncertain.** Identical summaries get
   P(A) = 0.80, not 0.5, and the bias is concentrated in exactly the close pairs where
   probabilities should add information. Judging both orientations and averaging
   removes it.
3. **In a controlled padding test it rewards content, not length.** Gemini penalises
   uninformative padding at least as strongly. Neither judge rewarded crude
   length-only padding.
4. **A yes/no (Noul) formulation with a mirror question handles ties correctly.**
   Asking "is A better than B?" and "is B better than A?" in one request and averaging
   gives P = 0.52 on identical summaries (Choice: 0.80). It halves the slot-A advantage
   on close pairs, never saturates, and gives the same ranking as Choice (ρ = 0.999) at
   about the same cost. Under the determinism, instruction and padding probes it
   matches or improves on Choice (Section 10).

Our starting hypothesis that the Gemini judge was mainly rewarding length is **not
supported** by the controlled tests so far. It is not ruled out either: subtle,
on-topic verbosity has not been tested yet (Section 9).

---

## Contents

1. [Background and motivation](#1-background-and-motivation)
2. [Data](#2-data)
3. [Judges and method](#3-judges-and-method)
4. [Experiment 1 — Jev pilot: cost, agreement, distribution, position](#4-experiment-1--jev-pilot)
5. [Experiment 2 — Length dependence in natural comparisons](#5-experiment-2--length-dependence-in-natural-comparisons)
6. [Experiment 3 — Prompt variants](#6-experiment-3--prompt-variants)
7. [Experiment 4 — Instruction-following probes](#7-experiment-4--instruction-following-probes)
8. [Experiment 5 — Determinism and identical-summary probes (slot bias)](#8-experiment-5--determinism-and-identical-summary-probes)
9. [Experiment 6 — Padding test (Jev and Gemini)](#9-experiment-6--padding-test)
10. [Experiment 7 — Noul (yes/no) formulation](#10-experiment-7--noul-yesno-formulation)
11. [Discussion, threats to validity, open questions](#11-discussion)
12. [Reproduction](#12-reproduction)
13. [Appendix: prompts and questions verbatim](#13-appendix-prompts-and-questions-verbatim)

---

## 1. Background and motivation

The task is to rank TREC RAG 2026 generation runs **without human assessments**. Our
approach is a pairwise tournament. For each topic, every run's summary is compared
against every other team's summary, and the comparison outcomes are aggregated into a
per-run win rate and a Bradley–Terry (BT) strength.

The first version (September 2026) used Gemini 3.5 Flash-Lite with a prompt that asks
for a single `A`/`B` token. Inspecting its outputs, the team suspected that **summary
length**, rather than quality, was driving its decisions.

Jev was attractive for two reasons:

- **Probabilistic output.** Jev answers a typed Choice question with a probability per
  option. This gives a graded outcome per comparison instead of 0/1. The hope was that
  close pairs would contribute partial credit rather than coin-flip noise.
- **Cost.** Jev bills input tokens only, at a rate that makes a full tournament much
  cheaper.

The research questions that came out of this work:

- **RQ1** Do Jev and Gemini agree on run rankings?
- **RQ2** Do Jev's probabilities change the ranking relative to binary outcomes?
- **RQ3** Is either judge biased toward longer summaries independent of content?
- **RQ4** Does Jev follow the question text, and does rewording the question change outcomes?
- **RQ5** How reliable are Jev's probabilities (determinism, position bias, calibration on ties)?

---

## 2. Data

TREC RAG 2026 AutoJudge test data, generation subset (`data/rag26/`):

| Property                       | Value                                   |
| ------------------------------ | --------------------------------------- |
| Generation runs                | 83                                      |
| Teams                          | 25                                      |
| Topics                         | 119                                     |
| Reports                        | 9,875                                   |
| Assessed topics (ground truth) | **0** (only `random.eval.jsonl` exists) |

- **Topics are long, multi-part information needs.** For example, rag2026-0 asks a nursing
  DEI council what to prioritise across five named areas, how to think about tradeoffs,
  which outcomes to measure, and which mistakes to avoid. A good answer is necessarily
  multi-faceted. This matters for interpreting length effects: a very short answer
  cannot cover all the facets.
- **Summary lengths on rag2026-0:** median 679 words, minimum 65, maximum 1,024. Lengths
  appear to be capped near 1,000 words.
- **Comparisons are cross-team only.** A team's runs are never compared with each other.
  Per topic this gives about 3,200 unordered cross-team pairs, or about 6,400 ordered
  comparisons.
- **The full tournament** is 380,443 unordered pairs (single orientation) or 760,886
  ordered comparisons.

**No ground truth.** Without assessments we cannot measure judge _accuracy_. Every
experiment below instead measures agreement, sensitivity to controlled manipulations,
or internal consistency.

**Topic selection.** The pilots used the lexicographically first topics: rag2026-0, -1,
-10, and rag2026-100 for the determinism probe. The selection was by sort order, not
cherry-picked, but it is also not random (see threats, Section 11).

---

## 3. Judges and method

### 3.1 Gemini binary judge (baseline)

- **Model:** `google/gemini-3.5-flash-lite` via OpenRouter, called in realtime with
  temperature 0 and `max_tokens` 8.
- **Prompt:** `judges/bonsai_judge/prompts/pairwise_summary_1.md` (verbatim in the
  appendix). The prompt contains the query and both summaries, and asks for exactly
  one character, `A` or `B`.
- **Parsing:** only a bare `A`/`B` is accepted, after stripping non-letters. Prose is
  marked unparseable. Across 249,419 comparisons only 1 was unparseable.
- **Orientation:** single direction. Each unordered pair is judged once, with the A/B
  orientation chosen by a deterministic hash of `(topic, run_x, run_y)`.
- **Full run status:** 78 of 119 topics scored and about $149 spent before a spend limit
  was hit. Measured cost was about $0.00058 per comparison on the full run, and $0.00072
  per call in the padding test, where the padded summaries are longer.

### 3.2 Jev probabilistic judge

- **Model and endpoint:** `typesafe/jev-1.13` through OpenRouter's Decisions API
  (`POST /api/alpha/decisions`). Every response reported the snapshot
  `typesafe/jev-1.13-20260917`.
- **Request shape:** Jev is not a chat model. Each request carries a `state` object,
  `{query, summary_a, summary_b}`, plus one or more typed **questions**. We use Choice
  questions with criteria `A` and `B` (verbatim in the appendix).
- **Response:** for each question, `choice`, `confidence` and
  `probabilities: {A: p, B: 1-p}`. Probabilities are reported to 2 decimals; we saw 101
  distinct values.
- **Several questions per request.** Questions are answered independently and the input
  state is billed once, so extra questions add only their own text to the cost. We use
  this to compare prompts at nearly no extra cost.
- **Cost:** input tokens only, about $0.042 per million input tokens. Mean input was
  2,184 tokens per comparison on rag2026-0, about $0.00010 per single-question call.
- **Implementation:** `judges/bonsai_judge/pairwise_jev.py` (class
  `BonsaiJevPairwiseJudge`) and `workflow.pairwise_jev.yml`.
- **Caching:** responses are cached on disk under the key
  `sha256(endpoint, model, state, questions)`. Only well-formed answers are cached;
  failures are retried on rerun. Identical summaries produce identical payloads and are
  asked once, which removed about 7% of calls per topic.

**Scoring.** In a comparison of runs a (slot A) and b (slot B), run a earns P(A) and run
b earns 1 − P(A).

- `PAIRWISE_WINRATE` = expected wins / games. This is the primary leaderboard measure.
- `PAIRWISE_EXP_WINS` = expected wins.
- `PAIRWISE_GAMES` = number of valid comparisons.
- `PAIRWISE_CONFIDENCE` = mean |2P − 1|, a measure of decisiveness. It is a diagnostic
  only.
- **Bradley–Terry** is fit by MM iteration on the same fractional outcome counts.

### 3.3 Abandoned variant: Jev Router

We first tried OpenRouter's `typesafe/jev-router`. This router uses Jev to pick a _chat
model and reasoning effort_ for each request. It is not a probabilistic judge.

- In a 12-comparison smoke test it routed to `deepseek/deepseek-v4.1-flash` (7 calls)
  and `openai/gpt-6-luna` (5 calls).
- Reasoning tokens made up about 92% of the cost, even though the answer was a single
  letter.
- We dropped it: the "judge" becomes a mixture of models that changes per request, and
  it returns no probabilities.

---

## 4. Experiment 1 — Jev pilot

**Setup:** topic rag2026-0, all 83 runs, both orientations (6,446 ordered comparisons;
5,978 unique calls after deduplication), question `better_summary`.

### 4.1 Operational results

| Metric            | Value                                                            |
| ----------------- | ---------------------------------------------------------------- |
| Valid answers     | 6,446 / 6,446 (100%)                                             |
| Wall-clock time   | 5.1 min (32 concurrent; two pauses of about 2 min each; no 429s) |
| Cost              | $0.5913 ($0.0000989 per call)                                    |
| Mean input tokens | 2,184                                                            |

### 4.2 Agreement with Gemini (RQ1)

Per-run win rates on rag2026-0 (Jev probabilistic, both orientations; Gemini binary,
single orientation; 83 runs in common):

|               | Spearman ρ | Kendall τ |
| ------------- | ---------- | --------- |
| Jev vs Gemini | **0.973**  | 0.868     |

Top of the board on rag2026-0:

| Rank | Jev (win rate) | Gemini (win rate) |
| ---- | -------------- | ----------------- |
| 1    | carmen 0.974   | hana 0.987        |
| 2    | edith 0.937    | xavi 0.987        |
| 3    | lars 0.931     | luca 0.974        |
| 4    | ariel 0.920    | carmen 0.961      |
| 5    | yara 0.919     | edith 0.961       |
| …    |                |                   |
| last | paula 0.005    | paula 0.000       |

Both judges place teams T569 (carmen, edith, lars, ariel) and T300 (hana, xavi, luca) at
the top. That matches the partial 78-topic Gemini leaderboard.

### 4.3 Distribution of Jev probabilities (RQ2)

| P(A) range   | Count | Share |
| ------------ | ----- | ----- |
| [0.00, 0.01) | 1,834 | 28.5% |
| [0.01, 0.10) | 801   | 12.4% |
| [0.10, 0.25) | 259   | 4.0%  |
| [0.25, 0.40) | 164   | 2.5%  |
| [0.40, 0.60) | 195   | 3.0%  |
| [0.60, 0.75) | 217   | 3.4%  |
| [0.75, 0.90) | 303   | 4.7%  |
| [0.90, 0.99) | 521   | 8.1%  |
| [0.99, 1.00] | 2,152 | 33.4% |

- 56.2% of answers are exactly 0 or 1.
- Mean decisiveness |2P − 1| is 0.890.
- Mean P(A) is 0.505.

**Probabilistic vs binary scoring.** We compared per-run probability-weighted win rate
with argmax (binary) win rate: **Spearman ρ = 0.9988**. On this topic, using
probabilities instead of binary outcomes barely changes the ranking. Jev is decisive
(P < 0.25 or P > 0.75) on 91% of comparisons (5,870 of 6,446), and the remaining pairs are too few to move the aggregate. Whether
probabilities improve _accuracy_ (as opposed to the ranking) cannot be tested without
ground truth.

### 4.4 Position effects: aggregate view

Over all 3,223 pairs judged in both orientations:

- Mean |ΔP| when the two summaries are swapped: **0.042**. The median is **0.000**.
- The winner (argmax) flips in **4.4%** of pairs.
- Mean P(A) is 0.505.

These aggregate numbers suggested that position bias was negligible and that judging
one orientation per pair would be enough. **That conclusion was wrong.** Section 8
shows the bias is concentrated in close pairs and hidden by the large number of
saturated ones. We note it as a methodological lesson: overall averages can hide a bias
that only shows up in a small subset of pairs.

### 4.5 Bradley–Terry on saturated outcomes

BT strengths stretch out when one run gets P ≈ 1 against nearly everyone: carmen 17.2,
yara 6.29, edith 6.09. A run that receives P = 0 in every comparison gets strength 0;
this happened to the weakest run in the kiddie smoke test. Win rate is the more stable
headline metric. BT should be reported on a log scale, or fit with a prior or
regularisation.

---

## 5. Experiment 2 — Length dependence in natural comparisons

**Setup:** for each comparison, the probability that the _longer_ summary (by word
count) wins, grouped by length ratio. We also report the Spearman correlation between
each run's win rate and its word count. Topic rag2026-0.

| Judge                               | P(longer wins) | Ratio < 1.25× | 1.25–2× | ≥ 2×  | ρ(win rate, words) |
| ----------------------------------- | -------------- | ------------- | ------- | ----- | ------------------ |
| Jev (P, both orientations)          | 0.694          | 0.546         | 0.617   | 0.844 | 0.524              |
| Gemini (binary, single orientation) | 0.662          | 0.520         | 0.571   | 0.817 | 0.436              |

The pattern is the same over 3 topics (rag2026-0, -1, -10) with Jev's control question:
0.705 overall, 0.529 at < 1.25×, 0.652 at 1.25–2×, 0.873 at ≥ 2×, ρ = 0.547.

**Interpretation.** Both judges favour longer summaries to a similar degree. The effect
is almost entirely in lopsided pairs: at ≥ 2× the longer summary wins about 82–87% of
the time, while near-equal lengths are close to a coin flip (0.52–0.55). This is
observational, so it cannot separate two explanations:

- **(a) Length bias:** the judge rewards length itself.
- **(b) Coverage:** short answers to multi-part questions leave sub-questions
  unanswered and are genuinely worse.

Sections 6–9 test this directly.

---

## 6. Experiment 3 — Prompt variants

**Setup:** 3 topics (rag2026-0, -1, -10), single orientation, 9,589 comparisons (9,209
unique calls), **4 questions in one request each**. Cost $1.12 total ($0.000122 per
call). The questions (verbatim in the appendix):

- `better_summary`: the control, our original prompt converted to a Choice question.
- `length_neutral`: says explicitly that length is not quality, says filler earns no
  credit, and breaks ties toward the more concise answer.
- `needs_coverage`: credits only content that answers what the user asked. Built to
  separate legitimate coverage from volume.
- `precision`: judges the _proportion_ of relevant, specific, trustworthy statements.
  Built so that a longer answer cannot win by adding more.

### 6.1 Length dependence by question

| Question       | P(longer) | < 1.25× | 1.25–2× | ≥ 2×  | ρ(win rate, words) | Exactly 0 or 1 | Decisiveness |
| -------------- | --------- | ------- | ------- | ----- | ------------------ | -------------- | ------------ |
| better_summary | 0.705     | 0.529   | 0.652   | 0.873 | 0.547              | 0.519          | 0.871        |
| length_neutral | 0.673     | 0.518   | 0.615   | 0.832 | 0.477              | 0.494          | 0.858        |
| needs_coverage | 0.705     | 0.527   | 0.654   | 0.873 | 0.544              | 0.601          | 0.886        |
| precision      | 0.672     | 0.522   | 0.618   | 0.823 | 0.471              | 0.459          | 0.849        |

### 6.2 Ranking agreement between questions

Spearman correlation of per-topic win rates, averaged over the 3 topics:

|                | better_summary | length_neutral | needs_coverage | precision |
| -------------- | -------------- | -------------- | -------------- | --------- |
| better_summary | 1.000          | 0.988          | 0.993          | 0.981     |
| length_neutral | 0.988          | 1.000          | 0.986          | 0.987     |
| needs_coverage | 0.993          | 0.986          | 1.000          | 0.971     |
| precision      | 0.981          | 0.987          | 0.971          | 1.000     |

### 6.3 Pair-level agreement with the control

Share of pairs where the argmax matches the control's:

| Question       | All pairs | Near-equal length (< 1.25×, n = 2,718) |
| -------------- | --------- | -------------------------------------- |
| length_neutral | 0.958     | 0.958                                  |
| needs_coverage | 0.971     | 0.961                                  |
| precision      | 0.950     | 0.958                                  |

### 6.4 Agreement with Gemini

Spearman correlation of per-topic win rates, averaged over 3 topics:

| Question       | ρ     |
| -------------- | ----- |
| better_summary | 0.955 |
| length_neutral | 0.976 |
| needs_coverage | 0.958 |
| precision      | 0.957 |

### 6.5 Top 8 on rag2026-0, by question (win rate, words)

| Question       | Top 8                                                                                                                                                 |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| better_summary | carmen (0.97, 1020), edith (0.94, 863), lars (0.94, 869), xavi (0.93, 760), hana (0.93, 760), ariel (0.92, 896), yara (0.91, 1000), rita (0.90, 1020) |
| length_neutral | edith (0.96, 863), carmen (0.96, 1020), ariel (0.95, 896), lars (0.95, 869), xavi (0.94, 760), hana (0.93, 760), june (0.92, 799), luca (0.90, 814)   |
| needs_coverage | edith (0.96, 863), lars (0.95, 869), carmen (0.94, 1020), ariel (0.94, 896), xavi (0.94, 760), hana (0.93, 760), rita (0.90, 1020), yara (0.89, 1000) |
| precision      | carmen (0.98, 1020), lars (0.95, 869), edith (0.95, 863), ariel (0.95, 896), xavi (0.92, 760), hana (0.92, 760), yara (0.91, 1000), darcy (0.90, 674) |

### 6.6 Test–retest of the control

`better_summary` asked again in the 4-question bundle, compared with the earlier
single-question pilot on the same oriented comparisons (n = 3,223):

- Mean |ΔP| 0.0042.
- 76.4% identical.
- Argmax agrees in 99.7% of pairs.

Bundling several questions in one request does not measurably change the answer.

**Findings (RQ4, part 1).**

- Explicit anti-length instructions (`length_neutral`, `precision`) reduce length
  dependence slightly. P(longer) falls about 3 points and ρ(win rate, words) falls
  from 0.55 to 0.47.
- The rankings barely change: question-to-question ρ is 0.97–0.99, and the same 6 runs
  make every top 8.
- The 4–5% pair-level disagreement between questions is much larger than the
  test–retest noise (0.3%), so the questions do produce different answers. They just
  don't change the outcome.

---

## 7. Experiment 4 — Instruction-following probes

**Motivation.** The near-identical rankings in Section 6 have two explanations. Either
Jev ignores the question text, or it reads the text and still reaches the same
judgment. We tested this with questions whose correct relation to the control is known.

**Setup:** same 3 topics and 9,589 comparisons, 4 questions per request. Cost $1.05
($0.000114 per call). The probes are verbatim in the appendix:

- `shorter`: "which summary has fewer words". The answer can be checked against true
  word counts.
- `worse_summary`: the control question negated. Expected P(A) ≈ 1 − P_control(A).
- `swapped_labels`: the control question with the option _descriptions_ swapped (option
  A's text describes summary_b). Expected P(A) ≈ 1 − P_control(A). If Jev keyed on the
  label or slot instead, we would see P(A) ≈ P_control(A).

### 7.1 Results

| Probe            | Key result                                                                                                                                                                                |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `worse_summary`  | Pearson with 1 − P_control **0.975**. Argmax inverted in 95.5% of all pairs and **99.6%** of decisive pairs (\|P_control − 0.5\| > 0.4, n = 7,613). Mean \|P − (1 − P_control)\| = 0.054. |
| `swapped_labels` | Pearson with 1 − P_control **0.976**. Argmax inverted in 94.4% of all pairs and **100%** of decisive pairs. Mean \|P − (1 − P_control)\| = 0.093.                                         |
| `shorter`        | Accuracy against true word counts: **85.8%** overall, 99.9% at ≥ 2×, 90.1% at 1.25–2×, 61.1% at < 1.25×. Spearman(P(A shorter), log(len_a/len_b)) = **−0.945**.                           |

**Quality leaking into the `shorter` question.**

- Spearman(P(A shorter), P_control(A)) = −0.675.
- Take the 2,765 pairs where the truly shorter summary is _also_ the one judged better.
  There, `shorter` picks the truly shorter summary only 67.2% of the time; the other
  32.8% it picks the worse summary instead.
- These pairs are probably skewed toward similar lengths, where `shorter` is only 61%
  accurate anyway, so we cannot separate the two causes.

**Uncertain pairs lean toward slot A.** In the 409 pairs where the control is between
0.4 and 0.6, both inverted probes give mean P(A) of about 0.39 (0.390 and 0.384). In
both probes, option B means "summary_a is better" or "summary_b is worse", so both
lean toward the summary _in slot A_. This was the first sign of the slot bias in
Section 8.

**Findings (RQ4, part 2).**

- Jev reads and follows the question text, including negation and which option text
  belongs to which label. It is not answering by label or position.
- So the agreement across prompt variants in Section 6 reflects a stable judgment, not
  ignored wording.
- Jev can perceive length easily (almost perfect on lopsided pairs), so length is
  _available_ to it as a signal. Whether it _uses_ length when judging quality is
  tested in Section 9.

---

## 8. Experiment 5 — Determinism and identical-summary probes

These probes bypass the cache: every call is a fresh request.

### 8.1 Determinism (RQ5)

**Setup:** topic rag2026-100, which had never been cached. All 3,223 single-orientation
pairs were asked twice with byte-identical payloads, in two sequential passes. Cost
$0.30 per pass.

| Metric                        | Value                                                 |
| ----------------------------- | ----------------------------------------------------- |
| Probability exactly identical | 75.6%                                                 |
| Mean / p95 / max \|ΔP\|       | 0.0043 / 0.020 / 0.090                                |
| Winner (argmax) flips         | 11 of 3,223 (0.34%), all near P = 0.5                 |
| Mean \|Δconfidence\|          | 0.0086                                                |
| Snapshot served               | `jev-1.13-20260917` in both passes                    |
| Largest changes               | 0.27→0.36, 0.53→0.45, 0.36→0.44, 0.64→0.56, 0.78→0.71 |

This independently reproduces the test–retest result from Section 6.6 (76.4% identical,
mean |ΔP| 0.0042). Jev has a small amount of jitter at its 2-decimal resolution and is
effectively deterministic for ranking purposes.

### 8.2 Identical summaries (RQ5)

**Setup:** each run's summary compared with itself (summary_a = summary_b). 248
self-comparisons over 3 topics, question `better_summary`. A judge without position
bias must return P(A) = 0.5.

| Metric                          | Value                                                                         |
| ------------------------------- | ----------------------------------------------------------------------------- |
| Mean P(A)                       | **0.799** (median 0.810)                                                      |
| Choice = A                      | **248 / 248**                                                                 |
| P(A) in [0.45, 0.55]            | 0                                                                             |
| Mean reported confidence        | 0.597                                                                         |
| Per topic, mean (sd)            | rag2026-0: 0.820 (0.057); rag2026-1: 0.790 (0.053); rag2026-10: 0.785 (0.062) |
| Distribution by decile, 0.0–1.0 | 0, 0, 0, 0, 0, 0, 14, 87, 142, 5                                              |

When the two options cannot be told apart, Jev does not report 0.5. It picks slot A
with about 0.8 probability and reports moderate confidence.

### 8.3 How slot bias depends on pair closeness

Using the both-orientation pilot (Section 4), we grouped the 3,223 pairs by their
orientation-averaged P(x better). For each group we report the mean probability that
the summary _in slot A_ wins, and how often the winner flips when the order is swapped:

| Mean P(x better) | Pairs   | Mean P(slot A wins) | Winner flips on swap |
| ---------------- | ------- | ------------------- | -------------------- |
| 0.00–0.05        | 1,361   | 0.500               | 0.0%                 |
| 0.05–0.25        | 238     | 0.506               | 0.0%                 |
| 0.25–0.40        | 107     | 0.527               | 29.0%                |
| **0.40–0.60**    | **111** | **0.582**           | **80.2%**            |
| 0.60–0.75        | 98      | 0.531               | 26.5%                |
| 0.75–0.95        | 217     | 0.507               | 0.0%                 |
| 0.95–1.00        | 1,091   | 0.500               | 0.0%                 |
| **All**          | 3,223   | 0.505               | 4.4%                 |

**Findings (RQ5).**

- Slot bias is invisible on decisive pairs (about 90% of pairs) and dominant on close
  ones. In the 0.4–0.6 group, the winner depends mainly on which summary was placed
  first.
- This is the subset where probabilistic scoring was supposed to add information. With
  a single orientation per pair, the probability on close pairs is substantially slot
  noise.
- **Mitigation:** judge both orientations and average P(x better). This cancels any
  constant slot offset by construction, at 2× the calls.
- Jev's probabilities are **not calibrated**: a true tie is reported as about 0.8.
  Even after averaging, P should be treated as a ranking signal, not as a probability
  that one summary is better.
- In the padding test (Section 9), which averages both orientations, the residual slot
  advantage is 0.507 for Jev and 0.476 for Gemini.

---

## 9. Experiment 6 — Padding test

**Question (RQ3):** if we add words to a summary without adding useful information,
does the judge prefer it more (length bias) or less (content judging)?

### 9.1 Design

- **Targets:** 16 mid-ranked runs per topic on rag2026-0, -1 and -10, i.e. runs with
  control win rate between 0.25 and 0.75, chosen evenly by rank. 48 targets in total;
  mean original length 685 words.
- **Opponents:** 8 fixed opponents per target, spread evenly over the topic's ranking,
  excluding the target's team and the other targets. Mean opponent length 766 words.
- **Variants of each target summary.** Padding adds about 50% more words. Added
  sentences are inserted at seeded random positions, not appended. Generic filler comes
  from a pool of 20 sentences and repeats when more is needed.

| Variant      | Construction                                                                   | Information added           | Prediction if the judge judges content | Prediction if it rewards length |
| ------------ | ------------------------------------------------------------------------------ | --------------------------- | -------------------------------------- | ------------------------------- |
| original     | unchanged                                                                      | —                           | —                                      | —                               |
| pad_repeat   | the summary's own sentences duplicated                                         | none                        | ↓                                      | ↑                               |
| pad_generic  | topic-agnostic platitudes                                                      | none                        | ↓                                      | ↑                               |
| pad_offtopic | sentences from summaries of rag2026-50                                         | irrelevant                  | ↓                                      | ↑                               |
| pad_relevant | sentences from the topic's top-ranked run (different team) not already present | relevant (positive control) | ↑                                      | ↑                               |
| truncate     | last third of sentences removed                                                | removed                     | ↓                                      | ↓                               |

- **Judging:** every (target variant, opponent) pair is judged in **both orientations**
  and P(target better) is averaged over the two. The change ΔP is measured against the
  original, averaged over opponents per target. 95% CIs are bootstrapped over targets
  (5,000 resamples).
- **Calls:** 48 × 6 × 8 × 2 = 4,608 per judge.
- **Gemini configuration:** identical to the original full run (same prompt, plain
  slug, `max_tokens` 8, temperature 0). Its binary outputs mean each
  orientation-averaged cell is 0, 0.5 or 1.
- **Cost:** Jev $0.53; Gemini $3.29.

### 9.2 Results

| Variant      | Length vs original | Jev P(win) | **Jev ΔP** [95% CI]         | Jev targets improved | Gemini P(win) | **Gemini ΔP** [95% CI]      | Gemini targets improved |
| ------------ | ------------------ | ---------- | --------------------------- | -------------------- | ------------- | --------------------------- | ----------------------- |
| original     | 1.00×              | 0.279      | —                           | —                    | 0.281         | —                           | —                       |
| pad_repeat   | 1.53×              | 0.228      | **−0.050** [−0.060, −0.041] | 1/48                 | 0.111         | **−0.171** [−0.210, −0.135] | 0/48                    |
| pad_generic  | 1.51×              | 0.190      | **−0.089** [−0.107, −0.073] | 0/48                 | 0.120         | **−0.161** [−0.206, −0.124] | 1/48                    |
| pad_offtopic | 1.53×              | 0.085      | **−0.194** [−0.220, −0.168] | 0/48                 | 0.057         | **−0.224** [−0.266, −0.184] | 0/48                    |
| pad_relevant | 1.53×              | 0.462      | **+0.183** [+0.164, +0.202] | 48/48                | 0.297         | **+0.016** [−0.042, +0.069] | 25/48                   |
| truncate     | 0.66×              | 0.192      | **−0.087** [−0.104, −0.071] | 2/48                 | 0.174         | **−0.107** [−0.138, −0.076] | 4/48                    |

ΔP by topic:

| Topic      | Judge  | pad_repeat | pad_generic | pad_offtopic | pad_relevant | truncate |
| ---------- | ------ | ---------- | ----------- | ------------ | ------------ | -------- |
| rag2026-0  | Jev    | −0.033     | −0.047      | −0.193       | +0.195       | −0.075   |
| rag2026-0  | Gemini | −0.125     | −0.105      | −0.191       | +0.102       | −0.082   |
| rag2026-1  | Jev    | −0.037     | −0.067      | −0.164       | +0.208       | −0.076   |
| rag2026-1  | Gemini | −0.117     | −0.102      | −0.188       | +0.109       | −0.078   |
| rag2026-10 | Jev    | −0.081     | −0.153      | −0.225       | +0.147       | −0.109   |
| rag2026-10 | Gemini | −0.270     | −0.277      | −0.293       | −0.164       | −0.160   |

### 9.3 Findings (RQ3)

1. **Neither judge rewards crude length-only padding.** Every variant that adds words
   without useful information _lowers_ P(win) for both judges, on all 3 topics. For
   each judge and variant, P(win) rises for at most 1 of the 48 targets. The size of the penalty follows how harmful the padding is:
   repeated < generic < off-topic.
2. **Gemini penalises uninformative padding more than Jev does:** −0.16 to −0.22,
   against Jev's −0.05 to −0.19. On this evidence Gemini is _not_ rewarding length for
   its own sake.
3. **Only Jev rewards added relevant content.** Jev gives +0.183, with all 48 targets
   improving. Gemini's change is +0.016 with a CI that includes 0: positive on
   rag2026-0 and -1 (+0.10), negative on rag2026-10 (−0.16).
   - **Hypothesis, not tested:** Gemini penalises the disruption of splicing in another
     summary's sentences (broken flow, overlap with existing points), while Jev credits
     the added coverage.
   - Which behaviour is "correct" is debatable: a human reader might also mark down a
     spliced answer.
4. **Truncation lowers both judges' preference by a similar amount** (−0.09 and −0.11).
5. **Gemini is more sensitive to any edit.** On rag2026-10 every manipulation costs
   Gemini −0.16 to −0.29, including adding relevant content.

**What this means for Section 5.** The length correlation in natural comparisons is at
least as well explained by **coverage** (short answers to multi-part questions omit
content) as by length bias. Both judges reject padding that is easy to recognise. The
remaining open possibility is **subtle verbosity**: fluent, on-topic restatement that
adds no facts. That is the most realistic form of verbosity and neither test covers it.

---

## 10. Experiment 7 — Noul (yes/no) formulation

**Motivation.** The Choice question makes Jev split probability between two options. On
ties it does not split evenly: it puts about 0.8 on option A (Section 8.2). Jev's other
primitive, **Noul**, asks a single yes/no question and returns one probability, P(yes).
We tested whether asking "Is summary_a better than summary_b?" gives cleaner,
less position-biased probabilities.

**Setup.**

- Same topic and design as the Choice pilot (Section 4): rag2026-0, both orientations,
  6,446 ordered comparisons, 5,978 unique calls.
- **Two Noul questions in each request**, verbatim in Section 13.3:
  - `noul_a_better`: "Does summary_a satisfy the information need … better than
    summary_b does …?"
  - `noul_b_better`: the mirror, "Does summary_b … better than summary_a …?"
- With yes/no questions there is a risk of a yes-bias (or no-bias). The mirror
  question measures it: with no such bias, P_yes(A better) + P_yes(B better) ≈ 1.
- Each question is tagged with local metadata, `x_polarity: a` or `b`, saying which
  summary a "yes" favours. This metadata is stripped before sending: the API rejects
  unknown keys, and stripping keeps the cache keys of existing questions unchanged
  (confirmed by a zero-call replay of the Choice pilot).
- We also re-ran the identical-summary probe (Section 8.2) for each Noul question
  separately: 248 calls each.
- **Cost:** $0.62 for the pilot ($0.000104 per call with both questions, against
  $0.000099 for one Choice question) and $0.05 for the two tie probes.
- All 6,446 answers were valid. Snapshot served: `jev-1.13-20260917`.

Three ways to estimate P(summary_a better) from the raw P(yes) values, written yA and yB:

| Estimator | Formula | Notes |
| --- | --- | --- |
| `noul_a` | yA | "is A better?" alone |
| `mirror` | (yA + 1 − yB) / 2 | averages with the mirror question |
| `ratio` | yA / (yA + yB) | relative preference |
| `choice` | Choice `better_summary` P(A) | reference, from Section 4 |

### 10.1 Identical summaries

| Question (248 self-comparisons, 3 topics) | Mean (median) P(yes) | P(yes) > 0.5 | Per topic: mean (sd) | Distribution by decile, 0.0–1.0 |
| --- | --- | --- | --- | --- |
| `noul_a_better` | 0.151 (0.150) | 0/248 | rag2026-0: 0.143 (0.029); -1: 0.149 (0.032); -10: 0.160 (0.041) | 6, 215, 27, 0, 0, 0, 0, 0, 0, 0 |
| `noul_b_better` | 0.103 (0.100) | 0/248 | rag2026-0: 0.099 (0.023); -1: 0.103 (0.028); -10: 0.105 (0.026) | 108, 140, 0, 0, 0, 0, 0, 0, 0, 0 |

Implied P(summary_a better) on ties:

| Estimator | P(A better) on identical summaries (ideal 0.5) |
| --- | --- |
| choice (Section 8.2) | **0.799** |
| noul_a | **0.151** |
| mirror | **0.524** |
| ratio | 0.595 |

- **For identical summaries Jev answers "no" to both questions.** That is logically
  correct: neither summary is better than the other.
- **So for Noul, "no" means "not better", which includes ties.** 1 − P(yes) is therefore
  not P(B better). Used alone, `noul_a` scores a tie as 0.15 for slot A: a tie bias
  about as large as Choice's, in the opposite direction.
- **The mirror average fixes ties.** It gives 0.524, close to 0.5.
- **There is still a small lean toward slot A.** The "A better" question gets more "yes"
  than the mirror on identical inputs (0.151 against 0.103).

### 10.2 Consistency of the two Noul answers on real pairs

| Metric | Value |
| --- | --- |
| Mean yA / mean yB | 0.503 / 0.479 |
| yA + yB: mean (median) | 0.983 (0.990) |
| yA + yB within [0.9, 1.1] | 97.1% (2.8% below 0.9; 0.1% above 1.1) |
| Both "no" (both < 0.5) | 33 comparisons (0.5%) |
| Both "yes" (both > 0.5) | 29 comparisons (0.4%) |
| Pearson(yA, 1 − yB) | 0.996 |
| Winner agrees between `noul_a` and Choice | 99.1% |

- On real, non-identical pairs the two Noul answers are almost exact complements, so
  there is no meaningful yes-bias.
- The mirror question only changes results near ties, which is where it is needed.

### 10.3 Distribution of Noul probabilities

`noul_a_better` raw P(yes), compared with Choice on the same 6,446 comparisons:

| P range | Noul | Choice (Section 4.3) |
| --- | --- | --- |
| [0.00, 0.01) | 0 | 1,834 |
| [0.01, 0.10) | 1,631 | 801 |
| [0.10, 0.25) | 941 | 259 |
| [0.25, 0.40) | 379 | 164 |
| [0.40, 0.60) | 415 | 195 |
| [0.60, 0.75) | 469 | 217 |
| [0.75, 0.90) | 950 | 303 |
| [0.90, 0.99) | 1,661 | 521 |
| [0.99, 1.00] | 0 | 2,152 |

- Noul never returns exactly 0 or 1. Its range is 0.02–0.98 (yB: 0.01–0.98), with 97
  distinct values.
- The most frequent values are 0.96 (320 comparisons), 0.95 (298), 0.04 (284), 0.03
  (259), 0.06 (221) and 0.05 (217).
- 80% of comparisons are decisive (P < 0.25 or P > 0.75), against 91% for Choice. 19.6%
  fall between 0.25 and 0.75 (1,263 of 6,446), against 8.9% for Choice (576).
- Noul's probabilities are more graded but still uncalibrated: they stop short of the
  extremes rather than reaching them.

### 10.4 Estimators compared

Pairs judged in both orientations: 3,223. "Swap" columns describe how P changes when
the two summaries are swapped.

| Estimator | Mean P(A) | Exactly 0/1 | Share in [0.4, 0.6] | Mean swap \|ΔP\| | Winner flips on swap | P(slot A wins) | Probability vs binary ranking ρ | ρ vs Choice | ρ vs Gemini | P(longer wins) | ρ(win rate, words) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| choice | 0.505 | 0.562 | 0.032 | 0.042 | 0.045 | 0.505 | 0.9988 | 1.000 | 0.973 | 0.694 | 0.524 |
| noul_a | 0.503 | 0.000 | 0.070 | 0.042 | 0.045 | 0.503 | 0.9972 | 0.999 | 0.970 | 0.669 | 0.532 |
| mirror | 0.512 | 0.000 | 0.063 | 0.036 | 0.039 | 0.512 | 0.9978 | 0.999 | 0.970 | 0.669 | 0.531 |
| ratio | 0.514 | 0.000 | 0.063 | 0.038 | 0.039 | 0.514 | 0.9985 | 0.999 | 0.968 | 0.671 | 0.533 |

Ranking agreement between estimators, after averaging both orientations: Spearman
**0.999–1.000** for every pair of estimators.

### 10.5 Slot bias by pair closeness

Pairs are grouped by orientation-averaged P(x better). Each cell is: number of pairs /
mean P(slot A wins) / winner-flip rate when swapped. Group boundaries are the same as
in Section 8.3. The groups hold different pairs for each estimator, because Noul
spreads probabilities differently from Choice.

| Estimator | 0.00–0.05 | 0.05–0.25 | 0.25–0.40 | **0.40–0.60** | 0.60–0.75 | 0.75–0.95 | 0.95–1.00 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| choice | 1360 / 0.500 / 0.00 | 239 / 0.507 / 0.00 | 107 / 0.527 / 0.29 | **111 / 0.582 / 0.80** | 98 / 0.531 / 0.27 | 217 / 0.507 / 0.00 | 1091 / 0.500 / 0.00 |
| noul_a | 410 / 0.498 / 0.00 | 1033 / 0.498 / 0.00 | 203 / 0.510 / 0.00 | **221 / 0.538 / 0.64** | 207 / 0.510 / 0.02 | 832 / 0.502 / 0.00 | 317 / 0.498 / 0.00 |
| mirror | 424 / 0.504 / 0.00 | 1027 / 0.508 / 0.00 | 200 / 0.526 / 0.00 | **220 / 0.542 / 0.57** | 195 / 0.521 / 0.01 | 851 / 0.511 / 0.00 | 306 / 0.504 / 0.00 |
| ratio | 528 / 0.503 / 0.00 | 934 / 0.511 / 0.00 | 194 / 0.528 / 0.01 | **213 / 0.545 / 0.58** | 191 / 0.524 / 0.01 | 793 / 0.514 / 0.00 | 370 / 0.504 / 0.00 |

- **In close pairs (0.4–0.6) Noul cuts the slot-A advantage roughly in half:** from
  0.582 to 0.538–0.545.
- **The flip rate overstates bias in this group.** When the averaged P is near 0.5, a
  small difference between the two orders flips the winner even without bias. P(slot A
  wins) is the bias measure.
- **The mirror average doesn't remove positional bias.** Asking the mirror question
  doesn't swap where the summaries sit in the input; summary_a is always first. Only
  judging both orders cancels it. Outside the close groups, `mirror` and `ratio` show a
  small constant slot-A offset (about 0.51) that `noul_a` does not have.

### 10.6 Top 8 on rag2026-0 (win rate, words)

| Estimator | Top 8 |
| --- | --- |
| choice | carmen (0.97, 1020), edith (0.94, 863), lars (0.93, 869), ariel (0.92, 896), yara (0.92, 1000), xavi (0.91, 760), hana (0.91, 760), todd (0.90, 1024) |
| noul_a | carmen (0.88, 1020), lars (0.86, 869), edith (0.85, 863), ariel (0.84, 896), yara (0.84, 1000), hana (0.84, 760), xavi (0.84, 760), todd (0.84, 1024) |
| mirror | carmen (0.89, 1020), lars (0.86, 869), edith (0.86, 863), ariel (0.85, 896), yara (0.84, 1000), hana (0.84, 760), xavi (0.84, 760), todd (0.84, 1024) |
| ratio | carmen (0.88, 1020), lars (0.86, 869), edith (0.86, 863), ariel (0.85, 896), rita (0.84, 1020), todd (0.84, 1024), yara (0.84, 1000), hana (0.84, 760) |

The choice row here is orientation-averaged on this topic, so it differs slightly from
the Section 6.5 row, which comes from the 3-topic single-orientation run.

### 10.7 Findings

1. **Noul with a mirror question handles ties correctly. Choice and Noul alone do not.**
   On identical summaries: Choice gives P(A better) = 0.80, `noul_a` alone 0.15, and
   the mirror average 0.52.
2. **It gives the same ranking as Choice.** ρ = 0.999, 99.1% of pair decisions agree,
   and ρ with Gemini is 0.970 against Choice's 0.973. Noul win rates are compressed
   (top run 0.88 against 0.97) because no answer is saturated.
3. **The probabilities are more graded, but it still makes no difference to the
   ranking.** No answer is exactly 0 or 1, twice as many fall in the middle range, and
   probability vs binary ranking ρ is still 0.997–0.999.
4. **It reduces position bias on close pairs but does not remove it** (0.54 against
   0.58). Judging both orientations is still required for an unbiased estimate.
5. **Length dependence is slightly lower** (P(longer wins) 0.669 against 0.694) and the
   run-level length correlation is unchanged (about 0.53).
6. **It costs about the same.** The mirror question adds about 5% per call, because the
   summaries dominate the input.
7. **It holds up under the same probes as Choice** (Section 10.9). It is at least as
   deterministic, follows negation and the length question as well or better, and gives
   the same padding results. The one weak point is the tie flag, which is not reliable
   on real pairs (Section 10.9.4).

### 10.8 Mirror scoring in the judge: ties and duplicates

The mirror average is now built into `BonsaiJevPairwiseJudge`, so the leaderboard can
use it directly. It is enabled per workflow with
`mirror: ["noul_a_better", "noul_b_better"]`.

- **Mirror score.** This adds a scored question, `mirror`, defined as
  (P(A better | noul_a_better) + P(A better | noul_b_better)) / 2 =
  (yA + 1 − yB) / 2. It becomes the primary (leaderboard) score unless
  `primary_question` overrides it. Its tables are written to `q_mirror/` alongside the
  per-question tables.
- **Validation.** The judge checks that `mirror` names exactly one polarity-a and one
  polarity-b Noul question taken from `questions`.
- **Tie flag.** A comparison is flagged `tie` when both raw P(yes) values are below
  `tie_threshold` (default 0.5), and `both_yes` when both are above it.
  - Ties still count as games and are scored by the mirror average (about 0.5).
  - Reported as the `PAIRWISE_TIE_RATE` leaderboard measure, `ties` and `tie_rate`
    columns in `leaderboard_overall.csv`, and `n_ties` / `n_both_yes` in
    `run_manifest.json`.
- **Duplicate check.** If summary_a and summary_b are identical text, no request is
  sent. Every question, including `mirror`, gets P = 0.5. The comparison is flagged
  `tie` and recorded with `result: "duplicate"` (`n_duplicate_ties` in the manifest).
- **Draws in diagnostics.** In the analysis scripts an exact P = 0.5 now counts as a
  draw, never as a B win and never as a winner flip. Before, `P > 0.5` was used, which
  silently counted 0.5 as a B win. The leaderboard was unaffected because it is
  fractional.

**Verification.**

- **Cache replay of the Noul pilot (rag2026-0, 6,446 comparisons).**
  - 0 new API calls.
  - The judge's `mirror` values match (yA + 1 − yB) / 2 for every comparison (0
    mismatches).
  - `mirror` is the primary score.
  - It reports 33 ties and 29 both-yes, matching Section 10.2.
  - Top of the mirror leaderboard: carmen 0.886, lars 0.861, edith 0.856.
- **Duplicate test on a kiddie copy** in which run4 (team `teamDUP`) is a copy of run1:
  - The 2 identical-text comparisons (both orders of run1 vs run4 on one topic) were
    skipped and scored 0.5. 6 API calls were made for the rest ($0.0003).
  - run1 and run4 get identical win rates (0.5692) and tie rates (2/6).
- **rag26 has no cross-team identical summaries** (0 of 380,443 pairs), so the
  duplicate check does not change any rag26 result. It guards other datasets and saves
  calls there.
- **Regression.** Reports for the earlier Jev and Gemini padding runs reproduce
  byte-identically after the script changes. pytest shows the same 3 failures as
  before (template README title and `autojudge-base` version pins), none from this
  judge.

### 10.9 Noul probe results

These are the probes from Sections 7–9, re-run with mirrored Noul. Each is compared with
the Choice result.

#### 10.9.1 Determinism

**Setup:** rag2026-100, the same 3,223 single-orientation pairs as Section 8.1, both
Noul questions per request. Two sequential passes with byte-identical payloads, cache
bypassed. Cost $0.32 per pass, 0 errors. Snapshot served: `jev-1.13-20260917`.

| Metric | Choice `better_summary` (8.1) | Noul `noul_a_better` (raw) | Noul `noul_b_better` (raw) | **Noul mirror** |
| --- | --- | --- | --- | --- |
| Probability exactly identical | 75.6% | 58.6% | 57.9% | 40.8% |
| Mean \|ΔP\| | 0.0043 | 0.0051 | 0.0052 | **0.0045** |
| p95 \|ΔP\| | 0.020 | — | — | **0.015** |
| Max \|ΔP\| | 0.090 | 0.050 | 0.060 | **0.050** |
| Winner flips | 11 (0.34%) | — | — | **8 (0.25%)** |

- **Each Noul answer jitters slightly more often than Choice, but by less.** Exact
  repeats are 58–59%, against Choice's 75.6%, but the maximum change is 0.05–0.06
  against 0.09.
- **Averaging the two adds the jitter of both** (40.8% exactly identical) but halves it
  (mean |ΔP| 0.0045).
- **Net effect:** the mirror score's 95th-percentile and maximum changes are smaller
  than Choice's, and fewer winners flip.
- **Largest mirror changes:** 0.49→0.44, 0.425→0.47, 0.49→0.45, 0.63→0.59,
  0.575→0.61. All are in close pairs.
- **Cross-bundle retest.** On rag2026-0, the Noul questions were asked once in a
  2-question request (Section 10 pilot) and once in a 4-question request (10.9.2).
  - `noul_a_better`: 60.2% identical, mean |ΔP| 0.0047, max 0.070, winner agrees 99.4%.
  - `noul_b_better`: 61.4% identical, mean |ΔP| 0.0046, max 0.060, winner agrees 99.5%.
  - Mirror: mean |ΔP| 0.0041; the winner flips in 4 of 3,223 pairs.
  - So, as with Choice (6.6), asking other questions in the same request does not
    measurably change the answers.

#### 10.9.2 Instruction following

**Setup:** rag2026-0, -1 and -10, single orientation, 9,589 comparisons (9,209 calls),
4 Noul questions per request: the mirror pair plus two probes, verbatim in 13.3. Cost
$1.04, 0 errors.

- `noul_a_worse`: "Does summary_a satisfy the information need … worse than
  summary_b?" This is the negation of `noul_a_better`. On decisive pairs we expect
  P_yes(worse) ≈ 1 − P_yes(A better) ≈ P_yes(B better). On true ties all three answers
  should be "no".
- `noul_a_shorter`: "Is summary_a shorter, i.e. does it contain fewer words?" The
  answer can be checked against true word counts.

**The mirror pair re-asked in the larger bundle (3 topics).**

| Metric | Value |
| --- | --- |
| yA + yB: mean | 0.989 |
| yA + yB within [0.9, 1.1] | 97.7% |
| Both "no" / both "yes" | 40 / 66 (0.42% / 0.69%) |
| Pearson(yA, 1 − yB) | 0.996 |

**Negation (`noul_a_worse`).**

| Metric | Noul (this run) | Choice `worse_summary` (7.1) |
| --- | --- | --- |
| Pearson with the expected value | 0.994 (with 1 − yA); 0.987 (with yB) | 0.975 |
| Mean \|P − expected\| | 0.025 (vs 1 − yA); 0.039 (vs yB) | 0.054 |
| Winner inverted, all pairs | 96.9% | 95.5% |
| Winner inverted, decisive pairs | **100%** (\|mirror − 0.5\| > 0.4, n = 4,337) | 99.6% (n = 7,613) |

- **Near-ties** (|mirror − 0.5| ≤ 0.1, n = 732): mean yA 0.508, yB 0.499, yW 0.471. Only
  10 of these pairs get "no" to all three questions.

**Length (`noul_a_shorter`).**

| Length ratio | n | Noul accuracy | Noul mean P(correct) | Choice `shorter` accuracy (7.1) |
| --- | --- | --- | --- | --- |
| All | 9,569 | **87.2%** | 0.802 | 85.8% |
| < 1.25× | 2,698 | 63.4% | 0.584 | 61.1% |
| 1.25–2× | 3,092 | 92.3% | 0.800 | 90.1% |
| ≥ 2× | 3,779 | **100.0%** | 0.958 | 99.9% |

- Spearman(P_yes(shorter), log(len_a / len_b)) = −0.946 (Choice: −0.945).
- **Quality still leaks into the length question.** Spearman(P_yes(shorter), mirror
  P(A better)) = −0.719 (Choice: −0.675). In the 2,726 conflict pairs, where the truly
  shorter summary is also judged better, the length question follows length 69.3% of
  the time and inverse quality 30.7% (Choice: 67.2% / 32.8%). The same caveat as 7.1
  applies: these pairs skew toward similar lengths.

**Finding:** Noul follows negation and the objective length question at least as
closely as Choice does. The same quality-leak caveat applies.

#### 10.9.3 Padding test

**Setup:** identical items to Section 9: 48 mid-ranked targets on rag2026-0, -1 and -10,
8 opponents each, 6 variants, both orientations, 4,608 calls. The score is the mirror
average. Cost $0.55, 0 errors.

| Variant | Length vs original | Noul P(win) | **Noul ΔP** [95% CI] | Noul targets improved | Choice Jev ΔP (9.2) | Gemini ΔP (9.2) |
| --- | --- | --- | --- | --- | --- | --- |
| original | 1.00× | 0.327 | — | — | — | — |
| pad_repeat | 1.53× | 0.280 | **−0.048** [−0.055, −0.040] | 1/48 | −0.050 | −0.171 |
| pad_generic | 1.51× | 0.231 | **−0.097** [−0.114, −0.081] | 0/48 | −0.089 | −0.161 |
| pad_offtopic | 1.53× | 0.136 | **−0.191** [−0.213, −0.170] | 0/48 | −0.194 | −0.224 |
| pad_relevant | 1.53× | 0.487 | **+0.160** [+0.144, +0.176] | 48/48 | +0.183 | +0.016 |
| truncate | 0.66× | 0.246 | **−0.082** [−0.098, −0.067] | 2/48 | −0.087 | −0.107 |

Noul ΔP by topic:

| Topic | pad_repeat | pad_generic | pad_offtopic | pad_relevant | truncate |
| --- | --- | --- | --- | --- | --- |
| rag2026-0 | −0.033 | −0.055 | −0.188 | +0.173 | −0.069 |
| rag2026-1 | −0.037 | −0.076 | −0.184 | +0.175 | −0.069 |
| rag2026-10 | −0.073 | −0.159 | −0.202 | +0.132 | −0.107 |

- **Mirrored Noul rewards content, not length, just as Choice does.** Every effect has
  the same direction, and the sizes are within about 0.02 of Choice (pad_relevant +0.160
  against +0.183). All 3 topics agree.
- **Baseline P(win) is higher** (0.327 against 0.279), because mirror-averaged
  probabilities are compressed toward 0.5 (10.3).
- **Slot bias is still present before averaging.** With both orientations averaged,
  slot A wins 0.518 of the time (Choice 0.507, Gemini 0.476). This matches the small
  constant slot-A offset of the mirror estimator (10.5). Judging both orders cancels it
  in ΔP.
- The tie flag fired on 19 of 4,608 calls (0.4%).

#### 10.9.4 How reliable is the tie flag?

We measured how many comparisons the flag catches at different thresholds, using the
instruction-check run (9,589 comparisons) and the two determinism passes:

| `tie_threshold` | Flagged (instruction run) | Of those, `noul_a_worse` also "no" | Flagged in determinism pass 1 / pass 2 / both |
| --- | --- | --- | --- |
| 0.50 (default) | 40 (0.42%) | 10 / 40 | 6 / 6 / 4 |
| 0.45 | 7 (0.07%) | 0 / 7 | 0 / 0 / 0 |
| 0.40 | 0 | — | 0 / 0 / 0 |
| 0.35 | 0 | — | 0 / 0 / 0 |

- **Jev almost never answers a clear "no" to both questions on real, different
  summaries.** At the default threshold the flag catches borderline pairs whose two
  answers both sit just under 0.5 (for flagged pairs, the mean P_yes(worse) is 0.550).
  It does not catch pairs Jev judges equal: only 10 of 40 also get "no" to "is A
  worse?".
- **The flag is unstable.** Across two identical determinism passes, 2 of the 6 flags
  in each pass did not recur in the other.
- **A clear "no" to both happens only for identical text** (10.1: 0.15 / 0.10). The
  duplicate check now handles that without calling Jev.
- **Scoring is unaffected.** The mirror average puts every flagged pair near 0.5 either
  way.
- **Recommendation:** treat `tie` and `PAIRWISE_TIE_RATE` as a *near-tie / borderline*
  diagnostic, not as a substantive tie rate. Do not report per-run tie rates as a
  finding. Duplicate-text ties (`result: "duplicate"`) are the only exact ties.

---

## 11. Discussion

### 11.1 Answers to the research questions

| RQ                                             | Answer (pilot scale)                                                                                                                                                                                 |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| RQ1 Jev vs Gemini agreement                    | High. ρ = 0.973 on rag2026-0 and 0.955–0.976 across 3 topics. Both judges have the same top teams.                                                                                                   |
| RQ2 Do probabilities matter?                   | Barely, for ranking: soft and binary win rates have ρ = 0.997–0.999 for both Choice (56% saturated) and Noul (never saturated, 0.02–0.98). Without ground truth we cannot say whether probabilities improve accuracy. |
| RQ3 Length bias                                | Not for crude padding, for either judge or either Jev question form; all penalise uninformative additions. The natural length correlation (ρ about 0.5) is consistent with coverage. Subtle on-topic verbosity is untested.     |
| RQ4 Instruction following / prompt sensitivity | Jev follows negation and option relabelling at ≥ 99.6% on decisive pairs for Choice, and follows negation at 100% for Noul. Rewording the quality question changes 4–5% of pair decisions but not the ranking (ρ ≥ 0.97). |
| RQ5 Reliability                                | Near-deterministic: Choice has 75.6% identical answers and 0.34% winner flips; the Noul mirror has mean |ΔP| 0.0045, max 0.05 and 0.25% flips. Choice has a strong first-slot bias on ties (identical summaries give P(A) = 0.80) that drives close-pair outcomes. A mirrored Noul pair gives 0.52 on ties and halves the close-pair slot advantage (0.58 → 0.54). Probabilities are uncalibrated in both forms. |

### 11.2 Recommended protocol for a full Jev run

1. **Judge both orientations and average P(x better) per pair.** For the full
   tournament this is 760,886 ordered comparisons, or **703,536 calls** after
   deduplicating identical payloads (92.5%). A cheaper alternative is to re-judge only
   pairs that are close after one pass (0.25 ≤ P ≤ 0.75) in the other order: about 9%
   of comparisons for Choice and about 20% for Noul.
   - **Cost estimate, mirrored Noul, both orientations: about $75.** This scales the
     measured pilot cost ($0.6192 for 5,978 calls) by the total input words over all
     unique payloads in all 119 topics. A typical call is 1,424 words across all
     topics, against 1,378 on rag2026-0. A naive calls × pilot-rate estimate gives
     $73.
   - **Single orientation: about $38.**
   - Choice is about 5% cheaper per call.
   - Budget about 10% headroom. Wall-clock time at the pilot's rate (about 18 calls/s
     at 32 concurrency) is about 10.5 h.
   - Check the OpenRouter key's total spend limit before starting. The Gemini run
     stopped on that limit (`HANDOFF-rag26-pairwise-judge.md`).
2. **Use orientation-averaged win rate as the headline metric.** Report BT on a log
   scale, or with regularisation, because saturated outcomes stretch the strengths.
3. **Choose the question form.**
   - **Mirrored Noul** (`noul_a_better` + `noul_b_better`, using the mirror average):
     correct on ties, less slot bias, graded probabilities.
   - **Choice** (`better_summary`): equally good for *ranking* (ρ = 0.999 between
     them).
   - Prefer mirrored Noul if per-pair probabilities will be analysed or reported.
   - With Choice, asking `worse_summary` in the same request is a cheap
     self-consistency check: average P(better) with 1 − P(worse).
4. **Pin `typesafe/jev-1.13` and record the served snapshot.** Cache everything, since
   reproducibility comes from the cache.

**Code status.** As of this writing, the scorer counts both orientations of a pair as
separate games. For win rate this is equivalent to averaging. BT should be changed to
fit on the orientation-averaged probability per pair.

The judge supports Noul questions (`x_polarity` metadata, raw answers stored as
`raw_p`). It also computes the mirror average as the primary score when `mirror:` is set,
flags near-ties, and scores identical-text pairs as 0.5 without a call (Section 10.8).
The `noul_pilot` and `noul_instruction_check` variants use it, and so does
**`full_noul`**, the full-run variant for the recommended protocol: all topics, both
orientations, scored by the mirror average. On one topic it replayed the pilot's cached
answers with 0 new calls.

```bash
auto-judge run --workflow judges/bonsai_judge/workflow.pairwise_jev.yml --variant full_noul \
  --rag-responses data/rag26/runs/generation/ \
  --rag-topics data/rag26/topics/trec_rag_2026_queries.jsonl --out-dir ./output-pairwise-jev-full-noul/
```

### 11.3 Threats to validity

- **No ground truth.** All conclusions are about agreement, consistency and controlled
  sensitivity, not accuracy. Two judges agreeing can share a bias.
- **Few topics.** 1–3 topics per experiment, chosen by lexicographic order (rag2026-0,
  -1, -10, -100), not at random. Effect sizes were consistent across these topics, but
  topic variety is limited.
- **Padding targets are mid-ranked only.** The behaviour of strong and weak runs under
  manipulation is untested.
- **Padding is easy to recognise.** Verbatim repeats, a 20-sentence generic pool and
  another topic's sentences may be easier to detect than real verbosity.
- **pad_relevant mixes two effects.** It adds both information and text from the
  strongest run. The direction of the effect is clear; its size does not isolate
  "more information".
- **The judges' outputs differ in type.** Gemini is binary and Jev is probabilistic, so
  ΔP values are on comparable scales but have different granularity. Gemini's
  orientation-averaged cells take values in {0, 0.5, 1}.
- **Gemini runs mixed orientation designs.** Gemini comparisons in Sections 4–6 come from
  the original single-orientation run. The padding test used both orientations for
  both judges.
- **Model drift.** Jev 1.13 (snapshot 20260917) and the Gemini slug may change
  server-side. Results are reproducible from the cache, not necessarily from fresh calls
  later.
- **Uncalibrated probabilities.** Any analysis that treats Jev's P as a true win
  probability (for example, expected-score models) inherits the tie bias in 8.2.

### 11.4 Open questions and next experiments

1. **On-topic verbosity padding.** Have an LLM rewrite each target to about 1.5× its
   length with no new facts (restatement, hedging, background), then judge with Jev and
   Gemini. This is the decisive test for RQ3.
2. **Splice-free relevant padding.** Have an LLM integrate missing facts smoothly. This
   separates "Gemini penalises disruption" from "Gemini doesn't credit coverage".
3. **Slot bias on more questions and topics.** Choice gives P(A) ≈ 0.8 on ties; mirrored
   Noul gives 0.52 (Section 10). Does mirrored Noul's residual close-pair bias (0.54)
   hold across topics? Could it be small enough to judge one orientation and re-judge
   only close pairs?
4. **Ground truth.** Hand-label about 50–100 pairs, stratified by closeness and length
   ratio, to estimate accuracy and test whether probabilities help on close pairs.
5. **Full-run comparison.** A full both-orientation Jev run against the completed
   Gemini leaderboard, with per-topic agreement and the distribution of disagreements.
6. **Length-controlled scoring.** Fit BT with a log-length-ratio covariate,
   logit P(i ≻ j) = s_i − s_j + β·log(len_i/len_j), and report β per judge as a
   summary statistic of length dependence. This uses existing data at no API cost.
7. **Noul under the other probes.** Done (Section 10.9): mirrored Noul matches or
   improves on Choice for determinism, instruction following and padding. Still open: a
   Noul prompt-variant study (Section 6 equivalent), and whether a clear "no" to both
   questions ever occurs for summaries that differ but are equally good.

### 11.5 Cost summary of this study

| Item                                        | Calls | Cost        |
| ------------------------------------------- | ----- | ----------- |
| Jev Router smoke test (kiddie)              | 12    | $0.003      |
| Jev smoke test (kiddie)                     | 12    | $0.0003     |
| Jev pilot (rag2026-0, both orientations)    | 5,978 | $0.59       |
| Prompt variants (3 topics × 4 questions)    | 9,209 | $1.12       |
| Instruction probes (3 topics × 4 questions) | 9,209 | $1.05       |
| Determinism (rag2026-100, 2 passes)         | 6,446 | $0.61       |
| Identical summaries                         | 248   | $0.02       |
| Padding test, Jev                           | 4,608 | $0.53       |
| Padding test, Gemini                        | 4,608 | $3.29       |
| Noul pilot (rag2026-0, both orientations, 2 questions) | 5,978 | $0.62 |
| Identical summaries, Noul (2 questions × 248) | 496 | $0.05 |
| Noul instruction probes (3 topics × 4 questions) | 9,209 | $1.04 |
| Noul determinism (rag2026-100, 2 passes × 2 questions) | 6,446 | $0.64 |
| Padding test, Noul mirror | 4,608 | $0.55 |
| Duplicate-check test (kiddie copy) | 6 | $0.0003 |
| **Total**                                   |       | **≈ $10.11** |

For reference, the partial Gemini tournament (78/119 topics, single orientation) cost
about $149.

---

## 12. Reproduction

Environment: `.env` provides `OPENAI_BASE_URL=https://openrouter.ai/api/v1`,
`OPENAI_API_KEY` (OpenRouter), `OPENAI_MODEL` (Gemini slug), and `CACHE_DIR=./cache`.
Load it with `set -a; source ./.env; set +a`.

| Section | Command                                                                                                                                                                                                                                                                                                                   | Output                                                                        |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| 4       | `auto-judge run --workflow judges/bonsai_judge/workflow.pairwise_jev.yml --variant pilot --rag-responses data/rag26/runs/generation/ --rag-topics data/rag26/topics/trec_rag_2026_queries.jsonl --out-dir ./output-pairwise-jev/`                                                                                         | `output-pairwise-jev/bonsai_pairwise_jev.pairwise/`; log `temp/jev_pilot.log` |
| 5       | `python temp/jev_length_bias.py <comparisons.jsonl> [topic]`                                                                                                                                                                                                                                                              | stdout                                                                        |
| 6       | same, `--variant prompts --out-dir ./output-pairwise-jev-prompts/`, then `python temp/jev_prompt_compare.py output-pairwise-jev-prompts/bonsai_pairwise_jev.pairwise/comparisons.jsonl output-pairwise-jev/bonsai_pairwise_jev.pairwise/comparisons.jsonl temp/rag26_pairwise/bonsai_pairwise.pairwise/comparisons.jsonl` | `temp/jev_prompt_compare.txt`                                                 |
| 7       | same, `--variant instruction_check --out-dir ./output-pairwise-jev-icheck/`, then `python temp/jev_instruction_check.py output-pairwise-jev-icheck/bonsai_pairwise_jev.pairwise/comparisons.jsonl`                                                                                                                        | `temp/jev_instruction_check.txt`                                              |
| 8.1     | `python temp/jev_probes.py determinism --topic rag2026-100` then `python temp/jev_probes.py report`                                                                                                                                                                                                                       | `temp/jev_probes/determinism_rag2026-100_pass{1,2}.jsonl`                     |
| 8.2     | `python temp/jev_probes.py identical` then `report`                                                                                                                                                                                                                                                                       | `temp/jev_probes/identical.jsonl`                                             |
| 8.3     | inline analysis of `output-pairwise-jev/bonsai_pairwise_jev.pairwise/pairs.csv` (columns `p_x_fwd`, `p_x_rev`)                                                                                                                                                                                                            | —                                                                             |
| 9       | `python temp/jev_padding.py run [--judge gemini]` then `report [--judge gemini]` (`dry` previews)                                                                                                                                                                                                                         | `temp/jev_probes/padding{,_gemini}.jsonl`, `temp/jev_padding{,_gemini}.txt`   |
| 10 | same `auto-judge run`, `--variant noul_pilot --out-dir ./output-pairwise-jev-noul/`, then `python temp/jev_noul_compare.py output-pairwise-jev-noul/bonsai_pairwise_jev.pairwise/comparisons.jsonl output-pairwise-jev/bonsai_pairwise_jev.pairwise/comparisons.jsonl temp/rag26_pairwise/bonsai_pairwise.pairwise/comparisons.jsonl`; ties: `python temp/jev_probes.py identical --question noul_a_better` (and `noul_b_better`), then `report --question …` | `output-pairwise-jev-noul/`, `temp/jev_noul_compare.txt`, `temp/jev_probes/identical_noul_{a,b}_better.jsonl`; log `temp/jev_noul_pilot.log` |
| 10.8 | cache replay: same `auto-judge run` with `--variant noul_pilot` (now with `mirror:`); duplicate test: a kiddie copy with one run duplicated under another team, `--variant noul_pilot -J max_topics=2` | `output-pairwise-jev-noul/bonsai_pairwise_jev.pairwise/{q_mirror,leaderboard_overall.csv,run_manifest.json}` |
| 10.9 | `bash temp/run_noul_probes.sh` (runs `--variant noul_instruction_check --out-dir ./output-pairwise-jev-noul-icheck/`, `python temp/jev_probes.py determinism --topic rag2026-100 --question mirror`, `python temp/jev_padding.py run --judge noul`); then `python temp/jev_noul_instruction_check.py output-pairwise-jev-noul-icheck/bonsai_pairwise_jev.pairwise/comparisons.jsonl output-pairwise-jev-noul/bonsai_pairwise_jev.pairwise/comparisons.jsonl`, `python temp/jev_probes.py report --question mirror`, `python temp/jev_padding.py report --judge noul` | `temp/jev_noul_instruction_check.txt`, `temp/jev_noul_determinism.txt`, `temp/jev_padding_noul.txt`; raw `temp/jev_probes/determinism_rag2026-100_mirror_pass{1,2}.jsonl`, `temp/jev_probes/padding_noul.jsonl`; logs `temp/jev_noul_{icheck,determinism,padding}.log` |

Gemini baseline comparisons: `temp/rag26_pairwise/bonsai_pairwise.pairwise/comparisons.jsonl`
(run 2026-09-10; see `HANDOFF-rag26-pairwise-judge.md`).

Cached fresh-call experiments: the Jev pilot, prompt, instruction and Noul runs, and
the Gemini padding run are in `cache/minima_llm.db`. The determinism, identical and Jev
padding probes deliberately bypass the cache; their raw responses are in the JSONL
files above.

---

## 13. Appendix: prompts and questions verbatim

### 13.1 Gemini prompt (`judges/bonsai_judge/prompts/pairwise_summary_1.md`)

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

### 13.2 Jev request shape

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

### 13.3 Jev questions (`judges/bonsai_judge/prompts/jev_questions.yml`)

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

**noul_a_shorter** (Noul probe, objective)

- _instructions:_ Ignore quality entirely. Is summary_a shorter than summary_b, i.e. does it contain fewer words?
- _true:_ summary_a contains fewer words than summary_b.
- _false:_ summary_a does not contain fewer words than summary_b.

`x_*` keys are local metadata and are stripped before the request is sent. The Noul
response shape is `{"type": "noul", "noul": 0.96}`.

### 13.4 Generic filler pool (padding test)

20 topic-agnostic sentences, for example: "It is important to consider multiple perspectives
when approaching this question."; "Every situation is different, so what works in one
context may not work in another."; "Ultimately, the right choice depends on individual
circumstances and goals." The full list is `GENERIC` in `temp/jev_padding.py`.
