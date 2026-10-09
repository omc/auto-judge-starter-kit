# Pairwise Summary Judging with Jev and Gemini on TREC RAG 2026 — Findings

**Status:** working notes for the team and raw material for a TREC paper. Pilot-scale
results (1–3 topics per experiment), replicated on a second dataset, ragtime26
(Section 11), plus full tournaments on all topics of both datasets (Section 13). All
numbers here come from runs in this repository between 2026-09-10 and 2026-10-09. Section 15 gives the
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
   about the same cost. It matches Choice under the instruction and padding probes but
   flips slightly more winners between identical requests (Section 10).
5. **Replication on ragtime26 (Section 11).** Several of these findings hold on a second
   dataset: judge agreement, ranking unaffected by probabilities, instruction following,
   the tie bias, and rejection of crude padding. Two do not:
   - Longer summaries no longer win (P(longer) ≈ 0.47, ρ ≈ −0.1).
   - Close real pairs no longer favour slot A.
     Both were properties of rag26, not of the judges. Gemini also now _penalises_ added
     relevant content.
6. **A depth-proportionate prompt adapts to the request (Section 12).**
   - Adding "conciseness" to the question acts as a uniform length penalty.
   - An explicit rule ("depth proportionate to what the query asks for: thorough when
     broad, brief and direct when narrow") judges like the original on broad requests.
   - When the request is narrowed to one sub-question, it moves 2–4 points further
     toward shorter answers than the original does, on all 4 test topics.
7. **Full runs (Section 13).** All topics of both datasets, mirrored proportionate Noul,
   both orientations: 971,304 comparisons, all valid, $93.84.
   - The leaderboards are highly stable: split-half reliability 0.998 (rag26) and
     0.989 (ragtime26).
   - They agree with Gemini on rag26 at Spearman 0.975 (80 topics).
   - The length effect stays dataset-specific.
   - Tie rates turn out to be a stable property of *topics*, replicating across
     ragtime's 28 duplicated requests.

Our starting hypothesis that the Gemini judge was mainly rewarding length is **not
supported** by the controlled tests so far, and on ragtime26 neither judge favours longer
reports at all (Section 11.3). It is not ruled out either: subtle, on-topic verbosity
has not been tested yet (Section 9).

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
11. [Experiment 8 — Replication on ragtime26](#11-experiment-8--replication-on-ragtime26)
12. [Experiment 9 — Depth-proportionate prompt](#12-experiment-9--depth-proportionate-prompt)
13. [Full runs: mirrored proportionate Noul on all topics](#13-full-runs-mirrored-proportionate-noul-on-all-topics)
14. [Discussion, threats to validity, open questions](#14-discussion)
15. [Reproduction](#15-reproduction)
16. [Appendix: prompts and questions verbatim](#16-appendix-prompts-and-questions-verbatim)

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
- **RQ6** Can the question make the judge reward depth proportionate to the need, rather
  than length in either direction?

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
cherry-picked, but it is also not random (see threats, Section 14).

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
  2,197 tokens per call on rag2026-0, about $0.00009 per single-question call.
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
| Cost              | $0.5516 ($0.0000923 per call; 5,978 calls)                       |
| Mean input tokens | 2,197 (per call)                                                 |

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
unique calls), **4 questions in one request each**. Cost $1.08 total ($0.000117 per
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
($0.000110 per call). The probes are verbatim in the appendix:

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
| Winner (argmax) flips         | 4 of 3,223 (0.12%), all near P = 0.5                  |
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
| 0.25–0.40        | 107     | 0.527               | 29%                  |
| **0.40–0.60**    | **111** | **0.582**           | **78%**              |
| 0.60–0.75        | 98      | 0.531               | 23%                  |
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
- **Two Noul questions in each request**, verbatim in Section 16.3:
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
- **Cost:** $0.58 for the pilot ($0.0000966 per call with both questions, against
  $0.0000923 for one Choice question) and $0.05 for the two tie probes.
- All 6,446 answers were valid. Snapshot served: `jev-1.13-20260917`.

Three ways to estimate P(summary_a better) from the raw P(yes) values, written yA and yB:

| Estimator | Formula                      | Notes                             |
| --------- | ---------------------------- | --------------------------------- |
| `noul_a`  | yA                           | "is A better?" alone              |
| `mirror`  | (yA + 1 − yB) / 2            | averages with the mirror question |
| `ratio`   | yA / (yA + yB)               | relative preference               |
| `choice`  | Choice `better_summary` P(A) | reference, from Section 4         |

### 10.1 Identical summaries

| Question (248 self-comparisons, 3 topics) | Mean (median) P(yes) | P(yes) > 0.5 | Per topic: mean (sd)                                            | Distribution by decile, 0.0–1.0  |
| ----------------------------------------- | -------------------- | ------------ | --------------------------------------------------------------- | -------------------------------- |
| `noul_a_better`                           | 0.151 (0.150)        | 0/248        | rag2026-0: 0.143 (0.029); -1: 0.149 (0.032); -10: 0.160 (0.041) | 6, 215, 27, 0, 0, 0, 0, 0, 0, 0  |
| `noul_b_better`                           | 0.103 (0.100)        | 0/248        | rag2026-0: 0.099 (0.023); -1: 0.103 (0.028); -10: 0.105 (0.026) | 108, 140, 0, 0, 0, 0, 0, 0, 0, 0 |

Implied P(summary_a better) on ties:

| Estimator            | P(A better) on identical summaries (ideal 0.5) |
| -------------------- | ---------------------------------------------- |
| choice (Section 8.2) | **0.799**                                      |
| noul_a               | **0.151**                                      |
| mirror               | **0.524**                                      |
| ratio                | 0.595                                          |

- **For identical summaries Jev answers "no" to both questions.** That is logically
  correct: neither summary is better than the other.
- **So for Noul, "no" means "not better", which includes ties.** 1 − P(yes) is therefore
  not P(B better). Used alone, `noul_a` scores a tie as 0.15 for slot A: a tie bias
  about as large as Choice's, in the opposite direction.
- **The mirror average fixes ties.** It gives 0.524, close to 0.5.
- **There is still a small lean toward slot A.** The "A better" question gets more "yes"
  than the mirror on identical inputs (0.151 against 0.103).

### 10.2 Consistency of the two Noul answers on real pairs

| Metric                                    | Value                                  |
| ----------------------------------------- | -------------------------------------- |
| Mean yA / mean yB                         | 0.503 / 0.479                          |
| yA + yB: mean (median)                    | 0.983 (0.990)                          |
| yA + yB within [0.9, 1.1]                 | 97.1% (2.8% below 0.9; 0.1% above 1.1) |
| Both "no" (both < 0.5)                    | 33 comparisons (0.5%)                  |
| Both "yes" (both > 0.5)                   | 29 comparisons (0.4%)                  |
| Pearson(yA, 1 − yB)                       | 0.996                                  |
| Winner agrees between `noul_a` and Choice | 99.1%                                  |

- On real, non-identical pairs the two Noul answers are almost exact complements, so
  there is no meaningful yes-bias.
- The mirror question only changes results near ties, which is where it is needed.

### 10.3 Distribution of Noul probabilities

`noul_a_better` raw P(yes), compared with Choice on the same 6,446 comparisons:

| P range      | Noul  | Choice (Section 4.3) |
| ------------ | ----- | -------------------- |
| [0.00, 0.01) | 0     | 1,834                |
| [0.01, 0.10) | 1,631 | 801                  |
| [0.10, 0.25) | 941   | 259                  |
| [0.25, 0.40) | 379   | 164                  |
| [0.40, 0.60) | 415   | 195                  |
| [0.60, 0.75) | 469   | 217                  |
| [0.75, 0.90) | 950   | 303                  |
| [0.90, 0.99) | 1,661 | 521                  |
| [0.99, 1.00] | 0     | 2,152                |

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
| --------- | --------- | ----------- | ------------------- | ---------------- | -------------------- | -------------- | ------------------------------- | ----------- | ----------- | -------------- | ------------------ |
| choice    | 0.505     | 0.562       | 0.032               | 0.042            | 0.044                | 0.505          | 0.9988                          | 1.000       | 0.973       | 0.694          | 0.524              |
| noul_a    | 0.503     | 0.000       | 0.070               | 0.042            | 0.042                | 0.503          | 0.9973                          | 0.999       | 0.970       | 0.669          | 0.532              |
| mirror    | 0.512     | 0.000       | 0.063               | 0.036            | 0.037                | 0.512          | 0.9979                          | 0.999       | 0.970       | 0.669          | 0.531              |
| ratio     | 0.514     | 0.000       | 0.063               | 0.038            | 0.037                | 0.514          | 0.9985                          | 0.999       | 0.968       | 0.671          | 0.533              |

Ranking agreement between estimators, after averaging both orientations: Spearman
**0.999–1.000** for every pair of estimators.

Flip columns here and in 10.5 count P = 0.5 as a draw (Section 10.8). Earlier drafts
counted 0.5 as a B win, which gave slightly higher flip rates.

### 10.5 Slot bias by pair closeness

Pairs are grouped by orientation-averaged P(x better). Each cell is: number of pairs /
mean P(slot A wins) / winner-flip rate when swapped. Group boundaries are the same as
in Section 8.3. The groups hold different pairs for each estimator, because Noul
spreads probabilities differently from Choice.

| Estimator | 0.00–0.05           | 0.05–0.25           | 0.25–0.40          | **0.40–0.60**          | 0.60–0.75          | 0.75–0.95          | 0.95–1.00           |
| --------- | ------------------- | ------------------- | ------------------ | ---------------------- | ------------------ | ------------------ | ------------------- |
| choice    | 1360 / 0.500 / 0.00 | 239 / 0.507 / 0.00  | 107 / 0.527 / 0.29 | **111 / 0.582 / 0.78** | 98 / 0.531 / 0.23  | 217 / 0.507 / 0.00 | 1091 / 0.500 / 0.00 |
| noul_a    | 410 / 0.498 / 0.00  | 1033 / 0.498 / 0.00 | 203 / 0.510 / 0.00 | **221 / 0.538 / 0.60** | 207 / 0.510 / 0.01 | 832 / 0.502 / 0.00 | 317 / 0.498 / 0.00  |
| mirror    | 424 / 0.504 / 0.00  | 1027 / 0.508 / 0.00 | 200 / 0.526 / 0.00 | **220 / 0.542 / 0.54** | 195 / 0.521 / 0.01 | 851 / 0.511 / 0.00 | 306 / 0.504 / 0.00  |
| ratio     | 528 / 0.503 / 0.00  | 934 / 0.511 / 0.00  | 194 / 0.528 / 0.01 | **213 / 0.545 / 0.55** | 191 / 0.524 / 0.01 | 793 / 0.514 / 0.00 | 370 / 0.504 / 0.00  |

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

| Estimator | Top 8                                                                                                                                                  |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| choice    | carmen (0.97, 1020), edith (0.94, 863), lars (0.93, 869), ariel (0.92, 896), yara (0.92, 1000), xavi (0.91, 760), hana (0.91, 760), todd (0.90, 1024)  |
| noul_a    | carmen (0.88, 1020), lars (0.86, 869), edith (0.85, 863), ariel (0.84, 896), yara (0.84, 1000), hana (0.84, 760), xavi (0.84, 760), todd (0.84, 1024)  |
| mirror    | carmen (0.89, 1020), lars (0.86, 869), edith (0.86, 863), ariel (0.85, 896), yara (0.84, 1000), hana (0.84, 760), xavi (0.84, 760), todd (0.84, 1024)  |
| ratio     | carmen (0.88, 1020), lars (0.86, 869), edith (0.86, 863), ariel (0.85, 896), rita (0.84, 1020), todd (0.84, 1024), yara (0.84, 1000), hana (0.84, 760) |

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
7. **It holds up under the same probes as Choice** (Section 10.9).
   - It follows negation and the length question as well as or better than Choice, and
     gives the same padding results.
   - Between identical requests its probabilities move less (smaller p95/max) but its
     winner flips slightly more (0.25% against 0.12%).
   - The tie flag is not reliable on real pairs (Section 10.9.4).

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
- **Draw rule and flips.** Treating P = 0.5 as a draw lowers previously reported
  Choice flip counts. On the rag26 determinism probe they fall from 11 (0.34%) to 4
  (0.12%). Sections 8.1, 8.3, 10.4, 10.5 and 10.9.1 use the corrected figures.
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

| Metric                        | Choice `better_summary` (8.1) | Noul `noul_a_better` (raw) | Noul `noul_b_better` (raw) | **Noul mirror** |
| ----------------------------- | ----------------------------- | -------------------------- | -------------------------- | --------------- |
| Probability exactly identical | 75.6%                         | 58.6%                      | 57.9%                      | 40.8%           |
| Mean \|ΔP\|                   | 0.0043                        | 0.0051                     | 0.0052                     | **0.0045**      |
| p95 \|ΔP\|                    | 0.020                         | —                          | —                          | **0.015**       |
| Max \|ΔP\|                    | 0.090                         | 0.050                      | 0.060                      | **0.050**       |
| Winner flips                  | 4 (0.12%)                     | —                          | —                          | **8 (0.25%)**   |

- **Each Noul answer jitters slightly more often than Choice, but by less.** Exact
  repeats are 58–59%, against Choice's 75.6%, but the maximum change is 0.05–0.06
  against 0.09.
- **Averaging the two adds the jitter of both** (40.8% exactly identical) but halves it
  (mean |ΔP| 0.0045).
- **Net effect:** the mirror score's 95th-percentile and maximum changes are smaller
  than Choice's, but **more winners flip**: 8 against 4. Its scores sit closer to 0.5,
  so the same jitter crosses 0.5 more often.
  - _Correction:_ an earlier version of this section compared against the Choice flip
    count under the old rule (11, 0.34%, counting 0.5 as a B win) and concluded that the
    mirror flips less. That was wrong.
  - ragtime26 confirms the ordering: Choice 0.59% against mirror 1.07% (Section 11.6).
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
4 Noul questions per request: the mirror pair plus two probes, verbatim in 16.3. Cost
$1.00, 0 errors.

- `noul_a_worse`: "Does summary_a satisfy the information need … worse than
  summary_b?" This is the negation of `noul_a_better`. On decisive pairs we expect
  P_yes(worse) ≈ 1 − P_yes(A better) ≈ P_yes(B better). On true ties all three answers
  should be "no".
- `noul_a_shorter`: "Is summary_a shorter, i.e. does it contain fewer words?" The
  answer can be checked against true word counts.

**The mirror pair re-asked in the larger bundle (3 topics).**

| Metric                    | Value                   |
| ------------------------- | ----------------------- |
| yA + yB: mean             | 0.989                   |
| yA + yB within [0.9, 1.1] | 97.7%                   |
| Both "no" / both "yes"    | 40 / 66 (0.42% / 0.69%) |
| Pearson(yA, 1 − yB)       | 0.996                   |

**Negation (`noul_a_worse`).**

| Metric                          | Noul (this run)                              | Choice `worse_summary` (7.1) |
| ------------------------------- | -------------------------------------------- | ---------------------------- |
| Pearson with the expected value | 0.994 (with 1 − yA); 0.987 (with yB)         | 0.975                        |
| Mean \|P − expected\|           | 0.025 (vs 1 − yA); 0.039 (vs yB)             | 0.054                        |
| Winner inverted, all pairs      | 96.9%                                        | 95.5%                        |
| Winner inverted, decisive pairs | **100%** (\|mirror − 0.5\| > 0.4, n = 4,337) | 99.6% (n = 7,613)            |

- **Near-ties** (|mirror − 0.5| ≤ 0.1, n = 732): mean yA 0.508, yB 0.499, yW 0.471. Only
  10 of these pairs get "no" to all three questions.

**Length (`noul_a_shorter`).**

| Length ratio | n     | Noul accuracy | Noul mean P(correct) | Choice `shorter` accuracy (7.1) |
| ------------ | ----- | ------------- | -------------------- | ------------------------------- |
| All          | 9,569 | **87.2%**     | 0.802                | 85.8%                           |
| < 1.25×      | 2,698 | 63.4%         | 0.584                | 61.1%                           |
| 1.25–2×      | 3,092 | 92.3%         | 0.800                | 90.1%                           |
| ≥ 2×         | 3,779 | **100.0%**    | 0.958                | 99.9%                           |

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

| Variant      | Length vs original | Noul P(win) | **Noul ΔP** [95% CI]        | Noul targets improved | Choice Jev ΔP (9.2) | Gemini ΔP (9.2) |
| ------------ | ------------------ | ----------- | --------------------------- | --------------------- | ------------------- | --------------- |
| original     | 1.00×              | 0.327       | —                           | —                     | —                   | —               |
| pad_repeat   | 1.53×              | 0.280       | **−0.048** [−0.055, −0.040] | 1/48                  | −0.050              | −0.171          |
| pad_generic  | 1.51×              | 0.231       | **−0.097** [−0.114, −0.081] | 0/48                  | −0.089              | −0.161          |
| pad_offtopic | 1.53×              | 0.136       | **−0.191** [−0.213, −0.170] | 0/48                  | −0.194              | −0.224          |
| pad_relevant | 1.53×              | 0.487       | **+0.160** [+0.144, +0.176] | 48/48                 | +0.183              | +0.016          |
| truncate     | 0.66×              | 0.246       | **−0.082** [−0.098, −0.067] | 2/48                  | −0.087              | −0.107          |

Noul ΔP by topic:

| Topic      | pad_repeat | pad_generic | pad_offtopic | pad_relevant | truncate |
| ---------- | ---------- | ----------- | ------------ | ------------ | -------- |
| rag2026-0  | −0.033     | −0.055      | −0.188       | +0.173       | −0.069   |
| rag2026-1  | −0.037     | −0.076      | −0.184       | +0.175       | −0.069   |
| rag2026-10 | −0.073     | −0.159      | −0.202       | +0.132       | −0.107   |

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
| --------------- | ------------------------- | ---------------------------------- | --------------------------------------------- |
| 0.50 (default)  | 40 (0.42%)                | 10 / 40                            | 6 / 6 / 4                                     |
| 0.45            | 7 (0.07%)                 | 0 / 7                              | 0 / 0 / 0                                     |
| 0.40            | 0                         | —                                  | 0 / 0 / 0                                     |
| 0.35            | 0                         | —                                  | 0 / 0 / 0                                     |

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
- **Recommendation:** treat `tie` and `PAIRWISE_TIE_RATE` as a _near-tie / borderline_
  diagnostic, not as a substantive tie rate. Do not report per-run tie rates as a
  finding. Duplicate-text ties (`result: "duplicate"`) are the only exact ties.

---

## 11. Experiment 8 — Replication on ragtime26

**Motivation.** Everything in Sections 4–10 comes from rag26, mostly from 1–3 topics. To
see which findings are properties of the judge and which belong to the dataset, we
re-ran every Jev experiment, and the Gemini comparisons, on a second TREC 2026
AutoJudge dataset.

### 11.1 Data and setup

ragtime26, task `repgen` (`data/ragtime26/runs/repgen/`):

| Property                                     | rag26 (generation)                        | ragtime26 (repgen)                                                            |
| -------------------------------------------- | ----------------------------------------- | ----------------------------------------------------------------------------- |
| Runs / teams / topics                        | 83 / 25 / 119                             | 49 / 10 / 103                                                                 |
| Reports (non-empty)                          | 9,875 (9,836)                             | 5,047 (5,041)                                                                 |
| Cross-team pairs per topic, single / ordered | about 3,200 / 6,400                       | 1,024 / 2,048                                                                 |
| Full tournament, ordered (unique calls)      | 760,886 (703,536)                         | 210,418 (196,958)                                                             |
| Report length, median (pilot topic range)    | 679 words (rag2026-0: 65–1,024)           | 602 words (2000: 235–857)                                                     |
| Topic text                                   | `title` only: a long, multi-part question | short `title` (a label) + `problem_statement` + `background`, about 118 words |

**Topic text.** On ragtime the title is just a label, for example "Anti-vaping
Legislation"; the actual request is in `problem_statement` and `background`. Both judges
now receive these fields:

- **Jev:** they are added to the `state` alongside `query`.
- **Gemini:** they are appended to `<%=topic_query%>`.

This is implemented as `topic_fields` / `topic_query_text` in
`judges/bonsai_judge/pairwise.py`. rag26 topics are title-only, so rag26 requests and
cache keys are unchanged. We verified this: both rag26 Jev pilots replay with 0 calls,
and 3,223 of 3,223 rag26 Gemini cache keys still hit.

**Design.** Each rag26 experiment was mirrored exactly, with the same workflow
variants, questions, scripts and parameters. Only the dataset is switched, via
`JEV_DATASET=ragtime26` (`temp/jev_dataset.py`).

| rag26               | ragtime26                                                                                | Used for                                                          |
| ------------------- | ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| rag2026-0           | 2000                                                                                     | pilots (Choice, Noul), both orientations                          |
| rag2026-0, -1, -10  | 2000, 2001, 2002                                                                         | prompt variants, instruction probes, identical summaries, padding |
| rag2026-100         | 2003                                                                                     | determinism (uncached)                                            |
| rag2026-50          | 2050                                                                                     | off-topic padding source                                          |
| existing Gemini run | **new** Gemini run on 2000–2002 (original prompt, single orientation, 3,072 comparisons) | agreement and length baseline                                     |

All 14 steps of `temp/run_ragtime26_suite.sh` finished with 0 errors and 0 invalid
answers. Cost: $1.29 for the Jev runs, $3.93 for the probes and padding (including
$2.66 for Gemini padding) and $1.66 for the Gemini pairwise run; **$6.89 in total**.
Jev cost $0.0000962 per Choice call and $0.0001005 per two-question Noul call on
ragtime; Gemini cost $0.000541 per comparison.

### 11.2 Pilot: distribution, agreement, position (Choice, topic 2000)

| Metric                                      | rag26 (4.x)   | ragtime26         |
| ------------------------------------------- | ------------- | ----------------- |
| Valid                                       | 6,446 / 6,446 | 2,048 / 2,048     |
| Exactly 0 or 1                              | 56.2%         | **24.4%**         |
| Decisive (P < 0.25 or > 0.75)               | 90.7%         | 78.7%             |
| In [0.25, 0.75] / in [0.4, 0.6]             | 9.3% / 3.2%   | 21.3% / 8.5%      |
| Mean decisiveness \|2P − 1\|                | 0.890         | 0.751             |
| Probability vs binary win-rate ranking ρ    | 0.9988        | 0.9961            |
| Swap: mean \|ΔP\| (median)                  | 0.042 (0.000) | 0.046 (0.020)     |
| Winner flips on swap                        | 4.4%          | 3.9%              |
| Jev vs Gemini win rates: Spearman / Kendall | 0.973 / 0.868 | **0.949 / 0.815** |

Choice P(A) distribution on topic 2000, by range: [0, 0.01) 286; [0.01, 0.10) 333;
[0.10, 0.25) 211; [0.25, 0.40) 116; [0.40, 0.60) 171; [0.60, 0.75) 135; [0.75, 0.90) 203;
[0.90, 0.99) 272; [0.99, 1.00] 321.

**Top of the board.**

- **Jev:** kurt 0.905 (T289), briar 0.891 (T289), marie 0.882 (T958), kent 0.874 (T289),
  alice 0.869 (T958).
- **Gemini:** alice, alyssa, lori and marie, all T958, are tied at 1.000, then briar
  0.911 (T289).
- **Both judges** put teams T289 and T958 at the top and the same two runs (eliza, hans)
  at the bottom.
- **Gemini's binary output saturates at the top.** Four runs win every comparison, so it
  cannot order them; Jev's probabilities can.

**Finding:** the judges still agree and probabilities still barely change the ranking.
Saturation, however, depends on the dataset: Jev is much less certain on ragtime.

### 11.3 Length dependence

Natural comparisons on the pilot topic (2000):

| Judge                          | P(longer wins) | Ratio < 1.25× | 1.25–2× | ≥ 2×    | ρ(win rate, words) |
| ------------------------------ | -------------- | ------------- | ------- | ------- | ------------------ |
| Jev Choice (both orientations) | **0.480**      | 0.386         | 0.629   | 0.438   | **−0.077**         |
| Gemini (single orientation)    | **0.455**      | 0.366         | 0.603   | 0.407   | **−0.157**         |
| _rag26, Jev (rag2026-0)_       | _0.694_        | _0.546_       | _0.617_ | _0.844_ | _0.524_            |
| _rag26, Gemini (rag2026-0)_    | _0.662_        | _0.520_       | _0.571_ | _0.817_ | _0.436_            |

Prompt variants, pooled over 2000–2002:

| Question       | P(longer) | < 1.25× | 1.25–2× | ≥ 2×  | ρ(win rate, words) | Exactly 0/1 | Decisiveness |
| -------------- | --------- | ------- | ------- | ----- | ------------------ | ----------- | ------------ |
| better_summary | 0.476     | 0.440   | 0.482   | 0.508 | −0.137             | 0.243       | 0.783        |
| length_neutral | 0.445     | 0.440   | 0.446   | 0.449 | −0.209             | 0.277       | 0.775        |
| needs_coverage | 0.489     | 0.447   | 0.492   | 0.535 | −0.075             | 0.317       | 0.786        |
| precision      | 0.425     | 0.424   | 0.429   | 0.423 | −0.262             | 0.213       | 0.776        |

**Finding: the rag26 length correlation does not replicate.**

- On ragtime neither judge favours the longer report. The run-level correlation is zero
  to slightly negative for every question and both judges.
- In particular, the pattern of lopsided pairs favouring the longer summary is gone:
  0.44 at ≥ 2×, against 0.84 on rag26.
- ragtime also has fewer lopsided pairs: 29% of pooled pairs are ≥ 2× apart, against 39%
  on rag26.
- The anti-length questions move further toward shorter reports, as on rag26.
- **Interpretation.** The question and the judges are the same, so the strong rag26
  length effect is a property of rag26's data. It fits the coverage explanation in 5
  and 9: rag26's very short answers to multi-part questions are genuinely incomplete,
  while ragtime's requests and length limits don't produce that pattern. It does not
  fit a fixed length bias in the judges.

### 11.4 Prompt variants

|                                                                                            | rag26                  | ragtime26               |
| ------------------------------------------------------------------------------------------ | ---------------------- | ----------------------- |
| Ranking agreement between questions (Spearman, mean over 3 topics)                         | 0.971–0.993            | **0.909–0.974**         |
| Argmax agreement with control, all pairs                                                   | 0.950–0.971            | 0.926–0.942             |
| Argmax agreement with control, near-equal length                                           | 0.958–0.961            | 0.937–0.961 (n = 1,021) |
| Agreement with Gemini, per question                                                        | 0.955–0.976            | 0.915–0.946             |
| Control re-asked in the 4-question bundle vs pilot: identical / mean \|ΔP\| / argmax agree | 76.4% / 0.0042 / 99.7% | 49.0% / 0.0109 / 98.8%  |

Ragtime ranking agreement in detail: better_summary vs length_neutral 0.971, vs
needs_coverage 0.970, vs precision 0.960; length_neutral vs needs_coverage 0.951, vs
precision 0.974; needs_coverage vs precision 0.909.

Top 8 on 2000:

- **better_summary:** kurt, briar, marie, alyssa, kent, alice, alana, joyce.
- **precision:** kurt, briar, kent, alana, marie, alyssa, lori, debra.

**Finding:** rankings remain stable across wordings, but ragtime is more sensitive to
wording than rag26. needs_coverage and precision, the two most opposed framings, still
agree at 0.909.

### 11.5 Instruction-following probes

| Probe                                                                 | rag26                      | ragtime26                     |
| --------------------------------------------------------------------- | -------------------------- | ----------------------------- |
| Choice `worse_summary`: Pearson with 1 − P_control                    | 0.975                      | 0.939                         |
| … inverted, all pairs / decisive pairs                                | 95.5% / 99.6%              | 90.6% / **98.3%** (n = 1,993) |
| Choice `swapped_labels`: Pearson / inverted on decisive pairs         | 0.976 / 100%               | 0.967 / **99.9%**             |
| Coin-flip control pairs (0.4–0.6): mean P(A) under worse / swapped    | 0.390 / 0.384              | 0.341 / 0.379 (n = 222)       |
| Choice `shorter`: accuracy all / < 1.25× / 1.25–2× / ≥ 2×             | 85.8 / 61.1 / 90.1 / 99.9% | 80.1 / 58.2 / 84.8 / 99.0%    |
| … Spearman with log length ratio                                      | −0.945                     | −0.872                        |
| … Spearman with P(A better), the quality leak                         | −0.675                     | **−0.154**                    |
| … conflict pairs: follows length / inverse quality                    | 67.2% / 32.8% (n = 2,765)  | 74.7% / 25.3% (n = 1,581)     |
| Noul `noul_a_worse`: Pearson with 1 − yA / inverted on decisive pairs | 0.994 / 100%               | 0.993 / **100%** (n = 513)    |
| Noul `noul_a_shorter`: accuracy all / < 1.25× / 1.25–2× / ≥ 2×        | 87.2 / 63.4 / 92.3 / 100%  | 81.3 / 58.9 / 86.9 / 99.4%    |
| … Spearman with mirror P(A better)                                    | −0.719                     | −0.154                        |

**Findings.**

- **Instruction following replicates.** Negation and label swaps invert decisive answers
  at ≥ 98.3% on both datasets.
- **The quality leak into the length question mostly disappears on ragtime**
  (−0.15 against −0.68). This is consistent with 11.3: on ragtime length and judged
  quality are nearly uncorrelated, so a judge that partly answered "shorter" by quality
  would show it less. The leak measured on rag26 was therefore partly a property of the
  data.
- **The slight lean toward summary_a on uncertain pairs** (inverted probes below 0.5)
  appears on both datasets.

### 11.6 Determinism and identical summaries

Determinism: two sequential passes with byte-identical payloads, single orientation.
rag26 topic rag2026-100 (3,223 pairs); ragtime topic 2003 (1,024 pairs). Winner flips
count P = 0.5 as a draw.

| Metric                                             | Choice rag26  | Choice ragtime | Noul mirror rag26             | Noul mirror ragtime           |
| -------------------------------------------------- | ------------- | -------------- | ----------------------------- | ----------------------------- |
| Exactly identical                                  | 75.6%         | 49.2%          | 40.8%                         | 25.4%                         |
| Mean \|ΔP\|                                        | 0.0043        | 0.0096         | 0.0045                        | 0.0071                        |
| p95 / max \|ΔP\|                                   | 0.020 / 0.090 | 0.040 / 0.090  | 0.015 / 0.050                 | 0.020 / 0.100                 |
| Winner flips                                       | 4 (0.12%)     | 6 (0.59%)      | 8 (0.25%)                     | 11 (1.07%)                    |
| Raw Noul answers (yA / yB): identical, mean \|ΔP\| | —             | —              | 58.6 / 57.9%, 0.0051 / 0.0052 | 41.4 / 42.1%, 0.0080 / 0.0082 |
| Tie flag, pass 1 / pass 2 / changed                | —             | —              | 6 / 6 / 4                     | 24 / 21 / 9                   |

Cross-bundle retest of the Noul pair on the pilot topic (2-question pilot vs 4-question
instruction run):

- **ragtime:** `noul_a_better` 42.8% identical, mean |ΔP| 0.0083, winner agrees 98.1%;
  `noul_b_better` 41.6%, 0.0086, 98.8%; mirror mean |ΔP| 0.0074, 3 of 1,024 winners
  flip.
- **rag26:** 60.2% / 61.4% identical, 0.0047 / 0.0046, 99.4% / 99.5%; mirror 0.0041,
  4 of 3,223 flips.

Identical summaries (each run's report judged against itself; rag26 248 calls, ragtime
147):

| Question                                            | rag26          | ragtime26                                            |
| --------------------------------------------------- | -------------- | ---------------------------------------------------- |
| Choice `better_summary`: mean P(A), answers of A    | 0.799, 248/248 | **0.785, 147/147** (per topic 0.736 / 0.833 / 0.787) |
| Noul `noul_a_better` / `noul_b_better`: mean P(yes) | 0.151 / 0.103  | 0.163 / 0.101                                        |
| Implied mirror P(A better) on ties                  | 0.524          | 0.531                                                |

**Findings.**

- **Jev is about twice as noisy on ragtime,** both for repeated requests and across
  question bundles. Winner flips stay around 1% or less.
- **The mirror average flips more often than Choice on both datasets** (0.25% against
  0.12% on rag26; 1.07% against 0.59% on ragtime). It averages two noisy answers into a
  score that sits closer to 0.5. Its p95 and maximum changes are smaller on rag26 but
  not on ragtime.
- **The tie behaviour replicates almost exactly.**
  - Choice picks slot A for every identical pair, at about 0.79 on both datasets.
  - Noul says "no" to both questions on both datasets.
  - The mirror average gives about 0.52–0.53 on both datasets.

### 11.7 Slot bias on real close pairs (pilot, both orientations)

Each cell: number of pairs / mean P(slot A wins) / winner-flip rate when swapped. Groups
are by orientation-averaged P(x better).

| Estimator | Dataset   | 0.05–0.25           | 0.25–0.40          | **0.40–0.60**          | 0.60–0.75          | 0.75–0.95          | All pairs: P(slot A wins) |
| --------- | --------- | ------------------- | ------------------ | ---------------------- | ------------------ | ------------------ | ------------------------- |
| choice    | rag26     | 239 / 0.507 / 0.00  | 107 / 0.527 / 0.29 | **111 / 0.582 / 0.78** | 98 / 0.531 / 0.23  | 217 / 0.507 / 0.00 | 0.505                     |
| choice    | ragtime26 | 141 / 0.486 / 0.00  | 56 / 0.482 / 0.04  | **86 / 0.491 / 0.43**  | 67 / 0.493 / 0.01  | 207 / 0.483 / 0.00 | 0.491                     |
| mirror    | rag26     | 1027 / 0.508 / 0.00 | 200 / 0.526 / 0.00 | **220 / 0.542 / 0.54** | 195 / 0.521 / 0.01 | 851 / 0.511 / 0.00 | 0.512                     |
| mirror    | ragtime26 | 254 / 0.504 / 0.00  | 133 / 0.513 / 0.00 | **145 / 0.520 / 0.28** | 203 / 0.511 / 0.00 | 289 / 0.502 / 0.00 | 0.508                     |

**Finding: the close-pair slot-A bias does not replicate for Choice.**

- On ragtime, Choice shows no slot-A advantage in close pairs (0.491) and a slight
  overall lean toward slot B. The mirror estimator keeps a small slot-A offset (0.52)
  on both datasets.
- **The two biases behave differently.**
  - The tie bias, 0.8 for slot A on identical inputs, is stable across datasets.
  - How much it leaks into real close pairs depends on the dataset.
- **So averaging both orientations is the only protection that works on both datasets.**
  A fixed correction calibrated on ties (open question 14.4 #3) would over-correct
  ragtime.

### 11.8 Noul formulation

| Metric                                                   | rag26 (10.x)                        | ragtime26                                |
| -------------------------------------------------------- | ----------------------------------- | ---------------------------------------- |
| Raw yA + yB: mean / within [0.9, 1.1]                    | 0.983 / 97.1%                       | **0.928 / 81.5%** (18.5% below 0.9)      |
| Pearson(yA, 1 − yB)                                      | 0.996                               | 0.993                                    |
| "No" to both questions (pilot / 3-topic instruction run) | 0.5% / 0.42%                        | **4.3% / 2.1%**                          |
| "Yes" to both questions (pilot / instruction run)        | 0.4% / 0.69%                        | 0% / 1.6%                                |
| Winner agrees between `noul_a` and Choice                | 99.1%                               | 97.3%                                    |
| Mirror: exactly 0/1, decisive, in [0.4, 0.6]             | 0%, 81%, 6.3% (339 distinct values) | 0%, **54%**, 13.9% (266 distinct values) |
| Mirror vs Choice ranking ρ (orientation-averaged)        | 0.999                               | 0.998                                    |
| Mirror vs Gemini (pilot topic): Spearman / Kendall       | 0.970 / 0.860                       | 0.955 / 0.832                            |
| Mirror swap: mean \|ΔP\| / flips                         | 0.036 / 3.7%                        | 0.033 / 4.0%                             |
| Top mirror win rate                                      | 0.886 (carmen)                      | 0.779 (kurt)                             |

**Findings.**

- **Noul ranks like Choice on both datasets** (ρ ≥ 0.998).
- **On ragtime Jev answers "no" to both questions far more often.** The two answers sum
  to noticeably less than 1, and mirror probabilities are much more compressed (only
  54% decisive). So "not better than" is a more frequent answer for real ragtime
  pairs.
- **The tie flag is somewhat more meaningful on ragtime but still unstable.** 9
  comparisons changed tie status between two identical passes on 2003; 18 were flagged
  in both.

### 11.9 Padding test

Same design as Section 9: 16 mid-ranked targets per topic on 2000–2002 (48 targets), 8
opponents, 6 variants, both orientations, 4,608 calls per judge. ΔP is relative to the
unmodified original, with 95% bootstrap CIs over targets.

| Variant          | Jev Choice: rag26 → **ragtime26** [95% CI] | Noul mirror: rag26 → **ragtime26** [95% CI] | Gemini: rag26 → **ragtime26** [95% CI] |
| ---------------- | ------------------------------------------ | ------------------------------------------- | -------------------------------------- |
| original, P(win) | 0.279 → 0.362                              | 0.327 → 0.412                               | 0.281 → 0.368                          |
| pad_repeat       | −0.050 → **−0.105** [−0.124, −0.088]       | −0.048 → **−0.087** [−0.100, −0.076]        | −0.171 → **−0.203** [−0.249, −0.160]   |
| pad_generic      | −0.089 → **−0.174** [−0.196, −0.152]       | −0.097 → **−0.156** [−0.172, −0.141]        | −0.161 → **−0.186** [−0.224, −0.150]   |
| pad_offtopic     | −0.194 → **−0.279** [−0.312, −0.246]       | −0.191 → **−0.274** [−0.298, −0.251]        | −0.224 → **−0.294** [−0.352, −0.238]   |
| pad_relevant     | +0.183 → **+0.094** [+0.072, +0.115]       | +0.160 → **+0.072** [+0.056, +0.087]        | +0.016 → **−0.104** [−0.146, −0.065]   |
| truncate         | −0.087 → **−0.117** [−0.137, −0.097]       | −0.082 → **−0.099** [−0.113, −0.085]        | −0.107 → **−0.152** [−0.195, −0.113]   |

Targets improved on ragtime (Jev / Noul / Gemini):

- pad_repeat: 0 / 0 / 0
- pad_generic: 0 / 0 / 0
- pad_offtopic: 0 / 0 / 0
- **pad_relevant: 42 / 43 / 4 of 48**
- truncate: 1 / 1 / 1

Residual P(slot A wins) after averaging both orientations: 0.494, 0.514 and 0.467. The
Noul tie flag fired on 66 of 4,608 calls (1.4%).

ΔP by ragtime topic:

| Topic | Judge  | pad_repeat | pad_generic | pad_offtopic | pad_relevant | truncate |
| ----- | ------ | ---------- | ----------- | ------------ | ------------ | -------- |
| 2000  | Jev    | −0.111     | −0.140      | −0.227       | +0.049       | −0.131   |
| 2000  | Noul   | −0.090     | −0.129      | −0.229       | +0.035       | −0.105   |
| 2000  | Gemini | −0.223     | −0.191      | −0.309       | −0.109       | −0.234   |
| 2001  | Jev    | −0.096     | −0.169      | −0.275       | +0.100       | −0.097   |
| 2001  | Noul   | −0.078     | −0.144      | −0.268       | +0.076       | −0.087   |
| 2001  | Gemini | −0.164     | −0.160      | −0.246       | −0.105       | −0.086   |
| 2002  | Jev    | −0.110     | −0.214      | −0.334       | +0.132       | −0.123   |
| 2002  | Noul   | −0.094     | −0.196      | −0.325       | +0.104       | −0.106   |
| 2002  | Gemini | −0.223     | −0.207      | −0.328       | −0.098       | −0.137   |

**Findings.**

1. **"Content, not length" replicates, and more strongly.** Every uninformative
   padding variant lowers P(win) for all three judges, on every topic, for every target
   but at most one. The Jev penalties are roughly twice as large on ragtime.
2. **Jev still credits added relevant content, but about half as much.** It improved
   42–43 of 48 targets, against 48 of 48 on rag26.
3. **Gemini now penalises relevant padding** (−0.104; only 4 of 48 targets improve; all
   3 topics negative). On rag26 the effect was null (+0.016). This strengthens the
   hypothesis from 9.3: Gemini penalises spliced or edited text regardless of the
   information it adds, while Jev credits the coverage. Judge splice-free relevant
   additions (open question 14.4 #2) to confirm.
4. **Baseline P(win) is higher on ragtime for all judges** (0.36–0.41 against
   0.28–0.33). The mid-ranked targets are closer to their opponents there.

### 11.10 What generalises

| Finding (rag26)                                        | ragtime26                                               | Status                                     |
| ------------------------------------------------------ | ------------------------------------------------------- | ------------------------------------------ |
| Jev and Gemini rank runs nearly identically            | ρ 0.949–0.955 (pilot), 0.915–0.946 (3 topics)           | **Generalises**, slightly weaker           |
| Probabilistic and binary scoring give the same ranking | ρ 0.996 (Choice), 0.993 (mirror)                        | **Generalises**                            |
| Jev's probabilities are mostly saturated               | 24% exactly 0/1 (rag26 56%)                             | **Dataset-dependent**                      |
| Ranking is robust to question wording                  | inter-question ρ 0.91–0.97                              | **Generalises**, more sensitive            |
| Jev follows negation and option relabelling            | ≥ 98.3% on decisive pairs                               | **Generalises**                            |
| Longer summaries win (P ≈ 0.69, ρ ≈ 0.5)               | P ≈ 0.46–0.48, ρ ≈ −0.08 to −0.16                       | **Does not generalise**: property of rag26 |
| Judges reject crude padding (content, not length)      | larger penalties, every target                          | **Generalises**, stronger                  |
| Jev credits relevant added content                     | +0.094 / +0.072                                         | **Generalises**, weaker                    |
| Gemini does not credit relevant padding                | now significantly negative                              | **Generalises**, stronger                  |
| Choice tie bias: identical inputs get P(A) ≈ 0.8       | 0.785, every pair slot A                                | **Generalises**                            |
| Close real pairs favour slot A (0.58)                  | 0.49                                                    | **Does not generalise**                    |
| Mirrored Noul maps ties to about 0.5                   | 0.531                                                   | **Generalises**                            |
| Mirrored Noul reduces close-pair slot bias             | Choice has none to reduce on ragtime; mirror keeps 0.52 | **Dataset-dependent**                      |
| Jev is near-deterministic                              | about 2× noisier; flips ≤ 1.1%                          | **Generalises**, noisier                   |
| Mirror score is at least as repeatable as Choice       | it flips more on both datasets                          | **Corrected**: Choice flips less           |
| Noul answers are near-complementary; ties rare         | sum 0.93, both "no" 2–4%                                | **Dataset-dependent**                      |

---

## 12. Experiment 9 — Depth-proportionate prompt

**Motivation.** Some information needs require long answers to cover the breadth of the
topic; others need a short, direct answer. A good judge should reward depth that is
*proportionate* to the need, rather than always preferring longer or shorter answers.
The `noul_*_better` pair has no explicit notion of this.

We tested two new mirrored Noul pairs. Both are verbatim in 16.3.

- **`noul_a_appropriate` / `noul_b_appropriate`.** "Does summary_a satisfy the
  information need … *more appropriately* than summary_b does, considering relevance,
  **conciseness**, completeness, and how well its claims are grounded?"
- **`noul_a_proportionate` / `noul_b_proportionate`.** The `better` question plus an
  explicit rule: "…considering relevance, completeness, **depth proportionate to what
  the query asks for (thorough when the query is broad or multi-part, brief and direct
  when it is narrow)**, and how well its claims are grounded?"

### 12.1 "Appropriate" pilot: adding "conciseness"

**Setup:** pilot topics rag2026-0 and 2000, both orientations. The `better` and
`appropriate` pairs were asked in the **same request** (4 questions), so every
comparison has both answers from identical input.

| Metric | rag26 (rag2026-0): better → appropriate | ragtime26 (2000): better → appropriate |
| --- | --- | --- |
| P(longer wins) | 0.669 → 0.653 | 0.478 → 0.468 |
| … ratio ≥ 2× | 0.803 → 0.783 | 0.445 → 0.425 |
| Spearman(win rate, words) | 0.531 → 0.494 | −0.088 → −0.151 |
| Mean per-comparison shift toward the longer summary | −0.015 | −0.010 |
| Spearman(that shift, length ratio) | −0.674 | −0.518 |
| Winner changes | 1.9% | 1.8% |
| Run ranking, better vs appropriate | 0.9955 | 0.9933 |
| Spearman vs Gemini | 0.970 → 0.980 | 0.954 → 0.962 |
| Tie flag ("no" to both) | 0.5% → 0.9% | 4.6% → 4.8% |

- **"Conciseness" works as a uniform length penalty.** It pushes every comparison
  slightly toward the shorter summary, with the shift growing with the length gap, on
  both datasets.
- **It penalises length even on rag26, where long answers are the complete ones.** The
  runs that dropped most on rag2026-0 are those at the 1,020-word ceiling: finn 11→19,
  eden 10→16, rita 8→14. darcy (674 words) rose 13→7.
- **The effect is real but small.** The winner changes in about 2% of comparisons,
  against about 0.1% between repeats of the same question.
- **Agreement with Gemini rises slightly** on both datasets. This does not validate the
  prompt: neither judge has been checked against human judgments.

### 12.2 Broad vs narrow test

The pilot topics both ask for broad, long-form answers, so they cannot show whether a
prompt *adapts*. Neither dataset has genuinely narrow needs: every rag26 topic asks for
an article, essay, guide or report, and every ragtime topic is a report request with the
same 5,000-character limit. rag24 and rag25 are not available locally.

**Design: a controlled query manipulation.** The *same* report pairs are judged under
two framings of the request:

- **Broad:** the original topic.
- **Narrow:** a single sub-question taken from the topic's own text, with no
  problem statement or background.

All three mirrored pairs (`better`, `appropriate`, `proportionate`) were asked in each
request, single orientation (hashed, identical across framings), with topics selected
with `--topic`.

| Dataset | Topic | Narrow question |
| --- | --- | --- |
| rag26 | rag2026-1 | "My 72-year-old mother was widowed six weeks ago and has chest tightness. What warning signs should we take seriously?" |
| rag26 | rag2026-107 | "What single post-395 CE turning point could realistically have ensured the Western Roman Empire's survival?" |
| ragtime26 | 2041 | "What percentage of Denmark's energy needs are now provided by wind power?" (verbatim from the request) |
| ragtime26 | 2046 | "What type of grapes are used to make Sherry?" |

**Measure.** For each pair, the change in P(longer wins) from broad to narrow. The key
number is the **difference-in-differences against `better`** on the same comparisons:
negative means the pair moves toward shorter answers on the narrow request *more* than
`better` does, which is the intended behaviour.

| Topic (n) | better: broad → narrow | appropriate: broad → narrow, DiD | **proportionate: broad → narrow, DiD** |
| --- | --- | --- | --- |
| rag2026-1 (3,223) | 0.670 → 0.627 | 0.650 → 0.603, −0.003 | 0.674 → 0.597, **−0.033** |
| rag2026-107 (3,143) | 0.621 → 0.615 | 0.597 → 0.591, −0.001 | 0.607 → 0.583, **−0.018** |
| 2041 (1,024) | 0.572 → 0.561 | 0.555 → 0.548, +0.004 | 0.574 → 0.539, **−0.024** |
| 2046 (1,024) | 0.541 → 0.475 | 0.527 → 0.452, −0.009 | 0.545 → 0.438, **−0.041** |
| **Pooled rag26 / ragtime26** | shift −0.025 / −0.039 | DiD −0.002 / −0.003 | **DiD −0.026 / −0.032** |

| Pooled metric | rag26: better / appropriate / proportionate | ragtime26: better / appropriate / proportionate |
| --- | --- | --- |
| Spearman(win rate, words), broad → narrow | 0.48→0.41 / 0.42→0.34 / **0.47→0.32** | 0.14→0.06 / 0.10→−0.07 / **0.15→−0.13** |
| Run ranking, broad vs narrow | 0.968 / 0.956 / 0.945 | 0.876 / 0.860 / 0.789 |
| Tie flag, broad → narrow | 0.5→2.2% / 0.8→2.6% / **0.7→5.1%** | 5.4→6.6% / 4.9→5.7% / **4.7→10.9%** |
| Winner differs from `better` — appropriate: broad / narrow | 2.4% / 3.2% | 2.2% / 4.5% |
| Winner differs from `better` — proportionate: broad / narrow | 1.6% / 3.4% | 1.5% / 6.3% |

**Findings.**

1. **"Proportionate" adapts to the request; "appropriate" does not.**
   - Under the broad request, proportionate behaves like `better`: pooled P(longer)
     0.641 against 0.646 on rag26, and 0.559 against 0.557 on ragtime.
   - Under the narrow question it moves 2–4 points further toward shorter answers than
     `better` does, on **all four topics**.
   - Appropriate's extra shift is about zero and its sign varies by topic: the same
     uniform length penalty in both framings.
2. **`better` also responds a little to a narrower request** (−0.025 / −0.039 pooled).
   The explicit proportionality wording roughly doubles that response.
3. **Under the narrow framing, the run–length correlation falls the most for
   proportionate** (rag26 0.47 → 0.32; ragtime 0.15 → −0.13). Its rankings also change
   most between framings, which is expected of a judge that responds to the request.
4. **Narrow questions raise the Noul tie rate most for proportionate** (to 5.1% and
   10.9%). For a one-line question, two full reports are often both "not
   proportionate", and Jev says "no" to both.

**Caveats.**

- The narrow requests are **synthetic**: the reports were written for the broad
  request, so this tests whether the judge responds to the request, not performance on
  real short-answer needs.
- The bootstrap CIs over comparisons (for example −0.0256 [−0.0267, −0.0245] pooled
  rag26) are **too narrow**, because comparisons share runs. The stronger evidence is
  that the direction holds on 4 of 4 topics.
- The effect is **modest** (2–4 points of P(longer)).
- On broad requests, proportionate ranks almost exactly like `better` (winner differs
  1.5–1.6%). On rag26 and ragtime26, which contain only broad requests, the choice
  barely matters; it would matter on datasets that mix narrow and broad needs.
- Validate on real narrow needs (for example rag24 or rag25) and against human
  judgments.

**Decision.** The full runs (Section 13) use the **proportionate** pair. It costs nothing
on broad requests and behaves correctly when a request is narrow.

**Cost:** the appropriate pilots $0.85 (7,948 calls); the broad/narrow test $1.85
(16,268 calls, 6 questions per request).

---

## 13. Full runs: mirrored proportionate Noul on all topics

**Setup.**

- Workflow variant `full_noul_proportionate`: questions
  `noul_a_proportionate` + `noul_b_proportionate`, mirror average as the score, **both
  orientations**, all topics.
- ragtime26 ran first, then rag26 (`temp/run_full_proportionate.sh`, under nohup).
- Snapshot served throughout: `typesafe/jev-1.13-20260917`.

| | ragtime26 | rag26 |
| --- | --- | --- |
| Topics / runs / teams | 103 / 49 / 10 | 119 / 83 / 25 |
| Comparisons (ordered) | 210,418, **all valid**, 0 missing | 760,886, **all valid**, 0 missing |
| Unique calls | 196,958 | 703,536 |
| Cost (unique calls) | **$18.95** ($0.0000962/call; estimate $18.5) | **$74.89** ($0.0001065/call; estimate $72) |
| Wall-clock | 3.0 h (about 18 calls/s) | 11.1 h (about 18 calls/s) |

- The first ragtime26 attempt (about 4,900 calls) was restarted after 8 minutes, only
  to get unbuffered progress output. Its answers were served from the cache; at most
  the ≤32 requests in flight were paid twice.
- The leaderboard is the **mean over topics of each run's per-topic mirror win rate**
  (every topic weighs equally). It is written to `bonsai_pairwise_jev.eval.txt` as
  `PAIRWISE_WINRATE` and related measures.

### 13.1 Leaderboards and stability

Rank CIs are 95% bootstraps over topics (500 resamples).

**rag26 (top 12 of 83).**

| Rank | Run | Team | Win rate | Rank CI |
| --- | --- | --- | --- | --- |
| 1 | carmen | T569 | 0.8611 | [1, 1] |
| 2 | lars | T569 | 0.8564 | [2, 2] |
| 3 | edith | T569 | 0.8456 | [3, 5] |
| 4 | ariel | T569 | 0.8453 | [3, 5] |
| 5 | hana | T300 | 0.8429 | [3, 5] |
| 6 | xavi | T300 | 0.8397 | [6, 6] |
| 7 | luca | T300 | 0.8295 | [7, 7] |
| 8 | yara | T649 | 0.8216 | [8, 8] |
| 9 | todd | T736 | 0.8047 | [9, 11] |
| 10 | eve | T341 | 0.8029 | [9, 12] |
| 11 | rita | T736 | 0.8024 | [9, 12] |
| 12 | eden | T736 | 0.8017 | [10, 12] |

- Bottom: carlo 0.116, annie 0.048, paula 0.044.
- Teams by best run: T569 0.861, T300 0.843, T649 0.822, T736 0.805, T341 0.803,
  T267 0.794, T410 0.744, T086 0.720, T448 0.707, T149 0.696.

**ragtime26 (top 12 of 49).**

| Rank | Run | Team | Win rate | Rank CI |
| --- | --- | --- | --- | --- |
| 1 | marie | T958 | 0.8093 | [1, 2] |
| 2 | alice | T958 | 0.8083 | [1, 2] |
| 3 | lori | T958 | 0.7956 | [3, 4] |
| 4 | alyssa | T958 | 0.7950 | [3, 4] |
| 5 | briar | T289 | 0.7127 | [5, 9] |
| 6 | kent | T289 | 0.7112 | [5, 10] |
| 7 | bree | T706 | 0.7065 | [5, 9] |
| 8 | dina | T846 | 0.7035 | [5, 11] |
| 9 | diana | T706 | 0.7021 | [6, 10] |
| 10 | susan | T706 | 0.6971 | [7, 11] |
| 11 | joyce | T706 | 0.6882 | [10, 12] |
| 12 | kurt | T289 | 0.6770 | [10, 13] |

- Bottom: hans 0.171, faye 0.110, adair 0.091.
- Teams by best run: T958 0.809, T289 0.713, T706 0.706, T846 0.704, T673 0.587,
  T149 0.575, T768 0.545, T986 0.522, T131 0.497, T047 0.470.

**Stability.**

| Metric | rag26 | ragtime26 |
| --- | --- | --- |
| Split-half reliability over topics (Spearman, 200 random splits) | **0.998** [0.997, 0.999] | **0.989** [0.979, 0.994] |
| Single topic vs overall leaderboard (Spearman): median [IQR], min | 0.968 [0.957, 0.976], 0.886 | 0.910 [0.855, 0.930], 0.592 |
| Rank-CI width: all runs / top 10 (median) | 2 / 1 | 2 / 4 |

- **rag26:** the top 8 positions are essentially fixed. T569 occupies ranks 1–4 and
  T300 ranks 5–7.
- **ragtime26:** T958's four runs are clearly separated at the top (0.80–0.81). Ranks
  5–12 form a tight cluster (0.68–0.71) whose order is not well determined.

**Duplicate topics (ragtime26).**

- ragtime26 contains **28 duplicated requests** (56 of its 103 topics). The systems
  produced *different* reports for each copy: only 2 of about 1,370 reports have
  identical text.
- This gives an independent judge-plus-system consistency check. Per-run win rates
  agree across the two copies at **Spearman median ≈ 0.90** (range 0.832–0.980 over
  28 pairs).

### 13.2 Agreement with Gemini

| | rag26 | ragtime26 |
| --- | --- | --- |
| Shared topics with Gemini judgments | 80 (all topics present in the partial Sept-2026 Gemini run, single orientation; 78 were complete) | 3 (2000–2002) |
| Overall leaderboard: Spearman / Kendall | **0.975 / 0.870** | 0.939 / 0.818 |
| Per-topic Spearman: median [IQR], min | 0.957 [0.946, 0.967], 0.883 | 0.929 [0.917, 0.957], 0.917 |
| Top-10 / top-20 overlap | 7/10, 18/20 | 6/10, 16/20 |

- On rag26 one Gemini topic had constant win rates and is excluded from the per-topic
  statistics.
- **Gemini top 8, rag26:** lars, hana, luca, ariel, edith, xavi, carmen (T569/T300),
  rick (T267).
- **Gemini top 8, ragtime26:** alyssa, lori, alice, marie (T958), briar (T289), jesse,
  alana, loren.

**Largest rank differences, Jev → Gemini.**

- **rag26:** rita 10→22, todd 9→20, finn 13→23 (all about 1,015 words), karl 54→67
  (912 words); in the other direction, jamie 21→11 (970 words).
- **ragtime26:** loren 20→8 (323 words) and rory 22→10 (315 words) rank much higher
  with Gemini; diana 11→20 (535 words), dina 12→21 (700 words) and kent 7→16 (564
  words) rank lower.

**Pattern (hypothesis):** on both datasets the disagreements point to Gemini penalising
length relative to Jev. That matches the padding tests (9.3, 11.9), where Gemini
penalised any added text. The ragtime evidence rests on only 3 topics.

### 13.3 Length, slot bias and ties at full scale

| Metric | rag26 | ragtime26 |
| --- | --- | --- |
| P(longer wins), orientation-averaged | **0.674** | **0.489** |
| … < 1.25× / 1.25–2× / ≥ 2× | 0.513 / 0.620 / 0.819 | 0.431 / 0.498 / 0.532 |
| Per-topic Spearman(win rate, words): median [IQR] | **+0.562** [0.526, 0.602] | **−0.082** [−0.193, 0.070] |
| Run-level Spearman(leaderboard, mean words) | +0.585 | −0.112 |
| P(slot A wins): all pairs / close pairs (0.4–0.6) | 0.518 / 0.543 | 0.508 / 0.514 |
| Winner flips on swap | 4.7% | 5.8% |
| Decisive (P < 0.25 or P > 0.75) / P in [0.4, 0.6] | 76.1% / 8.3% | 58.9% / 14.0% |
| Flagged near-ties ("no" to both) / both "yes" | 0.92% / 2.17% | 2.58% / 1.73% |
| Topic tie rate: median / max | 0.54% / 6.4% (rag2026-28) | 1.42% / 26.3% (2043) |

**Findings.**

1. **The length finding holds at full scale and is dataset-dependent.**
   - rag26 rewards longer reports, almost entirely in lopsided pairs (0.82 at ≥ 2×);
     ragtime26 does not (per-topic ρ ≈ −0.08).
   - The proportionate question does not remove rag26's length effect. That is
     consistent with rag26's long answers being the more complete ones (Sections 5, 9
     and 11.3), though human judgments are still needed to confirm it.
2. **Judging both orientations was necessary.** The mirror estimator keeps a residual
   slot-A lean (0.543 on rag26 close pairs). Averaging the two orientations cancels it
   in the win rates; a single-orientation run would have biased rag26's close pairs.
3. **Ties are a property of topics, not of pairs.**
   - The per-pair tie flag was unstable (10.9.4), but **topic-level tie rates
     replicate across duplicate topics** (Spearman 0.816 over 28 duplicate pairs; for
     example 2009/2013 at 10.7% and 9.7%, 2062/2096 at 6.5% and 6.3%).
   - The highest tie rates are the two copies of one request ("The Joyce Hatto Scam",
     2043 and 2023: 26.3% and 12.6%). These have unusually short reports (medians 279
     and 358 words) and some empty ones.
   - On ragtime, topic tie rate correlates moderately with shorter reports (Spearman
     with median words −0.35); on rag26 there is no relation (+0.05).
   - **Hypothesis:** a high topic tie rate flags topics where most systems failed the
     request, so Jev judges neither report proportionate. If confirmed, the per-topic
     tie rate is a useful topic-difficulty or system-failure diagnostic.

### 13.4 Caveats

- **No human judgments yet**, so all of this is agreement, consistency and stability,
  not accuracy.
- **The Gemini comparison is limited:** a partial run on 80 rag26 topics (single
  orientation, binary) and only 3 ragtime topics.
- **Bradley–Terry** counts each orientation as a separate fractional game, so
  strengths are on an uncalibrated scale; the leaderboard uses mean per-topic win rate.
- **The leaderboards are judge outputs on the evaluation set.** They are reported here
  as results, not used to tune the judge.

**Outputs** (excluded from git, 0.15 GB and 0.5 GB):

- `output-ragtime26-jev-full-prop/` and `output-pairwise-jev-full-prop/`, each with
  `bonsai_pairwise_jev.eval.txt` and `bonsai_pairwise_jev.pairwise/{comparisons.jsonl,
  leaderboard_overall.csv, bradley_terry.csv, q_mirror/, run_manifest.json}`.
- Analysis: `temp/jev_full_analysis.py`, with results in
  `temp/jev_full_analysis_{rag26,ragtime26}.txt`.

---

## 14. Discussion

### 14.1 Answers to the research questions

| RQ                                             | Answer                                                                                                                                                                                                                                                                         |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| RQ1 Jev vs Gemini agreement                    | High. ρ = 0.973 on rag2026-0 and 0.955–0.976 across 3 topics; full rag26 run vs Gemini on 80 topics 0.975 (13.2). Both judges have the same top teams.                                                                                                                                                                       |
| RQ2 Do probabilities matter?                   | Barely, for ranking: soft and binary win rates have ρ = 0.997–0.999 for both Choice (56% saturated) and Noul (never saturated, 0.02–0.98). Without ground truth we cannot say whether probabilities improve accuracy.                                                    |
| RQ3 Length bias                                | Not for crude padding, for either judge or either Jev question form; all penalise uninformative additions. The rag26 length correlation (ρ about 0.5) is consistent with coverage and does not replicate on ragtime26 (ρ ≈ −0.1). Subtle on-topic verbosity is untested. |
| RQ4 Instruction following / prompt sensitivity | Jev follows negation and option relabelling at ≥ 99.6% on decisive pairs for Choice, and follows negation at 100% for Noul. Rewording the quality question changes 4–5% of pair decisions but not the ranking (ρ ≥ 0.97).                                                |
| RQ5 Reliability | Near-deterministic: on rag26 Choice has 75.6% identical answers and 0.12% winner flips, and the Noul mirror 0.25% flips (mean absolute change 0.0045). ragtime26 is about 2× noisier (0.59% / 1.07% flips). Choice has a strong first-slot bias on ties on both datasets (identical summaries give P(A) ≈ 0.79). On rag26 it drives close-pair outcomes; on ragtime26 it does not (11.7). A mirrored Noul pair gives 0.52 on ties and halves the close-pair slot advantage (0.58 → 0.54). Probabilities are uncalibrated in both forms. Full runs: leaderboards have split-half reliability 0.998 / 0.989, and runs agree at about 0.90 across ragtime's duplicated topics (13.1). |
| RQ6 Depth proportionality | Partly. Adding "conciseness" acts as a uniform length penalty. An explicit proportionality rule matches the original on broad requests and shifts 2–4 points further toward shorter answers when the request is narrowed, on 4 of 4 topics (12.2). This is shown only with synthetic narrow requests, and accuracy is untested. |

### 14.2 Recommended protocol for a full Jev run

1. **Judge both orientations and average P(x better) per pair.** For the full
   tournament this is 760,886 ordered comparisons, or **703,536 calls** after
   deduplicating identical payloads (92.5%). A cheaper alternative is to re-judge only
   pairs that are close after one pass (0.25 ≤ P ≤ 0.75) in the other order: about 9%
   of comparisons for Choice and about 20% for Noul.
   - **Cost estimate, mirrored Noul, both orientations: rag26 about $70, ragtime26
     about $18.**
     - Method: scale the measured pilot cost (each actual call counted once) by the
       total input words over all unique requests.
     - rag26: $0.5775 for 5,978 calls; 703,536 calls; 1,424 words per call over all
       topics against 1,378 on rag2026-0. A naive calls × pilot-rate estimate gives $68.
     - ragtime26: $0.1980 for 1,970 calls; 196,958 calls; 1,236 words per call against
       1,363 on topic 2000. Naive estimate $20.
     - _Correction:_ earlier estimates ($75 for rag26) used a pilot cost that counted
       requests shared by several comparisons more than once ($0.6192 instead of
       $0.5775).
   - **Single orientation: about $35 (rag26) and $9 (ragtime26).**
   - Choice is about 5% cheaper per call.
   - Budget about 10% headroom. Wall-clock time at the pilot's rate (about 18 calls/s
     at 32 concurrency) is about 10.5 h for rag26 and about 3 h for ragtime26.
   - Check the OpenRouter key's total spend limit before starting. The Gemini run
     stopped on that limit (`HANDOFF-rag26-pairwise-judge.md`).
2. **Use orientation-averaged win rate as the headline metric.** Report BT on a log
   scale, or with regularisation, because saturated outcomes stretch the strengths.
3. **Choose the question form.** *Used for the full runs: mirrored proportionate Noul
   (Section 12), with both orientations averaged.*
   - **Mirrored Noul** (`noul_a_better` + `noul_b_better`, using the mirror average):
     correct on ties on both datasets and graded probabilities. It reduced close-pair
     slot bias on rag26, but it flips more winners between identical requests than
     Choice, and on ragtime26 Choice had no close-pair bias for it to reduce (11.7).
   - **Choice** (`better_summary`): equally good for _ranking_ (ρ ≥ 0.998 between
     them on both datasets) and slightly more repeatable.
   - With both orientations averaged, either form is defensible. Prefer mirrored Noul
     if per-pair probabilities or ties will be analysed or reported; prefer Choice for
     the most repeatable ranking.
   - With Choice, asking `worse_summary` in the same request is a cheap
     self-consistency check: average P(better) with 1 − P(worse).
4. **Pin `typesafe/jev-1.13` and record the served snapshot.** Cache everything, since
   reproducibility comes from the cache.

**Actual full-run cost** (Section 13): $18.95 for ragtime26 (estimate $18.5) and
$74.89 for rag26 (estimate $72). That is 3–4% above estimate, because the proportionate
question text is longer. Throughput was about 18 calls/s.

**Code status.** As of this writing, the scorer counts both orientations of a pair as
separate games. For win rate this is equivalent to averaging. BT should be changed to
fit on the orientation-averaged probability per pair.

The judge supports Noul questions (`x_polarity` metadata, raw answers stored as
`raw_p`). It also computes the mirror average as the primary score when `mirror:` is set,
flags near-ties, and scores identical-text pairs as 0.5 without a call (Section 10.8).
The `noul_pilot` and `noul_instruction_check` variants use it, and so does
**`full_noul`**, the full-run variant for the recommended protocol: all topics, both
orientations, scored by the mirror average. On one topic it replayed the pilot's cached
answers with 0 new calls. The full runs actually used **`full_noul_proportionate`**,
the same design with the proportionate pair; its runner is
`temp/run_full_proportionate.sh` (resumable).

```bash
auto-judge run --workflow judges/bonsai_judge/workflow.pairwise_jev.yml --variant full_noul \
  --rag-responses data/rag26/runs/generation/ \
  --rag-topics data/rag26/topics/trec_rag_2026_queries.jsonl --out-dir ./output-pairwise-jev-full-noul/
```

### 14.3 Threats to validity

- **No ground truth.** All conclusions are about agreement, consistency and controlled
  sensitivity, not accuracy. Two judges agreeing can share a bias.
- **Synthetic narrow requests (Section 12).** The proportionality test narrows existing
  broad topics; the reports were written for the broad request. Neither dataset
  contains real narrow information needs.
- **Few topics.** 1–3 topics per experiment, chosen by lexicographic order (rag2026-0,
  -1, -10, -100; ragtime 2000–2003), not at random. The ragtime26 replication
  (Section 11) shows that some effects are dataset-specific: the length correlation
  and close-pair slot bias. Conclusions drawn from one dataset should not be assumed
  to transfer.
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

### 14.4 Open questions and next experiments

1. **On-topic verbosity padding.** Have an LLM rewrite each target to about 1.5× its
   length with no new facts (restatement, hedging, background), then judge with Jev and
   Gemini. This is the decisive test for RQ3.
2. **Splice-free relevant padding.** Have an LLM integrate missing facts smoothly. This
   separates "Gemini penalises disruption" from "Gemini doesn't credit coverage".
3. **Slot bias on more questions and topics.** Choice gives P(A) ≈ 0.8 on ties on both
   datasets, but close-pair slot bias is 0.58 on rag26 and 0.49 on ragtime26 (11.7). A
   fixed correction calibrated on ties would therefore over-correct ragtime. Open: what
   drives the close-pair difference between datasets, and is it stable across more
   topics?
4. **Ground truth.** Hand-label about 50–100 pairs, stratified by closeness and length
   ratio, to estimate accuracy and test whether probabilities help on close pairs.
5. **Full-run comparison.** Done for Jev (Section 13). Still open: complete the Gemini
   tournament (41 rag26 topics missing, ragtime26 has only 3) to compare full
   leaderboards. Also test the observation that disagreements point to Gemini
   penalising length.
6. **Length-controlled scoring.** Fit BT with a log-length-ratio covariate,
   logit P(i ≻ j) = s_i − s_j + β·log(len_i/len_j), and report β per judge as a
   summary statistic of length dependence. This uses existing data at no API cost.
7. **Noul under the other probes.** Done (Section 10.9): mirrored Noul matches or
   improves on Choice for determinism, instruction following and padding. Still open: a
   Noul prompt-variant study (Section 6 equivalent), and whether a clear "no" to both
   questions ever occurs for summaries that differ but are equally good.
8. **Proportionality on real narrow needs.** Repeat the Section 12 test on a dataset
   with genuinely short-answer topics (for example rag24 or rag25), and check against
   human judgments.
9. **Topic tie rate as a diagnostic.** Test whether high topic tie rates (13.3) mark
   topics where most systems failed, for example against retrieval quality, empty or
   short reports, or eventual human scores.

### 14.5 Cost summary of this study

| Item                                                                                 | Calls  | Cost         |
| ------------------------------------------------------------------------------------ | ------ | ------------ |
| Jev Router smoke test (kiddie)                                                       | 12     | $0.003       |
| Jev smoke test (kiddie)                                                              | 12     | $0.0003      |
| Jev pilot (rag2026-0, both orientations)                                             | 5,978  | $0.55        |
| Prompt variants (3 topics × 4 questions)                                             | 9,209  | $1.08        |
| Instruction probes (3 topics × 4 questions)                                          | 9,209  | $1.01        |
| Determinism (rag2026-100, 2 passes)                                                  | 6,446  | $0.61        |
| Identical summaries                                                                  | 248    | $0.02        |
| Padding test, Jev                                                                    | 4,608  | $0.53        |
| Padding test, Gemini                                                                 | 4,608  | $3.29        |
| Noul pilot (rag2026-0, both orientations, 2 questions)                               | 5,978  | $0.58        |
| Identical summaries, Noul (2 questions × 248)                                        | 496    | $0.05        |
| Noul instruction probes (3 topics × 4 questions)                                     | 9,209  | $1.00        |
| Noul determinism (rag2026-100, 2 passes × 2 questions)                               | 6,446  | $0.64        |
| Padding test, Noul mirror                                                            | 4,608  | $0.55        |
| Duplicate-check test (kiddie copy)                                                   | 6      | $0.0003      |
| **Subtotal, rag26 study**                                                            |        | **≈ $9.91**  |
| ragtime26: Jev runs (Choice and Noul pilots, prompts, both instruction checks)       | 12,691 | $1.29        |
| ragtime26: identical summaries (Choice and Noul) and determinism (Choice and mirror) | 4,537  | $0.39        |
| ragtime26: padding (Jev, Noul, Gemini)                                               | 13,824 | $3.54        |
| ragtime26: Gemini pairwise (2000–2002, single orientation)                           | 3,072  | $1.66        |
| **Subtotal, ragtime26 replication**                                                  |        | **≈ $6.89**  |
| Appropriate pilots (rag2026-0 and 2000, both orientations, 4 questions)              | 7,948  | $0.85        |
| Broad vs narrow proportionality test (4 topics × 2 framings, 6 questions)            | 16,268 | $1.85        |
| **Subtotal, experiments**                                                            |        | **≈ $19.49** |
| Full run, ragtime26 (mirrored proportionate Noul, both orientations)                 | 196,958 | $18.95      |
| Full run, rag26 (mirrored proportionate Noul, both orientations)                     | 703,536 | $74.89      |
| **Subtotal, full runs**                                                              |        | **≈ $93.84** |
| **Total**                                                                            |        | **≈ $113.33** |

The OpenRouter key's usage rose by $93.96 over the full runs, which matches the
$93.84 computed from per-call `usage.cost` to within the few requests in flight when
the first ragtime attempt was restarted.

Jev costs count each actual call once. Earlier versions of this table summed per
comparison, which counted requests shared by several comparisons more than once and
overstated the Jev runs by 4–7% (old total ≈ $10.11 for rag26).

For reference, the partial Gemini tournament (78/119 topics, single orientation) cost
about $149.

---

## 15. Reproduction

**ragtime26 (Section 11):** every rag26 command below also runs on ragtime26 when
`JEV_DATASET=ragtime26` is set and the ragtime paths are used (`--rag-responses
data/ragtime26/runs/repgen/ --rag-topics
data/ragtime26/topics/topics.all.2026.v0625-fix.request.jsonl`, out-dirs
`output-ragtime26-*`). The whole replication is `bash temp/run_ragtime26_suite.sh`.
Analyses:

- `temp/jev_pilot_stats.py` (pilots; `--score mirror` for Noul)
- `temp/jev_prompt_compare.py`, `temp/jev_instruction_check.py`,
  `temp/jev_noul_compare.py`, `temp/jev_noul_instruction_check.py`
- `temp/jev_probes.py report [--question …]`
- `temp/jev_padding.py report --judge jev|noul|gemini`

Outputs:

- `temp/*_ragtime26.txt`
- `temp/jev_probes_ragtime26/`
- logs `temp/rt26_*.log`

Environment: `.env` provides `OPENAI_BASE_URL=https://openrouter.ai/api/v1`,
`OPENAI_API_KEY` (OpenRouter), `OPENAI_MODEL` (Gemini slug), and `CACHE_DIR=./cache`.
Load it with `set -a; source ./.env; set +a`.

| Section | Command                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Output                                                                                                                                                                                                                                                                 |
| ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 4       | `auto-judge run --workflow judges/bonsai_judge/workflow.pairwise_jev.yml --variant pilot --rag-responses data/rag26/runs/generation/ --rag-topics data/rag26/topics/trec_rag_2026_queries.jsonl --out-dir ./output-pairwise-jev/`                                                                                                                                                                                                                                                                                                                                    | `output-pairwise-jev/bonsai_pairwise_jev.pairwise/`; log `temp/jev_pilot.log`                                                                                                                                                                                          |
| 5       | `python temp/jev_length_bias.py <comparisons.jsonl> [topic]`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | stdout                                                                                                                                                                                                                                                                 |
| 6       | same, `--variant prompts --out-dir ./output-pairwise-jev-prompts/`, then `python temp/jev_prompt_compare.py output-pairwise-jev-prompts/bonsai_pairwise_jev.pairwise/comparisons.jsonl output-pairwise-jev/bonsai_pairwise_jev.pairwise/comparisons.jsonl temp/rag26_pairwise/bonsai_pairwise.pairwise/comparisons.jsonl`                                                                                                                                                                                                                                            | `temp/jev_prompt_compare.txt`                                                                                                                                                                                                                                          |
| 7       | same, `--variant instruction_check --out-dir ./output-pairwise-jev-icheck/`, then `python temp/jev_instruction_check.py output-pairwise-jev-icheck/bonsai_pairwise_jev.pairwise/comparisons.jsonl`                                                                                                                                                                                                                                                                                                                                                                   | `temp/jev_instruction_check.txt`                                                                                                                                                                                                                                       |
| 8.1     | `python temp/jev_probes.py determinism --topic rag2026-100` then `python temp/jev_probes.py report`                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | `temp/jev_probes/determinism_rag2026-100_pass{1,2}.jsonl`                                                                                                                                                                                                              |
| 8.2     | `python temp/jev_probes.py identical` then `report`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | `temp/jev_probes/identical.jsonl`                                                                                                                                                                                                                                      |
| 8.3     | inline analysis of `output-pairwise-jev/bonsai_pairwise_jev.pairwise/pairs.csv` (columns `p_x_fwd`, `p_x_rev`)                                                                                                                                                                                                                                                                                                                                                                                                                                                       | —                                                                                                                                                                                                                                                                      |
| 9       | `python temp/jev_padding.py run [--judge gemini]` then `report [--judge gemini]` (`dry` previews)                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | `temp/jev_probes/padding{,_gemini}.jsonl`, `temp/jev_padding{,_gemini}.txt`                                                                                                                                                                                            |
| 10      | same `auto-judge run`, `--variant noul_pilot --out-dir ./output-pairwise-jev-noul/`, then `python temp/jev_noul_compare.py output-pairwise-jev-noul/bonsai_pairwise_jev.pairwise/comparisons.jsonl output-pairwise-jev/bonsai_pairwise_jev.pairwise/comparisons.jsonl temp/rag26_pairwise/bonsai_pairwise.pairwise/comparisons.jsonl`; ties: `python temp/jev_probes.py identical --question noul_a_better` (and `noul_b_better`), then `report --question …`                                                                                                        | `output-pairwise-jev-noul/`, `temp/jev_noul_compare.txt`, `temp/jev_probes/identical_noul_{a,b}_better.jsonl`; log `temp/jev_noul_pilot.log`                                                                                                                           |
| 10.8    | cache replay: same `auto-judge run` with `--variant noul_pilot` (now with `mirror:`); duplicate test: a kiddie copy with one run duplicated under another team, `--variant noul_pilot -J max_topics=2`                                                                                                                                                                                                                                                                                                                                                               | `output-pairwise-jev-noul/bonsai_pairwise_jev.pairwise/{q_mirror,leaderboard_overall.csv,run_manifest.json}`                                                                                                                                                           |
| 10.9    | `bash temp/run_noul_probes.sh` (runs `--variant noul_instruction_check --out-dir ./output-pairwise-jev-noul-icheck/`, `python temp/jev_probes.py determinism --topic rag2026-100 --question mirror`, `python temp/jev_padding.py run --judge noul`); then `python temp/jev_noul_instruction_check.py output-pairwise-jev-noul-icheck/bonsai_pairwise_jev.pairwise/comparisons.jsonl output-pairwise-jev-noul/bonsai_pairwise_jev.pairwise/comparisons.jsonl`, `python temp/jev_probes.py report --question mirror`, `python temp/jev_padding.py report --judge noul` | `temp/jev_noul_instruction_check.txt`, `temp/jev_noul_determinism.txt`, `temp/jev_padding_noul.txt`; raw `temp/jev_probes/determinism_rag2026-100_mirror_pass{1,2}.jsonl`, `temp/jev_probes/padding_noul.jsonl`; logs `temp/jev_noul_{icheck,determinism,padding}.log` |
| 12.1 | `auto-judge run … --variant noul_appropriate_pilot` on each dataset (out-dirs `output-pairwise-jev-appropriate/`, `output-ragtime26-jev-appropriate/`), then `JEV_DATASET=<ds> python temp/jev_appropriate_compare.py <comparisons.jsonl> [gemini comparisons]` | `temp/jev_appropriate_compare{,_ragtime26}.txt`; logs `temp/jev_appropriate_{rag26,ragtime26}.log` |
| 12.2 | `bash temp/run_proportionate_test.sh` (`--variant noul_proportionate_test`, `--topic` filters, original topics vs `temp/narrow_topics_{rag26,ragtime26}.jsonl`), then `JEV_DATASET=<ds> python temp/jev_proportionate_compare.py <broad comparisons> <narrow comparisons>` (artifacts under `output-prop-*/tmp-bonsai_pairwise_jev.pairwise/`) | `temp/jev_proportionate_{rag26,ragtime26}.txt`; logs `temp/prop_*.log` |
| 13 | `bash temp/run_full_proportionate.sh` (`--variant full_noul_proportionate`; ragtime26 then rag26; resumable), then `JEV_DATASET=<ds> python temp/jev_full_analysis.py <comparisons.jsonl> [gemini comparisons]` | `output-ragtime26-jev-full-prop/`, `output-pairwise-jev-full-prop/` (git-ignored); `temp/jev_full_analysis_{rag26,ragtime26}.txt`; logs `temp/full_prop_*.log`, `temp/full_prop.out` |

Gemini baseline comparisons: `temp/rag26_pairwise/bonsai_pairwise.pairwise/comparisons.jsonl`
(run 2026-09-10; see `HANDOFF-rag26-pairwise-judge.md`).

Cached fresh-call experiments: the Jev pilot, prompt, instruction and Noul runs, and
the Gemini padding run are in `cache/minima_llm.db`. The determinism, identical and Jev
padding probes deliberately bypass the cache; their raw responses are in the JSONL
files above.

---

## 16. Appendix: prompts and questions verbatim

### 16.1 Gemini prompt (`judges/bonsai_judge/prompts/pairwise_summary_1.md`)

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

### 16.2 Jev request shape

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

### 16.3 Jev questions (`judges/bonsai_judge/prompts/jev_questions.yml`)

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

**noul_a_appropriate** / **noul_b_appropriate** (Section 12.1; `x_polarity` a / b)

- _instructions:_ You are a subject matter expert. Two RAG summaries (summary_a and summary_b) were generated for the query. Does summary_a [summary_b] satisfy the information need of the query more appropriately than summary_b [summary_a] does, considering relevance, conciseness, completeness, and how well its claims are grounded?
- _true:_ summary_a [summary_b] satisfies the information need of the query more appropriately than summary_b [summary_a].
- _false:_ summary_a [summary_b] does not satisfy the information need of the query more appropriately than summary_b [summary_a].

**noul_a_proportionate** / **noul_b_proportionate** (Sections 12.2 and 13; `x_polarity` a / b; used for the full runs)

- _instructions:_ You are a subject matter expert. Two RAG summaries (summary_a and summary_b) were generated for the query. Does summary_a [summary_b] satisfy the information need of the query better than summary_b [summary_a] does, considering relevance, completeness, depth proportionate to what the query asks for (thorough when the query is broad or multi-part, brief and direct when it is narrow), and how well its claims are grounded?
- _true:_ summary_a [summary_b] satisfies the information need of the query better than summary_b [summary_a], at a depth proportionate to what the query asks for.
- _false:_ summary_a [summary_b] does not satisfy the information need of the query better than summary_b [summary_a], at a depth proportionate to what the query asks for.

**noul_a_shorter** (Noul probe, objective)

- _instructions:_ Ignore quality entirely. Is summary_a shorter than summary_b, i.e. does it contain fewer words?
- _true:_ summary_a contains fewer words than summary_b.
- _false:_ summary_a does not contain fewer words than summary_b.

`x_*` keys are local metadata and are stripped before the request is sent. The Noul
response shape is `{"type": "noul", "noul": 0.96}`.

### 16.4 Generic filler pool (padding test)

20 topic-agnostic sentences, for example: "It is important to consider multiple perspectives
when approaching this question."; "Every situation is different, so what works in one
context may not work in another."; "Ultimately, the right choice depends on individual
circumstances and goals." The full list is `GENERIC` in `temp/jev_padding.py`.
