# OMC at TREC 2026 AutoJudge: Methods and Results (notebook paper draft)

**Working title:** Probabilistic pairwise, graded-rubric and grounding-discounted judges with the Jev decision model.

**Status:** draft, 2026-10-30.

- **Sources.** Every number comes from `FINDINGS-jev-pairwise-judge.md` (the findings doc, "notes §n" below) or from the figure scripts' printed checks (`papers/figures/CAPTIONS.md`).
- **Figures.** "Figure n" means `papers/figures/fNN_*.pdf`.
- **Data handling.** Results on individual comparisons or answers come from the permitted window only. Over all topics we report only the submitted leaderboards' aggregate rows.
- **Runs.** No run is named. The paper should report full-run results only in aggregate.

---

## 1. Overview

**Authors:** Max Irwin <max@onemorecloud.com>, Nathan Day <nday@eezy.com>, Daniel Worley <dworley@gmail.com>

**Introduction:** For TREC AutoJudge 2026, covering the rag26 and ragtime26 tracks, we took approaches to directly construct leaderboards for the tracks by grading of RAG summary output. For this meta-judge approach, we had two primary constraints to work with: (a) no existing golden set for the leaderboard task and (b) a limited window of responses to prevent LLM data leakage and overfitting. We submitted 10 hypotheses' outputs. We performed this task "in the dark" awaiting final results from the official TREC judges, and this forced us down several interesting paths using (1) new tools such as System One models (aka "Jev"), (2) SOTA practitioner techniques such as pairwise and bag-of-decisions auto-grading, and (3) classical tools such as SpaCy for concept hallucination risk. Along the way, we pulled apart Jev and discovered biases, found key restrictions of costs and speed with private LLMs, and saw interesting result correlations across independent experiments.

We submitted three judges to both TREC 2026 AutoJudge datasets (rag26 and ragtime26). None was trained or tuned on the evaluation topics outside a fixed 10-topic window. All three use TypeSafe's **Jev** decision model (`typesafe/jev-1.13`), which returns probabilities instead of generated text.

Our Judges and methods include pairwise preference with several variants, bag-of-decisions ("BOD") grading, and lexical concept overlap ("Hallucination Risk"). At first, we started with lexical experiments and then used Gemini. The System One model Jev was released during the competition, and given the high cost and high latency of Gemini, we pivoted to Jev. We were able to use the Gemini result data as an agreement point for some of our methods.

| Judge                                                                           | Unit judged                           | Primary measure                                            | Jev calls (rag26 / ragtime26) | API cost, both datasets |
| ------------------------------------------------------------------------------- | ------------------------------------- | ---------------------------------------------------------- | ----------------------------- | ----------------------- |
| **Pairwise:** mirrored, depth-proportionate yes/no questions, both orientations | pair of reports, cross-team           | `PAIRWISE_WINRATE`: mean over topics of per-topic win rate | 703,536 / 196,958 unique      | $93.84                  |
| **BOD:** bag-of-decisions graded rubric                                         | one report against a per-topic rubric | `BOD_MEAN`: mean expected grade / 3                        | 9,838 / 5,041 answers         | $1.31                   |
| **BOD × Hallucination Risk**                                                    | one report                            | `BOD_GROUNDED` = `BOD_MEAN` × (1 − `HALLUC_RISK`)          | 0 new (reuses the BOD grades) | $0                      |

**Main results** (details in §§4–8):

2. **Jev has a strong position bias on the first slot for choice**, and a mirrored pair of yes/no questions fixes the bias.
3. **None of our judges rewards uninformative length** in a controlled padding test. Each credits added relevant content, except our Gemini baseline on ragtime26.
4. **A "depth-proportionate" wording makes the pairwise judge respond to how narrow a request is,** on 10 of 10 window topics of each dataset.
5. **The rubric judge (BOD) agrees with the pairwise judge** at ρ 0.991 (rag26) and 0.943 (ragtime26) over all topics, at about 1/70 of the cost.
6. **The hallucination discount is small** (median risk about 1%) but changes the top-ranked run on both datasets.

---

## 2. Task and data

| Property                             | rag26                                     | ragtime26                                          |
| ------------------------------------ | ----------------------------------------- | -------------------------------------------------- |
| Runs / teams / topics                | 83 / 25 / 119                             | 49 / 10 / 103                                      |
| Reports (non-empty)                  | 9,875 (9,836)                             | 5,047 (5,041)                                      |
| Topic text                           | `title` only: a long, multi-part question | short `title` + `problem_statement` + `background` |
| Permitted window                     | rag2026-0 … rag2026-9                     | 2000 … 2009                                        |
| Window reports; median words (range) | 830; 738 (0–1,024)                        | 490; 562 (30–1,016)                                |
| Assessed topics available to us      | 0                                         | 0                                                  |

- **Pairs.** Comparisons are cross-team only: a team's runs are never compared with each other. Per topic there are about 3,200 unordered pairs on rag26 and 1,024 on ragtime26.
- **ragtime26 topic text.** The title is a label, so the problem statement and background are passed to every judge.
- **No ground truth.** We measured agreement, consistency, or sensitivity to a controlled manipulation.

---

## 3. Development protocol

### 3.1 Permitted window

The AutoJudge data policy lets participants inspect and tune on the first 10 topics of each topics file only. Every development run, probe and per-comparison statistic used those topics. We also used kiddie, a small synthetic, unrestricted dataset, for smoke tests.

The window is fixed by file order and not sampled at random, so results may not transfer to other topics.

### 3.2 Development and confirmation sets

- **Development (rag26 window).** All decisions were made on the rag26 window.
- **Confirmation (ragtime26 window).** It was run and read once, after the rag26 decisions were recorded.
- **Pre-registration.** Pass criteria were written down before results existed (§4.6 for the pairwise judge, §6.2 for the risk judge).

### 3.3 Statistics

- **Correlations.** Rankings are compared with Spearman's ρ and Kendall's τ over each run's mean score across the window topics (its "board" score).
- **Uncertainty.** It comes from bootstrapping over **topics**, or from the sign pattern across topics. Comparisons share runs, so bootstrap intervals over comparisons are too narrow and are marked where used.
- **Split-half reliability.** The Spearman correlation between boards built from two random 5-topic halves, over 200 splits.

### 3.4 Cost accounting

Cost is the change in the OpenRouter key's balance. All calls are cached on disk, keyed by a hash of the endpoint, model and request, so reruns cost nothing.

| Item                                             | Cost          |
| ------------------------------------------------ | ------------- |
| Window study, rag26 (development)                | $25.27        |
| Window study, ragtime26 (confirmation)           | $6.82         |
| Pairwise full runs, both datasets                | $93.84        |
| BOD: smoke test, pilots, padding test, full runs | $1.57         |
| BOD × Hallucination Risk (grades from cache)     | $0.00         |
| **Total**                                        | **≈ $127.50** |

---

## 4. Judge 1: pairwise tournament

### 4.1 The Jev decision model

Jev is a decision model served through OpenRouter's Decisions API (`POST /api/alpha/decisions`), not a chat model.

- **Request.** Each request carries a `state` object (query, plus problem statement and background on ragtime26, plus the two summaries) and one or more typed questions.
- **Question types.**
  - **Choice** returns a probability per option. Probabilities have 2 decimals; we saw 101 distinct values.
  - **Noul** (yes/no) returns P(yes).
- **Several questions per request.** They are answered independently, and the state is billed once.
- **Cost.** Billing is input-only: about $0.0001 per call, and the window Choice pilot used a mean of 2,317 input tokens per call on rag26.
- **Snapshot.** Every response reported the snapshot `typesafe/jev-1.13-20260917`.

**Test–retest.** Asking a question alongside three others barely changed its answer. Mean |ΔP| was 0.0050 on rag26 and 0.0095 on ragtime26, and the winner agreed in 99.5% / 99.0% of comparisons.

### 4.2 Baseline: Gemini binary judge

- **Model and prompt.** `google/gemini-3.5-flash-lite` at temperature 0, with a prompt asking for exactly one character, `A` or `B`.
- **Orientation.** Each unordered pair is judged once, with the slot order set by a hash.
- **Coverage on the window.** All 10 rag26 topics and 3 ragtime26 topics; every response parsed.
- **Cost.** About $0.00059 per comparison, roughly 6× Jev's.

We use it as a reference point for agreement, not as ground truth.

### 4.3 Scoring

- **Per comparison.** Run _a_ (slot A) earns P(A better) and run _b_ earns 1 − P(A better).
- **Per topic.** A run's win rate is its expected wins divided by its games.
- **Leaderboard.** The mean over topics of the per-topic win rates, so every topic weighs equally.
- **Empty reports.** They score 0 without a call.

### 4.4 From a Choice question to a mirrored yes/no pair

The first form asked one Choice question ("Which summary better satisfies the information need…?", options A/B) on all 10 window topics, in **both orientations**: 64,140 comparisons on rag26 and 20,480 on ragtime26.

**Choice probabilities saturate and don't change the ranking** (Figure 1).

| Metric                                            | rag26  | ragtime26 |
| ------------------------------------------------- | ------ | --------- |
| P exactly 0 or 1                                  | 49.8%  | 18.5%     |
| Decisive (P < 0.25 or > 0.75)                     | 88.5%  | 79.5%     |
| Probability-weighted vs binary ranking (Spearman) | 0.9992 | 0.9987    |

**Choice has a strong first-slot bias on ties** (Figure 3). We judged each run's report against itself in both slots:

| Form                             | rag26 (828 reports)          | ragtime26 (490)     |
| -------------------------------- | ---------------------------- | ------------------- |
| Choice: mean P(A)                | **0.792** (A chosen 828/828) | **0.789** (490/490) |
| Noul "is A better?": mean P(yes) | 0.177                        | 0.184               |
| Noul "is B better?": mean P(yes) | 0.108                        | 0.106               |
| **Mirror** (yA + 1 − yB) / 2     | **0.535**                    | **0.539**           |

**The mirrored Noul pair.** With yA = P(yes | "is A better than B?") and yB = P(yes | "is B better than A?"), asked in one request, the score is

> mirror = (yA + 1 − yB) / 2.

- **Near-ties.** "No" to both (yA, yB < 0.5) flags a near-tie, reported as `PAIRWISE_TIE_RATE`.
- **Identical text.** It is scored 0.5 without a call.

| Window metric, both orientations  | rag26 Choice | rag26 mirror | ragtime26 Choice | ragtime26 mirror |
| --------------------------------- | ------------ | ------------ | ---------------- | ---------------- |
| Exactly 0 or 1                    | 49.8%        | **0.0%**     | 18.5%            | **0.0%**         |
| Distinct values                   | 101          | 406          | 101              | 373              |
| Identical summaries               | 0.792        | **0.535**    | 0.789            | **0.539**        |
| P(slot A wins), close pairs       | 0.540        | 0.553        | 0.465            | 0.523            |
| Winner flips on swap, all pairs   | 4.3%         | 4.8%         | 6.1%             | 5.0%             |
| Winner flips on swap, close pairs | 71%          | 57%          | 64%              | 37%              |
| Board ρ vs Gemini                 | 0.976        | 0.974        | 0.937            | 0.939            |

- **Same ranking.** Choice and the mirror rank runs alike: ρ 0.998 on both datasets (Kendall 0.975 / 0.980).
- **Position effects remain in close pairs** (Figure 2). These are pairs whose orientation-averaged P is between 0.4 and 0.6.
  - Choice favours slot A on rag26 but slot **B** on ragtime26, so the direction depends on the dataset.
  - The mirror leans to slot A on both.
  - We therefore judge **both orientations** and average.
- **Determinism** (Figure 13). Asking the same mirror requests twice with the cache bypassed flipped the winner in 5 of 3,223 comparisons (0.155%) on rag26 and 7 of 1,024 (0.684%) on ragtime26. Mean |ΔP| was 0.0049 / 0.0067.
- **Tie flags.** The per-pair tie flag is unstable (17 of 43 flags changed on rag26), so it is used only in aggregate.

### 4.5 Question wording

**Prompt variants** (Figure 5). Four Choice wordings were asked in one request on all window topics (single orientation, hashed):

- `better_summary`, the control;
- `length_neutral`;
- `needs_coverage`;
- `precision`.

| Question       | rag26 P(longer wins) | ragtime26 P(longer wins) | Ranking vs control, rag26 / ragtime26 | Pair decisions agreeing with control |
| -------------- | -------------------- | ------------------------ | ------------------------------------- | ------------------------------------ |
| better_summary | 0.701                | 0.489                    | —                                     | —                                    |
| length_neutral | 0.669                | 0.450                    | 0.984 / 0.957                         | 95.4% / 93.4%                        |
| needs_coverage | 0.701                | 0.499                    | 0.992 / 0.944                         | 96.7% / 91.6%                        |
| precision      | 0.673                | 0.439                    | 0.982 / 0.938                         | 95.1% / 91.0%                        |

Rewording changes 3–9% of pair decisions but doesn't significantly impact the ranking.

**Depth-proportionate wording** (Figure 7). The submitted question adds one clause to the `better` question: "...considering relevance, completeness, **depth proportionate to what the query asks for (thorough when the query is broad or multi-part, brief and direct when it is narrow)**, and how well its claims are grounded?"

Neither dataset has genuinely narrow needs, so we changed the request instead.

- **Two framings of the same pairs.** The _same_ report pairs were judged under the original broad topic and under a narrow framing: one sub-question written from that topic's own text.
- **Both pairs in each request.** Both mirrored pairs (`better` and `proportionate`) were asked in every request.
- **Measure.** The difference-in-differences (DiD) of the broad→narrow change in P(longer wins), proportionate minus `better`. A negative value means proportionate moves further toward shorter answers when the request narrows.

|                                             | rag26                          | ragtime26                      |
| ------------------------------------------- | ------------------------------ | ------------------------------ |
| `better`: P(longer), broad → narrow         | 0.672 → 0.640                  | 0.486 → 0.477                  |
| `proportionate`: P(longer), broad → narrow  | 0.676 → 0.612                  | 0.487 → 0.461                  |
| **DiD, pooled**                             | **−0.032**                     | **−0.016**                     |
| Topics with DiD < 0                         | **10 / 10** (−0.015 to −0.053) | **10 / 10** (−0.008 to −0.027) |
| Winner differs from `better`, broad framing | 0.96%                          | 1.53%                          |

- **On the original requests,** proportionate judges like `better`.
- **On narrowed requests,** it moves toward shorter answers about twice as much as `better` (rag26) or nearly three times as much (ragtime26).
- **Caveat.** The narrow requests are synthetic: the reports were written for the broad request.

### 4.6 Length and padding

During early experiments on rag26, we found strong correlations with longer summaries being chosen as the preferred answer by the LLM. This section explores experiments on this hypothesis. The results show that length is not a strong indicator of preference.

**Natural length dependence is dataset-specific** (Figure 4). On rag26, longer reports win (Choice P(longer) = 0.700; 0.865 when one report is at least twice as long). On ragtime26 they don't (0.488), and within topics longer reports rank slightly lower. Every judge we ran shows the same pattern.

**Controlled padding test** (Figure 6). Design:

- **Targets.** On each topic, 16 mid-ranked target runs (win rate 0.25–0.75), giving 160 targets per dataset.
- **Opponents.** Each target was judged against 8 fixed cross-team opponents spread over the ranking, in both orientations.
- **Variants.** Each target was rewritten into five variants. Padding sentences were interleaved at seeded random positions, not appended.
  - pad_repeat: the report's own sentences repeated (+≈50% words);
  - pad_generic: topic-agnostic filler;
  - pad_offtopic: sentences from another window topic;
  - pad_relevant: unseen sentences from the topic's strongest run of another team (a positive control);
  - truncate: the last third of sentences removed.
- **Measure.** ΔP is the mean change in orientation-averaged P(target wins) from the original, with a 95% CI bootstrapped over targets.

| Variant          | Words × | rag26 ΔP [95% CI]           | rag26 targets ΔP > 0 | ragtime26 ΔP [95% CI]       | ragtime26 targets ΔP > 0 |
| ---------------- | ------- | --------------------------- | -------------------- | --------------------------- | ------------------------ |
| pad_repeat       | 1.53    | −0.048 [−0.053, −0.044]     | 1/160                | −0.074 [−0.080, −0.069]     | 1/160                    |
| pad_generic      | 1.51    | −0.095 [−0.102, −0.088]     | 0/160                | −0.135 [−0.144, −0.125]     | 1/160                    |
| pad_offtopic     | 1.53    | −0.196 [−0.209, −0.183]     | 0/160                | −0.258 [−0.271, −0.245]     | 0/160                    |
| **pad_relevant** | 1.53    | **+0.179** [+0.170, +0.188] | 160/160              | **+0.098** [+0.088, +0.107] | 155/160                  |
| truncate         | 0.67    | −0.081 [−0.088, −0.072]     | 2/160                | −0.078 [−0.088, −0.068]     | 16/160                   |

(Submitted pairwise form, all 10 window topics.)

**Gemini baseline on the first 3 window topics.**

| Variant      | rag26 ΔP [95% CI]           | ragtime26 ΔP [95% CI]       |
| ------------ | --------------------------- | --------------------------- |
| pad_repeat   | −0.132 [−0.160, −0.104]     | −0.203 [−0.249, −0.160]     |
| pad_generic  | −0.104 [−0.130, −0.079]     | −0.186 [−0.224, −0.150]     |
| pad_offtopic | −0.206 [−0.250, −0.165]     | −0.309 [−0.368, −0.250]     |
| pad_relevant | **+0.104** [+0.070, +0.138] | **−0.117** [−0.151, −0.085] |
| truncate     | −0.078 [−0.107, −0.049]     | −0.152 [−0.195, −0.113]     |

- **Jev.** It penalises all three uninformative paddings and credits relevant content on every topic of both datasets.
- **Gemini on ragtime26.** It penalises any added text, relevant or not. This test can't tell whether Gemini reads the splice as disruption or doesn't credit coverage.
- **Not tested.** Subtle on-topic verbosity: restatement or hedging with no new facts.

### 4.7 Pre-registered decisions

| #   | Criterion (written beforehand)                                                                        | rag26 (decision)                | ragtime26 (confirmation)        |
| --- | ----------------------------------------------------------------------------------------------------- | ------------------------------- | ------------------------------- |
| 1   | Proportionate DiD < 0 on ≥ 7/10 topics, topic-bootstrap CI excludes 0                                 | Pass: 10/10, −0.032             | Confirmed: 10/10, −0.016        |
| 2   | Identical summaries within [0.40, 0.60]                                                               | Pass: 0.535                     | Confirmed: 0.539                |
| 3   | Drop one orientation only if close-pair slot bias is non-significant and ties are within [0.45, 0.55] | Keep both: 0.553 [0.547, 0.559] | Confirmed: 0.523 [0.514, 0.532] |
| 4   | Uninformative padding ΔP < 0 (CI excludes 0); relevant ΔP > 0                                         | Pass                            | Confirmed                       |
| 5   | Reopen Choice only if the mirror flips > 2% of winners or ρ vs Choice < 0.95                          | Not reopened: 0.155%, ρ 0.998   | Confirmed: 0.684%, ρ 0.998      |

Decision 3 uses topic-bootstrap 95% CIs on P(slot A wins) for close pairs. The bias appeared on every topic: 0.537–0.572 on rag26 and 0.503–0.549 on ragtime26.

### 4.8 Submitted configuration and full runs

- **Questions.** `noul_a_proportionate` and `noul_b_proportionate` in one request, scored by their mirror average.
- **Coverage.** Both orientations of every cross-team pair, on every topic.

|                       | ragtime26          | rag26              |
| --------------------- | ------------------ | ------------------ |
| Comparisons (ordered) | 210,418, all valid | 760,886, all valid |
| Unique calls          | 196,958            | 703,536            |
| Cost                  | $18.95             | $74.89             |
| Wall-clock            | 3.0 h              | 11.1 h             |

**Window diagnostics of the full runs** (Figures 2, 4, 12):

| Metric                                               | rag26                     | ragtime26                |
| ---------------------------------------------------- | ------------------------- | ------------------------ |
| Split-half reliability                               | 0.974 [0.960, 0.982]      | 0.948 [0.915, 0.970]     |
| Board ρ / Kendall vs Gemini                          | 0.974 / 0.866 (10 topics) | 0.939 / 0.818 (3 topics) |
| P(longer wins); per-topic ρ(win rate, words), median | 0.675; +0.558             | 0.486; −0.090            |
| P(slot A wins): all / close pairs                    | 0.520 / 0.553             | 0.511 / 0.523            |
| Flagged near-ties                                    | 0.84%                     | 2.80%                    |

---

## 5. Judge 2: bag-of-decisions graded rubric (BOD)

### 5.1 Design

This judge adapts a "Bag of Decisions (BOD) Reranker" (well outlined in a post by Doug Turnbull). There, an LLM writes yes/no criteria per query, Jev answers them all in one request, and the score is the sum of P(yes). We change three things: the number of questions varies by topic, the questions are graded 0–3, and the grades are stored and tallied afterwards.

1. **Rubric.**
   - Gemini 2.5 Flash Lite at temperature 0 writes 5–15 "Does the answer ...?" questions per topic, sized to how many distinct parts the need has.
   - It sees the topic text only (query, problem statement, background), never a response.
   - Near-duplicates are dropped (token Jaccard ≥ 0.8).
   - The prompt is in Appendix A.2. It was settled on kiddie only, before any 2026 data was read. Its first version produced persona questions such as "Does the answer acknowledge the child's interest…?"; the final version allows content questions only.
2. **Grading.**
   - One Jev request per (run, topic). The state is the topic fields plus the answer.
   - Each rubric question is a 4-option Choice with anchored grades:
     - 0: not addressed;
     - 1: mentioned without substance;
     - 2: partially answered;
     - 3: fully answered with specific, accurate detail.
   - The same request carries a Noul padding question as a diagnostic.
3. **Scoring.**
   - Each question scores its **expected grade** Σ_g P(g)·g / 3, from the probabilities, as in G-Eval.
   - The overall score is the mean over the topic's questions, then the mean over topics.

| Measure              | Definition                                                                                |
| -------------------- | ----------------------------------------------------------------------------------------- |
| `BOD_MEAN` (primary) | mean expected grade over the topic's questions, divided by 3                              |
| `BOD_COVERAGE`       | fraction of questions with P(grade ≥ 2) > 0.5                                             |
| `BOD_FULL`           | fraction of questions with P(grade = 3) > 0.5                                             |
| `BOD_PADDING`        | P(yes) that the answer has substantial off-topic, repeated or generic filler (diagnostic) |
| `BOD_QUESTIONS`      | number of rubric questions                                                                |

Each answer is scored on its own, so a topic needs O(runs) calls instead of O(runs²).

### 5.2 Window results

We saw strong correlation with pairwise, justifying the significant reduction in cost without sacrificing perceived accuracy.

|                                      | rag26 window                | ragtime26 window            |
| ------------------------------------ | --------------------------- | --------------------------- |
| Answers / cost                       | 828 / $0.09                 | 490 / $0.05                 |
| Questions per topic                  | 5–13 (mean 9.0)             | 6–10                        |
| BOD_MEAN p10 / p50 / p90             | 0.420 / 0.756 / 0.983       | 0.514 / 0.710 / 0.894       |
| Most likely grade 0 / 1 / 2 / 3      | 8.2% / 9.2% / 40.3% / 42.3% | 8.4% / 7.8% / 46.4% / 37.5% |
| \|E[g] − most likely grade\| > 0.25  | 26.9% of questions          | 37.7%                       |
| Split-half (5 vs 5)                  | 0.969 [0.956, 0.979]        | 0.854 [0.789, 0.914]        |
| Per-topic ρ(BOD_MEAN, words), median | 0.518                       | −0.120                      |
| Board ρ vs pairwise judge            | **0.990** (Kendall 0.919)   | 0.918 (Figure 8)            |
| Board ρ vs Gemini                    | 0.972 (7 topics)            | —                           |

- **Probabilities.** The expected grade differs meaningfully from the most likely grade on a quarter to a third of questions, so here the probabilities carry information (Figure 9).
- **Ceiling.** Grades saturate near 3 on rag26, which compresses the top of the board.

### 5.3 Padding test (rag26 window)

The design is that of §4.6, with each variant graded on its own (160 targets × 6 variants). The criteria were written into the analysis script before the run.

| Variant      | Words × | ΔBOD_MEAN [95% CI]          | ΔBOD_PADDING [95% CI]   |
| ------------ | ------- | --------------------------- | ----------------------- |
| pad_repeat   | 1.52    | −0.023 [−0.025, −0.020]     | +0.510 [+0.474, +0.544] |
| pad_generic  | 1.51    | −0.032 [−0.035, −0.028]     | +0.537 [+0.499, +0.574] |
| pad_offtopic | 1.53    | −0.043 [−0.047, −0.039]     | +0.574 [+0.535, +0.612] |
| pad_relevant | 1.54    | **+0.132** [+0.121, +0.144] | +0.038 [+0.021, +0.055] |
| truncate     | 0.67    | **−0.103** [−0.114, −0.092] | −0.044 [−0.061, −0.027] |

Every criterion passes on each of the 10 topics individually (Figure 6, right).

### 5.4 Full runs

|                            | ragtime26       | rag26           |
| -------------------------- | --------------- | --------------- |
| Topics / runs              | 103 / 49        | 119 / 83        |
| Rubric questions per topic | 5–11 (mean 7.7) | 5–15 (mean 8.4) |
| Answers graded             | 5,041 / 5,041   | 9,838 / 9,838   |
| Wall-clock                 | 3.0 min         | 8.5 min         |

Cost: $1.31 for both datasets, including rubric generation.

---

## 6. Judge 3: BOD × Hallucination Risk

### 6.1 Definition

BOD rewards answering the rubric but doesn't check whether the answer's claims are supported by its citations. This judge discounts each answer by the share of its content that is unsupported _and_ that no other run produced.

**Concepts.**

- A concept is a normalized spaCy `en_core_web_lg` NOUN/PROPN phrase. We normalize by taking the lemma, lowercased.
- `sn` is the answer's concept set.
- `hn` is the union of the concept sets of the documents the answer cites.
- `df[t][c]` is the number of runs whose answer for topic _t_ states concept _c_.

**Measures.**

- **Ungrounded** concepts are `sn − hn`, those absent from every cited document. `UNGROUNDED_RATE` = |sn − hn| / |sn| = 1 − concept precision.
- **Private-ungrounded** concepts are the ungrounded ones that no other run states (`df` = 1).
- `HALLUC_RISK` = |private-ungrounded| / |sn|.
- `BOD_GROUNDED` = `BOD_MEAN` × (1 − `HALLUC_RISK`) per answer, then the mean over topics. There is no weight or threshold to set.
- **Undefined risk** (no citations, no cited-document text, or no concepts) is set to 1. This is a policy choice.

**Relation to prior work.**

- The ungrounded rate is the complement of the precision side of Irwin's concept F1, which compares the nouns in a summary with the nouns in its cited sources.
- The cross-run condition is our addition and is a cross-system analogue of SelfCheckGPT's consistency check across samples.
- The measure is lexical, but needs inference against the local spaCy model for each cited document and answer.

### 6.2 Pre-registered criteria and window results

| Criterion                                                         | rag26 window (dev)     | ragtime26 window (confirmation) |
| ----------------------------------------------------------------- | ---------------------- | ------------------------------- |
| 1 Pipeline: 0 new Jev calls; risk defined for ≥ 99% of answers    | 826/828: pass          | 490/490: pass                   |
| 2 Not redundant: board ρ(GROUNDED, MEAN) < 0.99                   | 0.992: **fail**        | 0.984: pass                     |
| 3 Not dominant: same ρ ≥ 0.70                                     | pass                   | pass                            |
| 4 Reliability: split-half ≥ BOD_MEAN's − 0.05                     | 0.968 vs 0.969: pass   | 0.820 vs 0.854: pass            |
| 5 Length: \|ρ(score, words)\| ≤ BOD_MEAN's + 0.15                 | +0.505 vs +0.523: pass | −0.163 vs −0.120: pass          |
| 6 Pool stability: half the other runs removed from `df`, ρ ≥ 0.90 | min 0.993: pass        | min 0.910: pass                 |

| Diagnostic (window)                                  | rag26                 | ragtime26             |
| ---------------------------------------------------- | --------------------- | --------------------- |
| HALLUC_RISK mean / median / p75                      | 0.021 / 0.009 / 0.027 | 0.025 / 0.013 / 0.036 |
| UNGROUNDED_RATE mean                                 | 0.104                 | 0.140                 |
| Per-topic ρ(risk, number of cited documents), median | +0.165                | +0.131                |
| Per-topic ρ(risk, BOD_MEAN), median                  | +0.331                | +0.347                |

- **Criterion 2 failed on rag26:** the discount barely changes the ranking. We submitted the judge anyway as a separate run, and recorded that decision before reading ragtime26.
- **Small effect.** About four in five unsupported concepts are also stated by some other run, so the private share is small (Figure 10).
- **No sign of citation gaming.** Risk rises slightly, rather than falls, with the number of cited documents.
- **Leans against stronger answers.** Risk correlates positively with BOD_MEAN, so the discount leans slightly against the stronger answers. Whether those private concepts are hallucinations or useful detail needs assessments.

### 6.3 Full runs

|                                    | ragtime26 | rag26     |
| ---------------------------------- | --------- | --------- |
| Answers graded / new Jev calls     | 5,041 / 0 | 9,838 / 0 |
| Risk defined                       | 5,033     | 9,791     |
| Cited documents tagged             | 10,394    | 61,151    |
| Wall-clock (spaCy, single process) | 10 min    | 146 min   |

---

## 7. Agreement between the submitted judges (all topics)

Read from the submitted leaderboards' aggregate rows after submission, for reporting only (Figure 11).

| Pair of judges           | rag26 Spearman / Kendall | ragtime26 Spearman / Kendall |
| ------------------------ | ------------------------ | ---------------------------- |
| BOD_MEAN vs pairwise     | 0.991 / 0.929            | 0.943 / 0.827                |
| BOD_GROUNDED vs BOD_MEAN | 0.993 / 0.952            | 0.975 / 0.884                |
| BOD_GROUNDED vs pairwise | 0.982 / 0.890            | 0.911 / 0.765                |

| Rank movement          | BOD vs pairwise: rag26 / ragtime26 | GROUNDED vs BOD: rag26 / ragtime26 |
| ---------------------- | ---------------------------------- | ---------------------------------- |
| Median \|rank change\| | 1 / 2                              | 1 / 2                              |
| Runs moving ≥ 5 places | 13 / 10                            | 12 / 6                             |
| Shared top-10          | 8 / 8                              | 9 / 9                              |
| Same #1                | no / no                            | no / no                            |

**Split-half reliability on the window** (Figure 12; 200 random 5/5 splits, mean):

| Judge             | rag26 | ragtime26 |
| ----------------- | ----- | --------- |
| Pairwise (mirror) | 0.974 | 0.948     |
| BOD               | 0.969 | 0.854     |
| BOD × Risk        | 0.968 | 0.820     |

- **rag26.** All three judges rank runs almost identically.
- **ragtime26.** They diverge more, and the per-answer judges are less stable across topic halves than the tournament, which compares every answer with every other.

---

## 8. Limitations

- **No ground truth.** Every result is about agreement, consistency or controlled sensitivity. Two judges that agree can share a bias, and LLM-judged leaderboards can be circular or gamed.
- **The window is small and not random.** It is 10 topics per dataset, fixed by file order. Effects whose sign depends on the dataset (length, close-pair slot bias) may also vary across topics.
- **Thin Gemini coverage.** The baseline covers 10 + 3 window topics, single orientation.
- **Synthetic manipulations.**
  - The narrow requests were written by us.
  - Padding splices may be easier to detect than real verbosity.
  - pad_relevant adds both information and text.
- **Model drift.** Results are reproducible from the cache. Fresh calls may differ: Jev 1.14 is now the latest version, and we pinned 1.13.
- **Rubrics.** BOD rubrics are generated from the topic only, so they can't credit useful content they didn't anticipate. Some needs are split into near-duplicate questions.
- **Hallucination Risk is a lexical proxy.** It penalises correct paraphrase, depends on the pool of runs, and its definition was first explored on topics outside the window.

---

## 9. Figures

| Figure | File                           | Section    | Shows                                                                    |
| ------ | ------------------------------ | ---------- | ------------------------------------------------------------------------ |
| 1      | `f01_probability_distribution` | §4.4       | Choice saturates at 0/1; the mirror does not                             |
| 2      | `f02_position_bias`            | §4.4, §4.7 | P(slot A wins) by closeness, Choice vs mirror                            |
| 3      | `f03_identical_summaries`      | §4.4       | Identical summaries: Choice ≈ 0.79, mirror ≈ 0.54                        |
| 4      | `f04_length`                   | §4.6, §5.2 | P(longer wins) by length ratio; per-topic ρ(score, words) for all judges |
| 5      | `f05_prompt_agreement`         | §4.5       | Ranking agreement between four wordings                                  |
| 6      | `f06_padding`                  | §4.6, §5.3 | Padding test: pairwise judges and BOD                                    |
| 7      | `f07_broad_narrow`             | §4.5       | Proportionate DiD per window topic                                       |
| 8      | `f08_judge_agreement`          | §4.2, §5.2 | Window boards: Jev vs Gemini, BOD vs Jev                                 |
| 9      | `f09_bod_grades`               | §5.2       | BOD grade distributions                                                  |
| 10     | `f10_hallucination_risk`       | §6.2       | Risk distributions; BOD_GROUNDED vs BOD_MEAN                             |
| 11     | `f11_full_leaderboards`        | §7         | Submitted all-topic boards against each other                            |
| 12     | `f12_split_half`               | §4.8, §7   | Split-half reliability per judge                                         |
| 13     | `f13_determinism`              | §4.4       | Repeated identical requests                                              |

Draft captions with exact values are in `papers/figures/CAPTIONS.md`.

---

## Appendix A. Prompts and questions (verbatim)

### A.1 Submitted pairwise question (Noul, asked as a mirrored pair)

> You are a subject matter expert. Two RAG summaries (summary_a and summary_b) were generated for the query. Does summary_a [summary_b] satisfy the information need of the query better than summary_b [summary_a] does, considering relevance, completeness, depth proportionate to what the query asks for (thorough when the query is broad or multi-part, brief and direct when it is narrow), and how well its claims are grounded?

- **true:** summary_a [summary_b] satisfies the information need of the query better than summary_b [summary_a], at a depth proportionate to what the query asks for.
- **false:** summary_a [summary_b] does not satisfy the information need of the query better than summary_b [summary_a], at a depth proportionate to what the query asks for.

Request shape: `{"model": "typesafe/jev-1.13", "state": {"query", ["problem_statement"], ["background"], "summary_a", "summary_b"}, "questions": {…}}`.

### A.2 BOD rubric prompt (Gemini 2.5 Flash Lite, temperature 0)

```
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
```

### A.3 BOD grade question (Jev Choice, one per rubric question)

```json
{
  "type": "choice",
  "instructions": "Grade the answer on this question: <rubric question>",
  "criteria": {
    "0": "The answer does not address this question at all.",
    "1": "The answer mentions this topic but gives no real substance or explanation.",
    "2": "The answer partially answers this question; important parts are missing or vague.",
    "3": "The answer fully answers this question with specific, accurate detail."
  }
}
```

Padding question (Noul): "Does the answer contain substantial content that is off-topic, repeated, or generic filler that does not help with the user's question?"

### A.4 Gemini baseline prompt

Verbatim in notes §18.1: the query, both summaries, and an instruction to answer with exactly one character, `A` or `B`.

---

## References to cite

Numbers in brackets are the reference numbers in notes §19, which has the full entries and URLs. `papers/prompt.md` uses the same list.

- **Tracks and infrastructure:** TREC AutoJudge [1], HowTo and data policy [2], TREC RAG site, Ragnarök baselines and 2025 overview [4, 5, 6], RAGTIME site and 2025 overview [7, 8], report evaluation [9], TIRA [10, 11], ir_measures [12], trec_eval [13].
- **RAG evaluation and nuggets:** AutoNuggetizer [14], Great Nugget Recall [15], support evaluation [16], TREC QA nuggets [17, 18], automatic nugget matching [19, 20], Pyramid method [21].
- **LLMs as assessors:** Thomas et al. [22], UMBRELA [23], LLMJudge [24], LLM4Eval [25], Faggioli et al. [26], Clarke and Dietz [27], Alaofi et al. [28], Arabzadeh and Clarke [29], Pencils Down [30].
- **Pairwise and preference judging:** Carterette et al. [31], Clarke et al. [32], MT-Bench [33], PRP [34], LLM-as-a-judge survey [35], Bradley–Terry [36], Chatbot Arena [37].
- **Position and length bias:** [33], Wang et al. [38], Shi et al. [39], Saito et al. [40], length-controlled AlpacaEval [41].
- **Jev:** OpenRouter documentation [3], Decisions API [42], Jev as a reranker [43], Bag of Decisions Reranker [44].
- **Statistics and protocol:** pre-registration [45], Sakai's bootstrap [46], Kendall [47], Voorhees [48].
- **Rubric judges:** Pencils Down [30], LLM-Rubric [49], Prometheus [50], FLASK [51], G-Eval [52].
- **Hallucination and grounding:** concept F1 [53], RAGAS [54], spaCy and `en_core_web_lg` [55, 56], SelfCheckGPT [57], FActScore [58], AIS [59], ALCE [60], ARES [61], verifiability [62], surveys [63, 64], ROUGE [65].
