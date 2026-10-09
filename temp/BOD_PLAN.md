# Bag-of-decisions graded judge (`judges/bonsai_judge/bod.py`, `workflow.bod.yml`) — plan

Source idea: softwaredoug.com/blog/2026/10/08/bag-of-decisions. There an LLM writes 10 yes/no
criteria per query, Jev Noul answers all of them in one `decide()` call, and the score is
ΣP(yes). This variant changes three things: the number of questions varies by topic, each
question is graded 0–3 instead of yes/no, and the granular grades are stored and tallied
later.

## Decisions taken (2026-10-09)
- Question generator: Gemini flash-lite via OpenRouter (`llm_config`), one call per topic.
- Scale: 0–3, anchored, as one 4-option Jev Choice per question.
- Number of questions: variable by topic, bounded to [5, 15]. The generator sizes the set
  to how many parts the information need has.
- Tally: mean expected grade, Σ P(g)·g / 3, averaged over the questions, then over topics.

## Components
1. **`BodQuestionCreator` (nugget creator)**
   - Built from the topic fields only, with `nugget_depends_on_responses: false`. The banks
     are therefore readable for all topics under the data policy.
   - Saved to `.nuggets.jsonl` and never regenerated across variants
     (`force_recreate_nuggets: false`).
   - Near-duplicate questions are removed before saving.
   - Each question is written as a check that an answer can satisfy to a greater or lesser
     degree.
2. **`BodJevJudge` (judge)**
   - One Jev decisions request per (run, topic).
     - state = topic fields + `answer`.
     - questions = `q1..qN`, each a Choice:
       - 0: not addressed
       - 1: mentioned without substance
       - 2: partially answered
       - 3: fully answered with specific, grounded detail
   - The same request also asks a diagnostic question about padding / off-topic content.
     It is reported as a separate measure and is not part of the primary score.
   - The cache key and retry handling are reused from `pairwise_jev.py`.
   - Empty reports get zero rows without a call.
3. **Storage and tally**
   - Each request writes rows to `bod.grades/grades.jsonl` as
     `{topic, run, qid, probs{0..3}, expected, argmax, valid}`.
   - `tally(grades) -> Leaderboard` is a pure function (0 calls), so aggregation is just a
     setting.
   - Measures:

     | Measure | Definition |
     | --- | --- |
     | `BOD_MEAN` (primary) | mean expected grade, divided by 3 |
     | `BOD_COVERAGE` | fraction of questions with P(grade ≥ 2) > 0.5 |
     | `BOD_FULL` | fraction of questions with P(grade = 3) > 0.5 |
     | `BOD_PADDING` | the diagnostic padding question's score |

## Order of work
1. Kiddie smoke test: does Jev accept a 4-option Choice, and are its probabilities usable?
   - Fallback if not: cumulative Nouls (P(≥2), P(=3)), checked for monotonicity.
   - **Result (2026-10-09, `temp/bod_smoke.py`, kiddie 'leaf', 4 calls, about $0.004):**
     PASS. A 4-option Choice with labels "0".."3" returns `probabilities` over all four
     labels, plus `choice` and `confidence`.
     - Answer quality: on-topic run1 scored BOD_MEAN 0.643; the off-topic answer scored 0.000
       (padding Noul 0.98).
     - Option order: listing the criteria 3..0 gave 0.642 instead of 0.643.
     - Short run2 scored 0.261, and its grades match the text.
     - Fallback not needed.
2. Implement, add pytest coverage, and run the kiddie end to end.
   - **Result (2026-10-09):**
     - Rubric: 8-11 questions per kiddie topic.
     - Jev: 20/20 answers graded for about $0.0015.
     - Meta-evaluation against kiddie_fake: Kendall 1.0 for BOD_MEAN and BOD_COVERAGE,
       -1.0 for BOD_PADDING (run4 is off-topic, padding 0.98).
     - Open: the generator adds audience/persona questions ("acknowledge the child's
       interest in ..."). Decide on the kiddie/window topics whether to keep them.
   - Prompt tightened (2026-10-09): background now only decides WHAT content is needed. There
     are no more questions about acknowledging interests, tone or reading level. Kiddie
     rubric: 7-9 questions per topic; the ranking is unchanged.
3. Window pilot on rag26 (10 topics), pre-registered in the style of `WINDOW_STUDY_PLAN.md`:
   - **Result (2026-10-09, `temp/bod_pilot_analysis.py`, window only):**
     - Run: 83 runs x 10 topics = 828 answers; 797 calls, $0.09, 8 s.
     - Rubric: 5-13 questions per topic.
     - Scores: BOD_MEAN median 0.756, p90 0.983, so it is near the ceiling. Argmax grade is
       3 on 42% of questions and 2 on 40%. The expected grade differs from the argmax by
       more than 0.25 on 27% of questions.
     - Split-half Spearman: 0.969.
     - Length: per-topic Spearman(BOD_MEAN, words) median 0.518. For comparison, Jev
       pairwise 0.558 and Gemini 0.451 (7 topics).
     - Agreement with Jev pairwise (mirror): board Spearman 0.990, Kendall 0.919;
       per-topic median 0.968.
     - Agreement with Gemini (7 window topics): board Spearman 0.972.
     - Padding: mean 0.43; Spearman with words -0.18.
   - split-half stability over topics
   - length dependence
   - padding probe
   - agreement with the Jev pairwise leaderboard on the window
   - **Padding test (2026-10-09, `temp/bod_padding.py`, rag26 window):** 160 targets x 6
     variants, $0.11. Every criterion written beforehand passed, on all 10 topics.

     | Variant | dMean | 95% CI | dPad | 95% CI |
     | --- | --- | --- | --- | --- |
     | repeat | -0.023 | -0.025..-0.020 | +0.51 | +0.47..+0.54 |
     | generic | -0.032 | -0.035..-0.028 | +0.54 | +0.50..+0.57 |
     | off-topic | -0.043 | -0.047..-0.039 | +0.57 | +0.54..+0.61 |
     | relevant | +0.132 | +0.121..+0.144 | +0.04 | |
     | truncate | -0.103 | -0.114..-0.092 | -0.04 | |

   - **ragtime26 window pilot (2026-10-09):**
     - Run: 49 runs x 10 topics = 490 answers; $0.05.
     - Rubric: 6-10 questions per topic, substantive and using the background as intended.
     - Scores: BOD_MEAN median 0.710, p90 0.894.
     - Split-half Spearman: 0.854 (0.789..0.914).
     - Length: Spearman(BOD_MEAN, words) median -0.12; Spearman(PADDING, words) median
       +0.49.
     - Agreement with the pairwise judges: NOT computed. The ragtime26 window is the
       pairwise study's confirmation set (WINDOW_STUDY_PLAN.md), to be read only after the
       rag26 decisions are recorded.
4. Price the full run from the pilot's per-call cost, then run both datasets in full,
   staying within the key's budget.
5. Split the submission per dataset, then do a TIRA dry run.

## Policy
- The per-response grades (`grades.jsonl`) are restricted outside the window. Every
  analysis reads them through `window_records`.
- Across all topics, read only `.eval.txt` and the nugget banks.
- No setting is chosen on restricted topics.
