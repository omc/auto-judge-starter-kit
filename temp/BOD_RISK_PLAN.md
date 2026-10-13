# BOD × Hallucination Risk — pre-registered plan

Written 2026-10-13, before any risk value was computed on 2026 data.

## Definition (fixed; not tuned)

Per (run, topic) answer, concepts are spaCy `en_core_web_lg` NOUN/PROPN lemmas (lowercased set),
as in concept-F1 and `temp/hallucination_cross.py`:

- `sn` = the answer's concepts; `hn` = union of the concepts of the documents the answer cites.
- `df[topic][c]` = number of runs whose answer for that topic states `c` (all runs judged).
- ungrounded = `sn − hn`; private-ungrounded = ungrounded concepts with `df == 1`.
- `HALLUC_RISK` = |private-ungrounded| / |sn|.
- Undefined risk (no citations, no cited-document text, or no concepts on either side):
  `HALLUC_RISK = 1` (setting `no_support_risk: 1.0`).
- Primary: `BOD_GROUNDED = BOD_MEAN × (1 − HALLUC_RISK)` per answer, mean over topics.
  No weight, no threshold. Empty reports score 0.

The BOD grades are the existing, cached ones (same rubric, same Jev requests), so no new Jev calls.

Provenance caveat: the risk definition was first explored in the withdrawn all-topic
`rag26_consensus` analysis. Everything below re-checks it on the window only; nothing is chosen
from it.

## Criteria (rag26 window = development; ragtime26 window = confirmation, read once)

These decide whether the combined judge is submitted. They do not tune it. If a criterion fails,
the judge is reported as failing that criterion and is not submitted unless the user decides
otherwise.

1. **Pipeline.** 0 new Jev calls on the window. Risk defined for ≥ 99% of non-empty answers.
2. **Not redundant.** Board Spearman(BOD_GROUNDED, BOD_MEAN) < 0.99. If ≥ 0.99, risk changes nothing
   that matters, so there's no point submitting it separately.
3. **Not dominant.** Board Spearman(BOD_GROUNDED, BOD_MEAN) ≥ 0.70. Below that, risk (a lexical
   proxy) overrides graded quality.
4. **Reliability.** Split-half board Spearman of BOD_GROUNDED (200 random topic halves) is at least
   BOD_MEAN's value minus 0.05 on the same window.
5. **Length.** Median per-topic Spearman(BOD_GROUNDED, words) is no further from 0 than BOD_MEAN's
   is, plus 0.15.
6. **Pool stability.** Recomputing df with a random half of the other runs removed (20 draws)
   gives a board Spearman of at least 0.90 against the full-pool BOD_GROUNDED.

Diagnostics reported without pass/fail: per-topic Spearman(risk, number of cited documents), the
gaming confound; Spearman(risk, BOD_MEAN); the risk distribution; ungrounded rate.

## Steps

1. Implement `BodRiskJudge` + `workflow.bod_risk.yml` (kiddie / pilot / full); unit tests.
2. Kiddie run (pipeline only).
3. rag26 window pilot → criteria 1–6 → record the results here.
4. ragtime26 window pilot → same criteria, confirmation, read once.
5. If they pass: full runs → `submissions/bonsai-bod-risk-jev-<date>/`, TIRA dry run. Upload only
   on the user's go-ahead.

## Results

Kiddie (2026-10-13): pipeline OK; 0 new Jev calls; risk defined 20/20.

rag26 window (2026-10-13; `temp/win_rag26_bod_risk.txt`): 10 topics, 83 runs, 828 answers.

| Criterion | Result | |
| --- | --- | --- |
| 1 Pipeline | 0 new calls; risk defined 826/828 | PASS |
| 2 Not redundant | board rho(GROUNDED, MEAN) 0.992 (needs < 0.99) | **FAIL** |
| 3 Not dominant | 0.992 >= 0.70 | PASS |
| 4 Reliability | split-half 0.968 vs 0.969 | PASS |
| 5 Length | +0.505 vs +0.523 | PASS |
| 6 Pool stability | half-pool rho min 0.993 | PASS |

Diagnostics: risk mean 0.021 (median 0.009, p75 0.027); ungrounded mean 0.104; median per-topic
rho(risk, cited docs) +0.165; rho(risk, BOD_MEAN) +0.331.

Decision per plan: the pre-registered definition fails criterion 2 on rag26 (it barely changes the
ranking), so it is not submitted as is. ragtime26 was not read.

User decision (2026-10-13): submit BOD_GROUNDED as a distinct judge regardless of criterion 2. A
near-BOD ranking is acceptable; the judge is part of an iterative process, not a gated check.
Criteria 1 and 3–6 are still reported for ragtime26.

ragtime26 window (confirmation, read once; `temp/win_ragtime26_bod_risk.txt`): 10 topics, 49 runs,
490 answers. All six criteria pass: 0 new calls, risk defined 490/490; board rho(GROUNDED, MEAN)
0.984; split-half 0.820 vs 0.854; length -0.163 vs -0.120; half-pool rho min 0.910. Diagnostics:
risk mean 0.025 (median 0.013, max 0.139); ungrounded mean 0.140; rho(risk, cited docs) +0.131;
rho(risk, BOD_MEAN) +0.347.
