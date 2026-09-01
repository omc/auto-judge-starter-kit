# RAG26 Concept Analysis — Session Handoff

**Branch:** `max/bonsai_judge_citations` · **Dataset:** TREC 2026 AutoJudge RAG subset (`data/rag26/`)
**Status:** exploratory analysis + a shipped judge change. Nothing committed yet.

This note walks a teammate through everything we discovered this session, in the order we
found it, with the tables and the caveats. Two threads run in parallel:

1. **Analysis** — understanding how 83 anonymized RAG systems behave on 119 shared topics.
2. **Engineering** — implementing the "concept-F1" metric and merging it into `bonsai_judge`.

> **Anonymization:** run names (`adela`, `alex`, …) and team IDs (`T448`, …) are randomized in
> this release. We use them only as grouping labels; no attempt is made to identify participants.

---

## 0. The data model

Each generation run is a JSONL file, one line per topic. Per record:

| field                  | meaning                                                                          |
| ---------------------- | -------------------------------------------------------------------------------- |
| `metadata`             | `run_id`, `topic_id` (`rag2026-0..118`), `team_id`                               |
| `answer` / `responses` | ~40 sentences, each `{text, citations}` where `citations` = `{shard_id: weight}` |
| `references`           | ordered list of cited shard IDs                                                  |
| `documents`            | `shard_id → {id, text}` for every cited shard                                    |

- **83 generation runs** across **25 teams**; **118 retrieval runs** across 28 teams; **119 topics**.
- A **shard** = one passage/document from the corpus (`shard_<corpusShard>_<offset>`), the retrievable/citable unit. It is the RAG analog of a "search result."
- A **nugget** (not shipped in this release) would be an atomic _fact the answer should contain_ — an evaluation-side rubric item, distinct from a shard (a source-side unit). We later reuse the word loosely for "answer concept."

All analysis outputs are CSVs in `temp/rag26_consensus/`. Scripts are in `temp/`.

---

## 1. Document consensus — which sources do runs agree on?

**Question:** For the same topic, do different systems cite the same shards?
**Script:** `temp/consensus.py` → `consensus_*.csv`

Across **62,143** distinct (topic, shard) citation pairs:

| overlap                                         | share     |
| ----------------------------------------------- | --------- |
| cited by exactly **1 run** (idiosyncratic)      | **49.3%** |
| cited by **≥ half** the runs (strong consensus) | **0.1%**  |

Heavy long tail: citations decay fast from 30.6k pairs at k=1 to a thin tail out to one shard cited by **73 of 83 runs**. A small set of "gold" documents dominate:

| topic       | shard             | #runs cited | Σweight |
| ----------- | ----------------- | ----------: | ------: |
| rag2026-5   | shard_04647_72515 |     73 / 73 |   330.8 |
| rag2026-67  | shard_01710_72297 |     66 / 66 |   215.3 |
| rag2026-115 | shard_04563_66689 |     59 / 59 |   422.5 |

Per-topic agreement varies widely: max-consensus fraction ranges **0.88** (rag2026-5, a canonical-source topic) down to **0.17** (rag2026-59, where systems scatter).

> **Data quirk:** `references` ≈ the cited set. Median citation-rate-among-referencers = 1.00, so a
> run essentially cites everything it lists. There's no un-cited retrieval tail here, so you
> **cannot** reproduce a "buried gold" position histogram from generation runs alone — you need
> `runs/retrieval/` for that (see §4).

**Takeaway:** systems disagree enormously on _which sources_ to cite (~half of all citations are unique to one run).

---

## 2. Run-vs-run citation overlap → system clusters

**Question:** Which runs cite _alike_? **Metric:** mean per-topic Jaccard of cited-shard sets.
**Script:** `temp/jaccard.py` → `jaccard_matrix.csv`, `jaccard_pairs.csv`, `run_centrality.csv`

Distribution over 3,403 run-pairs: **mean 0.052, median 0.031**, but a distinct high-similarity tail out to **0.971**. 78% of pairs sit below 0.05 — so overlap is low by default, with tight pockets of near-identical behavior.

**Clusters (Jaccard ≥ 0.7):**

| cluster | members                         |
| ------- | ------------------------------- |
| A       | alvin, arlo, isaac, ryan, steve |
| B       | damon, hilda, laura, tania      |
| C       | aron, jeff                      |
| D       | conor, geoff                    |
| E       | deena, opal                     |
| F       | finn, rita                      |
| G       | eden, todd                      |

- **Central** (cite like everyone, few citations each): chris, zora, conor (~0.09 mean).
- **Outliers:** carlo (0.015), annie (0.012) barely cite at all; but yara, carmen, jamie cite _heavily_ yet overlap no one — genuinely divergent retrieval.

**Takeaway:** the clusters looked like they might be same-team variants → tested next.

---

## 3. Clusters × retrieval runs — the retriever drives everything

**Question:** Are those clusters sharing a retrieval backend?
**Script:** `temp/xref.py` → `gen_to_retrieval.csv`

Retrieval and generation runs use **disjoint** anonymized namespaces, so there's no label linking
them. Structural test instead: a run can only cite what it retrieved, so **citation coverage** =
fraction of a run's cited shards present in a retrieval run's ranked list.

**Result — confirmed decisively.** 8 of 10 generation clusters have every member's citations
**100% covered by a single shared retrieval run:**

| gen cluster                      | shared retriever | coverage |
| -------------------------------- | ---------------- | -------: |
| alvin/arlo/isaac/rene/ryan/steve | **penny**        |     1.00 |
| aron/chris/eva/jeff/zora         | **zelda**        |     1.00 |
| adela/gwen/karl/marco            | **vera**         |     1.00 |
| damon/hilda/laura/tania          | **tom**          |     1.00 |
| conor/geoff, deena/opal          | **zelda**        |     1.00 |

Two findings beyond the hypothesis:

- **A few retrievers feed many generators.** `zelda` is the best match for ~12 generation runs;
  `troy`, `ken`, `vera`, `kirk` each feed several. The field is a handful of retrievers × many
  generation strategies layered on top.
- **`cov@20` resurfaces the "buried gold" story.** Same-retriever runs differ sharply in _depth_:
  `penny` cluster cites almost entirely from the top-20 (cov@20 ≈ 0.99); `vera`/`zelda` clusters
  pull **~60% of citations from below rank 20** (cov@20 ≈ 0.40). Different generators mine the
  retrieval list to different depths.
- **7 of 83 runs** (carmen, lars, edith, ari, ariel, sara, yara) have <50% coverage by _any_
  submitted retriever → they used a **private/in-house retriever**. These are exactly the
  "heavy but divergent" outliers from §2.

**Takeaway:** citation overlap ≈ retrieval overlap. The retriever is a shared substrate, not a per-team choice.

---

## 4. The concept-F1 metric

**Question:** implement the "concept-F1" metric from
[maxirwin.com/articles/llm-rag/](https://maxirwin.com/articles/llm-rag/) (author's reference:
`temp/concept-f1-reference.py`).

**Definition.** A _concept_ = spaCy `NOUN`/`PROPN` token **lemma** (lowercased), as a set.

- reference (`hn`) = concepts in the documents a report **cites**
- candidate (`sn`) = concepts in the report's **summary**
- `PRECISION = |sn ∩ hn| / |sn|` — grounding / anti-hallucination
- `RECALL = |sn ∩ hn| / |hn|` — coverage of cited-doc concepts
- `F1 = 2PR/(P+R)`

Mechanics (matched to the reference): `en_core_web_lg`, lemmatizer kept, HTML stripped/unescaped,
**lemmatized** (not surface form), records with empty either-side **skipped**.

**Key discovery — full documents break the metric's symmetry.** The blog cited short
title+description **snippets**; RAG26 cites **full documents** (~2,500 words). Measured:

| dataset                     |          precision (median) | recall (median) |
| --------------------------- | --------------------------: | --------------: |
| kiddie (short docs)         |                        0.97 |            0.05 |
| **rag26 slice** (full docs) | **0.982–0.984** (saturated) |       0.03–0.09 |

- **Precision saturates near 1.0** on full docs — nearly every summary noun appears _somewhere_ in
  a long cited document, so precision can't separate systems.
- **Recall is length-driven** — a short summary can't cover a full document's vocabulary, so recall
  ≈ 3–9% and mostly measures how much a system quotes, not quality.

**Decision:** report **precision and recall as separate measures**, not a single F1. On rag26 the
_false-positive_ (ungrounded) signal from precision is the useful one — which set up §8.

**Runtime reality** (193M tokens of cited-doc text): `en_core_web_trf` ≈ 6.5 h · `en_core_web_sm`
≈ 2 h · `en_core_web_lg` ≈ 2.3 h single-proc (IPC-bound; multiprocessing gives only ~3×).

---

## 5. Engineering — concept-F1 in `bonsai_judge`

Implemented, merged, and made self-contained (see `judges/bonsai_judge/`):

- **`BONSAI_SPEC` now emits 5 measures:** `RELEVANCE`, `COMPLETENESS` (LLM), plus
  `CONCEPT_PRECISION`, `CONCEPT_RECALL`, `CONCEPT_F1` (deterministic, no LLM).
- Concept helpers inlined into `bonsai_judge.py`; the standalone `concept_f1_judge/` was removed.
- `spacy>=3.7,<4` added to `pyproject.toml`; `Dockerfile` runs `python -m spacy download en_core_web_lg`.
- `judge_settings`: `spacy_model` (swap to `en_core_web_trf`), `include_propn`, `nproc`, `batch_size`.

**How `RELEVANCE` / `COMPLETENESS` are computed** (pre-existing, worth knowing):
one LLM call per (run, topic) that asks the model to emit `RELEVANCE: <0-1>` / `COMPLETENESS: <0-1>`
for `Query: {title}\n\nResponse: {report}`. Caveats:

- It's the LLM's **self-reported opinion**, not a formula.
- Query = **topic title only** (not the narrative).
- The generated **nuggets/qrels are unused** by the judge.
- Parse failure → **0.0** (conflates "bad" with "malformed"; biases means down).

**End-to-end verified** on kiddie and a rag26 slice against **`qwen/qwen3.8-flash`** (via OpenRouter;
HTTP 429s auto-retried by minima-llm, `failed=0`). All 5 measures populate. Example (rag26 slice):

| run   | RELEVANCE | COMPLETENESS | CONCEPT_PREC | CONCEPT_RECALL |
| ----- | --------: | -----------: | -----------: | -------------: |
| adela |      0.81 |         0.68 |        0.984 |          0.030 |
| alex  |      0.91 |         0.68 |        0.982 |          0.087 |
| zora  |      0.84 |         0.59 |        0.982 |          0.041 |

The LLM axis and the concept axis **can disagree** (on kiddie, run4 scored 0.02 relevance from the
LLM but had the best concept recall) — that independence is the point of having both.

---

## 6. Answer-concept consensus — which teams inject outlier concepts?

**Question:** treat each run's answer noun-concepts as its "nuggets"; per topic, which concepts are
shared vs private, and which teams are outliers? **Script:** `temp/concept_consensus.py` →
`answer_concept_by_{run,team,run_topic}.csv`

Across **1.44M** concept-mentions (83 runs × 119 topics):

|                                               | share of mentions |
| --------------------------------------------- | ----------------: |
| **private** (stated by exactly 1 run)         |          **5.6%** |
| **rare** (< 10% of runs)                      |             28.5% |
| mentions-weighted mean-share (0=alone, 1=all) |              0.31 |

**Answers agree far more than citations do** (5.6% private concepts vs 49.3% private _shards_).
Many retrieval paths, similar conceptual destinations.

**Outlier runs** (`private_rate` = fraction of a run's answer concepts unique to it; median = 0.049):

| run   | team | private_rate | ×median | avg concepts | note                  |
| ----- | ---- | -----------: | ------: | -----------: | --------------------- |
| brad  | T175 |        0.158 |    3.3× |          139 | verbose + distinctive |
| rene  | T047 |        0.158 |    3.2× |          166 |                       |
| june  | T300 |        0.121 |    2.5× |          200 |                       |
| april | T448 |        0.108 |    2.2× |          218 |                       |
| sofia | T578 |        0.105 |    2.2× |          204 |                       |
| paula | T907 |        0.095 |    2.0× |       **19** | tiny answers          |

**Mainstream:** team **T949** (nina/leon/nola/jon) ≈ 0.00–0.008; **T768** (deena/opal) ≈ 0.01.
**By team:** T614 (2.0×), T578 (1.9×), **T086 (1.6× across 7 runs** — most robust), … T949 (0.1×).

> **Two kinds of outlier:** by **addition** (T086, april: ~200 concepts, high consensus coverage
> _plus_ a unique layer) vs by **omission** (paula/T907: ~19 concepts, near-zero coverage — their
> little is skewed rare). Don't rank on `private_rate` alone; check `avg_concepts` and
> `consensus_coverage`.

**Caveat that motivated §7:** a private concept is unique _among answers_ — it may still be grounded
in that run's cited docs. Private ≠ hallucination until you check grounding.

---

## 7. Hallucination cross — private × ungrounded (the payoff)

**Question:** which concepts are both **ungrounded** (in the answer but _not_ the cited docs — the
precision false-positives) **and private** (no other run states them)? That intersection is the
strongest hallucination candidate. **Script:** `temp/hallucination_cross.py`

**Demo:** contrasting runs, first 40 topics, cross-run rarity computed over **all 83 runs' answers**;
cited docs tagged only for the selected runs (~8 min).

| run   | team | precision | ungrounded | **private-ung** | % of ungrounded that's private |
| ----- | ---- | --------: | ---------: | --------------: | -----------------------------: |
| june  | T300 |     0.755 |      0.245 |       **0.069** |                            30% |
| april | T448 |     0.836 |      0.164 |       **0.066** |                            43% |
| paula | T907 |     0.904 |      0.096 |           0.026 |                            15% |
| brad  | T175 | **0.998** |      0.002 |           0.001 |                            15% |
| rene  | T047 | **1.000** |      0.000 |           0.000 |                             3% |
| hana  | T300 |     0.768 |      0.232 |       **0.000** |                             0% |
| nina  | T949 |     0.895 |      0.105 |           0.000 |                             0% |

**The headline reversal:** `brad` and `rene` were the **#1 and #2 answer-concept outliers** (§6,
3.3×/3.2× median) — yet they are **near-perfectly grounded** (precision 0.998 / 1.000). Their
distinctiveness is _citing unique sources faithfully_, **not** hallucinating. **The answer-only
outlier metric falsely flags them.** You must cross privacy with grounding.

The real hallucination risks are **june** and **april**: high ungrounded rate _and_ a large private
share. Their top private-ungrounded concepts are **analytical/editorial framing**, not facts:

- **june:** `hindsight, posture, hybrid, mismatch, mode, analyst, quarter`
- **april:** `short‑term, near‑term, low‑cost, long‑term, trade‑off, equity, funder`

→ the model imposing interpretation/structure the sources don't contain.

Contrast **hana:** 23% ungrounded but **0% private** — its ungrounded concepts are common-knowledge
words _other runs also add_. That's systematic elaboration, not idiosyncratic fabrication.

**Conclusion:** **hallucination ≈ ungrounded ∩ private.** Neither grounding (precision) nor
cross-run rarity alone identifies it; the cross does.

---

## 8. Open items / next steps

1. **Full hallucination run:** `python temp/hallucination_cross.py --runs all --topics 119`
   — needs the full cited-doc tagging (~2 h). The demo used 7 runs / 40 topics.
2. **Snippet-analog reference** (`--snippet-chars N`): truncate each cited doc to its first ~2
   sentences before tagging, to restore precision's discriminative power on rag26 (§4). Cheap win;
   recommended before any full concept-F1 run.
3. **Wire nuggets into `RELEVANCE`/`COMPLETENESS`** — currently generated but unused; e.g.
   completeness = fraction of nugget questions answered.
4. **Persist concept sets** — the k-overlap histogram (§6) was lost to a report-stage bug after
   tagging; per-run/team/topic CSVs are complete, only the full distribution curve is missing.
5. **Log parse misses** in the LLM judge so `0.0`-from-failure is distinguishable from `0.0`-from-model.

## File map

| path                          | what                                     |
| ----------------------------- | ---------------------------------------- |
| `temp/consensus.py`           | §1 document consensus                    |
| `temp/jaccard.py`             | §2 run-vs-run overlap                    |
| `temp/xref.py`                | §3 cluster × retrieval                   |
| `temp/concept_f1.py`          | §4 standalone concept-F1 (P/R separated) |
| `judges/bonsai_judge/`        | §5 merged judge (5 measures)             |
| `temp/concept_consensus.py`   | §6 answer-concept outliers               |
| `temp/hallucination_cross.py` | §7 private × ungrounded                  |
| `temp/rag26_consensus/*.csv`  | all numeric outputs                      |
