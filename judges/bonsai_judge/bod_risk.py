#!/usr/bin/env python3
"""BOD x Hallucination Risk: the bag-of-decisions grade discounted by unsupported content.

Pre-registered in temp/BOD_RISK_PLAN.md (definition fixed before any 2026 data was read).

HALLUC_RISK per (run, topic) answer, after temp/hallucination_cross.py:
  concepts  = spaCy NOUN/PROPN lemmas (lowercased set), as in concept-F1
  sn        = the answer's concepts; hn = union of the concepts of its cited documents
  df[t][c]  = number of in-scope runs whose answer for topic t states c
  private-ungrounded = {c in sn - hn : df[t][c] == 1}  (unsupported AND no peer says it)
  HALLUC_RISK = |private-ungrounded| / |sn|; no_support_risk (default 1) when undefined
  (no citations, no cited-document text, or no concepts on either side).

BOD_GROUNDED (primary) = BOD_MEAN * (1 - HALLUC_RISK) per answer; the 'all' rows are
the mean over topics. No weight, no threshold. The BOD grades are the BodJevJudge's
(same rubric, same cached Jev requests).

The risk depends on the pool of runs (df), so values are comparable within one
dataset only. Artifacts: <filebase>.bod/risk.jsonl (numbers only, never concepts).
"""
from __future__ import annotations

import gc
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from autojudge_base import (
    Leaderboard,
    LeaderboardBuilder,
    LeaderboardSpec,
    LlmConfigProtocol,
    MeasureSpec,
    NuggetBanksProtocol,
    Qrels,
    Report,
    Request,
)

from .bod import BOD_SPEC, BodJevJudge, bod_values
from .bonsai_judge import _CONCEPT_DISABLE, _clean
from .pairwise import _add_empty_rows
from .pairwise_jev import JEV_MODEL

BOD_RISK_SPEC = LeaderboardSpec(measures=(
    MeasureSpec("BOD_GROUNDED", description=(
        "Primary. BOD_MEAN x (1 - HALLUC_RISK) per answer: the graded rubric score "
        "discounted by the share of the answer's concepts that are neither in its cited "
        "documents nor stated by any other run. Mean over topics.")),
    MeasureSpec("HALLUC_RISK", description=(
        "Share of the answer's noun/proper-noun concepts absent from its cited documents "
        "AND stated by no other run for the topic (1 when the answer has no usable "
        "citations). Higher is worse. Depends on the pool of runs.")),
    MeasureSpec("UNGROUNDED_RATE", description=(
        "Share of the answer's concepts absent from its cited documents "
        "(1 - CONCEPT_PRECISION). Diagnostic; higher is worse.")),
    *BOD_SPEC.measures,
))


# =============================================================================
# Concepts and risk
# =============================================================================

def _answer_and_citations(report: Report) -> Tuple[str, frozenset]:
    parts, cited = [], set()
    for sent in report.get_sentences_with_citations():
        parts.append(sent.text or "")
        cited.update(sent.citations or [])
    return _clean(" ".join(parts)), frozenset(cited)


def _tag(items: List[Tuple[str, Any]], spacy_model: str, include_propn: bool,
         chunk: int, batch_size: int, label: str) -> Dict[Any, frozenset]:
    """{key: concept set} for (text, key) items. The pipeline is reloaded every `chunk`
    items: spaCy's StringStore otherwise grows without bound over a large corpus."""
    import spacy
    tags = ("NOUN", "PROPN") if include_propn else ("NOUN",)
    out: Dict[Any, frozenset] = {}
    for start in range(0, len(items), chunk):
        nlp = spacy.load(spacy_model, disable=_CONCEPT_DISABLE)
        nlp.max_length = 3_000_000
        for doc, key in nlp.pipe(items[start:start + chunk], as_tuples=True,
                                 batch_size=batch_size):
            out[key] = frozenset(sys.intern(t.lemma_.lower()) for t in doc if t.pos_ in tags)
        del nlp
        gc.collect()
        print(f"[bod-risk] tagged {label} {min(start + chunk, len(items)):,}/{len(items):,}",
              flush=True)
    return out


def risk_from_concepts(answers: Dict[Tuple[str, str], frozenset],
                       cited: Dict[Tuple[str, str], frozenset],
                       doc_concepts: Dict[str, frozenset],
                       no_support_risk: float = 1.0) -> Dict[Tuple[str, str], dict]:
    """Pure risk computation. answers: {(run, topic): answer concepts}; cited:
    {(run, topic): cited doc ids}; doc_concepts: {doc id: concepts}. df counts every
    run in `answers` for the topic."""
    df: Dict[str, Counter] = defaultdict(Counter)
    for (_run, topic), sn in answers.items():
        df[topic].update(sn)
    out = {}
    for (run, topic), sn in answers.items():
        hn = frozenset().union(*(doc_concepts.get(d, frozenset())
                                 for d in cited.get((run, topic), ())))
        if not sn or not hn:
            out[(run, topic)] = {"risk": no_support_risk, "ungrounded": no_support_risk,
                                 "defined": False, "n_concepts": len(sn),
                                 "n_cited": len(cited.get((run, topic), ()))}
            continue
        ung = sn - hn
        private = sum(df[topic][c] == 1 for c in ung)
        out[(run, topic)] = {"risk": private / len(sn), "ungrounded": len(ung) / len(sn),
                             "defined": True, "n_concepts": len(sn),
                             "n_cited": len(cited.get((run, topic), ()))}
    return out


def hallucination_risk(reports: Iterable[Report], spacy_model: str = "en_core_web_lg",
                       include_propn: bool = True, no_support_risk: float = 1.0,
                       chunk: int = 5000, batch_size: int = 128,
                       doc_cache: Optional[Path] = None) -> Dict[Tuple[str, str], dict]:
    """HALLUC_RISK for every non-empty report. doc_cache: optional pickle of cited-doc
    concept sets (keyed by model settings) so reruns skip document tagging."""
    texts: List[Tuple[str, Tuple[str, str]]] = []
    cited: Dict[Tuple[str, str], frozenset] = {}
    available: set = set()                        # cited doc ids that carry a document
    for r in reports:
        text, cset = _answer_and_citations(r)
        if not text.strip():
            continue
        key = (r.metadata.run_id, r.metadata.topic_id)
        texts.append((text, key))
        cited[key] = cset
        docs = r.documents or {}
        available.update(d for d in cset if docs.get(d) is not None)
    tag = f"{spacy_model}|propn={include_propn}"
    doc_concepts: Dict[str, frozenset] = {}
    if doc_cache and doc_cache.exists():
        cached = pickle.loads(doc_cache.read_bytes())
        if cached.get("tag") == tag:
            doc_concepts = cached["docs"]
    need = available - set(doc_concepts)
    if need:
        items: List[Tuple[str, str]] = []
        for r in reports:
            for d, doc in (r.documents or {}).items():
                if d in need:
                    items.append((_clean(doc.get_document_text()), d))
                    need.discard(d)
        doc_concepts.update(_tag(items, spacy_model, include_propn, chunk, batch_size, "docs"))
        if doc_cache:
            doc_cache.write_bytes(pickle.dumps({"tag": tag, "docs": doc_concepts},
                                               protocol=pickle.HIGHEST_PROTOCOL))
    answers = _tag(texts, spacy_model, include_propn, chunk, batch_size, "answers")
    return risk_from_concepts(answers, cited, doc_concepts, no_support_risk)


# =============================================================================
# Tally and judge
# =============================================================================

def tally_risk(rows: Iterable[dict], risk: Dict[Tuple[str, str], dict],
               expected_topic_ids: Sequence[str], empties: Iterable[Tuple[str, str]] = (),
               on_missing: str = "default", threshold: float = 0.5,
               no_support_risk: float = 1.0) -> Leaderboard:
    """Leaderboard from stored BOD grade rows and per-answer risk (no LLM calls)."""
    builder = LeaderboardBuilder(BOD_RISK_SPEC)
    values = bod_values(rows, threshold)
    for (run, topic), v in values.items():
        rk = risk.get((run, topic), {"risk": no_support_risk, "ungrounded": no_support_risk})
        builder.add(run_id=run, topic_id=topic, values={
            **v, "BOD_GROUNDED": v["BOD_MEAN"] * (1 - rk["risk"]),
            "HALLUC_RISK": rk["risk"], "UNGROUNDED_RATE": rk["ungrounded"]})
    _add_empty_rows(builder, BOD_RISK_SPEC, empties, values)
    return builder.build(expected_topic_ids=list(expected_topic_ids), on_missing=on_missing)


class BodRiskJudge(BodJevJudge):
    """BodJevJudge grades, discounted per answer by HALLUC_RISK."""

    def judge(
        self,
        rag_responses: Iterable[Report],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        qrels: Optional[Qrels] = None,
        on_missing_evals: str = "default",
        model: Optional[str] = JEV_MODEL,
        padding: bool = True,
        threshold: float = 0.5,
        max_topics: Optional[int] = None,
        max_runs: Optional[int] = None,
        dev_topics: Optional[Sequence[str]] = None,   # dev variants: ONLY these topics
        max_outstanding: int = 32,
        max_attempts: int = 8,
        spacy_model: str = "en_core_web_lg",
        include_propn: bool = True,
        no_support_risk: float = 1.0,
        spacy_chunk: int = 5000,
        filebase: str = "default",
        outdir: Path = Path("."),
        **kwargs: Any,
    ) -> Leaderboard:
        rows, expected_topic_ids, empties, reports = self._grade(
            rag_responses, rag_topics, llm_config, nugget_banks, model, padding, max_topics,
            max_runs, dev_topics, max_outstanding, max_attempts, filebase, outdir)
        art = Path(outdir) / f"{Path(filebase).name}.bod"
        risk = hallucination_risk(reports, spacy_model, include_propn, no_support_risk,
                                  chunk=spacy_chunk, doc_cache=art / "doc_concepts.pkl")
        with open(art / "risk.jsonl", "w") as f:
            for (run, topic), v in sorted(risk.items()):
                f.write(json.dumps({"run_id": run, "topic_id": topic, **v}) + "\n")
        n_def = sum(v["defined"] for v in risk.values())
        print(f"[bod-risk] risk defined for {n_def:,}/{len(risk):,} non-empty answers "
              f"(undefined -> {no_support_risk})")
        return tally_risk(rows, risk, expected_topic_ids, empties, on_missing_evals,
                          threshold, no_support_risk)
