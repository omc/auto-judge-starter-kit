#!/usr/bin/env python3
"""
BonsaiJudge: Full-protocol LLM judge using OpenRouter-compatible API.

Three modular classes:
- BonsaiNuggetCreator: Generates nugget questions per topic via LLM
- BonsaiQrelsCreator: Assigns relevance grades via LLM
- BonsaiLeaderboardJudge: Scores responses and produces leaderboard
"""

import asyncio
import html
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type

from autojudge_base import (
    LlmConfigProtocol,
    Report,
    Request,
    Leaderboard,
    LeaderboardBuilder,
    LeaderboardSpec,
    MeasureSpec,
    Qrels,
    QrelsSpec,
    build_qrels,
    doc_id_md5,
    NuggetBanks,
    NuggetBanksProtocol,
)
from autojudge_base.nugget_data import NuggetBank, NuggetQuestion
from minima_llm import MinimaLlmConfig, MinimaLlmRequest, MinimaLlmResponse, OpenAIMinimaLlm


# =============================================================================
# Concept-F1 (deterministic, no LLM) — port of maxirwin.com/articles/llm-rag/
# =============================================================================
#
# Concept = spaCy NOUN/PROPN token lemma (lowercased), as a SET.
#   reference (hn) = lemma-noun concepts of the documents a report CITES
#   candidate (sn) = lemma-noun concepts of the report's summary text
#   CONCEPT_PRECISION = |sn & hn| / |sn|   grounding / anti-hallucination
#   CONCEPT_RECALL    = |sn & hn| / |hn|   coverage of cited-doc concepts
#   CONCEPT_F1        = 2PR/(P+R)
# Precision and recall are reported separately: with full-document citations
# recall is length-driven, so precision is the discriminative signal.

CONCEPT_MEASURES = (
    MeasureSpec("CONCEPT_PRECISION", description=(
        "Concept-F1 precision: fraction of the summary's noun/proper-noun lemma "
        "concepts that also occur in the cited documents (grounding / "
        "anti-hallucination). Higher is better.")),
    MeasureSpec("CONCEPT_RECALL", description=(
        "Concept-F1 recall: fraction of the cited documents' noun/proper-noun lemma "
        "concepts present in the summary (coverage). Length-driven when citing full "
        "documents; compare across runs rather than as an absolute.")),
    MeasureSpec("CONCEPT_F1", description=(
        "Harmonic mean of CONCEPT_PRECISION and CONCEPT_RECALL "
        "(concept-f1, after maxirwin.com/articles/llm-rag/).")),
)

_CONCEPT_DISABLE = ["parser", "senter", "ner", "entity_ruler", "textcat",
                    "morphologizer", "trainable_lemmatizer"]
_TAG_RE = re.compile(r"<.*?>")
_NLP_CACHE: dict = {}


def _get_nlp(model: str):
    nlp = _NLP_CACHE.get(model)
    if nlp is None:
        import spacy
        nlp = spacy.load(model, disable=_CONCEPT_DISABLE)
        nlp.max_length = 3_000_000
        _NLP_CACHE[model] = nlp
    return nlp


def _clean(text: str) -> str:
    """Strip HTML tags and unescape entities, as in the reference getnouns()."""
    return html.unescape(_TAG_RE.sub("", text or ""))


def _noun_lemmas(doc, include_propn: bool) -> frozenset:
    tags = ("NOUN", "PROPN") if include_propn else ("NOUN",)
    return frozenset(t.lemma_.lower() for t in doc if t.pos_ in tags)


def _prf(sn: frozenset, hn: frozenset) -> Tuple[float, float, float]:
    """Precision, recall, F1 on two non-empty concept sets (reference formulas)."""
    precision = 1 - len(sn - hn) / len(sn)
    recall = 1 - len(hn - sn) / len(hn)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def concept_scores(
    reports: Iterable[Report],
    spacy_model: str = "en_core_web_lg",
    include_propn: bool = True,
    nproc: int = 1,
    batch_size: int = 128,
) -> Dict[Tuple[str, str], Optional[Tuple[float, float, float]]]:
    """Compute concept-F1 per report.

    Returns {(run_id, topic_id): (precision, recall, f1)}, or None when either
    concept set is empty (the reference skips those). Cited-document concept sets
    are cached by shard id across reports.
    """
    nlp = _get_nlp(spacy_model)

    records: List[Tuple[str, str, frozenset]] = []
    candidate_texts: List[str] = []
    shard_text: dict = {}
    for resp in reports:
        cited = set()
        parts: List[str] = []
        for sent in resp.get_sentences_with_citations():
            parts.append(sent.text or "")
            for shard in (sent.citations or []):
                cited.add(shard)
        records.append((resp.metadata.run_id, resp.metadata.topic_id, frozenset(cited)))
        candidate_texts.append(_clean(" ".join(parts)))
        docs = resp.documents or {}
        for shard in cited:
            if shard not in shard_text:
                doc = docs.get(shard)
                if doc is not None:
                    shard_text[shard] = _clean(doc.get_document_text())

    shard_nouns: dict = {}
    for doc, sid in nlp.pipe(((t, sid) for sid, t in shard_text.items()),
                             as_tuples=True, batch_size=batch_size, n_process=nproc):
        shard_nouns[sid] = _noun_lemmas(doc, include_propn)
    shard_text.clear()

    cand_nouns: dict = {}
    for doc, i in nlp.pipe(((txt, i) for i, txt in enumerate(candidate_texts)),
                           as_tuples=True, batch_size=batch_size, n_process=nproc):
        cand_nouns[i] = _noun_lemmas(doc, include_propn)

    scores: Dict[Tuple[str, str], Optional[Tuple[float, float, float]]] = {}
    for i, (run, topic, cited) in enumerate(records):
        hn: set = set()
        for shard in cited:
            hn |= shard_nouns.get(shard, frozenset())
        sn = cand_nouns.get(i, frozenset())
        scores[(run, topic)] = _prf(sn, hn) if (hn and sn) else None
    return scores


# =============================================================================
# Specs
# =============================================================================

# LLM-judged measures plus the deterministic concept-F1 measures.
BONSAI_SPEC = LeaderboardSpec(measures=(
    MeasureSpec("RELEVANCE", description="LLM-judged relevance score (0.0-1.0)"),
    MeasureSpec("COMPLETENESS", description="How completely the response addresses the query (0.0-1.0)"),
    *CONCEPT_MEASURES,
))


class GradeRecord:
    """Record for qrels building."""
    def __init__(self, topic_id: str, text: str, grade: int):
        self.topic_id = topic_id
        self.text = text
        self.grade = grade


BONSAI_QRELS_SPEC = QrelsSpec[GradeRecord](
    topic_id=lambda r: r.topic_id,
    doc_id=lambda r: doc_id_md5(r.text),
    grade=lambda r: r.grade,
    on_duplicate="keep_max",
)


def _make_backend(llm_config: LlmConfigProtocol) -> OpenAIMinimaLlm:
    """Create LLM backend from config (works with OpenRouter via OPENAI_BASE_URL)."""
    full_config = MinimaLlmConfig.from_dict(llm_config.raw) if llm_config.raw else MinimaLlmConfig.from_env()
    return OpenAIMinimaLlm(full_config)


# =============================================================================
# BonsaiNuggetCreator
# =============================================================================

class BonsaiNuggetCreator:
    """Creates nugget questions for each topic using LLM."""

    nugget_banks_type: Type[NuggetBanksProtocol] = NuggetBanks

    def create_nuggets(
        self,
        rag_responses: Iterable[Report],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        questions_per_topic: int = 3,
        filebase: str = "default",
        outdir: Path = Path("."),
        **kwargs: Any,
    ) -> Optional[NuggetBanksProtocol]:
        banks: List[NuggetBank] = []

        # TODO: Replace template questions with LLM-generated nuggets
        for topic in rag_topics:
            bank = NuggetBank(
                query_id=topic.request_id,
                title_query=topic.title or topic.request_id,
            )

            questions: List[NuggetQuestion] = []
            for i in range(questions_per_topic):
                question = NuggetQuestion.from_lazy(
                    query_id=topic.request_id,
                    question=f"Q{i+1}: What information about '{topic.title}' is provided?",
                    gold_answers=[f"Answer about {topic.title}"],
                )
                questions.append(question)

            bank.add_nuggets(questions)
            banks.append(bank)

        nugget_banks = NuggetBanks.from_banks_list(banks)
        print(f"BonsaiNuggetCreator: Created nuggets for {len(banks)} topics")
        return nugget_banks


# =============================================================================
# BonsaiQrelsCreator
# =============================================================================

class BonsaiQrelsCreator:
    """Creates relevance judgments using LLM."""

    def create_qrels(
        self,
        rag_responses: Iterable[Report],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        grade_range: Tuple[int, int] = (0, 3),
        filebase: str = "default",
        outdir: Path = Path("."),
        **kwargs: Any,
    ) -> Optional[Qrels]:
        topic_titles: Dict[str, str] = {t.request_id: t.title or "" for t in rag_topics}
        backend = _make_backend(llm_config)

        responses_list = list(rag_responses)
        requests: List[MinimaLlmRequest] = []
        for i, response in enumerate(responses_list):
            query = topic_titles.get(response.metadata.topic_id, "")
            text = response.get_report_text()
            requests.append(MinimaLlmRequest(
                request_id=f"qrels_{i}",
                messages=[
                    {"role": "system", "content": (
                        f"You are a relevance assessor. Grade the response's relevance to the query "
                        f"on a scale of {grade_range[0]}-{grade_range[1]}. "
                        f"Respond with ONLY a single integer."
                    )},
                    {"role": "user", "content": f"Query: {query}\n\nResponse: {text}"},
                ],
                temperature=0.0,
            ))

        llm_results = asyncio.run(backend.run_batched(requests))

        grade_records: List[GradeRecord] = []
        for response, result in zip(responses_list, llm_results):
            grade = self._parse_grade(result, grade_range)
            grade_records.append(GradeRecord(
                topic_id=response.metadata.topic_id,
                text=response.get_report_text(),
                grade=grade,
            ))

        qrels = build_qrels(records=grade_records, spec=BONSAI_QRELS_SPEC)
        print(f"BonsaiQrelsCreator: Created qrels for {len(grade_records)} responses")
        return qrels

    def _parse_grade(self, result: Any, grade_range: Tuple[int, int]) -> int:
        if not isinstance(result, MinimaLlmResponse):
            return grade_range[0]
        try:
            grade = int(result.text.strip())
            return max(grade_range[0], min(grade, grade_range[1]))
        except ValueError:
            return grade_range[0]


# =============================================================================
# BonsaiLeaderboardJudge
# =============================================================================

class BonsaiLeaderboardJudge:
    """Scores responses via LLM and produces a leaderboard."""

    def judge(
        self,
        rag_responses: Iterable[Report],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        qrels: Optional[Qrels] = None,
        on_missing_evals: str = "fix_aggregate",
        spacy_model: str = "en_core_web_lg",
        include_propn: bool = True,
        nproc: int = 1,
        batch_size: int = 128,
        filebase: str = "default",
        outdir: Path = Path("."),
        **kwargs: Any,
    ) -> Leaderboard:
        topic_titles: Dict[str, str] = {t.request_id: t.title or "" for t in rag_topics}
        expected_topic_ids: List[str] = list(topic_titles.keys())
        backend = _make_backend(llm_config)

        responses_list = list(rag_responses)

        # deterministic concept-F1 (no LLM) computed once for all reports
        cscores = concept_scores(responses_list, spacy_model=spacy_model,
                                 include_propn=include_propn, nproc=nproc, batch_size=batch_size)
        requests: List[MinimaLlmRequest] = []
        for i, response in enumerate(responses_list):
            query = topic_titles.get(response.metadata.topic_id, "")
            text = response.get_report_text()
            requests.append(MinimaLlmRequest(
                request_id=f"judge_{i}",
                messages=[
                    {"role": "system", "content": (
                        "You are a response quality evaluator. "
                        "Rate the response on two dimensions:\n"
                        "1. RELEVANCE: How relevant is the response to the query? (0.0-1.0)\n"
                        "2. COMPLETENESS: How completely does it address the query? (0.0-1.0)\n\n"
                        "Respond in exactly this format:\n"
                        "RELEVANCE: <score>\n"
                        "COMPLETENESS: <score>"
                    )},
                    {"role": "user", "content": f"Query: {query}\n\nResponse: {text}"},
                ],
                temperature=0.0,
            ))

        llm_results = asyncio.run(backend.run_batched(requests))

        builder = LeaderboardBuilder(BONSAI_SPEC)
        for response, result in zip(responses_list, llm_results):
            relevance, completeness = self._parse_scores(result)
            values: Dict[str, float] = {"RELEVANCE": relevance, "COMPLETENESS": completeness}
            prf = cscores.get((response.metadata.run_id, response.metadata.topic_id))
            if prf is not None:   # omit concept keys when either concept set was empty
                p, r, f1 = prf
                values["CONCEPT_PRECISION"] = p
                values["CONCEPT_RECALL"] = r
                values["CONCEPT_F1"] = f1
            builder.add(
                run_id=response.metadata.run_id,
                topic_id=response.metadata.topic_id,
                values=values,
            )

        leaderboard = builder.build(
            expected_topic_ids=expected_topic_ids,
            on_missing=on_missing_evals,
        )
        print(f"BonsaiLeaderboardJudge: Built leaderboard with {len(leaderboard.entries)} entries")
        return leaderboard

    def _parse_scores(self, result: Any) -> Tuple[float, float]:
        if not isinstance(result, MinimaLlmResponse):
            return (0.0, 0.0)
        relevance = 0.0
        completeness = 0.0
        for line in result.text.strip().splitlines():
            line = line.strip().upper()
            if line.startswith("RELEVANCE:"):
                try:
                    relevance = max(0.0, min(float(line.split(":", 1)[1].strip()), 1.0))
                except ValueError:
                    pass
            elif line.startswith("COMPLETENESS:"):
                try:
                    completeness = max(0.0, min(float(line.split(":", 1)[1].strip()), 1.0))
                except ValueError:
                    pass
        return (relevance, completeness)
