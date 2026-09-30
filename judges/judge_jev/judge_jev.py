#!/usr/bin/env python3
"""
JudgeJev: qrels come from JEV; the leaderboard judge scores runs against those qrels
the way trec_eval scores a run against human qrels -- a deterministic function of
(responses, qrels), no LLM calls.

Three modular classes (same protocol as BonsaiJudge):
- JevNuggetCreator: Generates nugget questions per topic (template placeholder, no LLM)
- JevQrelsCreator: For every document cited anywhere in rag_responses, asks JEV a Noul
  (calibrated yes/no) question -- "is this citation relevant to the topic?" -- and
  quantizes the probability into an integer qrels grade. This is the only stage that
  calls JEV.
- JevLeaderboardJudge: Treats each response's cited documents (in citation order) as a
  "run" the way trec_eval treats a ranked retrieval list, and scores it against the
  qrels: num_ret/num_rel/num_rel_ret, map (AP), recip_rank, P_k/R_k/ndcg_cut_k, bpref.
  Requires qrels -- it does not call JEV itself, exactly as trec_eval requires a qrels
  file and does not judge documents on its own.

JEV (https://openrouter.ai/docs/guides/community/jev) is a typed-decision model, not a
chat model: it takes a `state` plus typed `questions` and returns probabilities, so
JevQrelsCreator calls it directly via POST {jev_base_url}/v1/systemone rather than
through minima. Prompt design follows hevmind.com/writing/jev-as-a-reranker/ (one Noul
per document, true/false criteria, score = calibrated probability).
"""

import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type

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


# =============================================================================
# Specs
# =============================================================================

# Fixed cutoff for the @k measures below. It is baked into JEV_SPEC's measure NAMES
# (P_5, R_5, ndcg_cut_5), so it is a module constant rather than a judge_settings knob
# -- changing it would desync the spec's names from what judge() actually emits.
CUTOFF_K = 5
DEFAULT_RELEVANCE_THRESHOLD_GRADE = 3

JEV_SPEC = LeaderboardSpec(measures=(
    MeasureSpec("num_ret", description=(
        "Number of unique documents the response cites for this topic.")),
    MeasureSpec("num_rel", description=(
        "Number of documents judged relevant (qrels grade >= relevance_threshold_grade) "
        "across ALL documents JEV scored for this topic -- the full qrels universe for "
        "the topic, not just this response's citations.")),
    MeasureSpec("num_rel_ret", description=(
        "Number of this response's cited documents that are judged relevant.")),
    MeasureSpec("map", description=(
        "Average Precision of the response's citations, in citation order, against the "
        "qrels -- this topic's contribution to Mean Average Precision.")),
    MeasureSpec("recip_rank", description=(
        "Reciprocal rank of the first relevant citation (0.0 if none of the response's "
        "citations are judged relevant).")),
    MeasureSpec(f"P_{CUTOFF_K}", description=(
        f"Precision at rank {CUTOFF_K}: relevant citations among the first {CUTOFF_K}, "
        f"divided by {CUTOFF_K} (trec_eval convention -- fewer than {CUTOFF_K} citations "
        f"still divides by {CUTOFF_K}).")),
    MeasureSpec(f"R_{CUTOFF_K}", description=(
        f"Recall at rank {CUTOFF_K}: fraction of all relevant documents for the topic "
        f"found within the response's first {CUTOFF_K} citations.")),
    MeasureSpec(f"ndcg_cut_{CUTOFF_K}", description=(
        f"nDCG at rank {CUTOFF_K}: graded gain (2^grade - 1) discounted by rank, "
        f"normalized against the ideal ranking of every judged document for the topic.")),
    MeasureSpec("bpref", description=(
        "Binary preference: like map but robust to incomplete judgments -- only counts "
        "citations that were actually JEV-scored (present in qrels), so a citation JEV "
        "never got to score does not count against the response.")),
))


class GradeRecord:
    """Record for qrels building. One per unique cited document, not per response."""
    def __init__(self, topic_id: str, doc_id: str, grade: int):
        self.topic_id = topic_id
        self.doc_id = doc_id
        self.grade = grade


JEV_QRELS_SPEC = QrelsSpec[GradeRecord](
    topic_id=lambda r: r.topic_id,
    doc_id=lambda r: r.doc_id,
    grade=lambda r: r.grade,
    on_duplicate="keep_max",
)


# =============================================================================
# JEV client (Noul over /v1/systemone) -- used only by JevQrelsCreator
# =============================================================================

NOUL_QUESTION = "citation_relevant"

NOUL_INSTRUCTIONS = (
    "The cited document `document` is relevant to the `query`: it contains information "
    "that answers or directly addresses what the query asks about."
)
# Criteria are folded into the instructions: the documented Noul shape is
# {"type": "noul", "instructions": ...}.
NOUL_CRITERIA = (
    " True: the document contains information that answers the query or directly "
    "addresses what it asks about. False: the document is only loosely related, on a "
    "similar topic, or does not address what the query asks."
)

Transport = Callable[[str, Dict[str, Any], Dict[str, str], float], Dict[str, Any]]


def _http_post(url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: float) -> Dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class JevClient:
    """Minimal JEV System One client with retries and an on-disk probability cache."""

    def __init__(
        self,
        api_key: Optional[str],
        model: str = "typesafe/jev-1.13",
        base_url: str = "https://openrouter.ai/api",
        cache_dir: Optional[Path] = None,
        timeout_s: float = 60.0,
        max_retries: int = 4,
        concurrency: int = 16,
        transport: Optional[Transport] = None,
    ):
        self.model = model
        self.url = base_url.rstrip("/") + "/v1/systemone"
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.concurrency = concurrency
        self._post = transport or _http_post
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, query: str, document: str) -> str:
        blob = json.dumps([self.model, NOUL_INSTRUCTIONS + NOUL_CRITERIA, query, document])
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> Optional[float]:
        if self.cache_dir:
            p = self.cache_dir / f"{key}.json"
            if p.exists():
                try:
                    return float(json.loads(p.read_text())["noul"])
                except (ValueError, KeyError, OSError):
                    return None
        return None

    def _cache_put(self, key: str, p: float) -> None:
        if self.cache_dir:
            (self.cache_dir / f"{key}.json").write_text(json.dumps({"noul": p}))

    @staticmethod
    def parse(resp: Dict[str, Any]) -> float:
        ans = resp["answers"][NOUL_QUESTION]
        return max(0.0, min(float(ans["noul"]), 1.0))

    def score(self, query: str, document: str) -> Optional[float]:
        """P(document relevant to query), or None if the call failed after retries."""
        key = self._key(query, document)
        hit = self._cache_get(key)
        if hit is not None:
            return hit
        payload = {
            "model": self.model,
            "state": {"query": query, "document": document},
            "questions": {NOUL_QUESTION: {
                "type": "noul",
                "instructions": NOUL_INSTRUCTIONS + NOUL_CRITERIA,
            }},
        }
        for attempt in range(self.max_retries + 1):
            try:
                p = self.parse(self._post(self.url, payload, self.headers, self.timeout_s))
                self._cache_put(key, p)
                return p
            except urllib.error.HTTPError as e:
                if e.code not in (408, 429) and e.code < 500:
                    print(f"JevClient: HTTP {e.code} (not retrying): {e.read()[:300]!r}")
                    return None
            except (urllib.error.URLError, TimeoutError, KeyError, ValueError, TypeError):
                pass
            time.sleep(min(2 ** attempt, 30))
        return None

    def score_many(self, pairs: Sequence[Tuple[str, str]]) -> List[Optional[float]]:
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(pool.map(lambda qd: self.score(*qd), pairs))


def _get_api_key(llm_config: LlmConfigProtocol) -> Optional[str]:
    return getattr(llm_config, "api_key", None) or (llm_config.raw or {}).get("api_key")


def _make_jev_client(
    llm_config: LlmConfigProtocol,
    jev_model: str,
    jev_base_url: str,
    cache_dir: Optional[str],
    timeout_s: float,
    max_retries: int,
    concurrency: int,
) -> JevClient:
    return JevClient(
        api_key=_get_api_key(llm_config),
        model=jev_model,
        base_url=jev_base_url,
        cache_dir=Path(cache_dir) if cache_dir else Path(__file__).parent / ".cache",
        timeout_s=timeout_s,
        max_retries=max_retries,
        concurrency=concurrency,
    )


def _collect_cited_documents(
    responses_list: Sequence[Report],
    topic_titles: Dict[str, str],
    max_doc_chars: int,
) -> Tuple[Dict[Tuple[str, str], int], List[Tuple[str, str]], List[str]]:
    """Unique (topic_id, shard) cited-document pairs across all reports, deduplicated
    so each is scored by JEV once regardless of how many responses cite it.

    Returns (pair_index, pairs, doc_ids):
      pair_index[(topic_id, shard)] -> index into `pairs` / `doc_ids`
      pairs[i]   = (query, truncated_document_text) -- what's sent to JEV
      doc_ids[i] = doc_id_md5(full_document_text) -- stable identity, independent of
                   max_doc_chars, so it matches what JevLeaderboardJudge computes.
    """
    pair_index: Dict[Tuple[str, str], int] = {}
    pairs: List[Tuple[str, str]] = []
    doc_ids: List[str] = []
    for response in responses_list:
        topic_id = response.metadata.topic_id
        docs = response.documents or {}
        seen = set()
        for sent in response.get_sentences_with_citations():
            for shard in (sent.citations or []):
                if shard in seen:
                    continue
                seen.add(shard)
                doc = docs.get(shard)
                if doc is None:
                    continue
                key = (topic_id, shard)
                if key not in pair_index:
                    full_text = doc.get_document_text() or ""
                    pair_index[key] = len(pairs)
                    pairs.append((topic_titles.get(topic_id, ""), full_text[:max_doc_chars]))
                    doc_ids.append(doc_id_md5(full_text))
    return pair_index, pairs, doc_ids


def _collect_cited_doc_ids(responses_list: Sequence[Report]) -> List[List[str]]:
    """For each response (same order as responses_list), the doc_ids of documents it
    cites, in first-citation order, deduplicated. doc_id = doc_id_md5(full document
    text) -- the same identity JevQrelsCreator uses, independent of any truncation
    applied before sending text to JEV."""
    result: List[List[str]] = []
    for response in responses_list:
        docs = response.documents or {}
        seen: set = set()
        ranked: List[str] = []
        for sent in response.get_sentences_with_citations():
            for shard in (sent.citations or []):
                doc = docs.get(shard)
                if doc is None:
                    continue
                doc_id = doc_id_md5(doc.get_document_text() or "")
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                ranked.append(doc_id)
        result.append(ranked)
    return result


def _quantize(p: float, grade_range: Tuple[int, int]) -> int:
    """Linearly map a JEV probability in [0,1] onto an integer grade in grade_range."""
    lo, hi = grade_range
    return max(lo, min(round(lo + p * (hi - lo)), hi))


def _write_qrels_scores(
    pair_index: Dict[Tuple[str, str], int],
    pairs: List[Tuple[str, str]],
    doc_ids: List[str],
    probs: List[Optional[float]],
    grade_range: Tuple[int, int],
    jev_model: str,
    outdir: Path,
    filebase: str,
) -> Path:
    """Sidecar JSONL, one row per unique cited document, recording the raw JEV
    probability alongside the quantized qrels grade -- everything build_qrels()
    collapses away. Row: topic_id, shard, doc_id (matches the qrels doc id),
    jev_probability (null if the call failed), grade (null if failed/omitted)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{Path(filebase).name}.qrels_scores.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for (topic_id, shard), i in pair_index.items():
            p = probs[i]
            row = {
                "topic_id": topic_id,
                "shard": shard,
                "doc_id": doc_ids[i],
                "jev_model": jev_model,
                "jev_probability": p,
                "grade": _quantize(p, grade_range) if p is not None else None,
            }
            f.write(json.dumps(row) + "\n")
    return path


def _qrels_by_topic(qrels: Qrels) -> Dict[str, Dict[str, int]]:
    """{topic_id: {doc_id: grade}} -- the full judged universe per topic, needed for
    num_rel / recall / nDCG's ideal ranking, exactly as trec_eval needs the whole
    qrels file rather than just what one run retrieved."""
    by_topic: Dict[str, Dict[str, int]] = {}
    for row in qrels.rows:
        by_topic.setdefault(row.topic_id, {})[row.doc_id] = row.grade
    return by_topic


def _dcg(grades: Sequence[int], k: int) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades[:k]))


def _trec_eval_measures(
    ranked_doc_ids: List[str],
    grades: Dict[str, int],
    threshold: int,
) -> Dict[str, float]:
    """trec_eval-style measures for one response's cited-document ranking against that
    topic's qrels. `ranked_doc_ids` is this response's citations in citation order
    (the "run"); `grades` is {doc_id: grade} for the WHOLE topic (every document JEV
    scored for it), used as ground truth the same way trec_eval uses a qrels file.
    Docs cited but never judged (JEV failed, or omitted) are treated as grade 0 for
    gain/relevance purposes, but as UNJUDGED (skipped, not "non-relevant") for bpref.
    """
    def is_rel(doc_id: str) -> bool:
        return grades.get(doc_id, 0) >= threshold

    all_grades = list(grades.values())
    R = sum(1 for g in all_grades if g >= threshold)
    N = sum(1 for g in all_grades if g < threshold)
    ideal_desc = sorted(all_grades, reverse=True)

    rel_flags = [is_rel(d) for d in ranked_doc_ids]
    judged_flags = [d in grades for d in ranked_doc_ids]
    doc_grades = [grades.get(d, 0) for d in ranked_doc_ids]

    # ---- map (AP) ----
    if R == 0:
        ap = 0.0
    else:
        hits = 0
        total = 0.0
        for i, rel in enumerate(rel_flags, start=1):
            if rel:
                hits += 1
                total += hits / i
        ap = total / R

    # ---- recip_rank ----
    rr = 0.0
    for i, rel in enumerate(rel_flags, start=1):
        if rel:
            rr = 1.0 / i
            break

    # ---- P_k / R_k (trec_eval fixed-denominator convention for P_k) ----
    top_k_rel = rel_flags[:CUTOFF_K]
    p_at_k = sum(top_k_rel) / CUTOFF_K
    r_at_k = (sum(top_k_rel) / R) if R > 0 else 0.0

    # ---- ndcg_cut_k ----
    dcg = _dcg(doc_grades, CUTOFF_K)
    idcg = _dcg(ideal_desc, CUTOFF_K)
    ndcg_at_k = (dcg / idcg) if idcg > 0 else 0.0

    # ---- bpref ----
    if R == 0:
        bpref = 0.0
    else:
        denom = min(R, N)
        n_above = 0
        total = 0.0
        for judged, rel in zip(judged_flags, rel_flags):
            if not judged:
                continue   # unjudged citations don't count toward n_above or the sum
            if rel:
                total += 1.0 - (n_above / denom if denom else 0.0)
            else:
                n_above += 1
        bpref = total / R

    return {
        "num_ret": float(len(ranked_doc_ids)),
        "num_rel": float(R),
        "num_rel_ret": float(sum(rel_flags)),
        "map": ap,
        "recip_rank": rr,
        f"P_{CUTOFF_K}": p_at_k,
        f"R_{CUTOFF_K}": r_at_k,
        f"ndcg_cut_{CUTOFF_K}": ndcg_at_k,
        "bpref": bpref,
    }


# =============================================================================
# JevNuggetCreator
# =============================================================================

class JevNuggetCreator:
    """Creates nugget questions for each topic."""

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
        print(f"JevNuggetCreator: Created nuggets for {len(banks)} topics")
        return nugget_banks


# =============================================================================
# JevQrelsCreator
# =============================================================================

class JevQrelsCreator:
    """Creates document-level relevance judgments from JEV citation-relevance calls.

    Every document cited anywhere in rag_responses is asked the same Noul question
    ("is this citation relevant to the topic?"); its probability is linearly quantized
    into grade_range. One qrels row per unique cited document (not per response) --
    documents cited by multiple responses are scored once. This is the only class in
    the module that calls JEV.
    """

    def create_qrels(
        self,
        rag_responses: Iterable[Report],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        grade_range: Tuple[int, int] = (0, 3),
        jev_model: str = "typesafe/jev-1.13",
        jev_base_url: str = "https://openrouter.ai/api",
        max_doc_chars: int = 24000,
        concurrency: int = 16,
        timeout_s: float = 60.0,
        max_retries: int = 4,
        cache_dir: Optional[str] = None,
        filebase: str = "default",
        outdir: Path = Path("."),
        jev_client: Optional[JevClient] = None,   # injectable for tests
        **kwargs: Any,
    ) -> Optional[Qrels]:
        topic_titles: Dict[str, str] = {t.request_id: t.title or "" for t in rag_topics}
        responses_list = sorted(rag_responses, key=lambda r: (r.metadata.run_id, r.metadata.topic_id))

        pair_index, pairs, doc_ids = _collect_cited_documents(responses_list, topic_titles, max_doc_chars)
        if jev_client is None:
            jev_client = _make_jev_client(llm_config, jev_model, jev_base_url, cache_dir,
                                          timeout_s, max_retries, concurrency)

        print(f"JevQrelsCreator: scoring {len(pairs)} unique cited documents with {jev_client.model}")
        probs = jev_client.score_many(pairs)
        failed = sum(p is None for p in probs)
        if pairs and failed == len(pairs):
            raise RuntimeError("JevQrelsCreator: every JEV call failed; check API key / jev_base_url / model")
        if failed:
            print(f"JevQrelsCreator: WARNING {failed}/{len(pairs)} JEV calls failed; those documents are omitted")

        scores_path = _write_qrels_scores(pair_index, pairs, doc_ids, probs, grade_range,
                                          jev_client.model, outdir, filebase)
        print(f"JevQrelsCreator: wrote per-document JEV probabilities to {scores_path}")

        grade_records: List[GradeRecord] = []
        for (topic_id, _shard), i in pair_index.items():
            p = probs[i]
            if p is None:
                continue
            grade_records.append(GradeRecord(topic_id=topic_id, doc_id=doc_ids[i], grade=_quantize(p, grade_range)))

        qrels = build_qrels(records=grade_records, spec=JEV_QRELS_SPEC)
        print(f"JevQrelsCreator: Created qrels for {len(grade_records)} unique cited documents")
        return qrels


# =============================================================================
# JevLeaderboardJudge
# =============================================================================

class JevLeaderboardJudge:
    """Scores each response's citations against qrels, trec_eval-style.

    No JEV calls, no LLM calls: it is a deterministic function of (responses, qrels).
    Requires qrels -- pass a Qrels built by JevQrelsCreator (judge_uses_qrels: true in
    the same workflow run), or load a previously created one via --qrels-input.
    """

    def judge(
        self,
        rag_responses: Iterable[Report],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        qrels: Optional[Qrels] = None,
        on_missing_evals: str = "fix_aggregate",
        relevance_threshold_grade: int = DEFAULT_RELEVANCE_THRESHOLD_GRADE,
        filebase: str = "default",
        outdir: Path = Path("."),
        **kwargs: Any,
    ) -> Leaderboard:
        if qrels is None:
            raise RuntimeError(
                "JevLeaderboardJudge: qrels are required -- this judge scores each response's "
                "citations against JEV-derived qrels the way trec_eval scores a run against "
                "qrels; it makes no JEV calls itself. Run JevQrelsCreator first (create_qrels: "
                "true, judge_uses_qrels: true), or pass --qrels-input pointing at a previously "
                "created qrels file."
            )

        topic_titles: Dict[str, str] = {t.request_id: t.title or "" for t in rag_topics}
        expected_topic_ids: List[str] = list(topic_titles.keys())

        # Sort for deterministic output order.
        responses_list = sorted(rag_responses, key=lambda r: (r.metadata.run_id, r.metadata.topic_id))
        report_doc_ids = _collect_cited_doc_ids(responses_list)
        topic_grades = _qrels_by_topic(qrels)

        builder = LeaderboardBuilder(JEV_SPEC)
        for response, ranked in zip(responses_list, report_doc_ids):
            topic_id = response.metadata.topic_id
            values = _trec_eval_measures(ranked, topic_grades.get(topic_id, {}), relevance_threshold_grade)
            builder.add(run_id=response.metadata.run_id, topic_id=topic_id, values=values)

        leaderboard = builder.build(
            expected_topic_ids=expected_topic_ids,
            on_missing=on_missing_evals,
        )
        print(f"JevLeaderboardJudge: Built leaderboard with {len(leaderboard.entries)} entries "
              f"(trec_eval-style measures vs. qrels, cutoff k={CUTOFF_K}, "
              f"relevance_threshold_grade={relevance_threshold_grade})")
        return leaderboard
