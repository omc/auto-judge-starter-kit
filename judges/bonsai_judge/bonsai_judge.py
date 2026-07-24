#!/usr/bin/env python3
"""
BonsaiJudge: Full-protocol LLM judge using OpenRouter-compatible API.

Three modular classes:
- BonsaiNuggetCreator: Generates nugget questions per topic via LLM
- BonsaiQrelsCreator: Assigns relevance grades via LLM
- BonsaiLeaderboardJudge: Scores responses and produces leaderboard
"""

import asyncio
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
# Specs
# =============================================================================

BONSAI_SPEC = LeaderboardSpec(measures=(
    MeasureSpec("RELEVANCE", description="LLM-judged relevance score (0.0-1.0)"),
    MeasureSpec("COMPLETENESS", description="How completely the response addresses the query (0.0-1.0)"),
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
        filebase: str = "default",
        outdir: Path = Path("."),
        **kwargs: Any,
    ) -> Leaderboard:
        topic_titles: Dict[str, str] = {t.request_id: t.title or "" for t in rag_topics}
        expected_topic_ids: List[str] = list(topic_titles.keys())
        backend = _make_backend(llm_config)

        responses_list = list(rag_responses)
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
            builder.add(
                run_id=response.metadata.run_id,
                topic_id=response.metadata.topic_id,
                values={"RELEVANCE": relevance, "COMPLETENESS": completeness},
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
