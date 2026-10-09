#!/usr/bin/env python3
"""Bag-of-decisions (BOD) graded judge: per-topic rubric questions, graded by Jev.

After softwaredoug.com/blog/2026/10/08/bag-of-decisions (yes/no criteria summed as
P(yes)), changed to a graded scale and a stored, re-tallied grade table:

1. BodQuestionCreator (nugget creator): a chat LLM writes 5-15 "Does the answer ...?"
   questions per topic from the TOPIC TEXT ONLY (never the responses), sized to how
   many distinct parts the need has; near-duplicates are dropped. Saved as the
   run's nugget banks and reused by every variant (force_recreate_nuggets: false).
2. BodJevJudge: ONE Jev Decisions request per (run, topic): state = topic fields +
   the answer; questions = one 4-option Choice per rubric question (grades 0-3,
   anchored) plus a Noul padding question. Jev bills the input (state), so the
   number of questions barely changes cost.
3. Every graded question is written to <filebase>.bod/grades.jsonl; the leaderboard
   is tallied from those rows by tally() with no further calls.

Score per question = expected grade sum_g P(g)*g / 3 (probabilities, not argmax).
BOD_MEAN (primary) = mean over the topic's questions; leaderboard 'all' rows are the
mean over topics, so topics with more questions do not weigh more. BOD_PADDING is a
separate diagnostic and is NOT part of BOD_MEAN.

Caching: Jev answers share the minima PromptCache with the pairwise Jev judge (key =
sha256 over endpoint, model, state, questions); question generation uses minima's
normal chat cache. Reruns cost nothing.
"""
from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import re
import time
from collections import defaultdict
from dataclasses import replace
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
    NuggetBanks,
    NuggetBanksProtocol,
)
from autojudge_base.nugget_data import NuggetBank, NuggetQuestion
from minima_llm import MinimaLlmConfig, MinimaLlmRequest, OpenAIMinimaLlm

from .pairwise import (
    DEFAULT_CACHE_DIR,
    _add_empty_rows,
    _empty_reports,
    _render,
    _summaries_by_topic,
    select_topics,
    topic_fields,
    topic_query_text,
)
from .pairwise_jev import JEV_MODEL, BonsaiJevPairwiseJudge, decisions_url

_PROMPT_PATH = Path(__file__).parent / "prompts" / "bod_questions.md"

# Grade anchors (kiddie smoke test, temp/bod_smoke.py: listing them 3..0 changed
# expected grades by < 0.03). Editing any text re-asks every Jev call.
SCALE: Dict[str, str] = {
    "0": "The answer does not address this question at all.",
    "1": "The answer mentions this topic but gives no real substance or explanation.",
    "2": "The answer partially answers this question; important parts are missing or vague.",
    "3": "The answer fully answers this question with specific, accurate detail.",
}
MAX_GRADE = 3
PADDING_ID = "padding"
PADDING_QUESTION = ("Does the answer contain substantial content that is off-topic, repeated, "
                    "or generic filler that does not help with the user's question?")

BOD_SPEC = LeaderboardSpec(measures=(
    MeasureSpec("BOD_MEAN", description=(
        "Primary. Mean over the topic's rubric questions of Jev's expected grade "
        "sum_g P(g)*g on the 0-3 scale, divided by 3 (0 = nothing addressed, 1 = every "
        "question fully answered).")),
    MeasureSpec("BOD_COVERAGE", description=(
        "Fraction of the topic's rubric questions with P(grade >= 2) > 0.5, i.e. at "
        "least partially answered.")),
    MeasureSpec("BOD_FULL", description=(
        "Fraction of the topic's rubric questions with P(grade = 3) > 0.5, i.e. fully "
        "answered.")),
    MeasureSpec("BOD_PADDING", description=(
        "Jev's P(yes) that the answer contains substantial off-topic, repeated or "
        "generic filler. Diagnostic; higher is worse; not part of BOD_MEAN.")),
    MeasureSpec("BOD_QUESTIONS", int, description=(
        "Number of rubric questions for the topic (0 for an empty report).")),
))


# =============================================================================
# 1. Question generation
# =============================================================================

def _tokens(text: str) -> frozenset:
    return frozenset(re.findall(r"[a-z0-9]+", text.lower()))


def dedupe_questions(questions: Sequence[str], jaccard: float = 0.8) -> List[str]:
    """Drop exact and near-duplicate questions (token Jaccard >= threshold), keeping
    the first occurrence and the original order."""
    kept: List[Tuple[str, frozenset]] = []
    for q in questions:
        q = " ".join(str(q).split())
        tok = _tokens(q)
        if not q or not tok:
            continue
        if any(len(tok & t) / len(tok | t) >= jaccard for _, t in kept):
            continue
        kept.append((q, tok))
    return [q for q, _ in kept]


def parse_questions(text: str) -> List[str]:
    """JSON array of strings from the model's reply (tolerates ```json fences)."""
    m = re.search(r"\[.*\]", text or "", re.S)
    if not m:
        raise ValueError("no JSON array in the generator's reply")
    data = json.loads(m.group(0))
    if not isinstance(data, list):
        raise ValueError("generator reply is not a JSON array")
    return [d for d in data if isinstance(d, str)]


def _chat_backend(llm_config: LlmConfigProtocol, model: Optional[str]) -> OpenAIMinimaLlm:
    """Endpoint + key from llm_config; ``model`` overrides OPENAI_MODEL (None = env).
    A ':batch' suffix is batch-only on OpenRouter, so it is stripped for realtime use."""
    cfg = MinimaLlmConfig.from_dict(llm_config.raw) if llm_config.raw else MinimaLlmConfig.from_env()
    if not cfg.cache_dir:
        cfg = replace(cfg, cache_dir=DEFAULT_CACHE_DIR)
    cfg = replace(cfg, model=(model or cfg.model).replace(":batch", ""))
    Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)
    return OpenAIMinimaLlm(cfg)


class BodQuestionCreator:
    """Rubric questions per topic, from the topic text only."""

    nugget_banks_type: Type[NuggetBanksProtocol] = NuggetBanks

    def create_nuggets(
        self,
        rag_responses: Optional[Iterable[Report]],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        generator_model: Optional[str] = None,
        min_questions: int = 5,
        max_questions: int = 15,
        dedupe_jaccard: float = 0.8,
        temperature: float = 0.0,
        max_tokens: int = 2000,
        dev_topics: Optional[Sequence[str]] = None,
        max_topics: Optional[int] = None,
        filebase: str = "default",
        outdir: Path = Path("."),
        **kwargs: Any,
    ) -> Optional[NuggetBanksProtocol]:
        # Responses are deliberately ignored (nugget_depends_on_responses: false): the
        # rubric must not depend on what any run wrote.
        topics = {t.request_id: t for t in rag_topics}
        order = select_topics(rag_topics, topics, dev_topics, max_topics)
        template = _PROMPT_PATH.read_text()
        reqs = [MinimaLlmRequest(
            request_id=tid,
            messages=[{"role": "user", "content": _render(
                template, topic_query=topic_query_text(topic_fields(topics[tid])),
                min_questions=str(min_questions), max_questions=str(max_questions))}],
            temperature=temperature, max_tokens=max_tokens) for tid in order]

        backend = _chat_backend(llm_config, generator_model)

        async def run():
            try:
                return await backend.run_batched(reqs)
            finally:
                await backend.aclose()
        results = asyncio.run(run())

        banks: List[NuggetBank] = []
        failed: List[str] = []
        sizes: List[int] = []
        for tid, res in zip(order, results):
            text = getattr(res, "text", None)
            try:
                qs = dedupe_questions(parse_questions(text), dedupe_jaccard)[:max_questions]
            except (ValueError, json.JSONDecodeError) as e:
                print(f"[bod] {tid}: unusable generator reply ({type(e).__name__}: {e})")
                qs = []
            if not qs:
                failed.append(tid)
                continue
            if len(qs) < min_questions:
                print(f"[bod] {tid}: only {len(qs)} questions after de-duplication "
                      f"(min {min_questions}); kept")
            bank = NuggetBank(query_id=tid, title_query=topics[tid].title or tid)
            bank.add_nuggets([NuggetQuestion.from_lazy(query_id=tid, question=q,
                                                       question_id=f"q{i:02d}")
                              for i, q in enumerate(qs, 1)])
            banks.append(bank)
            sizes.append(len(qs))
        if failed:
            raise RuntimeError(f"[bod] question generation failed for topics {failed}; "
                               f"rerun (successful replies are cached)")
        print(f"[bod] rubric questions for {len(banks)} topics: "
              f"min {min(sizes)}, max {max(sizes)}, mean {sum(sizes) / len(sizes):.1f}")
        return NuggetBanks.from_banks_list(banks)


def bank_questions(nugget_banks: Optional[NuggetBanksProtocol], topic_id: str) -> List[str]:
    """Rubric question texts for a topic, in question_id order."""
    if not nugget_banks or topic_id not in nugget_banks.banks:
        return []
    nuggets = nugget_banks.banks[topic_id].nuggets_as_list()
    return [n.question for n in sorted(nuggets, key=lambda n: n.question_id or "")]


# =============================================================================
# 2-3. Judging and tally
# =============================================================================

def jev_questions(rubric: Sequence[str], padding: bool = True) -> Dict[str, dict]:
    qs = {f"q{i:02d}": {"type": "choice",
                        "instructions": f"Grade the answer on this question: {q}",
                        "criteria": dict(SCALE)}
          for i, q in enumerate(rubric, 1)}
    if padding:
        qs[PADDING_ID] = {"type": "noul", "instructions": PADDING_QUESTION}
    return qs


def grade_probs(answer: Optional[dict]) -> Optional[Dict[int, float]]:
    """{grade: P} renormalised over 0..MAX_GRADE; None if unusable."""
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        return None
    raw = answer.get("probabilities") or {}
    try:
        p = {g: float(raw[str(g)]) for g in range(MAX_GRADE + 1)}
    except (KeyError, TypeError, ValueError):
        return None
    total = sum(p.values())
    if total <= 0 or any(v < 0 for v in p.values()):
        return None
    return {g: v / total for g, v in p.items()}


def _noul(answer: Optional[dict]) -> Optional[float]:
    if not isinstance(answer, dict) or answer.get("type") != "noul":
        return None
    try:
        v = float(answer["noul"])
    except (KeyError, TypeError, ValueError):
        return None
    return v if 0.0 <= v <= 1.0 else None


def _valid_answer(answer: Optional[dict], qid: str) -> bool:
    return (_noul(answer) if qid == PADDING_ID else grade_probs(answer)) is not None


def grade_rows(topic_id: str, run_id: str, team_id: str, rubric: Sequence[str],
               answers: dict, key: str) -> List[dict]:
    """One row per rubric question (+ the padding row) from a Jev response."""
    rows = []
    for i, q in enumerate(rubric, 1):
        qid = f"q{i:02d}"
        a = answers.get(qid)
        p = grade_probs(a)
        rows.append({
            "topic_id": topic_id, "run_id": run_id, "team_id": team_id, "qid": qid,
            "question": q,
            "probs": None if p is None else {str(g): round(v, 6) for g, v in p.items()},
            "expected": None if p is None else sum(g * v for g, v in p.items()),
            "argmax": (a or {}).get("choice"), "confidence": (a or {}).get("confidence"),
            "cache_key": key,
        })
    if PADDING_ID in answers:
        rows.append({"topic_id": topic_id, "run_id": run_id, "team_id": team_id,
                     "qid": PADDING_ID, "question": PADDING_QUESTION,
                     "p_yes": _noul(answers[PADDING_ID]), "cache_key": key})
    return rows


def tally(rows: Iterable[dict], expected_topic_ids: Sequence[str],
          empties: Iterable[Tuple[str, str]] = (), on_missing: str = "default",
          threshold: float = 0.5) -> Leaderboard:
    """Leaderboard from stored grade rows (no LLM calls). A (run, topic) is scored
    only if every one of its rubric rows is valid."""
    by_rt: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    pad: Dict[Tuple[str, str], float] = {}
    for r in rows:
        k = (r["run_id"], r["topic_id"])
        if r["qid"] == PADDING_ID:
            if r.get("p_yes") is not None:
                pad[k] = r["p_yes"]
        else:
            by_rt[k].append(r)
    builder = LeaderboardBuilder(BOD_SPEC)
    scored = set()
    for (run, topic), rs in by_rt.items():
        if not rs or any(r.get("probs") is None for r in rs):
            continue
        n = len(rs)
        builder.add(run_id=run, topic_id=topic, values={
            "BOD_MEAN": sum(r["expected"] for r in rs) / (n * MAX_GRADE),
            "BOD_COVERAGE": sum((r["probs"]["2"] + r["probs"]["3"]) > threshold for r in rs) / n,
            "BOD_FULL": sum(r["probs"]["3"] > threshold for r in rs) / n,
            "BOD_PADDING": pad.get((run, topic), 0.0),
            "BOD_QUESTIONS": n,
        })
        scored.add((run, topic))
    _add_empty_rows(builder, BOD_SPEC, empties, scored)
    return builder.build(expected_topic_ids=list(expected_topic_ids), on_missing=on_missing)


class BodJevJudge:
    """Grades each (run, topic) answer on the topic's rubric with Jev."""

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
        filebase: str = "default",
        outdir: Path = Path("."),
        **kwargs: Any,
    ) -> Leaderboard:
        art = Path(outdir) / f"{Path(filebase).name}.bod"
        art.mkdir(parents=True, exist_ok=True)

        topic_info = {t.request_id: topic_fields(t) for t in rag_topics}
        expected_topic_ids = list(topic_info)
        reports = list(rag_responses)
        answers = _summaries_by_topic(reports)          # topic -> run -> (team, text)
        empties = _empty_reports(reports)
        if max_runs is not None:
            keep = set(sorted({r for tr in answers.values() for r in tr})[:max_runs])
            answers = {t: {r: v for r, v in tr.items() if r in keep} for t, tr in answers.items()}
            empties = [(t, r) for t, r in empties if r in keep]
        topic_order = select_topics(rag_topics, set(answers) | {t for t, _ in empties},
                                    dev_topics, max_topics)
        if dev_topics is not None or max_topics is not None:
            keep_t = set(topic_order)
            empties = [(t, r) for t, r in empties if t in keep_t]
            expected_topic_ids = [t for t in expected_topic_ids if t in keep_t]
        topic_order = [t for t in topic_order if t in answers]

        rubrics = {t: bank_questions(nugget_banks, t) for t in topic_order}
        missing = [t for t, qs in rubrics.items() if not qs]
        if missing:
            raise ValueError(f"[bod] no rubric questions for topics {missing}; "
                             f"run with create_nuggets: true")

        helper = BonsaiJevPairwiseJudge()
        backend, cfg = helper._make_jev_backend(llm_config, model)
        url = decisions_url(cfg.base_url)
        rows, stats = asyncio.run(self._run(helper, backend, cfg.model, url, topic_order,
                                            answers, topic_info, rubrics, padding,
                                            max_outstanding, max_attempts, art))
        lb = tally(rows, expected_topic_ids, empties, on_missing_evals, threshold)
        self._write_artifacts(rows, stats, art, cfg, url, topic_order, rubrics, padding)
        return lb

    async def _run(self, helper, backend, model, url, topic_order, answers, topic_info,
                   rubrics, padding, max_outstanding, max_attempts, art):
        cache = backend._ensure_cache()
        if cache is None:
            raise RuntimeError("BOD judge requires a cache_dir")
        items = []
        for t in topic_order:
            qs = jev_questions(rubrics[t], padding)
            for run in sorted(answers[t]):                    # deterministic order
                team, text = answers[t][run]
                payload = {"model": model, "state": {**topic_info[t], "answer": text},
                           "questions": qs}
                items.append((t, run, team, payload, helper._cache_key(payload)))
        todo = {k: p for *_x, p, k in items if cache.get(k) is None}
        print(f"[bod] model={model} {len(topic_order)} topics, {len(items):,} answers, "
              f"{len(todo):,} new Jev calls")
        sem = asyncio.Semaphore(max_outstanding)
        errors: Dict[str, int] = defaultdict(int)
        t0 = time.time()

        async def one(key, payload):
            async with sem:
                data, err = await helper._call(backend, url, payload, max_attempts,
                                               check=_valid_answer)
            if data is not None:
                cache.put(key, json.dumps(data["answers"]), data)
            else:
                errors[err.split(":")[0]] += 1

        await asyncio.gather(*(one(k, p) for k, p in todo.items()))

        rows: List[dict] = []
        cost = tokens = 0.0
        n_ok = 0
        served = set()
        for t, run, team, _p, key in items:
            hit = cache.get(key)
            raw = hit[1] if hit else None
            if not isinstance(raw, dict):
                continue
            n_ok += 1
            usage = raw.get("usage") or {}
            cost += usage.get("cost") or 0.0
            tokens += usage.get("input_tokens") or 0
            if raw.get("model"):
                served.add(raw["model"])
            rows.extend(grade_rows(t, run, team, rubrics[t], raw.get("answers") or {}, key))
        await backend.aclose()                 # closes the cache too: read it first
        if errors:
            print(f"[bod] call errors (left uncached, retried on rerun): {dict(errors)}")
        stats = {"n_answers": len(items), "n_valid": n_ok, "n_missing": len(items) - n_ok,
                 "new_calls": len(todo), "errors": dict(errors), "cost_usd": cost,
                 "input_tokens": int(tokens), "served_models": sorted(served),
                 "minutes": round((time.time() - t0) / 60, 2)}
        print(f"[bod] {n_ok:,}/{len(items):,} answers graded "
              f"({stats['minutes']}m, ${cost:.4f} incl. cached)")
        return rows, stats

    @staticmethod
    def _write_artifacts(rows, stats, art, cfg, url, topic_order, rubrics, padding):
        with open(art / "grades.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        # overall table: mean over topics of BOD_MEAN / BOD_PADDING per run
        per: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
        pad: Dict[Tuple[str, str], float] = {}
        team: Dict[str, str] = {}
        for r in rows:
            team[r["run_id"]] = r["team_id"]
            if r["qid"] == PADDING_ID:
                pad[(r["run_id"], r["topic_id"])] = r.get("p_yes")
            elif r.get("expected") is not None:
                per[(r["run_id"], r["topic_id"])].append(r)
        run_scores: Dict[str, List[float]] = defaultdict(list)
        run_pad: Dict[str, List[float]] = defaultdict(list)
        for (run, t), rs in per.items():
            run_scores[run].append(sum(r["expected"] for r in rs) / (len(rs) * MAX_GRADE))
            if pad.get((run, t)) is not None:
                run_pad[run].append(pad[(run, t)])
        table = sorted(((run, sum(v) / len(v), len(v),
                         sum(run_pad[run]) / len(run_pad[run]) if run_pad[run] else None)
                        for run, v in run_scores.items()), key=lambda x: -x[1])
        with open(art / "leaderboard_overall.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rank", "run_id", "team_id", "bod_mean", "topics", "bod_padding"])
            for i, (run, m, n, p) in enumerate(table, 1):
                w.writerow([i, run, team.get(run, ""), f"{m:.4f}", n,
                            "" if p is None else f"{p:.4f}"])
        sizes = [len(rubrics[t]) for t in topic_order]
        manifest = {
            "model": cfg.model, "url": url, "cache_dir": cfg.cache_dir,
            "scale_sha256": hashlib.sha256(json.dumps(SCALE, sort_keys=True).encode()).hexdigest(),
            "padding_question": PADDING_QUESTION if padding else None,
            "n_topics": len(topic_order),
            "questions_per_topic": {"min": min(sizes, default=0), "max": max(sizes, default=0),
                                    "mean": sum(sizes) / len(sizes) if sizes else 0},
            **stats,
        }
        (art / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"[bod] artifacts in {art}")
