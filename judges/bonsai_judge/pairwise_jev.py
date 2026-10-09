#!/usr/bin/env python3
"""BonsaiJevPairwiseJudge: probabilistic pairwise summary tournament judged by
TypeSafe's Jev decision model (``typesafe/jev-1.13``) on OpenRouter.

Jev is not a chat model: it answers typed questions about a ``state`` via the
Decisions API (``POST /api/alpha/decisions``) and returns probabilities, not text.
Each comparison asks one or more Choice questions (options ``A``/``B``, loaded
from ``prompts/jev_questions.yml``) in a single request; Jev answers them
independently and bills mostly for the shared state, so extra questions are nearly
free. Each answer's ``probabilities`` give P(A better). That probability -- not
the argmax -- is the score; ``primary_question`` drives the leaderboard and every
question gets its own tables under ``<filebase>.pairwise/q_<id>/``:

- run A earns P(A) "expected wins", run B earns 1 - P(A)
- PAIRWISE_WINRATE  = expected wins / comparisons played
- Bradley-Terry is fit on the same fractional outcomes

Same comparison plan as BonsaiPairwiseJudge (cross-team only, runs sorted by
run_id, ``direction`` single|ordered with hashed A/B orientation in single mode).

Caching: results go into the minima PromptCache under a key derived from
(endpoint, model, state, questions), so reruns cost nothing and never collide with
the chat-model judges' entries. Only well-formed 2xx answers are cached; failures
stay missing and are retried on the next run.
"""
from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import random
import time

import yaml
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from autojudge_base import (
    LlmConfigProtocol,
    Report,
    Request,
    Leaderboard,
    LeaderboardBuilder,
    LeaderboardSpec,
    MeasureSpec,
    Qrels,
    NuggetBanksProtocol,
)
from minima_llm import MinimaLlmConfig, OpenAIMinimaLlm

from .pairwise import (
    DEFAULT_CACHE_DIR,
    BonsaiPairwiseJudge,
    Comparison,
    _add_empty_rows,
    _empty_reports,
    _plan_comparisons,
    _summaries_by_topic,
    select_topics,
    topic_fields,
)

JEV_MODEL = "typesafe/jev-1.13"

JEV_SPEC = LeaderboardSpec(measures=(
    MeasureSpec("PAIRWISE_WINRATE", description=(
        "Probability-weighted win rate: sum over this run's cross-team comparisons "
        "of Jev's P(this summary is better), divided by comparisons played. Primary "
        "metric; comparable across runs with different opponent counts.")),
    MeasureSpec("PAIRWISE_EXP_WINS", description=(
        "Expected wins for the topic: sum of Jev's P(this summary is better) over "
        "its comparisons (numerator of PAIRWISE_WINRATE).")),
    MeasureSpec("PAIRWISE_GAMES", int, description=(
        "Number of valid cross-team comparisons this run took part in for the topic "
        "(as A or B). Denominator of PAIRWISE_WINRATE.")),
    MeasureSpec("PAIRWISE_CONFIDENCE", description=(
        "Mean Jev decisiveness |2*P(A)-1| over this run's comparisons for the topic: "
        "0 = coin flip, 1 = certain. Diagnostic, not a quality score.")),
    MeasureSpec("PAIRWISE_TIE_RATE", description=(
        "Fraction of this run's comparisons for the topic flagged as ties: the two "
        "summaries are identical text (scored 0.5, no call), or -- with mirrored Noul "
        "questions -- Jev answered 'no' to both 'is A better?' and 'is B better?'. "
        "Ties still count as games. Diagnostic.")),
))

MIRROR_ID = "mirror"   # virtual question: mean of the mirrored questions' P(A better)

_QUESTIONS_PATH = Path(__file__).parent / "prompts" / "jev_questions.yml"


def _bank(ids: Sequence[str]) -> Dict[str, dict]:
    bank = yaml.safe_load(_QUESTIONS_PATH.read_text())
    unknown = [q for q in ids if q not in bank]
    if unknown:
        raise KeyError(f"unknown Jev question(s) {unknown}; known: {sorted(bank)}")
    return {q: bank[q] for q in ids}


def load_questions(ids: Sequence[str]) -> Dict[str, dict]:
    """Questions by id, in the order given, as sent to Jev. Local metadata keys
    (prefixed ``x_``) are stripped: the API rejects unknown keys, and stripping keeps
    cache keys of existing questions unchanged. Question text is part of the cache key."""
    return {q: {k: v for k, v in d.items() if not k.startswith("x_")}
            for q, d in _bank(ids).items()}


def question_polarity(ids: Sequence[str]) -> Dict[str, str]:
    """Which summary a 'yes' (Noul) favours: 'a' (default) or 'b' (``x_polarity: b``)."""
    return {q: d.get("x_polarity", "a") for q, d in _bank(ids).items()}


RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


def decisions_url(base_url: str) -> str:
    """https://openrouter.ai/api/v1 -> https://openrouter.ai/api/alpha/decisions."""
    b = base_url.rstrip("/")
    if "/api/v1" in b:
        b = b.split("/api/v1")[0] + "/api"
    elif b.endswith("/v1"):
        b = b[: -len("/v1")]
    return b + "/alpha/decisions"


def _prob_a(answer: Optional[dict], polarity: str = "a") -> Optional[float]:
    """P(summary_a better) from a Jev answer. None if unusable.

    Choice: probabilities renormalised over {A, B} (option text sets the meaning).
    Noul:   P(yes); for a polarity-'b' question ("is summary_b better?") that is
            P(B better), so P(A better) = 1 - P(yes)."""
    if not isinstance(answer, dict):
        return None
    if answer.get("type") == "noul":
        try:
            py = float(answer["noul"])
        except (KeyError, TypeError, ValueError):
            return None
        if not 0.0 <= py <= 1.0:
            return None
        return 1.0 - py if polarity == "b" else py
    if answer.get("type") != "choice":
        return None
    probs = answer.get("probabilities") or {}
    try:
        pa, pb = float(probs["A"]), float(probs["B"])
    except (KeyError, TypeError, ValueError):
        return None
    total = pa + pb
    if total <= 0:
        return None
    return pa / total


class BonsaiJevPairwiseJudge(BonsaiPairwiseJudge):
    """Pairwise tournament scored by Jev's choice probabilities."""

    def judge(
        self,
        rag_responses: Iterable[Report],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        qrels: Optional[Qrels] = None,
        on_missing_evals: str = "fix_aggregate",
        model: Optional[str] = JEV_MODEL,
        questions: Sequence[str] = ("better_summary",),
        primary_question: Optional[str] = None,
        mirror: Optional[Sequence[str]] = None,
        tie_threshold: float = 0.5,
        direction: str = "single",
        max_topics: Optional[int] = None,
        max_runs: Optional[int] = None,
        dev_topics: Optional[Sequence[str]] = None,   # dev variants: ONLY these topics
        topics_per_batch: int = 20,
        max_outstanding: int = 32,
        max_attempts: int = 8,
        filebase: str = "default",
        outdir: Path = Path("."),
        **kwargs: Any,
    ) -> Leaderboard:
        outdir = Path(outdir)
        art = outdir / f"{Path(filebase).name}.pairwise"
        art.mkdir(parents=True, exist_ok=True)

        # {query, [problem_statement], [background]} per topic -> merged into Jev's state
        topic_titles: Dict[str, dict] = {t.request_id: topic_fields(t) for t in rag_topics}
        expected_topic_ids = list(topic_titles.keys())
        reports = list(rag_responses)
        summaries = _summaries_by_topic(reports)
        empties = _empty_reports(reports)          # scored as zero rows, never compared

        if max_runs is not None:
            keep = set(sorted({r for tr in summaries.values() for r in tr})[:max_runs])
            summaries = {t: {r: v for r, v in tr.items() if r in keep} for t, tr in summaries.items()}
            empties = [(t, r) for t, r in empties if r in keep]
        topic_order = select_topics(rag_topics, set(summaries) | {t for t, _ in empties},
                                    dev_topics, max_topics)
        if dev_topics is not None or max_topics is not None:
            keep_t = set(topic_order)
            empties = [(t, r) for t, r in empties if t in keep_t]
            # leaderboard covers only the judged topics (no zero-filled rows for the rest)
            expected_topic_ids = [t for t in expected_topic_ids if t in keep_t]
        topic_order = [t for t in topic_order if t in summaries]   # comparable topics

        plan: List[Comparison] = []
        for t in topic_order:
            plan.extend(_plan_comparisons(summaries[t], t, direction=direction))
        print(f"[jev] {len(topic_order)} topics -> {len(plan):,} cross-team "
              f"comparisons (direction={direction})")
        if not plan:
            print(f"[jev] nothing to compare; {len(empties)} empty-report rows only")
            builder = LeaderboardBuilder(JEV_SPEC)
            _add_empty_rows(builder, JEV_SPEC, empties, ())
            return builder.build(expected_topic_ids=expected_topic_ids, on_missing=on_missing_evals)

        qs = load_questions(list(questions))
        self._polarity = question_polarity(list(questions))
        self._mirror = list(mirror or [])
        self._tie_threshold = tie_threshold
        if self._mirror:
            # Mirrored Noul pair: "is A better?" + "is B better?". Score = mean of their
            # P(A better); a 'no' to both (P(yes) < tie_threshold) is a tie.
            missing = [q for q in self._mirror if q not in qs]
            pols = sorted(self._polarity[q] for q in self._mirror if q in qs)
            if missing or pols != ["a", "b"] or any(qs[q]["type"] != "noul" for q in self._mirror):
                raise ValueError(f"mirror must name one polarity-a and one polarity-b Noul "
                                 f"question from questions {list(qs)}, got {self._mirror}")
        scored = list(qs) + ([MIRROR_ID] if self._mirror else [])
        primary = primary_question or (MIRROR_ID if self._mirror else next(iter(qs)))
        if primary not in scored:
            raise ValueError(f"primary_question {primary!r} is not in {scored}")

        backend, cfg = self._make_jev_backend(llm_config, model)
        url = decisions_url(cfg.base_url)
        records = asyncio.run(self._run_jev(
            backend, cfg.model, url, plan, summaries, topic_titles, qs, primary,
            topics_per_batch, max_outstanding, max_attempts, art))

        lb = None
        for qid in scored:
            question = qs[qid] if qid in qs else {q: qs[q] for q in self._mirror}
            qlb = self._score_jev(records, qid, question, topic_order, expected_topic_ids,
                                  on_missing_evals, art / f"q_{qid}", cfg, url, direction,
                                  empties)
            if qid == primary:
                lb = qlb
                # primary's tables also at the top level (same layout as before)
                for f in (art / f"q_{qid}").iterdir():
                    (art / f.name).write_bytes(f.read_bytes())
        print(f"[jev] primary question: {primary}")
        print(f"[jev] artifacts in {art}")
        return lb

    # ----- backend -----

    def _make_jev_backend(self, llm_config: LlmConfigProtocol, model: Optional[str]
                          ) -> Tuple[OpenAIMinimaLlm, MinimaLlmConfig]:
        """Endpoint + key come from llm_config (OPENAI_BASE_URL / OPENAI_API_KEY).
        ``model`` overrides OPENAI_MODEL (Jev is not a chat model, so the env's chat
        model is never the right value here); model=None defers to the env."""
        cfg = MinimaLlmConfig.from_dict(llm_config.raw) if llm_config.raw else MinimaLlmConfig.from_env()
        if not cfg.cache_dir:
            cfg = replace(cfg, cache_dir=DEFAULT_CACHE_DIR)
        if model:
            cfg = replace(cfg, model=model)
        if "openrouter.ai" not in (cfg.base_url or ""):
            raise ValueError(f"Jev's Decisions API is served by OpenRouter; "
                             f"OPENAI_BASE_URL is {cfg.base_url!r}")
        Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)
        return OpenAIMinimaLlm(cfg), cfg

    # ----- request / cache key -----

    @staticmethod
    def _state(comp: Comparison, summaries, titles) -> dict:
        """titles[topic] is a topic_fields() dict, or a bare title string."""
        info = titles.get(comp.topic_id, "")
        state = {"query": info} if isinstance(info, str) else dict(info)
        state["summary_a"] = summaries[comp.topic_id][comp.a_run][1]
        state["summary_b"] = summaries[comp.topic_id][comp.b_run][1]
        return state

    @staticmethod
    def _payload(model: str, state: dict, questions: Dict[str, dict]) -> dict:
        return {"model": model, "state": state, "questions": questions}

    @staticmethod
    def _cache_key(payload: dict) -> str:
        canon = json.dumps({"api": "alpha/decisions", **payload},
                           sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()

    # ----- execution -----

    async def _call(self, backend, url, payload, max_attempts) -> Tuple[Optional[dict], str]:
        qids = list(payload["questions"])
        """POST with retry on transient statuses. Returns (response_json, error)."""
        err = ""
        for attempt in range(1, max_attempts + 1):
            try:
                status, headers, raw = await backend._post_json(url, payload)
            except Exception as e:                      # network / timeout
                status, headers, raw, err = 0, {}, b"", f"{type(e).__name__}: {e}"
            if 200 <= status < 300:
                try:
                    data = json.loads(raw.decode("utf-8"))
                except Exception as e:
                    return None, f"non-JSON 2xx: {e}"
                answers = data.get("answers") or {}
                if any(_prob_a(answers.get(q)) is None for q in qids):
                    return None, f"malformed answer: {raw[:200]!r}"
                return data, ""
            if status:
                err = f"HTTP {status}: {raw[:200].decode('utf-8', 'replace')}"
            if status and status not in RETRY_STATUS:
                return None, err                        # 4xx: retrying won't help
            try:
                wait = float(headers.get("retry-after", ""))
            except ValueError:
                wait = min(60.0, 2.0 ** attempt) * (0.5 + random.random())
            await asyncio.sleep(wait)
        return None, err

    async def _run_jev(self, backend, model, url, plan, summaries, titles, qs, primary,
                       topics_per_batch, max_outstanding, max_attempts, art) -> List[dict]:
        cache = backend._ensure_cache()
        if cache is None:
            raise RuntimeError("Jev pairwise judge requires a cache_dir")
        print(f"[jev] model={model} url={url} questions={list(qs)} "
              f"max_outstanding={max_outstanding}")
        sem = asyncio.Semaphore(max_outstanding)

        by_topic: Dict[str, List[Comparison]] = defaultdict(list)
        for c in plan:
            by_topic[c.topic_id].append(c)
        topics = list(by_topic)
        groups = [topics[i:i + topics_per_batch] for i in range(0, len(topics), topics_per_batch)]

        records: List[dict] = []
        errors: Dict[str, int] = defaultdict(int)
        t0 = time.time()
        with open(art / "comparisons.jsonl", "w") as cf:
            for gi, group in enumerate(groups):
                comps = [c for t in group for c in by_topic[t]]
                items, dups = [], []
                for c in comps:
                    state = self._state(c, summaries, titles)
                    if state["summary_a"] == state["summary_b"]:
                        dups.append(c)        # identical text: a tie by definition, no call
                        continue
                    payload = self._payload(model, state, qs)
                    items.append((c, payload, self._cache_key(payload)))

                # Identical summaries => identical payloads; ask once per key.
                todo: Dict[str, dict] = {k: p for _c, p, k in items if cache.get(k) is None}
                done = 0

                async def one(key: str, payload: dict) -> None:
                    nonlocal done
                    async with sem:
                        data, err = await self._call(backend, url, payload, max_attempts)
                    if data is not None:
                        cache.put(key, json.dumps(data["answers"]), data)
                    else:
                        errors[err.split(":")[0]] += 1
                    done += 1
                    if done % 500 == 0:
                        print(f"[jev] group {gi+1}: {done:,}/{len(todo):,} calls "
                              f"({(time.time()-t0)/60:.1f}m)")

                await asyncio.gather(*(one(k, p) for k, p in todo.items()))

                n_ok = 0
                for c, _p, key in items:
                    hit = cache.get(key)
                    rec = self._jev_record(c, hit[1] if hit else None, key, list(qs), primary,
                                           self._polarity, self._mirror, self._tie_threshold)
                    n_ok += rec["valid"]
                    records.append(rec)
                    cf.write(json.dumps(rec) + "\n")
                for c in dups:
                    rec = self._duplicate_record(c, list(qs), primary, self._mirror)
                    n_ok += 1
                    records.append(rec)
                    cf.write(json.dumps(rec) + "\n")
                print(f"[jev] group {gi+1}/{len(groups)} ({len(group)} topics): "
                      f"{len(todo):,} new calls, {n_ok:,}/{len(items) + len(dups):,} valid, "
                      f"{len(dups):,} identical-text ties "
                      f"({(time.time()-t0)/60:.1f}m elapsed)")
        if errors:
            print(f"[jev] call errors (left uncached, retried on rerun): {dict(errors)}")
        await backend.aclose()
        return records

    @staticmethod
    def _jev_record(comp: Comparison, raw: Optional[dict], key: str,
                    qids: List[str], primary: str,
                    polarity: Optional[Dict[str, str]] = None,
                    mirror: Optional[List[str]] = None,
                    tie_threshold: float = 0.5) -> dict:
        raw = raw if isinstance(raw, dict) else None
        answers = (raw or {}).get("answers") or {}
        polarity = polarity or {}
        p = {q: _prob_a(answers.get(q), polarity.get(q, "a")) for q in qids}
        raw_p = {q: (answers.get(q) or {}).get(                 # as Jev reported it
            "noul", ((answers.get(q) or {}).get("probabilities") or {}).get("A"))
            for q in qids}
        tie = both_yes = False
        if mirror:
            ok = all(p[q] is not None for q in mirror)
            p[MIRROR_ID] = sum(p[q] for q in mirror) / len(mirror) if ok else None
            if ok:
                tie = all(raw_p[q] < tie_threshold for q in mirror)       # 'no' to both
                both_yes = all(raw_p[q] > tie_threshold for q in mirror)  # contradictory
        answer, pa = answers.get(primary), p[primary]
        usage = (raw or {}).get("usage") or {}
        return {
            "comp_id": comp.comp_id,
            "topic_id": comp.topic_id,
            "a_run": comp.a_run, "a_team": comp.a_team,
            "b_run": comp.b_run, "b_team": comp.b_team,
            "p_a": pa,                    # primary question
            "p": p,                       # every question: P(A better)
            "raw_p": raw_p,
            "tie": tie,
            "both_yes": both_yes,
            "choice": (answer or {}).get("choice"),
            "confidence": (answer or {}).get("confidence"),
            "valid": all(v is not None for v in p.values()),
            "served_model": (raw or {}).get("model"),
            "input_tokens": usage.get("input_tokens", 0),
            "cost": usage.get("cost"),
            "cache_key": key,
            "result": ("ok" if all(v is not None for v in p.values())
                       else "missing" if raw is None else "malformed"),
        }

    @staticmethod
    def _duplicate_record(comp: Comparison, qids: List[str], primary: str,
                          mirror: Optional[List[str]] = None) -> dict:
        """Identical summary text: P(A better) = 0.5 for every question, no API call."""
        p = {q: 0.5 for q in qids + ([MIRROR_ID] if mirror else [])}
        return {
            "comp_id": comp.comp_id, "topic_id": comp.topic_id,
            "a_run": comp.a_run, "a_team": comp.a_team,
            "b_run": comp.b_run, "b_team": comp.b_team,
            "p_a": p[primary], "p": p, "raw_p": {q: None for q in qids},
            "tie": True, "both_yes": False,
            "choice": None, "confidence": None, "valid": True,
            "served_model": None, "input_tokens": 0, "cost": 0.0,
            "cache_key": None, "result": "duplicate",
        }

    # ----- scoring -----

    def _score_jev(self, records, qid, question, topic_order, expected_topic_ids,
                   on_missing, art, cfg, url, direction, empties=()) -> Leaderboard:
        """Leaderboard + tables for ONE question, from records[*]["p"][qid]."""
        art.mkdir(parents=True, exist_ok=True)
        exp_wins: Dict[Tuple[str, str], float] = defaultdict(float)
        games: Dict[Tuple[str, str], int] = defaultdict(int)
        decisive: Dict[Tuple[str, str], float] = defaultdict(float)
        ties: Dict[Tuple[str, str], int] = defaultdict(int)
        beat: Dict[Tuple[str, str], float] = defaultdict(float)   # soft BT counts
        team: Dict[str, str] = {}
        pair_pa: Dict[Tuple[str, str, str], Dict[str, float]] = defaultdict(dict)
        valid = [r for r in records if r["valid"]]

        for r in valid:
            t, a, b, pa = r["topic_id"], r["a_run"], r["b_run"], r["p"][qid]
            team[a], team[b] = r["a_team"], r["b_team"]
            exp_wins[(a, t)] += pa
            exp_wins[(b, t)] += 1.0 - pa
            for run in (a, b):
                games[(run, t)] += 1
                decisive[(run, t)] += abs(2 * pa - 1)
                ties[(run, t)] += int(bool(r.get("tie")))
            beat[(a, b)] += pa
            beat[(b, a)] += 1.0 - pa
            x, y = sorted((a, b))
            # P(x better) in each orientation; ordered mode gives both.
            pair_pa[(t, x, y)]["fwd" if a == x else "rev"] = pa if a == x else 1.0 - pa

        builder = LeaderboardBuilder(JEV_SPEC)
        for (run, t), g in games.items():
            builder.add(run_id=run, topic_id=t, values={
                "PAIRWISE_WINRATE": exp_wins[(run, t)] / g,
                "PAIRWISE_EXP_WINS": exp_wins[(run, t)],
                "PAIRWISE_GAMES": g,
                "PAIRWISE_CONFIDENCE": decisive[(run, t)] / g,
                "PAIRWISE_TIE_RATE": ties[(run, t)] / g,
            })
        _add_empty_rows(builder, JEV_SPEC, empties, games)
        leaderboard = builder.build(expected_topic_ids=expected_topic_ids, on_missing=on_missing)

        # ---- overall soft win-rate ----
        tot_w: Dict[str, float] = defaultdict(float)
        tot_g: Dict[str, int] = defaultdict(int)
        tot_t: Dict[str, int] = defaultdict(int)
        for (run, _t), g in games.items():
            tot_w[run] += exp_wins[(run, _t)]
            tot_g[run] += g
            tot_t[run] += ties[(run, _t)]
        rows = sorted(((run, team[run], tot_w[run], tot_g[run], tot_w[run] / tot_g[run], tot_t[run])
                       for run in tot_g), key=lambda x: -x[4])
        with open(art / "leaderboard_overall.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["rank", "run_id", "team_id", "expected_wins", "games", "win_rate",
                         "ties", "tie_rate"])
            for i, (run, tm, w, g, rate, nt) in enumerate(rows, 1):
                wr.writerow([i, run, tm, f"{w:.4f}", g, f"{rate:.4f}", nt, f"{nt / g:.4f}"])

        # ---- Bradley-Terry on fractional outcomes ----
        self._write_bradley_terry(beat, art)

        # ---- pairs.csv: P(x better) per orientation; ordered mode -> position bias ----
        with open(art / "pairs.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["topic_id", "run_x", "run_y", "p_x_fwd", "p_x_rev", "abs_diff"])
            for (t, x, y), d in sorted(pair_pa.items()):
                fwd, rev = d.get("fwd"), d.get("rev")
                diff = abs(fwd - rev) if fwd is not None and rev is not None else ""
                wr.writerow([t, x, y,
                             "" if fwd is None else f"{fwd:.4f}",
                             "" if rev is None else f"{rev:.4f}",
                             diff if diff == "" else f"{diff:.4f}"])

        # ---- manifest ----
        n_missing = sum(r["result"] == "missing" for r in records)
        n_malformed = sum(r["result"] == "malformed" for r in records)
        mean_pa = sum(r["p"][qid] for r in valid) / len(valid) if valid else None
        cost = sum(r["cost"] or 0.0 for r in valid)
        served = sorted({r["served_model"] for r in valid if r["served_model"]})
        manifest = {
            "question_id": qid,
            "model": cfg.model,
            "served_models": served,
            "url": url,
            "question_sha256": hashlib.sha256(
                json.dumps(question, sort_keys=True).encode()).hexdigest(),
            "direction": direction,
            "cache_dir": cfg.cache_dir,
            "n_comparisons": len(records),
            "n_valid": len(valid), "n_missing": n_missing, "n_malformed": n_malformed,
            "n_duplicate_ties": sum(r["result"] == "duplicate" for r in records),
            "n_ties": sum(bool(r.get("tie")) for r in valid),
            "n_both_yes": sum(bool(r.get("both_yes")) for r in valid),
            "n_topics": len(topic_order),
            # Orientation is hash-randomised, so mean P(A) far from 0.5 = slot bias.
            "mean_p_a": mean_pa,
            "usage_input_tokens": sum(r["input_tokens"] for r in valid),
            "cost_usd": cost,
        }
        (art / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"[jev] {qid}: {len(valid):,} valid / {n_malformed:,} malformed / {n_missing:,} missing "
              f"| mean P(A)={mean_pa if mean_pa is None else round(mean_pa, 4)} "
              f"| ties {manifest['n_ties']:,} (dup {manifest['n_duplicate_ties']:,}) "
              f"both-yes {manifest['n_both_yes']:,} | cost ${cost:.4f} | served {served}")
        if n_missing:
            print(f"[jev] WARNING: {n_missing:,} comparisons missing (excluded from the "
                  f"tally). Re-run to retry them.")
        return leaderboard
