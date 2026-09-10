#!/usr/bin/env python3
"""BonsaiPairwiseJudge: LLM-powered pairwise summary comparison tournament.

For each topic, every team-run's summary is compared head-to-head against every
other team's runs, in BOTH orderings (A,B) and (B,A) -- i.e. N*(N-1) ordered
comparisons per topic, excluding same-team pairs. The chosen summary earns one
point per comparison. Points are tallied into a per-(run,topic) win-rate and
aggregated into a leaderboard; a Bradley-Terry fit gives a schedule-corrected
strength ranking.

Design goals (see the plan): use the Parasail batch API + disk cache so expensive
inference is never redone, minimize output to a single 'A'/'B' token, and capture
rich per-comparison data for later analysis -- not just the final ranking.

Cost/scale is real: 83 runs / 25 teams / 119 topics ~= 760k ordered comparisons.
Stage with judge_settings.max_topics (pilot -> scale -> full); all stages share the
same cache, so nothing is paid for twice.
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
from minima_llm import MinimaLlmConfig, MinimaLlmRequest, MinimaLlmResponse, OpenAIMinimaLlm

from .bonsai_judge import _clean  # reuse HTML-strip/unescape for deterministic text

# -----------------------------------------------------------------------------
# Spec
# -----------------------------------------------------------------------------

PAIRWISE_SPEC = LeaderboardSpec(measures=(
    MeasureSpec("PAIRWISE_WINRATE", description=(
        "Fraction of ordered cross-team pairwise comparisons this run won for the "
        "topic (points / comparisons played). Primary metric: unlike raw wins it is "
        "comparable across runs whose opponent counts differ because same-team pairs "
        "are excluded.")),
    MeasureSpec("PAIRWISE_WINS", int, description=(
        "Raw count of pairwise comparisons won by this run for the topic (1 point "
        "per chosen comparison).")),
    MeasureSpec("PAIRWISE_GAMES", int, description=(
        "Number of ordered cross-team comparisons this run participated in for the "
        "topic (as either A or B). Denominator of PAIRWISE_WINRATE.")),
))

DEFAULT_CACHE_DIR = "judges/bonsai_judge/.cache/pairwise"
_PROMPT_PATH = Path(__file__).parent / "prompts" / "pairwise_summary_1.md"
_VAR_RE = re.compile(r"<%=\s*(\w+)\s*%>")


def _render(template: str, **vars: str) -> str:
    """Substitute <%=name%> placeholders. Fails fast on an unknown placeholder."""
    def sub(m: "re.Match[str]") -> str:
        name = m.group(1)
        if name not in vars:
            raise KeyError(f"prompt references unknown variable <%={name}%>")
        return str(vars[name])
    return _VAR_RE.sub(sub, template)


# -----------------------------------------------------------------------------
# Comparison plan
# -----------------------------------------------------------------------------

class Comparison:
    """One ordered head-to-head: does the judge prefer summary a or summary b?"""
    __slots__ = ("topic_id", "a_run", "a_team", "b_run", "b_team")

    def __init__(self, topic_id, a_run, a_team, b_run, b_team):
        self.topic_id = topic_id
        self.a_run, self.a_team = a_run, a_team
        self.b_run, self.b_team = b_run, b_team

    @property
    def comp_id(self) -> str:
        return f"{self.topic_id}|{self.a_run}|{self.b_run}"


def _summaries_by_topic(reports: Iterable[Report]) -> Dict[str, Dict[str, Tuple[str, str]]]:
    """topic_id -> {run_id: (team_id, summary_text)}. Empty summaries are dropped."""
    out: Dict[str, Dict[str, Tuple[str, str]]] = defaultdict(dict)
    n_empty = 0
    for r in reports:
        text = _clean(r.get_report_text()).strip()
        if not text:
            n_empty += 1
            continue
        out[r.metadata.topic_id][r.metadata.run_id] = (r.metadata.team_id, text)
    if n_empty:
        print(f"[pairwise] dropped {n_empty} empty summaries (no text to compare)")
    return out


def _plan_comparisons(topic_runs: Dict[str, Tuple[str, str]], topic_id: str,
                      direction: str = "ordered") -> List[Comparison]:
    """Cross-team comparison plan for one topic. Runs sorted by run_id for a
    deterministic prompt order (stable cache keys).

    direction="ordered": both (a,b) and (b,a) -> N*(N-1); position bias cancels in
      the tally and fwd/rev agreement is measurable.
    direction="single": each unordered pair judged ONCE, with A/B orientation chosen
      deterministically by a hash of (topic,x,y) so neither slot is systematically
      favored -- ~half the calls, for when volume/latency/budget is the constraint."""
    runs = sorted(topic_runs)  # deterministic
    comps: List[Comparison] = []
    if direction == "single":
        for i, x in enumerate(runs):
            for y in runs[i + 1:]:
                if topic_runs[x][0] == topic_runs[y][0]:      # same team
                    continue
                h = int(hashlib.sha1(f"{topic_id}|{x}|{y}".encode()).hexdigest(), 16)
                a, b = (x, y) if h % 2 == 0 else (y, x)       # deterministic orientation
                comps.append(Comparison(topic_id, a, topic_runs[a][0], b, topic_runs[b][0]))
    else:
        for a in runs:
            a_team = topic_runs[a][0]
            for b in runs:
                if a == b or topic_runs[b][0] == a_team:      # skip self and same-team
                    continue
                comps.append(Comparison(topic_id, a, a_team, b, topic_runs[b][0]))
    return comps


# -----------------------------------------------------------------------------
# Judge
# -----------------------------------------------------------------------------

class BonsaiPairwiseJudge:
    """Scores summaries by a batched pairwise-comparison tournament."""

    def judge(
        self,
        rag_responses: Iterable[Report],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        qrels: Optional[Qrels] = None,
        on_missing_evals: str = "fix_aggregate",
        max_tokens: int = 1,
        temperature: float = 0.0,
        topics_per_batch: int = 20,
        max_topics: Optional[int] = None,
        max_runs: Optional[int] = None,
        batch_prefix: Optional[str] = None,
        submit_mode: str = "openrouter",       # "openrouter" | "live" | "parasail"
        poll_interval_s: float = 30.0,
        max_requests_per_batch: int = 5000,     # OpenRouter per-batch cap
        max_inflight_requests: int = 20000,     # OpenRouter in-flight-per-entity cap
        wait: bool = False,                     # False: submit + one harvest pass, exit
                                                # if not all done. True: block until done.
        poll_timeout_min: float = 0.0,          # cap for wait=True (0 => max_poll_hours)
        direction: str = "ordered",             # "ordered" (N*(N-1)) | "single" (half)
        live_max_outstanding: int = 48,         # realtime concurrency (submit_mode=live)
        live_rpm: int = 0,                      # realtime pacing (0 = unlimited)
        filebase: str = "default",
        outdir: Path = Path("."),
        **kwargs: Any,
    ) -> Leaderboard:
        outdir = Path(outdir)
        # basename: judge_runner sets filebase to "{out_dir}/{name}", which would
        # nest artifacts under output/output/... -- strip the dir part.
        art = outdir / f"{Path(filebase).name}.pairwise"
        art.mkdir(parents=True, exist_ok=True)

        topic_titles: Dict[str, str] = {t.request_id: (t.title or "") for t in rag_topics}
        expected_topic_ids = list(topic_titles.keys())

        summaries = _summaries_by_topic(rag_responses)

        # ---- optional staging limits (pilot -> scale -> full) ----
        if max_runs is not None:
            keep = set(sorted({r for tr in summaries.values() for r in tr})[:max_runs])
            summaries = {t: {r: v for r, v in tr.items() if r in keep} for t, tr in summaries.items()}
        topic_order = sorted(summaries)
        if max_topics is not None:
            topic_order = topic_order[:max_topics]

        # ---- build the full comparison plan ----
        plan: List[Comparison] = []
        for t in topic_order:
            plan.extend(_plan_comparisons(summaries[t], t, direction=direction))
        print(f"[pairwise] {len(topic_order)} topics -> {len(plan):,} cross-team "
              f"comparisons (direction={direction})")
        if not plan:
            print("[pairwise] nothing to compare; emitting empty leaderboard")
            return LeaderboardBuilder(PAIRWISE_SPEC).build(
                expected_topic_ids=expected_topic_ids, on_missing=on_missing_evals)

        backend, cfg = self._make_backend(
            llm_config, submit_mode, live_max_outstanding, live_rpm)
        # Use a STABLE, filesystem-safe prefix keyed to the judge (filebase), NOT the
        # out_dir. judge_runner pollutes cfg.parasail.{prefix,llm_batch_prefix} with the
        # out_dir, which both breaks state filenames (slashes) and makes batch state
        # non-reusable across runs with different out_dirs. batch_prefix overrides.
        prefix = re.sub(r"[^A-Za-z0-9_.-]", "_",
                        batch_prefix or Path(filebase).name)

        # ---- run in topic-groups to bound memory + give durable progress ----
        template = _PROMPT_PATH.read_text()
        records, all_ready = asyncio.run(self._run(
            backend, cfg, plan, summaries, topic_titles, template,
            prefix, temperature, max_tokens, topics_per_batch, art,
            submit_mode, poll_interval_s, max_requests_per_batch,
            wait, poll_timeout_min, max_inflight_requests))

        if not all_ready:
            # Decoupled mode: batches still in flight. Do NOT score (would be partial).
            n_cached = sum(1 for r in records if r["result"] == "ok")
            status = {"ready": False, "comparisons": len(records), "cached": n_cached,
                      "pending": len(records) - n_cached, "prefix": prefix}
            (art / "harvest_status.json").write_text(json.dumps(status, indent=2))
            raise SystemExit(
                f"[pairwise] BATCHES NOT READY: {n_cached:,}/{len(records):,} comparisons "
                f"harvested. Re-run the same command to harvest more once OpenRouter "
                f"finishes (status in {art}/harvest_status.json). No scoring yet.")

        # ---- all batches terminal -> score + write artifacts ----
        leaderboard = self._score_and_write(
            records, summaries, topic_order, expected_topic_ids,
            on_missing_evals, art, cfg, temperature, max_tokens, template)
        print(f"[pairwise] artifacts in {art}")
        return leaderboard

    # ----- backend / config -----

    def _make_backend(self, llm_config: LlmConfigProtocol, submit_mode: str = "openrouter",
                      live_max_outstanding: int = 48, live_rpm: int = 0
                      ) -> Tuple[OpenAIMinimaLlm, MinimaLlmConfig]:
        cfg = MinimaLlmConfig.from_dict(llm_config.raw) if llm_config.raw else MinimaLlmConfig.from_env()
        if not cfg.cache_dir:  # pin a STABLE cache path so reruns resume (not the timestamped outdir)
            cfg = replace(cfg, cache_dir=DEFAULT_CACHE_DIR)
        Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)
        if submit_mode == "live":
            # Realtime path: the ':batch' variant 404s on realtime, so use the plain
            # slug. Crank concurrency, and DON'T abort the batch on failures -- a long
            # run will have sporadic ones; they stay uncached and are retried on re-run.
            from minima_llm import BatchConfig
            cfg = replace(
                cfg,
                model=cfg.model.replace(":batch", ""),
                max_outstanding=live_max_outstanding,
                rpm=live_rpm,
                batch=replace(cfg.batch, max_failures=None,
                              num_workers=max(cfg.batch.num_workers, live_max_outstanding)),
            )
        Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)
        return OpenAIMinimaLlm(cfg), cfg

    def _request(self, comp, summaries, titles, template, temperature, max_tokens) -> MinimaLlmRequest:
        a_text = summaries[comp.topic_id][comp.a_run][1]
        b_text = summaries[comp.topic_id][comp.b_run][1]
        content = _render(template, topic_query=titles.get(comp.topic_id, ""),
                          summary_a=a_text, summary_b=b_text)
        return MinimaLlmRequest(
            request_id=comp.comp_id,
            messages=[{"role": "user", "content": content}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # ----- batched execution -----

    async def _run(self, backend, cfg, plan, summaries, titles, template,
                   prefix, temperature, max_tokens, topics_per_batch, art,
                   submit_mode, poll_interval_s, max_requests_per_batch,
                   wait, poll_timeout_min, max_inflight_requests) -> Tuple[List[dict], bool]:
        """Per topic-group: populate the disk cache via the chosen submit path, then
        read every result straight from cache so scoring is path-agnostic and no
        surprise live calls happen during retrieval. Returns (records, all_ready);
        all_ready is False when OpenRouter batches are still in flight."""
        cache = backend._ensure_cache()
        if cache is None:
            raise RuntimeError("pairwise judge requires a cache_dir (disk cache) to be set")
        print(f"[pairwise] submit_mode={submit_mode} model={cfg.model} wait={wait}")
        all_ready = True

        # group the plan by topic so each batch session covers whole topics
        by_topic: Dict[str, List[Comparison]] = defaultdict(list)
        for c in plan:
            by_topic[c.topic_id].append(c)
        topics = list(by_topic)
        groups = [topics[i:i + topics_per_batch] for i in range(0, len(topics), topics_per_batch)]

        comp_path = art / "comparisons.jsonl"
        records: List[dict] = []
        t0 = time.time()
        with open(comp_path, "w") as cf:
            for gi, group in enumerate(groups):
                comps = [c for t in group for c in by_topic[t]]
                reqs = [self._request(c, summaries, titles, template, temperature, max_tokens)
                        for c in comps]
                # Scope the group's batch-state to its TOPIC CONTENT, not its ordinal.
                # Otherwise different runs (pilot 1-topic vs scale 5-topic) collide on
                # "{prefix}-g0" and positional chunk indices get cross-matched, silently
                # dropping comparisons. Same topic set => same gprefix => clean resume.
                ghash = __import__("hashlib").sha1(",".join(sorted(group)).encode()).hexdigest()[:8]
                gprefix = f"{prefix}-g{gi}-{ghash}"
                # Phase 1: populate cache for this group via the chosen submit path.
                if submit_mode == "openrouter":
                    from .openrouter_batch import OpenRouterBatch
                    orb = OpenRouterBatch(
                        cfg, cache, state_dir=Path(cfg.cache_dir), prefix=gprefix,
                        poll_interval_s=poll_interval_s,
                        max_requests_per_batch=max_requests_per_batch,
                        max_inflight_requests=max_inflight_requests)
                    keys = [backend._make_cache_key(r) for r in reqs]
                    # Synchronous: minima's sqlite PromptCache connection is bound to
                    # this (main) thread; to_thread would trip sqlite's thread check.
                    sub = orb.submit(list(zip(reqs, keys)))
                    ready, hv = orb.harvest(wait=wait, poll_timeout_s=poll_timeout_min * 60)
                    # Not ready unless every chunk is BOTH submitted and terminal.
                    all_ready = all_ready and ready and sub["submitted_all"]
                    print(f"[pairwise] group {gi}: submit={sub} harvest(ready={ready})={hv}")
                elif submit_mode == "parasail":
                    async with backend.batch_mode(gprefix):
                        await backend.run_batched(reqs)
                elif submit_mode == "live":
                    await backend.run_batched(reqs)   # realtime HTTP; still cached
                else:
                    raise ValueError(f"unknown submit_mode: {submit_mode!r}")
                # Phase 2: read every result straight from cache; missing => not returned.
                n_cached = n_fail = 0
                for c, req in zip(comps, reqs):
                    key = backend._make_cache_key(req)
                    hit = cache.get(key)
                    if hit is None:
                        n_fail += 1
                        rec = self._record(c, None, None, key)
                    else:
                        n_cached += 1
                        rec = self._record(c, hit[0], hit[1], key)
                    records.append(rec)
                    cf.write(json.dumps(rec) + "\n")
                print(f"[pairwise] group {gi+1}/{len(groups)} ({len(group)} topics): "
                      f"{n_cached:,} results, {n_fail:,} missing "
                      f"({(time.time()-t0)/60:.1f}m elapsed)")
        await backend.aclose()
        return records, all_ready

    @staticmethod
    def _record(comp: Comparison, text: Optional[str], raw: Optional[dict], key: str) -> dict:
        winner, valid = BonsaiPairwiseJudge._parse_choice(text)
        usage = (raw or {}).get("usage", {}) if isinstance(raw, dict) else {}
        return {
            "comp_id": comp.comp_id,
            "topic_id": comp.topic_id,
            "a_run": comp.a_run, "a_team": comp.a_team,
            "b_run": comp.b_run, "b_team": comp.b_team,
            "winner_run": (comp.a_run if winner == "A" else comp.b_run) if valid else None,
            "choice": winner,            # "A" | "B" | None
            "valid": valid,
            "raw_text": (text or "")[:64],
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "cache_key": key,
            "result": "ok" if valid else ("missing" if text is None else "unparseable"),
        }

    @staticmethod
    def _parse_choice(text: Optional[str]) -> Tuple[Optional[str], bool]:
        if not text:
            return None, False
        # Strip whitespace/punctuation/markdown; accept ONLY when the response is
        # a bare 'A' or 'B'. Prose (e.g. "The answer is B.") is flagged unparseable
        # rather than guessed from a stray letter -- misattribution is worse.
        cleaned = re.sub(r"[^A-Za-z]", "", text).upper()
        if cleaned in ("A", "B"):
            return cleaned, True
        return None, False

    # ----- scoring + artifacts -----

    def _score_and_write(self, records, summaries, topic_order, expected_topic_ids,
                         on_missing, art, cfg, temperature, max_tokens, template) -> Leaderboard:
        wins: Dict[Tuple[str, str], int] = defaultdict(int)     # (run,topic) -> wins
        games: Dict[Tuple[str, str], int] = defaultdict(int)    # (run,topic) -> games
        pair_choice: Dict[Tuple[str, str, str], Dict[str, Optional[str]]] = defaultdict(dict)
        beat: Dict[Tuple[str, str], int] = defaultdict(int)     # (i,j) -> #times i beat j (BT)
        n_valid = n_invalid = 0

        for r in records:
            t, a, b = r["topic_id"], r["a_run"], r["b_run"]
            games[(a, t)] += 1
            games[(b, t)] += 1
            if not r["valid"]:
                n_invalid += 1
                continue
            n_valid += 1
            w = r["winner_run"]
            wins[(w, t)] += 1
            loser = b if w == a else a
            beat[(w, loser)] += 1
            # forward/reverse agreement diagnostic (unordered pair keyed by sorted runs)
            key = (t,) + tuple(sorted((a, b)))
            direction = "fwd" if (a, b) == (key[1], key[2]) else "rev"
            pair_choice[key][direction] = w

        # ---- leaderboard: per-(run,topic) win-rate ----
        builder = LeaderboardBuilder(PAIRWISE_SPEC)
        seen: set = set()
        for (run, topic), g in games.items():
            w = wins.get((run, topic), 0)
            builder.add(run_id=run, topic_id=topic, values={
                "PAIRWISE_WINRATE": (w / g) if g else 0.0,
                "PAIRWISE_WINS": w,
                "PAIRWISE_GAMES": g,
            })
            seen.add((run, topic))
        leaderboard = builder.build(expected_topic_ids=expected_topic_ids, on_missing=on_missing)

        # ---- pairs.csv: position-bias / agreement per unordered pair ----
        with open(art / "pairs.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["topic_id", "run_x", "run_y", "fwd_winner", "rev_winner",
                         "agree", "position_bias"])
            for (t, x, y), d in sorted(pair_choice.items()):
                fwd, rev = d.get("fwd"), d.get("rev")
                agree = (fwd is not None and fwd == rev)
                # position bias: both directions favored the slot (A won fwd AND A won rev)
                bias = (fwd is not None and rev is not None and fwd != rev)
                wr.writerow([t, x, y, fwd or "", rev or "", int(agree), int(bias)])

        # ---- overall win-rate + Bradley-Terry strengths ----
        self._write_overall(records, art)
        self._write_bradley_terry(beat, art)

        # ---- manifest (reproducibility + real cost inputs) ----
        n_missing = sum(1 for r in records if r["result"] == "missing")
        in_tok = sum(r["input_tokens"] for r in records)
        out_tok = sum(r["output_tokens"] for r in records)
        manifest = {
            "model": cfg.model,
            "base_url": cfg.base_url,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_sha256": __import__("hashlib").sha256(template.encode()).hexdigest(),
            "cache_dir": cfg.cache_dir,
            "n_comparisons": len(records),
            "n_valid": n_valid, "n_invalid": n_invalid, "n_missing": n_missing,
            "n_topics": len(topic_order),
            "usage_input_tokens": in_tok, "usage_output_tokens": out_tok,
        }
        (art / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"[pairwise] {n_valid:,} valid / {n_invalid:,} invalid / {n_missing:,} missing "
              f"| usage in={in_tok:,} out={out_tok:,} tokens")
        if n_missing:
            print(f"[pairwise] WARNING: {n_missing:,} comparisons missing from batch results "
                  f"(excluded from tally, not silently). Re-run to resubmit them.")
        return leaderboard

    def _write_overall(self, records, art) -> None:
        wins: Dict[str, int] = defaultdict(int)
        games: Dict[str, int] = defaultdict(int)
        team: Dict[str, str] = {}
        for r in records:
            team[r["a_run"]] = r["a_team"]; team[r["b_run"]] = r["b_team"]
            games[r["a_run"]] += 1; games[r["b_run"]] += 1
            if r["valid"]:
                wins[r["winner_run"]] += 1
        rows = [(run, team[run], wins.get(run, 0), games[run],
                 wins.get(run, 0) / games[run] if games[run] else 0.0)
                for run in games]
        rows.sort(key=lambda x: -x[4])
        with open(art / "leaderboard_overall.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["rank", "run_id", "team_id", "wins", "games", "win_rate"])
            for i, (run, tm, w, g, wr_) in enumerate(rows, 1):
                wr.writerow([i, run, tm, w, g, f"{wr_:.4f}"])

    def _write_bradley_terry(self, beat: Dict[Tuple[str, str], int], art: Path,
                             iters: int = 200) -> None:
        """MM algorithm for Bradley-Terry strengths. Schedule-corrected ranking:
        strength_i s.t. P(i beats j) = s_i/(s_i+s_j). Handles the unequal opponent
        counts from same-team exclusion better than raw win totals."""
        runs = sorted({r for pair in beat for r in pair})
        if not runs:
            return
        idx = {r: i for i, r in enumerate(runs)}
        n = len(runs)
        W = [[0]* n for _ in range(n)]
        for (i, j), c in beat.items():
            W[idx[i]][idx[j]] += c
        wins = [sum(W[i]) for i in range(n)]
        s = [1.0] * n
        for _ in range(iters):
            ns = [0.0] * n
            for i in range(n):
                denom = 0.0
                for j in range(n):
                    nij = W[i][j] + W[j][i]
                    if nij:
                        denom += nij / (s[i] + s[j])
                ns[i] = wins[i] / denom if denom > 0 else s[i]
            g = sum(ns) / n or 1.0
            s = [x / g for x in ns]  # normalize to mean 1 (identifiability)
        ranked = sorted(zip(runs, s), key=lambda x: -x[1])
        with open(art / "bradley_terry.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["rank", "run_id", "bt_strength", "bt_wins"])
            for r, (run, strength) in enumerate(ranked, 1):
                wr.writerow([r, run, f"{strength:.6f}", wins[idx[run]]])
