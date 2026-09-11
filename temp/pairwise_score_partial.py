#!/usr/bin/env python3
"""Score the pairwise tournament from the disk cache ONLY (no API calls).

Rebuilds the single-direction plan, reads each comparison's verdict from the cache,
keeps only topics whose comparisons are (near-)fully cached, and writes a labeled
partial leaderboard (win-rate + Bradley-Terry). For when the full run is incomplete
(e.g. hit a budget wall) but we want a clean, honest result over the finished topics.
"""
import json, os, sys, time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from minima_llm import MinimaLlmConfig, OpenAIMinimaLlm
from judges.bonsai_judge import pairwise as P

COVERAGE = 0.99   # a topic is "complete enough" if >= this fraction of its comps are cached
IN_PRICE, OUT_PRICE = 0.15, 1.25   # $/M (note: nominal; actual OpenRouter billing differs)


def main():
    cfg = MinimaLlmConfig.from_env()
    cfg = replace(cfg, model=cfg.model.replace(":batch", ""))   # live run used the plain slug
    be = OpenAIMinimaLlm(cfg)
    cache = be._ensure_cache()
    art = Path("output-pairwise/bonsai_pairwise.pairwise"); art.mkdir(parents=True, exist_ok=True)

    # summaries + titles, exactly as the judge builds them
    GEN = "data/rag26/runs/generation"
    summaries = defaultdict(dict)
    for run in sorted(os.listdir(GEN)):
        p = os.path.join(GEN, run)
        if not os.path.isfile(p):
            continue
        for line in open(p):
            d = json.loads(line)
            t = d["metadata"].get("topic_id")
            txt = P._clean(" ".join(o.get("text", "") for o in (d.get("answer") or d.get("responses") or []))).strip()
            if txt:
                summaries[t][d["metadata"]["run_id"]] = (d["metadata"]["team_id"], txt)
    titles = {}
    for line in open("data/rag26/topics/trec_rag_2026_queries.jsonl"):
        r = json.loads(line); rid = r.get("request_id") or r.get("query_id") or r.get("id")
        titles[rid] = r.get("title") or r.get("query") or r.get("text") or ""
    template = P._PROMPT_PATH.read_text()
    judge = P.BonsaiPairwiseJudge()

    # walk every topic; read cached verdicts; decide which topics are complete enough
    complete_topics, per_topic_cov = [], {}
    records = []
    in_tok = out_tok = 0
    for t in sorted(summaries):
        comps = P._plan_comparisons(summaries[t], t, direction="single")
        if not comps:
            continue
        hits = []
        for c in comps:
            key = be._make_cache_key(judge._request(c, summaries, titles, template, 0.0, 8))
            hit = cache.get(key)
            hits.append((c, key, hit))
        n_cached = sum(1 for _c, _k, h in hits if h is not None)
        cov = n_cached / len(comps)
        per_topic_cov[t] = (n_cached, len(comps), cov)
        if cov >= COVERAGE:
            complete_topics.append(t)
            for c, key, hit in hits:
                if hit is None:
                    continue
                rec = judge._record(c, hit[0], hit[1], key)
                records.append(rec)
                in_tok += rec["input_tokens"]; out_tok += rec["output_tokens"]

    # score over complete topics only
    beat = defaultdict(int)
    valid = invalid = 0
    for r in records:
        if r["valid"]:
            valid += 1
            w = r["winner_run"]; loser = r["b_run"] if w == r["a_run"] else r["a_run"]
            beat[(w, loser)] += 1
        else:
            invalid += 1

    judge._write_overall(records, art)
    judge._write_bradley_terry(dict(beat), art)

    manifest = {
        "PARTIAL": True,
        "model": cfg.model,
        "topics_complete": len(complete_topics),
        "topics_total": len(summaries),
        "coverage_threshold": COVERAGE,
        "comparisons_scored": len(records),
        "valid": valid, "invalid": invalid,
        "usage_input_tokens": in_tok, "usage_output_tokens": out_tok,
        "nominal_cost_scored": round((in_tok*IN_PRICE + out_tok*OUT_PRICE)/1e6, 2),
        "note": "single-direction, randomized A/B; scored over complete topics only; "
                "excludes topics whose calls were incomplete (budget wall).",
    }
    (art / "run_manifest_partial.json").write_text(json.dumps(manifest, indent=2))

    # console summary
    import csv
    print(f"\n=== PARTIAL leaderboard: {len(complete_topics)}/{len(summaries)} topics "
          f"({valid:,} valid / {invalid:,} invalid comparisons) ===")
    ov = list(csv.DictReader(open(art / "leaderboard_overall.csv")))
    bt = list(csv.DictReader(open(art / "bradley_terry.csv")))
    print("\n-- Top 10 by WIN-RATE --")
    print(f"  {'run':<9}{'team':<7}{'win_rate':>9}{'wins':>7}{'games':>7}")
    for r in ov[:10]:
        print(f"  {r['run_id']:<9}{r['team_id']:<7}{float(r['win_rate']):>9.3f}{r['wins']:>7}{r['games']:>7}")
    print("\n-- Top 10 by BRADLEY-TERRY strength --")
    print(f"  {'run':<9}{'bt_strength':>12}{'bt_wins':>9}")
    for r in bt[:10]:
        print(f"  {r['run_id']:<9}{float(r['bt_strength']):>12.3f}{r['bt_wins']:>9}")
    incomplete = sorted([t for t in summaries if per_topic_cov.get(t, (0,1,0))[2] < COVERAGE])
    print(f"\nexcluded (incomplete) topics: {len(incomplete)}  e.g. {incomplete[:8]}")
    print(f"artifacts -> {art}/  (leaderboard_overall.csv, bradley_terry.csv, run_manifest_partial.json)")


if __name__ == "__main__":
    main()
