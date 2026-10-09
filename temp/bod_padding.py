"""Padding test for the BOD judge (pointwise): does padding raise the rubric score?

Same variants as temp/jev_padding.py (same generic pool, off-topic source = window
topic i+5, relevant padding = unseen sentences from the topic's top BOD run of another
team, seeded interleaving), but each variant is graded on its own against the topic's
rubric (from the pilot's nugget banks). Effect = per-target change vs the original,
bootstrapped over targets. PERMITTED window topics only.

Expected outcome, written BEFORE the results (2026-10-09):
  BOD_MEAN:    pad_repeat / pad_generic / pad_offtopic dMean <= 0 (95% CI upper bound
               <= +0.01: no reward for uninformative length); pad_relevant dMean > 0
               (CI excludes 0); truncate dMean < 0.
  BOD_PADDING: pad_repeat / pad_generic / pad_offtopic dPadding > 0 (CI excludes 0).

Usage: set -a; source ./.env; set +a
       JEV_DATASET=rag26 python temp/bod_padding.py run|report
"""
import argparse
import asyncio
import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from jev_dataset import DS, assert_window, offtopic_for, topic_info, window_records  # noqa: E402
from jev_padding import GENERIC, VARIANTS, load_sentences, variants  # noqa: E402

from judges.bonsai_judge.bod import (  # noqa: E402
    MAX_GRADE, PADDING_ID, _valid_answer, grade_probs, jev_questions, _noul)
from judges.bonsai_judge.pairwise_jev import BonsaiJevPairwiseJudge, decisions_url  # noqa: E402

PILOT = Path(f"{DS['out']}-bod")
OUT = Path(DS["probes"]) / "bod_padding.jsonl"
N_TARGETS = 16


def rubrics():
    out = {}
    for line in open(PILOT / "bonsai_bod.nuggets.jsonl"):
        b = json.loads(line)
        if b["query_id"] in DS["window"]:
            ns = sorted(b["nugget_bank"].values(), key=lambda n: n["question_id"])
            out[b["query_id"]] = [n["question"] for n in ns]
    return out


def pilot_scores():
    """(topic, run) -> BOD_MEAN from the pilot's grades (window only)."""
    by = defaultdict(list)
    for r in window_records(PILOT / "bonsai_bod.bod" / "grades.jsonl"):
        if r["qid"] != PADDING_ID:
            by[(r["topic_id"], r["run_id"])].append(r["expected"])
    return {k: sum(v) / (MAX_GRADE * len(v)) for k, v in by.items()}


def build():
    topics = DS["probe_topics"]
    assert_window(topics)
    data = load_sentences(set(topics) | {offtopic_for(t) for t in topics})
    sc, info, rub = pilot_scores(), topic_info(), rubrics()
    items = []
    for t in topics:
        offtopic_pool = [s for _team, ss in data[offtopic_for(t)].values() for s in ss]
        ranked = sorted((r for r in data[t] if (t, r) in sc), key=lambda r: -sc[(t, r)])
        q = len(ranked)
        mid = ranked[q // 4: 3 * q // 4]                   # interquartile: room to move
        targets = list(dict.fromkeys(mid[round(i * (len(mid) - 1) / (N_TARGETS - 1))]
                                     for i in range(N_TARGETS)))
        for tgt in targets:
            team, sents = data[t][tgt]
            src = next(r for r in ranked if data[t][r][0] != team)   # top run, other team
            rng = random.Random(f"{t}|{tgt}")
            for v, vs in variants(sents, rng, GENERIC, offtopic_pool, data[t][src][1]).items():
                text = " ".join(vs)
                items.append(({"topic_id": t, "target": tgt, "variant": v,
                               "words": len(text.split()), "orig_words": len(" ".join(sents).split())},
                              {**info[t], "answer": text}, rub[t]))
    return items


async def ask(items):
    helper = BonsaiJevPairwiseJudge()
    backend, cfg = helper._make_jev_backend(SimpleNamespace(raw=None), "typesafe/jev-1.13")
    url, cache = decisions_url(cfg.base_url), backend._ensure_cache()
    sem = asyncio.Semaphore(32)
    payloads = [{"model": cfg.model, "state": s, "questions": jev_questions(rub)} for _t, s, rub in items]
    keys = [helper._cache_key(p) for p in payloads]
    todo = {k: p for k, p in zip(keys, payloads) if cache.get(k) is None}
    print(f"[bod-padding] {len(items)} items, {len(todo)} new calls")

    async def one(k, p):
        async with sem:
            data, err = await helper._call(backend, url, p, 8, check=_valid_answer)
        if data is not None:
            cache.put(k, json.dumps(data["answers"]), data)
        else:
            print("error:", err.split(":")[0])
    await asyncio.gather(*(one(k, p) for k, p in todo.items()))
    rows, cost = [], 0.0
    for (tag, _s, rub), k in zip(items, keys):
        hit = cache.get(k)
        raw = hit[1] if hit else None
        ans = (raw or {}).get("answers") or {}
        ps = [grade_probs(ans.get(f"q{i:02d}")) for i in range(1, len(rub) + 1)]
        ok = raw is not None and all(p is not None for p in ps)
        cost += ((raw or {}).get("usage") or {}).get("cost") or 0.0
        rows.append({**tag, "bod_mean": (sum(sum(g * v for g, v in p.items()) for p in ps)
                                         / (MAX_GRADE * len(ps))) if ok else None,
                     "padding": _noul(ans.get(PADDING_ID)), "cache_key": k})
    await backend.aclose()
    print(f"[bod-padding] cost ${cost:.4f} incl. cached")
    return rows


def run():
    rows = asyncio.run(ask(build()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[bod-padding] wrote {OUT}: {sum(r['bod_mean'] is None for r in rows)} invalid")


def report():
    rows = [r for r in window_records(OUT) if r["bod_mean"] is not None]
    cell = {(r["topic_id"], r["target"], r["variant"]): r for r in rows}
    targets = sorted({(t, g) for t, g, _v in cell})
    rng = random.Random(0)
    print(f"{DS['name']}: {len(targets)} targets over {len({t for t, _ in targets})} window topics\n")
    print(f"{'variant':14} {'words x':>8} {'BOD_MEAN':>9} {'dMean':>7} {'95% CI':>17} "
          f"{'PADDING':>8} {'dPad':>7} {'95% CI':>17}")
    for v in VARIANTS:
        ts = [x for x in targets if x + (v,) in cell and x + ("original",) in cell]
        line = f"{v:14} {st.mean(cell[x + (v,)]['words'] / cell[x + (v,)]['orig_words'] for x in ts):8.2f}"
        for m in ("bod_mean", "padding"):
            d = [cell[x + (v,)][m] - cell[x + ("original",)][m] for x in ts]
            b = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(5000))
            line += (f" {st.mean(cell[x + (v,)][m] for x in ts):8.3f} {st.mean(d):+7.3f} "
                     f"[{b[125]:+.3f},{b[4874]:+.3f}]")
        print(line)
    print("\nper topic dMean / dPad:")
    for t in DS["window"]:
        ts = [x for x in targets if x[0] == t]
        if ts:
            print(f"  {t}: " + "  ".join(
                f"{v}={st.mean(cell[x + (v,)]['bod_mean'] - cell[x + ('original',)]['bod_mean'] for x in ts):+.3f}"
                f"/{st.mean(cell[x + (v,)]['padding'] - cell[x + ('original',)]['padding'] for x in ts):+.3f}"
                for v in VARIANTS[1:]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    run() if ap.parse_args().cmd == "run" else report()
