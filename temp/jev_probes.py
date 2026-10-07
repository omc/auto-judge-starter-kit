"""Jev reliability probes (bypass the cache; every call is a fresh API request).

  determinism: every cross-team pair of one topic asked twice with the IDENTICAL
               payload, in two sequential passes. Do probabilities change?
  identical:   each run's summary judged against ITSELF (summary_a == summary_b).
               An unbiased judge must return P(A) = 0.5.

Usage:
  set -a; source ./.env; set +a
  python temp/jev_probes.py determinism --topic rag2026-100
  python temp/jev_probes.py identical --topics rag2026-0 rag2026-1 rag2026-10
  python temp/jev_probes.py report
"""
import argparse
import asyncio
import json
import statistics as st
import time
from pathlib import Path

from autojudge_base import load_report
from minima_llm import MinimaLlmConfig, OpenAIMinimaLlm

from judges.bonsai_judge.pairwise import Comparison, _plan_comparisons, _summaries_by_topic
from judges.bonsai_judge.pairwise_jev import (
    JEV_MODEL, BonsaiJevPairwiseJudge, _prob_a, decisions_url, load_questions)

OUT = Path("temp/jev_probes")
RUNS = Path("data/rag26/runs/generation")
TOPICS = Path("data/rag26/topics/trec_rag_2026_queries.jsonl")
J = BonsaiJevPairwiseJudge()


def load(topics):
    reports = [r for f in sorted(RUNS.iterdir()) for r in load_report(f)
               if r.metadata.topic_id in topics]
    titles = {}
    for line in open(TOPICS):
        t = json.loads(line)
        titles[t["request_id"]] = t.get("title", "")
    return _summaries_by_topic(reports), titles


def backend():
    cfg = MinimaLlmConfig.from_env()
    from dataclasses import replace
    cfg = replace(cfg, model=JEV_MODEL)
    return OpenAIMinimaLlm(cfg), decisions_url(cfg.base_url)


async def ask_all(items, qs, conc=32):
    """items: [(tag_dict, state)] -> list of result rows. No cache reads/writes."""
    be, url = backend()
    sem = asyncio.Semaphore(conc)
    t0 = time.time()

    async def one(tag, state):
        async with sem:
            data, err = await J._call(be, url, J._payload(JEV_MODEL, state, qs), 8)
        ans = (data or {}).get("answers") or {}
        return {**tag, "p": {q: _prob_a(ans.get(q)) for q in qs},
                "raw": {q: ans.get(q) for q in qs}, "err": err,
                "served_model": (data or {}).get("model"),
                "cost": ((data or {}).get("usage") or {}).get("cost"),
                "t": round(time.time() - t0, 2)}

    rows = await asyncio.gather(*(one(t, s) for t, s in items))
    await be.aclose()
    return rows


def write(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    cost = sum(r["cost"] or 0 for r in rows)
    errs = sum(bool(r["err"]) for r in rows)
    print(f"[probe] wrote {OUT/name}: {len(rows)} rows, {errs} errors, ${cost:.4f}")


MIRROR = ("noul_a_better", "noul_b_better")


def _score(row, question):
    """P(A better) for a question; 'mirror' = mean of the Noul pair (raw P(yes) stored)."""
    p = row["p"]
    if question == "mirror":
        if any(p.get(q) is None for q in MIRROR):
            return None
        return (p[MIRROR[0]] + 1 - p[MIRROR[1]]) / 2
    return p.get(question)


def determinism(topic, question):
    summaries, titles = load({topic})
    qs = load_questions(list(MIRROR) if question == "mirror" else [question])
    plan = _plan_comparisons(summaries[topic], topic, direction="single")
    items = [({"comp_id": c.comp_id}, J._state(c, summaries, titles)) for c in plan]
    print(f"[probe] determinism: {topic}, {len(items)} comparisons x 2 passes")
    for i in (1, 2):   # sequential passes, identical payloads
        tag = "" if question == "better_summary" else f"_{question}"
        write(f"determinism_{topic}{tag}_pass{i}.jsonl", asyncio.run(ask_all(items, qs)))


def identical(topics, question):
    summaries, titles = load(set(topics))
    qs = load_questions([question])
    items = []
    for t in topics:
        for run, (team, _txt) in sorted(summaries[t].items()):
            c = Comparison(t, run, team, run, team)   # self vs self
            items.append(({"topic_id": t, "run_id": run, "words": len(_txt.split())},
                          J._state(c, summaries, titles)))
    print(f"[probe] identical: {len(items)} self-comparisons over {topics}")
    write(_identical_name(question), asyncio.run(ask_all(items, qs)))


def _identical_name(question):
    return "identical.jsonl" if question == "better_summary" else f"identical_{question}.jsonl"


def report(question):
    print("== determinism ==")
    for p1 in sorted(OUT.glob("determinism_*_pass1.jsonl")):
        tagged = question != "better_summary"
        if tagged != p1.name.endswith(f"_{question}_pass1.jsonl"):
            continue
        a = {r["comp_id"]: r for r in map(json.loads, open(p1))}
        b = {r["comp_id"]: r for r in map(json.loads, open(str(p1).replace("pass1", "pass2")))}
        pairs = [(_score(a[k], question), _score(b[k], question)) for k in a
                 if k in b and _score(a[k], question) is not None and _score(b[k], question) is not None]
        d = [abs(x - y) for x, y in pairs]
        flips = sum((x - .5) * (y - .5) < 0 for x, y in pairs)   # draws never flip
        conf_d = [abs(((a[k]["raw"].get(question) or {}).get("confidence") or 0) -
                      ((b[k]["raw"].get(question) or {}).get("confidence") or 0)) for k in a if k in b]
        models = {a[k]["served_model"] for k in a} | {b[k]["served_model"] for k in b}
        print(f"{p1.name}: n={len(pairs)} identical={sum(x == 0 for x in d)/len(d):.3f} "
              f"mean|dp|={st.mean(d):.4f} p95|dp|={sorted(d)[int(.95*len(d))]:.3f} "
              f"max|dp|={max(d):.3f} argmax flips={flips} ({flips/len(pairs):.3%}) "
              f"mean|dconf|={st.mean(conf_d):.4f} served={models}")
        big = sorted(((abs(_score(a[k], question) - _score(b[k], question)), k) for k in a if k in b), reverse=True)[:5]
        print("  largest changes:", [(k, round(_score(a[k], question), 3), round(_score(b[k], question), 3)) for _, k in big])
        if question == "mirror":
            for q in MIRROR:
                d = [abs(a[k]["p"][q] - b[k]["p"][q]) for k in a if k in b]
                print(f"  raw {q}: identical={sum(x == 0 for x in d)/len(d):.3f} mean|dp|={st.mean(d):.4f} max={max(d):.3f}")
            ties = [(all(a[k]["p"][q] < .5 for q in MIRROR), all(b[k]["p"][q] < .5 for q in MIRROR)) for k in a if k in b]
            print(f"  tie flag: pass1={sum(x for x, _ in ties)} pass2={sum(y for _, y in ties)} "
                  f"changed={sum(x != y for x, y in ties)}")
    f = OUT / _identical_name(question)
    if f.exists() and question != "mirror":
        print("\n== identical summaries (expected P(A)=0.5) ==")
        rows = [r for r in map(json.loads, open(f)) if r["p"][question] is not None]
        ps = [r["p"][question] for r in rows]
        print(f"n={len(ps)} mean P(A)={st.mean(ps):.3f} median={st.median(ps):.3f} "
              f"exactly 0.5={sum(p == .5 for p in ps)/len(ps):.3f} "
              f"within .45-.55={sum(.45 <= p <= .55 for p in ps)/len(ps):.3f} "
              f"P(A)>.9={sum(p > .9 for p in ps)} P(A)<.1={sum(p < .1 for p in ps)}")
        if (rows[0]["raw"][question] or {}).get("type") == "choice":
            print(f"mean confidence={st.mean((r['raw'][question] or {}).get('confidence', 0) for r in rows):.3f} "
                  f"choices: A={sum(r['raw'][question]['choice']=='A' for r in rows)} "
                  f"B={sum(r['raw'][question]['choice']=='B' for r in rows)}")
        else:
            print(f"P(yes)>0.5: {sum(p > .5 for p in ps)}/{len(ps)}  (raw Noul P(yes) shown above)")
        for t in sorted({r["topic_id"] for r in rows}):
            tp = [r["p"][question] for r in rows if r["topic_id"] == t]
            print(f"  {t}: mean P(A)={st.mean(tp):.3f} sd={st.pstdev(tp):.3f}")
        hist = [0] * 10
        for p in ps:
            hist[min(9, int(p * 10))] += 1
        print("  histogram P(A) by decile:", hist)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("probe", choices=["determinism", "identical", "report"])
    ap.add_argument("--topic", default="rag2026-100")
    ap.add_argument("--topics", nargs="+", default=["rag2026-0", "rag2026-1", "rag2026-10"])
    ap.add_argument("--question", default="better_summary")
    a = ap.parse_args()
    {"determinism": lambda: determinism(a.topic, a.question),
     "identical": lambda: identical(a.topics, a.question),
     "report": lambda: report(a.question)}[a.probe]()
