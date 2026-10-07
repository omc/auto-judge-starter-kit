"""Compare Jev question variants from a multi-question pairwise run.
Usage: python temp/jev_prompt_compare.py <comparisons.jsonl> [baseline_pilot.jsonl] [gemini.jsonl]"""
import json, sys, statistics as st
from collections import defaultdict
from pathlib import Path
from scipy.stats import spearmanr, kendalltau

def lengths():
    L = defaultdict(dict)
    for f in Path("data/rag26/runs/generation").iterdir():
        for line in open(f):
            r = json.loads(line)
            L[r["metadata"]["topic_id"]][r["metadata"]["run_id"]] = \
                len(" ".join(s["text"] for s in r["responses"]).split())
    return L

recs = [json.loads(l) for l in open(sys.argv[1])]
recs = [r for r in recs if r["valid"]]
Q = list(recs[0]["p"])
topics = sorted({r["topic_id"] for r in recs})
L = lengths()

def winrates(rs, get):
    w, g = defaultdict(float), defaultdict(int)
    for r in rs:
        p = get(r)
        w[r["a_run"]] += p; w[r["b_run"]] += 1 - p; g[r["a_run"]] += 1; g[r["b_run"]] += 1
    return {x: w[x] / g[x] for x in g}

print(f"topics {topics} | comparisons {len(recs)}\n")
print("== length dependence (pooled over topics) ==")
print(f"{'question':16} {'P(longer)':>9} {'<1.25x':>7} {'1.25-2x':>8} {'>=2x':>6} {'rho(wr,len)':>11} {'exact0/1':>8} {'decisive':>8}")
for q in Q:
    lw, bins = [], defaultdict(list)
    for r in recs:
        la, lb, p = L[r["topic_id"]][r["a_run"]], L[r["topic_id"]][r["b_run"]], r["p"][q]
        if la == lb: continue
        pl = p if la > lb else 1 - p
        lw.append(pl)
        ratio = max(la, lb) / max(1, min(la, lb))
        bins["a" if ratio < 1.25 else "b" if ratio < 2 else "c"].append(pl)
    rhos = []
    for t in topics:
        wr = winrates([r for r in recs if r["topic_id"] == t], lambda r: r["p"][q])
        runs = sorted(wr); rhos.append(spearmanr([wr[x] for x in runs], [L[t][x] for x in runs])[0])
    ps = [r["p"][q] for r in recs]
    print(f"{q:16} {st.mean(lw):9.3f} {st.mean(bins['a']):7.3f} {st.mean(bins['b']):8.3f} {st.mean(bins['c']):6.3f} "
          f"{st.mean(rhos):11.3f} {sum(p in (0,1) for p in ps)/len(ps):8.3f} {st.mean(abs(2*p-1) for p in ps):8.3f}")
print("  rho(wr,len) = mean over topics of Spearman(run win-rate, run word count)")

print("\n== ranking agreement between questions (Spearman of per-topic win-rates, mean over topics) ==")
print(" " * 16 + "".join(f"{q[:14]:>15}" for q in Q))
for q1 in Q:
    row = []
    for q2 in Q:
        vals = []
        for t in topics:
            rs = [r for r in recs if r["topic_id"] == t]
            a, b = winrates(rs, lambda r: r["p"][q1]), winrates(rs, lambda r: r["p"][q2])
            runs = sorted(a); vals.append(spearmanr([a[x] for x in runs], [b[x] for x in runs])[0])
        row.append(st.mean(vals))
    print(f"{q1:16}" + "".join(f"{v:15.3f}" for v in row))

print("\n== pairwise decision agreement with control (argmax agrees, all pairs / near-equal-length pairs) ==")
for q in Q[1:]:
    allp = [(r["p"][q] > .5) == (r["p"][Q[0]] > .5) for r in recs]
    near = [(r["p"][q] > .5) == (r["p"][Q[0]] > .5) for r in recs
            if max(L[r["topic_id"]][r["a_run"]], L[r["topic_id"]][r["b_run"]]) /
               max(1, min(L[r["topic_id"]][r["a_run"]], L[r["topic_id"]][r["b_run"]])) < 1.25]
    print(f"{q:16} {st.mean(allp):.3f}  {st.mean(near):.3f} (n_near={len(near)})")

if len(sys.argv) > 2:
    print("\n== test-retest: control vs earlier single-question pilot (same oriented comparison) ==")
    old = {json.loads(l)["comp_id"]: json.loads(l) for l in open(sys.argv[2])}
    pairs = [(r["p"][Q[0]], old[r["comp_id"]]["p_a"]) for r in recs if r["comp_id"] in old]
    d = [abs(a - b) for a, b in pairs]
    print(f"n={len(pairs)} mean|dp|={st.mean(d):.4f} identical={sum(x == 0 for x in d)/len(d):.3f} "
          f"argmax agree={st.mean([(a > .5) == (b > .5) for a, b in pairs]):.3f}")

if len(sys.argv) > 3:
    print("\n== agreement with Gemini binary judge (Spearman of per-topic win-rates) ==")
    gem = defaultdict(list)
    for l in open(sys.argv[3]):
        r = json.loads(l)
        if r["topic_id"] in topics and r["valid"]: gem[r["topic_id"]].append(r)
    for q in Q:
        vals = []
        for t in topics:
            if not gem[t]: continue
            a = winrates([r for r in recs if r["topic_id"] == t], lambda r: r["p"][q])
            b = winrates(gem[t], lambda r: float(r["winner_run"] == r["a_run"]))
            runs = sorted(set(a) & set(b)); vals.append(spearmanr([a[x] for x in runs], [b[x] for x in runs])[0])
        print(f"{q:16} {st.mean(vals):.3f} over {len(vals)} topics")

print("\n== top 8 per question, topic", topics[0], "(win-rate, words) ==")
rs = [r for r in recs if r["topic_id"] == topics[0]]
for q in Q:
    wr = winrates(rs, lambda r: r["p"][q])
    top = sorted(wr, key=lambda x: -wr[x])[:8]
    print(f"{q:16} " + ", ".join(f"{x}({wr[x]:.2f},{L[topics[0]][x]})" for x in top))
