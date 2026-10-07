"""Instruction-following probes for Jev. Usage: python temp/jev_instruction_check.py <comparisons.jsonl>"""
import json, sys, statistics as st, math
from collections import defaultdict
from pathlib import Path
from scipy.stats import spearmanr, pearsonr

L = defaultdict(dict)
for f in Path("data/rag26/runs/generation").iterdir():
    for line in open(f):
        r = json.loads(line)
        L[r["metadata"]["topic_id"]][r["metadata"]["run_id"]] = len(" ".join(s["text"] for s in r["responses"]).split())
recs = [json.loads(l) for l in open(sys.argv[1])]
recs = [r for r in recs if r["valid"]]
la = lambda r: L[r["topic_id"]][r["a_run"]]
lb = lambda r: L[r["topic_id"]][r["b_run"]]
P = lambda q: [r["p"][q] for r in recs]
better = P("better_summary")

print(f"n={len(recs)}\n== 1. shorter: objective accuracy vs word counts ==")
rows = [(r["p"]["shorter"], la(r) < lb(r), max(la(r), lb(r)) / max(1, min(la(r), lb(r))), r["p"]["better_summary"])
        for r in recs if la(r) != lb(r)]
for name, lo, hi in [("all", 1, 1e9), ("<1.25x", 1, 1.25), ("1.25-2x", 1.25, 2), (">=2x", 2, 1e9)]:
    sub = [x for x in rows if lo <= x[2] < hi]
    acc = st.mean((p > .5) == a_short for p, a_short, _, _ in sub)
    print(f"  {name:8} n={len(sub):5} accuracy={acc:.3f}  mean P(correct)={st.mean(p if s else 1-p for p, s, _, _ in sub):.3f}")
lr = [math.log(la(r) / lb(r)) for r in recs if la(r) != lb(r)]
ps = [r["p"]["shorter"] for r in recs if la(r) != lb(r)]
pb = [r["p"]["better_summary"] for r in recs if la(r) != lb(r)]
print(f"  Spearman(P(A shorter), log(len_a/len_b)) = {spearmanr(ps, lr)[0]:.3f}  (ideal: strongly negative)")
print(f"  Spearman(P(A shorter), P(A better))     = {spearmanr(ps, pb)[0]:.3f}  (strongly negative => answering quality)")
# partial: does shorter track length after controlling for better? residual check via agreement split
agree_len = st.mean(((p > .5) == (a < 0)) for p, a in zip(ps, lr))
agree_inv_quality = st.mean(((p > .5) == (b < .5)) for p, b in zip(ps, pb))
print(f"  argmax matches true-shorter {agree_len:.3f} | matches 'NOT better' {agree_inv_quality:.3f}")
# disagreement cases: where length and quality point different ways
conf = [(p, a, b) for p, a, b in zip(ps, lr, pb) if (a < 0) != (b < .5)]
if conf:
    print(f"  conflict pairs (shorter one is ALSO judged better), n={len(conf)}: "
          f"follows length {st.mean((p > .5) == (a < 0) for p, a, b in conf):.3f} vs follows inverse-quality "
          f"{st.mean((p > .5) == (b < .5) for p, a, b in conf):.3f}")

for q in ("worse_summary", "swapped_labels"):
    print(f"\n== {q}: expected P(A) = 1 - P_better(A) ==")
    pq = P(q)
    inv = [1 - b for b in better]
    print(f"  mean P(A)={st.mean(pq):.3f} (better: {st.mean(better):.3f})")
    print(f"  Pearson with 1-P_better {pearsonr(pq, inv)[0]:.3f} | with P_better {pearsonr(pq, better)[0]:.3f}")
    print(f"  mean |P - (1-P_better)| {st.mean(abs(a-b) for a, b in zip(pq, inv)):.3f}  |  mean |P - P_better| {st.mean(abs(a-b) for a, b in zip(pq, better)):.3f}")
    print(f"  argmax = inverted-better {st.mean((a > .5) == (b < .5) for a, b in zip(pq, better)):.3f}  | "
          f"= same-as-better {st.mean((a > .5) == (b > .5) for a, b in zip(pq, better)):.3f}")
    # where better is decisive
    dec = [(a, b) for a, b in zip(pq, better) if abs(b - .5) > .4]
    print(f"  on decisive control pairs (|P_better-.5|>.4, n={len(dec)}): inverted {st.mean((a > .5) == (b < .5) for a, b in dec):.3f}")
    # A-slot tendency for label probe
    print(f"  P(A) when better is ~coin flip (0.4-0.6, n={sum(.4<=b<=.6 for b in better)}): "
          f"{st.mean([a for a, b in zip(pq, better) if .4 <= b <= .6] or [float('nan')]):.3f}")
