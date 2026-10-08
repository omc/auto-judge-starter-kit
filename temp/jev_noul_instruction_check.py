"""Noul instruction-following probes.
Usage: python temp/jev_noul_instruction_check.py <noul-icheck comparisons.jsonl> [noul-pilot comparisons.jsonl]

Uses raw P(yes) (``raw_p``):
  yA = noul_a_better, yB = noul_b_better, yW = noul_a_worse, yS = noul_a_shorter
  mirror = (yA + 1 - yB) / 2
"""
import json, math, statistics as st, sys
from collections import defaultdict
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from jev_dataset import runs_dir  # JEV_DATASET=rag26|ragtime26

L = defaultdict(dict)
for f in runs_dir().iterdir():
    for line in open(f):
        r = json.loads(line)
        L[r["metadata"]["topic_id"]][r["metadata"]["run_id"]] = len(" ".join(s["text"] for s in r["responses"]).split())

recs = [r for r in map(json.loads, open(sys.argv[1])) if r["valid"] and r["result"] == "ok"]
y = lambda r, q: r["raw_p"][q]
yA = lambda r: y(r, "noul_a_better"); yB = lambda r: y(r, "noul_b_better")
yW = lambda r: y(r, "noul_a_worse"); yS = lambda r: y(r, "noul_a_shorter")
mir = lambda r: (yA(r) + 1 - yB(r)) / 2
la = lambda r: L[r["topic_id"]][r["a_run"]]; lb = lambda r: L[r["topic_id"]][r["b_run"]]
sign = lambda p: (p > .5) - (p < .5)      # +1 A, -1 B, 0 draw
print(f"n={len(recs)} topics={sorted({r['topic_id'] for r in recs})}\n")

print("== mirror pair (re-asked in a 4-question bundle) ==")
s = [yA(r) + yB(r) for r in recs]
print(f"yA+yB mean {st.mean(s):.3f} | within [0.9,1.1] {sum(.9 <= x <= 1.1 for x in s)/len(s):.3f} | "
      f"both-no {sum(yA(r) < .5 and yB(r) < .5 for r in recs)} | both-yes {sum(yA(r) > .5 and yB(r) > .5 for r in recs)} "
      f"| Pearson(yA,1-yB) {pearsonr([yA(r) for r in recs], [1 - yB(r) for r in recs])[0]:.3f}")

print("\n== noul_a_worse: expected yW ~= 1 - yA (negation), i.e. ~= yB ==")
print(f"mean yW {st.mean(map(yW, recs)):.3f} (mean yA {st.mean(map(yA, recs)):.3f}, mean yB {st.mean(map(yB, recs)):.3f})")
print(f"Pearson(yW, 1-yA) {pearsonr([yW(r) for r in recs], [1 - yA(r) for r in recs])[0]:.3f} | "
      f"Pearson(yW, yB) {pearsonr([yW(r) for r in recs], [yB(r) for r in recs])[0]:.3f} | "
      f"Pearson(yW, yA) {pearsonr([yW(r) for r in recs], [yA(r) for r in recs])[0]:.3f}")
print(f"mean |yW-(1-yA)| {st.mean(abs(yW(r) - (1 - yA(r))) for r in recs):.3f} | mean |yW-yB| {st.mean(abs(yW(r) - yB(r)) for r in recs):.3f}")
print(f"argmax: 'A worse' agrees with mirror saying B better {st.mean(sign(yW(r) - .5 + .5) == -sign(mir(r)) for r in recs if sign(mir(r))):.3f}")
dec = [r for r in recs if abs(mir(r) - .5) > .4]
print(f"  on decisive pairs (|mirror-.5|>.4, n={len(dec)}): {st.mean((yW(r) > .5) == (mir(r) < .5) for r in dec):.3f}")
near = [r for r in recs if abs(mir(r) - .5) <= .1]
if near:
    print(f"  near-ties (|mirror-.5|<=.1, n={len(near)}): mean yA {st.mean(map(yA, near)):.3f} yB {st.mean(map(yB, near)):.3f} "
          f"yW {st.mean(map(yW, near)):.3f} | all three 'no' {sum(yA(r) < .5 and yB(r) < .5 and yW(r) < .5 for r in near)}")
tie = [r for r in recs if yA(r) < .5 and yB(r) < .5]
if tie:
    print(f"  flagged ties (n={len(tie)}): mean yW {st.mean(map(yW, tie)):.3f} | yW<.5 {sum(yW(r) < .5 for r in tie)}")

print("\n== noul_a_shorter: accuracy vs word counts ==")
rows = [(yS(r), la(r) < lb(r), max(la(r), lb(r)) / max(1, min(la(r), lb(r))), mir(r)) for r in recs if la(r) != lb(r)]
for name, lo, hi in [("all", 1, 1e9), ("<1.25x", 1, 1.25), ("1.25-2x", 1.25, 2), (">=2x", 2, 1e9)]:
    sub = [x for x in rows if lo <= x[2] < hi]
    print(f"  {name:8} n={len(sub):5} accuracy={st.mean((p > .5) == short for p, short, _, _ in sub):.3f} "
          f"mean P(correct)={st.mean(p if short else 1 - p for p, short, _, _ in sub):.3f}")
lr = [math.log(la(r) / lb(r)) for r in recs if la(r) != lb(r)]
ps = [x[0] for x in rows]; pm = [x[3] for x in rows]
print(f"  Spearman(yS, log(len_a/len_b)) = {spearmanr(ps, lr)[0]:.3f} (ideal strongly negative)")
print(f"  Spearman(yS, mirror P(A better)) = {spearmanr(ps, pm)[0]:.3f} (strongly negative => answering quality)")
conf = [(p, a, m) for p, a, m in zip(ps, lr, pm) if (a < 0) != (m < .5)]
if conf:
    print(f"  conflict pairs (shorter one ALSO judged better), n={len(conf)}: follows length "
          f"{st.mean((p > .5) == (a < 0) for p, a, m in conf):.3f} vs follows inverse-quality "
          f"{st.mean((p > .5) == (m < .5) for p, a, m in conf):.3f}")

if len(sys.argv) > 2:
    print("\n== test-retest vs noul pilot (same oriented comparison, different question bundle) ==")
    old = {r["comp_id"]: r for r in map(json.loads, open(sys.argv[2]))}
    for q in ("noul_a_better", "noul_b_better"):
        pr = [(y(r, q), old[r["comp_id"]]["raw_p"][q]) for r in recs if r["comp_id"] in old]
        d = [abs(a - b) for a, b in pr]
        print(f"  {q}: n={len(pr)} identical={sum(x == 0 for x in d)/len(d):.3f} mean|dp|={st.mean(d):.4f} "
              f"max={max(d):.3f} winner agree={st.mean(sign(a) == sign(b) for a, b in [(a - .5 + .5, b) for a, b in pr]):.3f}")
    pm2 = [(mir(r), (old[r['comp_id']]['raw_p']['noul_a_better'] + 1 - old[r['comp_id']]['raw_p']['noul_b_better']) / 2)
           for r in recs if r["comp_id"] in old]
    print(f"  mirror: mean|dp|={st.mean(abs(a - b) for a, b in pm2):.4f} winner flips={sum((a - .5) * (b - .5) < 0 for a, b in pm2)}")
