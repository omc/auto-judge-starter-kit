"""Noul vs Choice formulation on the same ordered pilot topic.
Usage: python temp/jev_noul_compare.py <noul comparisons.jsonl> <choice comparisons.jsonl> <gemini comparisons.jsonl>

Estimators of P(summary_a better), from raw Noul P(yes):
  noul_a   : yA                          ("is A better?" alone)
  mirror   : (yA + 1 - yB) / 2           (average with the mirror question)
  ratio    : yA / (yA + yB)              (relative preference; ties -> yA/(yA+yB))
  choice   : Choice better_summary P(A)  (reference)
"""
import json, sys, statistics as st
from collections import defaultdict
from pathlib import Path
from scipy.stats import spearmanr, kendalltau, pearsonr
from jev_dataset import runs_dir  # JEV_DATASET=rag26|ragtime26

noul = [json.loads(l) for l in open(sys.argv[1])]
choice = {r["comp_id"]: r for r in map(json.loads, open(sys.argv[2]))}
noul = [r for r in noul if r["valid"] and r["comp_id"] in choice]
topic = noul[0]["topic_id"]
L = {}
for f in runs_dir().iterdir():
    for line in open(f):
        r = json.loads(line)
        if r["metadata"]["topic_id"] == topic:
            L[r["metadata"]["run_id"]] = len(" ".join(s["text"] for s in r["responses"]).split())

def outcome(p):            # 1 = A wins, 0 = B wins, 0.5 = draw (P exactly 0.5)
    return 1.0 if p > .5 else 0.0 if p < .5 else 0.5

def flipped(f, r):         # winner changes between orientations; draws never "flip"
    return (f - .5) * (r - .5) < 0

def ya(r): return r["raw_p"]["noul_a_better"]
def yb(r): return r["raw_p"]["noul_b_better"]
EST = {
    "choice": lambda r: choice[r["comp_id"]]["p_a"],
    "noul_a": lambda r: ya(r),
    "mirror": lambda r: (ya(r) + 1 - yb(r)) / 2,
    "ratio": lambda r: ya(r) / (ya(r) + yb(r)) if ya(r) + yb(r) > 0 else 0.5,
}

print(f"topic {topic}, ordered comparisons {len(noul)}\n")
print("== raw Noul answers ==")
s = [ya(r) + yb(r) for r in noul]
print(f"mean P(yes) A-better {st.mean(map(ya, noul)):.3f} | B-better {st.mean(map(yb, noul)):.3f}")
print(f"yA + yB: mean {st.mean(s):.3f} median {st.median(s):.3f} | <0.9: {sum(x < .9 for x in s)/len(s):.3f} "
      f"| 0.9-1.1: {sum(.9 <= x <= 1.1 for x in s)/len(s):.3f} | >1.1: {sum(x > 1.1 for x in s)/len(s):.3f}")
both_no = sum(ya(r) < .5 and yb(r) < .5 for r in noul); both_yes = sum(ya(r) > .5 and yb(r) > .5 for r in noul)
print(f"both 'no' (<.5): {both_no} ({both_no/len(noul):.3f}) | both 'yes' (>.5): {both_yes} ({both_yes/len(noul):.3f})")
print(f"Pearson(yA, 1-yB) = {pearsonr([ya(r) for r in noul], [1-yb(r) for r in noul])[0]:.3f}")
print(f"argmax agreement noul_a vs choice: {st.mean(outcome(ya(r)) == outcome(choice[r['comp_id']]['p_a']) for r in noul):.3f}")

# pairs: P(x better) in each orientation for each estimator
def pairs(est):
    d = defaultdict(dict)
    for r in noul:
        a, b = r["a_run"], r["b_run"]; x, y = sorted((a, b)); p = est(r)
        d[(x, y)]["fwd" if a == x else "rev"] = p if a == x else 1 - p
    return {k: v for k, v in d.items() if len(v) == 2}

def winrate(est, avg=True):
    w, g = defaultdict(float), defaultdict(int)
    for r in noul:
        p = est(r); w[r["a_run"]] += p; w[r["b_run"]] += 1 - p; g[r["a_run"]] += 1; g[r["b_run"]] += 1
    return {k: w[k] / g[k] for k in g}

def hard(est):
    return winrate(lambda r: outcome(est(r)))

gw, gg = defaultdict(int), defaultdict(int)
for l in open(sys.argv[3]):
    r = json.loads(l)
    if r["topic_id"] == topic:
        gg[r["a_run"]] += 1; gg[r["b_run"]] += 1
        if r["valid"]: gw[r["winner_run"]] += 1
gem = {k: gw[k] / gg[k] for k in gg}
cwr = winrate(EST["choice"])

print("\n== per estimator ==")
print(f"{'estimator':8} {'mean P(A)':>9} {'exact0/1':>8} {'in.4-.6':>7} {'swap|dp|':>8} {'flips':>6} "
      f"{'P(slotA)':>8} {'soft~hard':>9} {'~choice':>7} {'~gemini':>7} {'P(longer)':>9} {'rho len':>7}")
for name, est in EST.items():
    ps = [est(r) for r in noul]
    pr = pairs(est)
    dp = [abs(v["fwd"] - v["rev"]) for v in pr.values()]
    flips = st.mean(flipped(v["fwd"], v["rev"]) for v in pr.values())
    slotA = st.mean((v["fwd"] + 1 - v["rev"]) / 2 for v in pr.values())
    wr = winrate(est); hw = hard(est); runs = sorted(wr)
    lw = [(est(r) if L[r["a_run"]] > L[r["b_run"]] else 1 - est(r)) for r in noul if L[r["a_run"]] != L[r["b_run"]]]
    print(f"{name:8} {st.mean(ps):9.3f} {sum(p in (0, 1) for p in ps)/len(ps):8.3f} "
          f"{sum(.4 <= p <= .6 for p in ps)/len(ps):7.3f} {st.mean(dp):8.3f} {flips:6.3f} {slotA:8.3f} "
          f"{spearmanr([wr[x] for x in runs], [hw[x] for x in runs])[0]:9.4f} "
          f"{spearmanr([wr[x] for x in runs], [cwr[x] for x in runs])[0]:7.3f} "
          f"{spearmanr([wr[x] for x in runs if x in gem], [gem[x] for x in runs if x in gem])[0]:7.3f} "
          f"{st.mean(lw):9.3f} {spearmanr([wr[x] for x in runs], [L[x] for x in runs])[0]:7.3f}")
print("  swap|dp| / flips / P(slotA): over pairs judged in both orientations; P(slotA) ideal 0.5")

print("\n== slot bias by pair closeness (orientation-averaged P(x better)) ==")
bins = [(0, .05), (.05, .25), (.25, .4), (.4, .6), (.6, .75), (.75, .95), (.95, 1.01)]
print(f"{'estimator':8} " + " ".join(f"{f'{lo:.2f}-{hi:.2f}':>16}" for lo, hi in bins))
for name, est in EST.items():
    pr = pairs(est); cells = []
    for lo, hi in bins:
        s_ = [v for v in pr.values() if lo <= (v["fwd"] + v["rev"]) / 2 < hi]
        cells.append(f"{len(s_):4} {st.mean((v['fwd'] + 1 - v['rev'])/2 for v in s_):.3f}/{st.mean(flipped(v['fwd'], v['rev']) for v in s_):.2f}" if s_ else f"{'-':>16}")
    print(f"{name:8} " + " ".join(f"{c:>16}" for c in cells))
print("  cell = n_pairs P(slot A wins)/winner-flip-rate")

print("\n== orientation-averaged ranking agreement (both orientations per pair) ==")
for n1 in EST:
    print(f"{n1:8} " + " ".join(f"{n2}={spearmanr([winrate(EST[n1])[x] for x in sorted(cwr)], [winrate(EST[n2])[x] for x in sorted(cwr)])[0]:.3f}" for n2 in EST if n2 != n1))

print("\n== top 8 (win rate, words) ==")
for name, est in EST.items():
    wr = winrate(est); top = sorted(wr, key=lambda x: -wr[x])[:8]
    print(f"{name:8} " + ", ".join(f"{x}({wr[x]:.2f},{L[x]})" for x in top))
