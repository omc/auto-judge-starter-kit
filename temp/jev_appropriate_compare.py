"""Paired comparison of two mirrored Noul pairs asked in the SAME request.
Usage: JEV_DATASET=rag26|ragtime26 python temp/jev_appropriate_compare.py <comparisons.jsonl> [gemini comparisons.jsonl]

Pairs: better = (noul_a_better, noul_b_better), appropriate = (noul_a_appropriate, noul_b_appropriate).
Score per pair = mirror average (yA + 1 - yB) / 2 from raw P(yes).
"""
import json, statistics as st, sys
from collections import defaultdict
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
sys.path.insert(0, str(Path(__file__).parent))
from jev_dataset import word_lengths  # noqa: E402

PAIRS = {"better": ("noul_a_better", "noul_b_better"),
         "appropriate": ("noul_a_appropriate", "noul_b_appropriate")}
recs = [r for r in map(json.loads, open(sys.argv[1])) if r["valid"] and r["result"] == "ok"]
topic = recs[0]["topic_id"]
L = word_lengths({topic})[topic]
y = lambda r, q: r["raw_p"][q]
mir = lambda r, k: (y(r, PAIRS[k][0]) + 1 - y(r, PAIRS[k][1])) / 2
print(f"topic {topic} | ordered comparisons {len(recs)} | words median {st.median(L.values())} "
      f"min {min(L.values())} max {max(L.values())}\n")

def winrate(k):
    w, g = defaultdict(float), defaultdict(int)
    for r in recs:
        p = mir(r, k); w[r["a_run"]] += p; w[r["b_run"]] += 1 - p; g[r["a_run"]] += 1; g[r["b_run"]] += 1
    return {x: w[x] / g[x] for x in g}

W = {k: winrate(k) for k in PAIRS}
runs = sorted(W["better"])
print(f"{'pair':12} {'yA+yB':>6} {'both-no':>8} {'both-yes':>8} {'decisive':>8} {'[.4,.6]':>7} "
      f"{'P(longer)':>9} {'<1.25x':>7} {'1.25-2x':>8} {'>=2x':>6} {'rho len':>7} {'P(slotA)':>8}")
for k, (qa, qb) in PAIRS.items():
    s = [y(r, qa) + y(r, qb) for r in recs]
    ps = [mir(r, k) for r in recs]
    lw, bins = [], defaultdict(list)
    for r in recs:
        la, lb = L[r["a_run"]], L[r["b_run"]]
        if la == lb: continue
        pl = mir(r, k) if la > lb else 1 - mir(r, k); lw.append(pl)
        ratio = max(la, lb) / max(1, min(la, lb))
        bins["a" if ratio < 1.25 else "b" if ratio < 2 else "c"].append(pl)
    pr = defaultdict(dict)
    for r in recs:
        a, b, p = r["a_run"], r["b_run"], mir(r, k); x, z = sorted((a, b))
        pr[(x, z)]["f" if a == x else "r"] = p if a == x else 1 - p
    slot = st.mean((v["f"] + 1 - v["r"]) / 2 for v in pr.values() if len(v) == 2)
    rho = spearmanr([W[k][x] for x in runs], [L[x] for x in runs])[0]
    print(f"{k:12} {st.mean(s):6.3f} {sum(y(r, qa) < .5 and y(r, qb) < .5 for r in recs)/len(recs):8.3f} "
          f"{sum(y(r, qa) > .5 and y(r, qb) > .5 for r in recs)/len(recs):8.3f} "
          f"{sum(p < .25 or p > .75 for p in ps)/len(ps):8.3f} {sum(.4 <= p <= .6 for p in ps)/len(ps):7.3f} "
          f"{st.mean(lw):9.3f} {st.mean(bins['a']):7.3f} {st.mean(bins['b']):8.3f} {st.mean(bins['c']) if bins['c'] else float('nan'):6.3f} "
          f"{rho:7.3f} {slot:8.3f}")

pb = [mir(r, "better") for r in recs]; pa = [mir(r, "appropriate") for r in recs]
print(f"\nper-comparison: Pearson(better, appropriate) {pearsonr(pb, pa)[0]:.3f} | mean |diff| "
      f"{st.mean(abs(a - b) for a, b in zip(pa, pb)):.3f} | winner differs "
      f"{sum((a - .5) * (b - .5) < 0 for a, b in zip(pa, pb))/len(pa):.3f}")
# when the two disagree, which way does length go?
d = [(a - b, L[r['a_run']] - L[r['b_run']]) for r, a, b in zip(recs, pa, pb) if L[r['a_run']] != L[r['b_run']]]
shift = [da if dl > 0 else -da for da, dl in d]   # + = 'appropriate' moves toward the LONGER summary
print(f"mean shift toward the longer summary (appropriate - better): {st.mean(shift):+.4f} | "
      f"Spearman(shift_A, log len ratio) {spearmanr([x for x, _ in d], [dl for _, dl in d])[0]:+.3f}")
print(f"run ranking: Spearman(better, appropriate) {spearmanr([W['better'][x] for x in runs], [W['appropriate'][x] for x in runs])[0]:.4f}")

# runs whose rank moved most, with length
rb = {x: i for i, x in enumerate(sorted(runs, key=lambda x: -W['better'][x]), 1)}
ra = {x: i for i, x in enumerate(sorted(runs, key=lambda x: -W['appropriate'][x]), 1)}
mv = sorted(runs, key=lambda x: -abs(rb[x] - ra[x]))[:8]
print("largest rank moves (better rank -> appropriate rank, words):",
      ", ".join(f"{x} {rb[x]}->{ra[x]} ({L[x]}w)" for x in mv))
print("winrate change vs words: Spearman(appropriate - better, words) "
      f"{spearmanr([W['appropriate'][x] - W['better'][x] for x in runs], [L[x] for x in runs])[0]:+.3f}")
for k in PAIRS:
    top = sorted(runs, key=lambda x: -W[k][x])[:8]
    print(f"top 8 {k:12}: " + ", ".join(f"{x}({W[k][x]:.2f},{L[x]}w)" for x in top))

if len(sys.argv) > 2 and Path(sys.argv[2]).exists():
    gw, gg = defaultdict(int), defaultdict(int)
    for l in open(sys.argv[2]):
        r = json.loads(l)
        if r["topic_id"] != topic: continue
        gg[r["a_run"]] += 1; gg[r["b_run"]] += 1
        if r["valid"]: gw[r["winner_run"]] += 1
    common = [x for x in runs if gg.get(x)]
    for k in PAIRS:
        print(f"vs Gemini ({k}): Spearman {spearmanr([W[k][x] for x in common], [gw[x]/gg[x] for x in common])[0]:.3f}")
