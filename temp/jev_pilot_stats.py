"""Section 4 / 8.3 pilot statistics for an ordered (both-orientation) Jev pilot.
Usage: JEV_DATASET=rag26|ragtime26 python temp/jev_pilot_stats.py <pilot comparisons.jsonl> [gemini comparisons.jsonl] [--score mirror]

Reports: P distribution and saturation, probabilistic-vs-binary ranking, position
effects on swap, slot-A bias by pair closeness, agreement with Gemini, top/bottom runs.
"""
import json, statistics as st, sys
from collections import defaultdict
from pathlib import Path
from scipy.stats import kendalltau, spearmanr

args = [a for a in sys.argv[1:] if not a.startswith("--")]
key = "mirror" if "--score" in sys.argv and sys.argv[sys.argv.index("--score") + 1] == "mirror" else None
recs = [r for r in map(json.loads, open(args[0])) if r["valid"]]
P = (lambda r: r["p"][key]) if key else (lambda r: r["p_a"])
topic = recs[0]["topic_id"]
ps = [P(r) for r in recs]
print(f"pilot {args[0]} | topic {topic} | score {key or 'p_a (primary)'} | n={len(recs)}")
calls = sum(1 for r in recs if r.get("cost"))
cost = sum(r["cost"] or 0 for r in recs)
uniq = {r["cache_key"]: r for r in recs if r.get("cache_key")}
print(f"unique calls {len(uniq):,} | cost (unique) ${sum(r['cost'] or 0 for r in uniq.values()):.4f} "
      f"| $/call {sum(r['cost'] or 0 for r in uniq.values())/max(1,len(uniq)):.7f} "
      f"| mean input tokens {st.mean(r['input_tokens'] for r in uniq.values()):.0f}")

bins = [0, .01, .1, .25, .4, .6, .75, .9, .99, 1.0001]
h = [sum(bins[i] <= p < bins[i + 1] for p in ps) for i in range(len(bins) - 1)]
print("histogram:", ", ".join(f"[{bins[i]:.2f},{min(bins[i+1],1):.2f}):{h[i]}" for i in range(len(h))))
print(f"exact 0/1 {sum(p in (0, 1) for p in ps)/len(ps):.3f} | decisive (<.25|>.75) {sum(p < .25 or p > .75 for p in ps)/len(ps):.3f} "
      f"| [.25,.75] {sum(.25 <= p <= .75 for p in ps)/len(ps):.3f} | [.4,.6] {sum(.4 <= p <= .6 for p in ps)/len(ps):.3f} "
      f"| mean P(A) {st.mean(ps):.4f} | decisiveness {st.mean(abs(2*p-1) for p in ps):.3f} | distinct {len(set(ps))}")

def outcome(p): return 1.0 if p > .5 else 0.0 if p < .5 else 0.5
soft, hard, g, team = defaultdict(float), defaultdict(float), defaultdict(int), {}
for r in recs:
    p = P(r)
    for run, q in ((r["a_run"], p), (r["b_run"], 1 - p)):
        soft[run] += q; hard[run] += outcome(q); g[run] += 1
    team[r["a_run"]], team[r["b_run"]] = r["a_team"], r["b_team"]
runs = sorted(g); wr = {x: soft[x] / g[x] for x in runs}
print(f"soft vs binary win-rate Spearman {spearmanr([wr[x] for x in runs], [hard[x]/g[x] for x in runs])[0]:.4f}")

pairs = defaultdict(dict)
for r in recs:
    a, b, p = r["a_run"], r["b_run"], P(r); x, y = sorted((a, b))
    pairs[(x, y)]["fwd" if a == x else "rev"] = p if a == x else 1 - p
both = [v for v in pairs.values() if len(v) == 2]
d = [abs(v["fwd"] - v["rev"]) for v in both]
flip = lambda v: (v["fwd"] - .5) * (v["rev"] - .5) < 0
print(f"pairs both orientations {len(both)} | swap mean|dp| {st.mean(d):.3f} median {st.median(d):.3f} "
      f"| winner flips {st.mean(flip(v) for v in both):.3f} | P(slot A wins) {st.mean((v['fwd'] + 1 - v['rev'])/2 for v in both):.4f}")
print("slot bias by closeness (n / P(slot A wins) / flip rate):")
for lo, hi in [(0, .05), (.05, .25), (.25, .4), (.4, .6), (.6, .75), (.75, .95), (.95, 1.01)]:
    s = [v for v in both if lo <= (v["fwd"] + v["rev"]) / 2 < hi]
    if s:
        print(f"  {lo:.2f}-{hi:.2f}: {len(s):5} / {st.mean((v['fwd'] + 1 - v['rev'])/2 for v in s):.3f} / {st.mean(flip(v) for v in s):.2f}")

if len(args) > 1 and Path(args[1]).exists():
    gw, gg = defaultdict(int), defaultdict(int)
    for l in open(args[1]):
        r = json.loads(l)
        if r["topic_id"] != topic: continue
        gg[r["a_run"]] += 1; gg[r["b_run"]] += 1
        if r["valid"]: gw[r["winner_run"]] += 1
    common = [x for x in runs if gg.get(x)]
    x = [wr[k] for k in common]; y = [gw[k] / gg[k] for k in common]
    print(f"Jev vs Gemini ({len(common)} runs, {sum(gg.values())//2} Gemini comparisons): "
          f"Spearman {spearmanr(x, y)[0]:.3f} Kendall {kendalltau(x, y)[0]:.3f}")
    gtop = sorted(common, key=lambda k: -gw[k] / gg[k])
    print("Gemini top 5:", ", ".join(f"{k}({gw[k]/gg[k]:.3f},{team[k]})" for k in gtop[:5]),
          "| bottom:", ", ".join(f"{k}({gw[k]/gg[k]:.3f})" for k in gtop[-2:]))
top = sorted(runs, key=lambda k: -wr[k])
print("Jev top 5:", ", ".join(f"{k}({wr[k]:.3f},{team[k]})" for k in top[:5]),
      "| bottom:", ", ".join(f"{k}({wr[k]:.3f})" for k in top[-2:]))
