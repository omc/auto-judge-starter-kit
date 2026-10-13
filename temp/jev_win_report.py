"""Window-only report for one Jev pairwise comparisons file (multi-topic aware).
Usage: JEV_DATASET=rag26|ragtime26 python temp/jev_win_report.py <comparisons.jsonl> <score> [gemini comparisons.jsonl]
  score: p_a (primary) | a question id in r["p"] (e.g. better_summary, mirror)

All records are read through window_records (permitted topics only). Leaderboards are the
mean over topics of per-topic win rates, as in the judge. Reports: probability
distribution; probabilistic vs binary ranking; swap/slot statistics (pairs judged in both
orientations); slot bias by closeness; length dependence; split-half stability; agreement
with Gemini on shared topics.
"""
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from jev_dataset import DS, window_records, word_lengths  # noqa: E402

path, score = sys.argv[1], sys.argv[2]
gem_path = sys.argv[3] if len(sys.argv) > 3 else None
recs = [r for r in window_records(path) if r["valid"]]
P = (lambda r: r["p_a"]) if score == "p_a" else (lambda r: r["p"][score])
recs = [r for r in recs if P(r) is not None]
topics = [t for t in DS["window"] if any(r["topic_id"] == t for r in recs)]
ps = [P(r) for r in recs]
uniq = {r["cache_key"]: r for r in recs if r.get("cache_key")}
cost = sum(r.get("cost") or 0 for r in uniq.values())
print(f"{DS['name']} | {Path(path).parent.parent.name} | score={score} | {len(recs):,} comparisons over "
      f"{len(topics)} window topics | unique calls {len(uniq):,} ${cost:.2f} (${cost/max(1,len(uniq)):.7f}/call)")
print(f"distribution: exact 0/1 {sum(p in (0, 1) for p in ps)/len(ps):.3f} | decisive (<.25|>.75) "
      f"{sum(p < .25 or p > .75 for p in ps)/len(ps):.3f} | [.4,.6] {sum(.4 <= p <= .6 for p in ps)/len(ps):.3f} "
      f"| mean P(A) {st.mean(ps):.4f} | distinct values {len(set(ps))}")


def outcome(p):
    return 1.0 if p > .5 else 0.0 if p < .5 else 0.5


def per_topic(f):
    w, g = defaultdict(float), defaultdict(int)
    for r in recs:
        p = f(r); t = r["topic_id"]
        w[(t, r["a_run"])] += p; w[(t, r["b_run"])] += 1 - p
        g[(t, r["a_run"])] += 1; g[(t, r["b_run"])] += 1
    return {k: w[k] / g[k] for k in g}


def board(d, ts):
    s = defaultdict(list)
    for (t, x), v in d.items():
        if t in ts:
            s[x].append(v)
    return {x: st.mean(v) for x, v in s.items()}


soft = per_topic(P)
hard = per_topic(lambda r: outcome(P(r)))
B, H = board(soft, set(topics)), board(hard, set(topics))
runs = sorted(B)
print(f"ranking: {len(runs)} runs | probabilistic vs binary board Spearman "
      f"{spearmanr([B[x] for x in runs], [H[x] for x in runs])[0]:.4f}")

pairs = defaultdict(dict)
for r in recs:
    a, b, p = r["a_run"], r["b_run"], P(r); x, y = sorted((a, b))
    pairs[(r["topic_id"], x, y)]["fwd" if a == x else "rev"] = p if a == x else 1 - p
both = [v for v in pairs.values() if len(v) == 2]
if both:
    flip = lambda v: (v["fwd"] - .5) * (v["rev"] - .5) < 0
    print(f"swap ({len(both):,} pairs in both orientations): mean|dp| {st.mean(abs(v['fwd'] - v['rev']) for v in both):.3f} "
          f"| winner flips {st.mean(flip(v) for v in both):.3f} | P(slot A wins) "
          f"{st.mean((v['fwd'] + 1 - v['rev']) / 2 for v in both):.4f}")
    print("slot bias by closeness of orientation-averaged P (n / P(slot A wins) / flip rate):")
    for lo, hi in [(0, .05), (.05, .25), (.25, .4), (.4, .6), (.6, .75), (.75, .95), (.95, 1.01)]:
        s = [v for v in both if lo <= (v["fwd"] + v["rev"]) / 2 < hi]
        if s:
            print(f"  {lo:.2f}-{hi:.2f}: {len(s):6,} / {st.mean((v['fwd'] + 1 - v['rev']) / 2 for v in s):.3f} "
                  f"/ {st.mean(flip(v) for v in s):.2f}")
else:
    print("swap: single orientation (no pair judged both ways)")

L = word_lengths(set(topics))
lw = [(P(r) if L[r["topic_id"]][r["a_run"]] > L[r["topic_id"]][r["b_run"]] else 1 - P(r))
      for r in recs if L[r["topic_id"]][r["a_run"]] != L[r["topic_id"]][r["b_run"]]]
rho = []
for t in topics:
    xs = [x for x in runs if (t, x) in soft]
    rho.append(spearmanr([soft[(t, x)] for x in xs], [L[t][x] for x in xs])[0])
print(f"length: P(longer wins) {st.mean(lw):.3f} | per-topic Spearman(win rate, words) median "
      f"{st.median(rho):.3f} [min {min(rho):.3f}, max {max(rho):.3f}]")

if len(topics) >= 4:
    sh = []
    for i in range(200):
        rr = random.Random(i); tt = topics[:]; rr.shuffle(tt)
        h1, h2 = board(soft, set(tt[:len(tt) // 2])), board(soft, set(tt[len(tt) // 2:]))
        c = [x for x in runs if x in h1 and x in h2]
        sh.append(spearmanr([h1[x] for x in c], [h2[x] for x in c])[0])
    sh.sort()
    print(f"split-half ({len(topics)//2} vs {len(topics) - len(topics)//2} topics, 200 splits): Spearman "
          f"{st.mean(sh):.3f} [p2.5 {sh[5]:.3f}, p97.5 {sh[194]:.3f}]")

top = sorted(runs, key=lambda x: -B[x])
print("top 5:", ", ".join(f"{x}({B[x]:.3f})" for x in top[:5]))

if gem_path and Path(gem_path).exists():
    gw, gg = defaultdict(int), defaultdict(int)
    for r in window_records(gem_path):
        gg[(r["topic_id"], r["a_run"])] += 1; gg[(r["topic_id"], r["b_run"])] += 1
        if r["valid"]:
            gw[(r["topic_id"], r["winner_run"])] += 1
    gwr = {k: gw[k] / gg[k] for k in gg}
    gt = [t for t in topics if any(k[0] == t for k in gwr)]
    jb, gb = board(soft, set(gt)), board(gwr, set(gt))
    xs = [x for x in runs if x in jb and x in gb]
    per = []
    for t in gt:
        ys = [x for x in runs if (t, x) in soft and (t, x) in gwr]
        per.append(spearmanr([soft[(t, x)] for x in ys], [gwr[(t, x)] for x in ys])[0])
    print(f"vs Gemini on {len(gt)} shared window topics, {len(xs)} runs: board Spearman "
          f"{spearmanr([jb[x] for x in xs], [gb[x] for x in xs])[0]:.3f} Kendall "
          f"{kendalltau([jb[x] for x in xs], [gb[x] for x in xs])[0]:.3f} | per-topic median {st.median(per):.3f}")
