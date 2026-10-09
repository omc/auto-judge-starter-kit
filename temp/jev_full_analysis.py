"""Analysis of a FULL Jev pairwise run (both orientations, mirror score).
Usage: JEV_DATASET=rag26|ragtime26 python temp/jev_full_analysis.py <comparisons.jsonl> [gemini comparisons.jsonl]

Streams the comparisons (rag26 has ~760k) and reports: operations/cost, leaderboard
(runs and best-run-per-team), ranking stability over topics (bootstrap rank CIs and
split-half reliability), agreement with Gemini, decisiveness/ties, slot bias, length.
"""
import json, random, statistics as st, sys
from collections import defaultdict
from pathlib import Path
from scipy.stats import kendalltau, spearmanr
sys.path.insert(0, str(Path(__file__).parent))
from jev_dataset import DS, word_lengths  # noqa: E402

path = sys.argv[1]
gem_path = sys.argv[2] if len(sys.argv) > 2 else None

# ---- stream: per (topic, run) expected wins/games; per-pair orientation map; costs ----
ew = defaultdict(float); gm = defaultdict(int); team = {}
ties = defaultdict(int); both_yes = 0; n = 0
pair = {}                       # (t, x, y) -> [p_x fwd, p_x rev]
uniq_cost = {}; mean_pa = 0.0
hist = [0] * 10
dec = mid = exact = 0
for line in open(path):
    r = json.loads(line)
    if not r["valid"]:
        continue
    n += 1
    t, a, b, p = r["topic_id"], r["a_run"], r["b_run"], r["p_a"]
    team[a], team[b] = r["a_team"], r["b_team"]
    ew[(t, a)] += p; ew[(t, b)] += 1 - p; gm[(t, a)] += 1; gm[(t, b)] += 1
    ties[t] += bool(r.get("tie")); both_yes += bool(r.get("both_yes"))
    mean_pa += p; hist[min(9, int(p * 10))] += 1
    dec += p < .25 or p > .75; mid += .4 <= p <= .6; exact += p in (0.0, 1.0)
    if r.get("cache_key"):
        uniq_cost[r["cache_key"]] = r.get("cost") or 0.0
    x, y = sorted((a, b))
    v = pair.setdefault((t, x, y), [None, None])
    v[0 if a == x else 1] = p if a == x else 1 - p
topics = sorted({t for t, _ in gm})
runs = sorted(team)
print(f"== {DS['name']}: {n:,} valid comparisons, {len(topics)} topics, {len(runs)} runs, "
      f"{len(set(team.values()))} teams ==")
print(f"unique calls {len(uniq_cost):,} | cost ${sum(uniq_cost.values()):.2f} "
      f"(${sum(uniq_cost.values())/len(uniq_cost):.7f}/call) | mean P(A) {mean_pa/n:.4f}")
print("P(A) by decile:", hist)
print(f"decisive (<.25|>.75) {dec/n:.3f} | in [.4,.6] {mid/n:.3f} | exact 0/1 {exact/n:.3f} | flagged ties "
      f"{sum(ties.values()):,} ({sum(ties.values())/n:.2%}) | both-yes {both_yes:,} ({both_yes/n:.2%})")
tie_rate = sorted(((ties[t] / (sum(gm[(t, x)] for x in runs if (t, x) in gm) / 2), t) for t in topics), reverse=True)
print("topic tie rates: median", f"{st.median(x for x, _ in tie_rate):.3%}", "| top 5:",
      ", ".join(f"{t} {x:.1%}" for x, t in tie_rate[:5]))

# ---- leaderboard: mean over topics of per-topic win rate (each topic weighs equally) ----
wr_t = {k: ew[k] / gm[k] for k in gm}
def board(tset):
    s, c = defaultdict(float), defaultdict(int)
    for (t, x), w in wr_t.items():
        if t in tset:
            s[x] += w; c[x] += 1
    return {x: s[x] / c[x] for x in s}
B = board(set(topics))
order = sorted(B, key=lambda x: -B[x])
# bootstrap topics -> rank CI
rnd = random.Random(0); ranks = defaultdict(list)
for _ in range(500):
    bb = board(set(rnd.choices(topics, k=len(topics))))
    for i, x in enumerate(sorted(bb, key=lambda x: -bb[x]), 1):
        ranks[x].append(i)
ci = {x: (sorted(v)[12], sorted(v)[487]) for x, v in ranks.items()}
print("\nleaderboard (mean per-topic win rate; 95% bootstrap rank CI over topics):")
for i, x in enumerate(order[:12], 1):
    print(f"  {i:2} {x:8} {team[x]:6} {B[x]:.4f}  rank CI [{ci[x][0]},{ci[x][1]}]")
print("  ...", ", ".join(f"{len(order)-k} {x} {B[x]:.3f}" for k, x in enumerate(reversed(order[-3:]))))
tb = defaultdict(float)
for x in runs:
    tb[team[x]] = max(tb[team[x]], B[x])
print("teams by best run:", ", ".join(f"{tm} {v:.3f}" for tm, v in sorted(tb.items(), key=lambda z: -z[1])[:10]))
widths = [ci[x][1] - ci[x][0] for x in runs]
print(f"rank-CI width: median {st.median(widths)} | top-10 runs median "
      f"{st.median(ci[x][1] - ci[x][0] for x in order[:10])}")

# split-half reliability over topics
sh = []
for i in range(200):
    rr = random.Random(i); tt = topics[:]; rr.shuffle(tt)
    h1, h2 = board(set(tt[: len(tt) // 2])), board(set(tt[len(tt) // 2:]))
    sh.append(spearmanr([h1[x] for x in runs], [h2[x] for x in runs])[0])
print(f"split-half (topics) Spearman: mean {st.mean(sh):.3f} [p2.5 {sorted(sh)[5]:.3f}, p97.5 {sorted(sh)[194]:.3f}]")
# per-topic agreement with the overall board
pt = [spearmanr([wr_t[(t, x)] for x in runs if (t, x) in wr_t], [B[x] for x in runs if (t, x) in wr_t])[0] for t in topics]
print(f"per-topic vs overall Spearman: median {st.median(pt):.3f} IQR [{sorted(pt)[len(pt)//4]:.3f}, {sorted(pt)[3*len(pt)//4]:.3f}] "
      f"min {min(pt):.3f}")

# ---- slot bias (both orientations) ----
both = [v for v in pair.values() if None not in v]
slot = st.mean((f + 1 - rv) / 2 for f, rv in both)
print(f"\nslot bias: pairs with both orientations {len(both):,} | P(slot A wins) {slot:.4f} | "
      f"winner flips on swap {st.mean((f - .5) * (rv - .5) < 0 for f, rv in both):.3f}")
for lo, hi in [(0, .05), (.05, .25), (.25, .4), (.4, .6), (.6, .75), (.75, .95), (.95, 1.01)]:
    s = [(f, rv) for f, rv in both if lo <= (f + rv) / 2 < hi]
    if s:
        print(f"  {lo:.2f}-{hi:.2f}: {len(s):7,} / P(slot A) {st.mean((f + 1 - rv) / 2 for f, rv in s):.3f} / "
              f"flips {st.mean((f - .5) * (rv - .5) < 0 for f, rv in s):.2f}")

# ---- length ----
L = word_lengths(set(topics))
lw, bins = [], defaultdict(list)
for (t, x, y), (f, rv) in pair.items():
    if None in (f, rv) or L[t][x] == L[t][y]:
        continue
    px = (f + rv) / 2                                  # orientation-averaged P(x better)
    pl = px if L[t][x] > L[t][y] else 1 - px
    lw.append(pl)
    ratio = max(L[t][x], L[t][y]) / max(1, min(L[t][x], L[t][y]))
    bins["<1.25x" if ratio < 1.25 else "1.25-2x" if ratio < 2 else ">=2x"].append(pl)
rho_t = [spearmanr([wr_t[(t, x)] for x in runs if (t, x) in wr_t], [L[t][x] for x in runs if (t, x) in wr_t])[0]
         for t in topics]
mw = {x: st.mean(L[t][x] for t in topics if x in L[t]) for x in runs}
print(f"\nlength: P(longer wins) {st.mean(lw):.3f} | " + " | ".join(f"{k} {st.mean(v):.3f} (n={len(v):,})"
      for k, v in sorted(bins.items())))
print(f"  per-topic Spearman(win rate, words): median {st.median(rho_t):.3f} IQR "
      f"[{sorted(rho_t)[len(rho_t)//4]:.3f}, {sorted(rho_t)[3*len(rho_t)//4]:.3f}] | "
      f"run-level Spearman(board, mean words) {spearmanr([B[x] for x in runs], [mw[x] for x in runs])[0]:.3f}")

# ---- agreement with Gemini ----
if gem_path and Path(gem_path).exists():
    gw, gg = defaultdict(int), defaultdict(int)
    for line in open(gem_path):
        r = json.loads(line)
        gg[(r["topic_id"], r["a_run"])] += 1; gg[(r["topic_id"], r["b_run"])] += 1
        if r["valid"]:
            gw[(r["topic_id"], r["winner_run"])] += 1
    gtopics = sorted({t for t, _ in gg} & set(topics))
    gwr = {k: gw[k] / gg[k] for k in gg}
    per = [spearmanr([wr_t[(t, x)] for x in runs if (t, x) in wr_t and (t, x) in gwr],
                     [gwr[(t, x)] for x in runs if (t, x) in wr_t and (t, x) in gwr])[0] for t in gtopics]
    jb, gb = board(set(gtopics)), defaultdict(list)
    for (t, x), v in gwr.items():
        if t in gtopics: gb[x].append(v)
    gbm = {x: st.mean(v) for x, v in gb.items()}
    common = [x for x in runs if x in gbm]
    print(f"\nGemini: {len(gtopics)} shared topics | overall board Spearman "
          f"{spearmanr([jb[x] for x in common], [gbm[x] for x in common])[0]:.3f} Kendall "
          f"{kendalltau([jb[x] for x in common], [gbm[x] for x in common])[0]:.3f} | per-topic median "
          f"{st.median(per):.3f} IQR [{sorted(per)[len(per)//4]:.3f}, {sorted(per)[3*len(per)//4]:.3f}] min {min(per):.3f}")
    gtop = sorted(common, key=lambda x: -gbm[x])
    jtop = sorted(common, key=lambda x: -jb[x])
    print("  Gemini top 8:", ", ".join(f"{x}({team[x]})" for x in gtop[:8]))
    print("  Jev    top 8:", ", ".join(f"{x}({team[x]})" for x in jtop[:8]))
    print(f"  top-10 overlap {len(set(gtop[:10]) & set(jtop[:10]))}/10 | top-20 overlap {len(set(gtop[:20]) & set(jtop[:20]))}/20")
    diff = sorted(common, key=lambda x: -abs(jtop.index(x) - gtop.index(x)))[:6]
    print("  largest rank differences (Jev -> Gemini):", ", ".join(
        f"{x} {jtop.index(x)+1}->{gtop.index(x)+1} ({mw[x]:.0f}w)" for x in diff))
