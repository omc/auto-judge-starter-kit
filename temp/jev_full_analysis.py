"""Report on a FULL Jev pairwise run (both orientations, mirror score).
Usage: JEV_DATASET=rag26|ragtime26 python temp/jev_full_analysis.py <out-dir> [gemini comparisons.jsonl]

DATA POLICY (CLAUDE.md, rule 4): over the whole evaluation set we only read the SCORED
outputs (<filebase>.eval.txt, run_manifest.json), and only to (a) verify the run completed
and (b) report the leaderboard. Every diagnostic -- probability distribution, ties, slot
bias, length dependence, ranking stability, agreement with Gemini -- is computed on the
PERMITTED WINDOW topics only (first 10 of the topics file), from comparisons.jsonl lines
filtered by topic id. These diagnostics are development feedback, not a basis for choosing
the submitted judge on evaluation-set numbers.
"""
import json, random, statistics as st, sys
from collections import defaultdict
from pathlib import Path
from scipy.stats import kendalltau, spearmanr
sys.path.insert(0, str(Path(__file__).parent))
from jev_dataset import DS, in_window, window, window_records, word_lengths  # noqa: E402

out = Path(sys.argv[1])
gem_path = sys.argv[2] if len(sys.argv) > 2 else None
art = next(out.glob("*.pairwise"))
eval_txt = next(out.glob("*.eval.txt"))
WIN = window()

# =========================== (a) completeness (scored outputs) ===========================
man = json.loads((art / "run_manifest.json").read_text())
rows = [l.rstrip("\n").split("\t") for l in open(eval_txt)]
runs = sorted({r for r, *_ in rows}); topics = sorted({t for _, t, *_ in rows} - {"all"})
measures = sorted({m for _, _, m, _ in rows})
have = {(r, t, m) for r, t, m, _ in rows}
complete = all((r, t, m) in have for r in runs for t in topics for m in measures)
print(f"== {DS['name']} full run: {man['n_comparisons']:,} comparisons, {man['n_valid']:,} valid, "
      f"{man['n_missing']:,} missing, {man['n_malformed']:,} malformed | served {man['served_models']}")
print(f"leaderboard {eval_txt.name}: {len(runs)} runs x {len(topics)} topics x {len(measures)} measures "
      f"| complete (every run x topic x measure): {complete}")

# =========================== (b) leaderboard report (scored outputs) =====================
allwr = {r: float(v) for r, t, m, v in rows if t == "all" and m == "PAIRWISE_WINRATE"}
order = sorted(allwr, key=lambda x: -allwr[x])
print("\nleaderboard (PAIRWISE_WINRATE, 'all' rows):")
for i, x in enumerate(order[:12], 1):
    print(f"  {i:2} {x:8} {allwr[x]:.4f}")
print("  ...", ", ".join(f"{len(order)-k} {x} {allwr[x]:.3f}" for k, x in enumerate(reversed(order[-3:]))))

# =========================== (c) diagnostics: PERMITTED WINDOW ONLY =======================
print(f"\n== diagnostics on the permitted window only: {WIN[0]} .. {WIN[-1]} ==")
recs = [r for r in window_records(art / "comparisons.jsonl") if r["valid"]]
wt = sorted({r["topic_id"] for r in recs})
ew = defaultdict(float); gm = defaultdict(int); team = {}; ties = defaultdict(int); both_yes = 0
pair = {}; ps = []
for r in recs:
    t, a, b, p = r["topic_id"], r["a_run"], r["b_run"], r["p_a"]
    team[a], team[b] = r["a_team"], r["b_team"]
    ew[(t, a)] += p; ew[(t, b)] += 1 - p; gm[(t, a)] += 1; gm[(t, b)] += 1
    ties[t] += bool(r.get("tie")); both_yes += bool(r.get("both_yes")); ps.append(p)
    x, y = sorted((a, b)); v = pair.setdefault((t, x, y), [None, None])
    v[0 if a == x else 1] = p if a == x else 1 - p
n = len(ps); wruns = sorted(team)
print(f"window comparisons {n:,} over {len(wt)} topics, {len(wruns)} runs")
print(f"decisive (<.25|>.75) {sum(p < .25 or p > .75 for p in ps)/n:.3f} | in [.4,.6] "
      f"{sum(.4 <= p <= .6 for p in ps)/n:.3f} | near-ties {sum(ties.values())/n:.2%} | both-yes {both_yes/n:.2%}")

wr_t = {k: ew[k] / gm[k] for k in gm}
def board(tset):
    s, c = defaultdict(float), defaultdict(int)
    for (t, x), w in wr_t.items():
        if t in tset: s[x] += w; c[x] += 1
    return {x: s[x] / c[x] for x in s}
B = board(set(wt))
sh = []
for i in range(200):
    rr = random.Random(i); tt = wt[:]; rr.shuffle(tt)
    h1, h2 = board(set(tt[: len(tt) // 2])), board(set(tt[len(tt) // 2:]))
    common = [x for x in wruns if x in h1 and x in h2]
    sh.append(spearmanr([h1[x] for x in common], [h2[x] for x in common])[0])
print(f"split-half over window topics (5 vs 5): Spearman mean {st.mean(sh):.3f} "
      f"[p2.5 {sorted(sh)[5]:.3f}, p97.5 {sorted(sh)[194]:.3f}]")

both = [v for v in pair.values() if None not in v]
print(f"slot bias: P(slot A wins) {st.mean((f + 1 - rv) / 2 for f, rv in both):.4f} | close pairs (.4-.6) "
      f"{st.mean([(f + 1 - rv) / 2 for f, rv in both if .4 <= (f + rv) / 2 < .6] or [float('nan')]):.3f} | "
      f"winner flips on swap {st.mean((f - .5) * (rv - .5) < 0 for f, rv in both):.3f}")

L = word_lengths(set(wt))
lw = [((f + rv) / 2 if L[t][x] > L[t][y] else 1 - (f + rv) / 2)
      for (t, x, y), (f, rv) in pair.items() if None not in (f, rv) and L[t][x] != L[t][y]]
rho_t = [spearmanr([wr_t[(t, x)] for x in wruns if (t, x) in wr_t],
                   [L[t][x] for x in wruns if (t, x) in wr_t])[0] for t in wt]
print(f"length: P(longer wins) {st.mean(lw):.3f} | per-topic Spearman(win rate, words) median {st.median(rho_t):.3f}")

if gem_path and Path(gem_path).exists():
    gw, gg = defaultdict(int), defaultdict(int)
    for r in window_records(gem_path):                 # permitted topics only
        gg[(r["topic_id"], r["a_run"])] += 1; gg[(r["topic_id"], r["b_run"])] += 1
        if r["valid"]: gw[(r["topic_id"], r["winner_run"])] += 1
    gt = sorted({t for t, _ in gg} & set(wt))
    gwr = {k: gw[k] / gg[k] for k in gg}
    per = []
    for t in gt:
        xs = [x for x in wruns if (t, x) in wr_t and (t, x) in gwr]
        a, b = [wr_t[(t, x)] for x in xs], [gwr[(t, x)] for x in xs]
        if len(set(b)) > 1: per.append(spearmanr(a, b)[0])
    jb = board(set(gt)); gb = defaultdict(list)
    for (t, x), v in gwr.items():
        if t in gt: gb[x].append(v)
    gbm = {x: st.mean(v) for x, v in gb.items()}
    common = [x for x in wruns if x in gbm and x in jb]
    print(f"Gemini agreement on {len(gt)} window topics: board Spearman "
          f"{spearmanr([jb[x] for x in common], [gbm[x] for x in common])[0]:.3f} Kendall "
          f"{kendalltau([jb[x] for x in common], [gbm[x] for x in common])[0]:.3f} | per-topic median "
          f"{st.median(per):.3f} (n={len(per)})")
