"""Depth-adaptivity test: the same report pairs judged under a broad request and under a
narrowed one-sub-question request, with three mirrored Noul pairs in each request.
Usage: JEV_DATASET=rag26|ragtime26 python temp/jev_proportionate_compare.py <broad comparisons.jsonl> <narrow comparisons.jsonl>

For each pair (better / appropriate / proportionate), score = mirror average. "Longer
preference" of a comparison = P(the longer summary wins). Reports, per topic and pooled:
  - P(longer) under broad and narrow, and the shift narrow - broad
  - difference-in-differences vs 'better': (pair shift) - (better shift) on the SAME
    comparisons, with a bootstrap CI over comparisons. Negative = the pair moves toward
    shorter answers on the narrow request MORE than 'better' does (the intended behaviour).
  - run-ranking agreement broad vs narrow, and Spearman(win rate, words)
"""
import json, random, statistics as st, sys
from collections import defaultdict
from pathlib import Path
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).parent))
from jev_dataset import word_lengths  # noqa: E402

PAIRS = {"better": ("noul_a_better", "noul_b_better"),
         "appropriate": ("noul_a_appropriate", "noul_b_appropriate"),
         "proportionate": ("noul_a_proportionate", "noul_b_proportionate")}
load = lambda p: {r["comp_id"]: r for r in map(json.loads, open(p)) if r["valid"] and r["result"] == "ok"}
B, N = load(sys.argv[1]), load(sys.argv[2])
ids = sorted(set(B) & set(N))
topics = sorted({B[i]["topic_id"] for i in ids})
L = word_lengths(set(topics))
mir = lambda r, k: (r["raw_p"][PAIRS[k][0]] + 1 - r["raw_p"][PAIRS[k][1]]) / 2

def longer_pref(r, k):
    la, lb = L[r["topic_id"]][r["a_run"]], L[r["topic_id"]][r["b_run"]]
    if la == lb: return None
    p = mir(r, k)
    return p if la > lb else 1 - p

def boot(xs, n=4000, seed=0):
    rnd = random.Random(seed); m = sorted(st.mean(rnd.choices(xs, k=len(xs))) for _ in range(n))
    return m[int(.025 * n)], m[int(.975 * n)]

def winrate(R, k, t):
    w, g = defaultdict(float), defaultdict(int)
    for r in R.values():
        if r["topic_id"] != t: continue
        p = mir(r, k); w[r["a_run"]] += p; w[r["b_run"]] += 1 - p; g[r["a_run"]] += 1; g[r["b_run"]] += 1
    return {x: w[x] / g[x] for x in g}

print(f"matched comparisons {len(ids)} (broad {len(B)}, narrow {len(N)}) | topics {topics}\n")
for scope in topics + ["pooled"]:
    sel = [i for i in ids if scope == "pooled" or B[i]["topic_id"] == scope]
    print(f"== {scope} (n={len(sel)}) ==")
    print(f"{'pair':14} {'P(longer) broad':>15} {'narrow':>7} {'shift':>7} {'DiD vs better [95% CI]':>28} "
          f"{'rho len broad':>13} {'narrow':>7} {'rank broad~narrow':>17} {'tie% b/n':>9}")
    for k in PAIRS:
        rows = [(longer_pref(B[i], k), longer_pref(N[i], k), longer_pref(B[i], "better"), longer_pref(N[i], "better"))
                for i in sel]
        rows = [x for x in rows if x[0] is not None]
        pb, pn = st.mean(x[0] for x in rows), st.mean(x[1] for x in rows)
        did = [(x[1] - x[0]) - (x[3] - x[2]) for x in rows]
        lo, hi = boot(did) if k != "better" else (0.0, 0.0)
        tl = [t for t in topics if scope in (t, "pooled")]
        rb = st.mean(spearmanr([winrate(B, k, t)[x] for x in sorted(L[t]) if x in winrate(B, k, t)],
                               [L[t][x] for x in sorted(L[t]) if x in winrate(B, k, t)])[0] for t in tl)
        rn = st.mean(spearmanr([winrate(N, k, t)[x] for x in sorted(L[t]) if x in winrate(N, k, t)],
                               [L[t][x] for x in sorted(L[t]) if x in winrate(N, k, t)])[0] for t in tl)
        ra = st.mean(spearmanr([winrate(B, k, t)[x] for x in sorted(winrate(B, k, t))],
                               [winrate(N, k, t)[x] for x in sorted(winrate(B, k, t))])[0] for t in tl)
        qa, qb = PAIRS[k]
        tie = lambda R: 100 * st.mean(R[i]["raw_p"][qa] < .5 and R[i]["raw_p"][qb] < .5 for i in sel)
        didtxt = "—" if k == "better" else f"{st.mean(did):+.4f} [{lo:+.4f},{hi:+.4f}]"
        print(f"{k:14} {pb:15.3f} {pn:7.3f} {pn - pb:+7.3f} {didtxt:>28} {rb:13.3f} {rn:7.3f} {ra:17.3f} "
              f"{tie(B):4.1f}/{tie(N):4.1f}")
    print()

# within each framing: how far apart are the pairs?
for name, R in (("broad", B), ("narrow", N)):
    sel = ids
    for k in ("appropriate", "proportionate"):
        d = [mir(R[i], k) - mir(R[i], "better") for i in sel]
        flips = sum((mir(R[i], k) - .5) * (mir(R[i], "better") - .5) < 0 for i in sel)
        print(f"{name:6} {k:13} vs better: mean diff {st.mean(d):+.4f} | mean |diff| {st.mean(abs(x) for x in d):.4f} "
              f"| winner differs {flips/len(sel):.3%}")

print("\ntop 6 per topic and framing (win rate, words):")
for t in topics:
    for name, R in (("broad", B), ("narrow", N)):
        for k in PAIRS:
            wr = winrate(R, k, t); top = sorted(wr, key=lambda x: -wr[x])[:6]
            print(f"  {t} {name:6} {k:13} " + ", ".join(f"{x}({wr[x]:.2f},{L[t][x]}w)" for x in top))
