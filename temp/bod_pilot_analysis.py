"""BOD pilot report, PERMITTED WINDOW ONLY (data policy, CLAUDE.md).
Usage: JEV_DATASET=rag26|ragtime26 python temp/bod_pilot_analysis.py <bod out-dir> [jev pairwise comparisons] [gemini comparisons]

Every per-answer record is read through window_records (restricted topics are skipped
after reading their topic id). Reports:
  - rubric size and score distribution; how peaked Jev's grade probabilities are
  - split-half ranking stability over window topics (5 vs 5)
  - length dependence of BOD_MEAN and BOD_PADDING (per-topic Spearman with word count)
  - agreement with the Jev pairwise (mirror) and Gemini pairwise judges on the window
"""
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from jev_dataset import DS, window, window_records, word_lengths  # noqa: E402

out = Path(sys.argv[1])
jev_path = sys.argv[2] if len(sys.argv) > 2 else None
gem_path = sys.argv[3] if len(sys.argv) > 3 else None
WIN = window()

rows = window_records(next(out.glob("*.bod")) / "grades.jsonl")
q = [r for r in rows if r["qid"] != "padding"]
pad = {(r["topic_id"], r["run_id"]): r["p_yes"] for r in rows if r["qid"] == "padding"}
by = defaultdict(list)
for r in q:
    by[(r["topic_id"], r["run_id"])].append(r)
score = {k: sum(r["expected"] for r in v) / (3 * len(v)) for k, v in by.items()}
topics = [t for t in WIN if any(k[0] == t for k in score)]
runs = sorted({k[1] for k in score})
nq = {t: len(next(v for k, v in by.items() if k[0] == t)) for t in topics}
print(f"== BOD pilot, {DS['name']} window: {len(topics)} topics, {len(runs)} runs, "
      f"{len(score):,} graded answers")
print("questions per topic: " + ", ".join(f"{t.split('-')[-1]}:{nq[t]}" for t in topics))

vals = sorted(score.values())
print(f"BOD_MEAN quantiles p10 {vals[len(vals)//10]:.3f} p50 {st.median(vals):.3f} "
      f"p90 {vals[9*len(vals)//10]:.3f} | mean padding {st.mean(pad.values()):.3f}")
peak = [max(r["probs"].values()) for r in q]
gap = [abs(r["expected"] - int(r["argmax"])) for r in q if r["argmax"] is not None]
dist = defaultdict(int)
for r in q:
    dist[r["argmax"]] += 1
print(f"grade probs: max P > 0.9 on {sum(p > .9 for p in peak)/len(peak):.1%} of questions | "
      f"|E[g] - argmax| > 0.25 on {sum(g > .25 for g in gap)/len(gap):.1%} | argmax distribution "
      + " ".join(f"{g}:{dist[g]/len(q):.1%}" for g in "0123"))


def board(tset, sc):
    s, c = defaultdict(float), defaultdict(int)
    for (t, x), v in sc.items():
        if t in tset:
            s[x] += v; c[x] += 1
    return {x: s[x] / c[x] for x in s}


B = board(set(topics), score)
sh = []
for i in range(200):
    rr = random.Random(i); tt = topics[:]; rr.shuffle(tt)
    h1, h2 = board(set(tt[:len(tt)//2]), score), board(set(tt[len(tt)//2:]), score)
    common = [x for x in runs if x in h1 and x in h2]
    sh.append(spearmanr([h1[x] for x in common], [h2[x] for x in common])[0])
sh.sort()
print(f"split-half (5 vs 5 topics, 200 splits): Spearman mean {st.mean(sh):.3f} "
      f"[p2.5 {sh[5]:.3f}, p97.5 {sh[194]:.3f}]")

L = word_lengths(set(topics))
def per_topic_len(sc):
    out = []
    for t in topics:
        xs = [x for x in runs if (t, x) in sc and x in L[t]]
        out.append(spearmanr([sc[(t, x)] for x in xs], [L[t][x] for x in xs])[0])
    return out
rl, rp = per_topic_len(score), per_topic_len(pad)
print(f"length: per-topic Spearman(BOD_MEAN, words) median {st.median(rl):.3f} "
      f"[min {min(rl):.3f}, max {max(rl):.3f}] | Spearman(PADDING, words) median {st.median(rp):.3f}")


def pairwise_winrates(path, prob_key):
    """(topic, run) -> win rate from a pairwise comparisons.jsonl, window topics only."""
    w, g = defaultdict(float), defaultdict(int)
    for r in window_records(path):
        if not r.get("valid"):
            continue
        t, a, b = r["topic_id"], r["a_run"], r["b_run"]
        if prob_key == "p_a":
            pa = r["p_a"]
        else:                                   # Gemini: binary winner
            pa = 1.0 if r.get("winner_run") == a else 0.0
        w[(t, a)] += pa; w[(t, b)] += 1 - pa; g[(t, a)] += 1; g[(t, b)] += 1
    return {k: w[k] / g[k] for k in g}


def agreement(name, wr):
    tt = [t for t in topics if any(k[0] == t for k in wr)]
    if not tt:
        print(f"{name}: no window topics in common"); return
    b1, b2 = board(set(tt), score), board(set(tt), wr)
    xs = [x for x in runs if x in b1 and x in b2]
    per, lw = [], []
    for t in tt:
        ys = [x for x in runs if (t, x) in score and (t, x) in wr]
        if len({wr[(t, x)] for x in ys}) > 1:
            per.append(spearmanr([score[(t, x)] for x in ys], [wr[(t, x)] for x in ys])[0])
        lw.append(spearmanr([wr[(t, x)] for x in ys], [L[t][x] for x in ys])[0])
    print(f"{name} on {len(tt)} window topics, {len(xs)} runs: board Spearman "
          f"{spearmanr([b1[x] for x in xs], [b2[x] for x in xs])[0]:.3f} Kendall "
          f"{kendalltau([b1[x] for x in xs], [b2[x] for x in xs])[0]:.3f} | per-topic median "
          f"{st.median(per):.3f} | its own length Spearman median {st.median(lw):.3f}")


if jev_path and Path(jev_path).exists():
    agreement("vs Jev pairwise (mirror)", pairwise_winrates(jev_path, "p_a"))
if gem_path and Path(gem_path).exists():
    agreement("vs Gemini pairwise", pairwise_winrates(gem_path, "winner"))
