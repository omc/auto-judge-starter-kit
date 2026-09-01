#!/usr/bin/env python3
"""Run-vs-run Jaccard overlap for RAG26 generation runs.

For each topic, each run has a set of CITED shards. Pairwise Jaccard =
|A∩B| / |A∪B| over the shards two runs cite for the same topic. We average
Jaccard across all topics both runs answered -> a run-similarity matrix.
High similarity = two systems tend to cite the same sources.
"""
import json, os, sys, csv, itertools, statistics
from collections import defaultdict

GEN = "data/rag26/runs/generation"
OUT = "temp/rag26_consensus"
os.makedirs(OUT, exist_ok=True)

# run -> topic -> set(cited shards)
cites = defaultdict(lambda: defaultdict(set))
runs = []
for run in sorted(os.listdir(GEN)):
    path = os.path.join(GEN, run)
    if not os.path.isfile(path):
        continue
    runs.append(run)
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            topic = d["metadata"]["topic_id"]
            rid = d["metadata"]["run_id"]
            ans = d.get("answer") or d.get("responses") or []
            s = cites[rid][topic]
            for sent in ans:
                for shard, w in (sent.get("citations") or {}).items():
                    if w and w > 0:
                        s.add(shard)
    print(f"  parsed {run}", file=sys.stderr)

run_ids = sorted(cites.keys())
n = len(run_ids)
idx = {r: i for i, r in enumerate(run_ids)}
print(f"\n{n} runs", file=sys.stderr)

# pairwise mean Jaccard
import math
sim = [[float("nan")] * n for _ in range(n)]
for i in range(n):
    sim[i][i] = 1.0
pair_rows = []
for a, b in itertools.combinations(run_ids, 2):
    ta, tb = cites[a], cites[b]
    shared = set(ta) & set(tb)
    js = []
    for t in shared:
        A, B = ta[t], tb[t]
        if not A and not B:
            continue
        u = len(A | B)
        js.append(len(A & B) / u if u else 0.0)
    if js:
        m = sum(js) / len(js)
        sim[idx[a]][idx[b]] = sim[idx[b]][idx[a]] = m
        pair_rows.append((a, b, m, len(js)))

# ---- write full matrix ----
with open(f"{OUT}/jaccard_matrix.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run"] + run_ids)
    for i, r in enumerate(run_ids):
        w.writerow([r] + [("" if math.isnan(sim[i][j]) else f"{sim[i][j]:.4f}") for j in range(n)])

# ---- per-run centrality (mean sim to all others) ----
central = []
for i, r in enumerate(run_ids):
    vals = [sim[i][j] for j in range(n) if j != i and not math.isnan(sim[i][j])]
    central.append((r, sum(vals) / len(vals) if vals else 0.0,
                    statistics.mean(len(cites[r][t]) for t in cites[r]) if cites[r] else 0))
central.sort(key=lambda x: -x[1])
with open(f"{OUT}/run_centrality.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["run", "mean_jaccard_to_others", "avg_cited_shards_per_topic"])
    for r, c, a in central:
        w.writerow([r, f"{c:.4f}", f"{a:.1f}"])

pair_rows.sort(key=lambda x: -x[2])
with open(f"{OUT}/jaccard_pairs.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["run_a", "run_b", "mean_jaccard", "n_shared_topics"])
    for a, b, m, k in pair_rows:
        w.writerow([a, b, f"{m:.4f}", k])

# ---- greedy clustering at a threshold (connected components) ----
def components(threshold):
    parent = {r: r for r in run_ids}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b, m, k in pair_rows:
        if m >= threshold:
            parent[find(a)] = find(b)
    comp = defaultdict(list)
    for r in run_ids:
        comp[find(r)].append(r)
    return sorted((c for c in comp.values()), key=len, reverse=True)

# ================= report =================
allvals = [m for _, _, m, _ in pair_rows]
print("\n" + "=" * 70)
print(f"RUN-vs-RUN JACCARD  (cited-shard overlap, {n} runs, {len(pair_rows)} pairs)")
print("=" * 70)
print(f"\nPairwise mean-Jaccard distribution:")
print(f"  min={min(allvals):.3f}  median={statistics.median(allvals):.3f}  "
      f"mean={statistics.mean(allvals):.3f}  max={max(allvals):.3f}  sd={statistics.pstdev(allvals):.3f}")
# histogram
buckets = defaultdict(int)
for m in allvals:
    buckets[min(int(m * 20), 19)] += 1
mx = max(buckets.values())
print("\n  Jaccard  count")
for b in range(20):
    lo = b / 20
    c = buckets.get(b, 0)
    if c:
        print(f"  {lo:.2f}-{lo+0.05:.2f} | {c:4d} {'#'*int(round(40*c/mx))}")

print("\n--- Top 20 most-similar run pairs (cite the same shards) ---")
print(f"  {'run_a':<10}{'run_b':<10}{'Jaccard':>8}{'shared_topics':>15}")
for a, b, m, k in pair_rows[:20]:
    print(f"  {a:<10}{b:<10}{m:>8.3f}{k:>15}")

print("\n--- Bottom 10 least-similar run pairs (cite disjoint sources) ---")
for a, b, m, k in pair_rows[-10:]:
    print(f"  {a:<10}{b:<10}{m:>8.3f}{k:>15}")

print("\n--- Most 'central' runs (highest mean similarity to all others) ---")
print(f"  {'run':<10}{'meanJacc':>9}{'avgCited/topic':>15}")
for r, c, a in central[:10]:
    print(f"  {r:<10}{c:>9.3f}{a:>15.1f}")
print("  --- most 'outlier' runs (lowest mean similarity) ---")
for r, c, a in central[-10:]:
    print(f"  {r:<10}{c:>9.3f}{a:>15.1f}")

for thr in (0.5, 0.6, 0.7):
    comps = [c for c in components(thr) if len(c) > 1]
    print(f"\n--- Clusters at Jaccard >= {thr} (connected components, size>1) ---")
    if not comps:
        print("   (none)")
    for c in comps:
        print(f"   [{len(c)}] " + ", ".join(sorted(c)))

print(f"\nCSVs: {OUT}/jaccard_matrix.csv  jaccard_pairs.csv  run_centrality.csv")
