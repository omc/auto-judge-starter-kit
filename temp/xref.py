#!/usr/bin/env python3
"""Cross-reference generation-run clusters against retrieval runs.

Namespaces are disjoint, so there is no label linking a generation run to its
retriever. Structural test instead: a run can only cite what was retrieved, so
if a generation cluster shares a retrieval backend, every member's cited shards
should be COVERED by the same retrieval run's ranked list.

coverage(g, r, topic) = |cited_g ∩ retrieved_r| / |cited_g|   (citation recall)
Best-match retriever for g = argmax_r mean_topic coverage(g, r).
Then: do co-clustered generation runs share the same best-match retriever?
"""
import json, os, sys, itertools, statistics
from collections import defaultdict

GEN = "data/rag26/runs/generation"
RET = "data/rag26/runs/retrieval"
OUT = "temp/rag26_consensus"

# ---------- load generation cited sets ----------
gcites = defaultdict(lambda: defaultdict(set))   # gen_run -> topic -> set(cited)
for run in sorted(os.listdir(GEN)):
    p = os.path.join(GEN, run)
    if not os.path.isfile(p):
        continue
    with open(p) as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            t = d["metadata"]["topic_id"]
            s = gcites[run][t]
            for sent in (d.get("answer") or d.get("responses") or []):
                for shard, w in (sent.get("citations") or {}).items():
                    if w and w > 0:
                        s.add(shard)
    print(f"  gen {run}", file=sys.stderr)
gen_ids = sorted(gcites)

# ---------- load retrieval sets (full + top-20) ----------
rfull = defaultdict(lambda: defaultdict(set))    # ret_run -> topic -> set(all retrieved)
rtop20 = defaultdict(lambda: defaultdict(set))
for run in sorted(os.listdir(RET)):
    p = os.path.join(RET, run)
    if not os.path.isfile(p):
        continue
    with open(p) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 4:
                continue
            topic, _q0, shard, rank = parts[0], parts[1], parts[2], parts[3]
            rfull[run][topic].add(shard)
            try:
                if int(rank) <= 20:
                    rtop20[run][topic].add(shard)
            except ValueError:
                pass
    print(f"  ret {run}", file=sys.stderr)
ret_ids = sorted(rfull)
print(f"\n{len(gen_ids)} generation, {len(ret_ids)} retrieval runs", file=sys.stderr)

# ---------- recompute generation clusters (Jaccard >= 0.5) ----------
def gen_clusters(threshold=0.5):
    parent = {g: g for g in gen_ids}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in itertools.combinations(gen_ids, 2):
        ta, tb = gcites[a], gcites[b]
        shared = set(ta) & set(tb)
        js = []
        for t in shared:
            A, B = ta[t], tb[t]
            u = len(A | B)
            if u:
                js.append(len(A & B) / u)
        if js and sum(js) / len(js) >= threshold:
            parent[find(a)] = find(b)
    comp = defaultdict(list)
    for g in gen_ids:
        comp[find(g)].append(g)
    return sorted((sorted(c) for c in comp.values()), key=len, reverse=True)

clusters = gen_clusters(0.5)

# ---------- best-match retriever per generation run ----------
def coverage(gset_by_topic, rset_by_topic):
    """mean over topics of |cited ∩ retrieved| / |cited|"""
    covs = []
    for t, cited in gset_by_topic.items():
        if not cited:
            continue
        rr = rset_by_topic.get(t)
        if rr is None:
            covs.append(0.0)
        else:
            covs.append(len(cited & rr) / len(cited))
    return statistics.mean(covs) if covs else 0.0

best = {}   # gen -> (ret_run, cov_full, cov_top20_of_that_ret, second_ret, second_cov)
for g in gen_ids:
    scored = []
    for r in ret_ids:
        scored.append((coverage(gcites[g], rfull[r]), r))
    scored.sort(reverse=True)
    top_r = scored[0][1]
    cov20 = coverage(gcites[g], rtop20[top_r])
    best[g] = (top_r, scored[0][0], cov20, scored[1][1], scored[1][0])
    print(f"  matched {g}", file=sys.stderr)

# ---------- retrieval self-clustering (Jaccard >= 0.9 => near-identical retrievers) ----------
def ret_clusters(threshold=0.9):
    parent = {r: r for r in ret_ids}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in itertools.combinations(ret_ids, 2):
        ta, tb = rfull[a], rfull[b]
        shared = set(ta) & set(tb)
        js = []
        for t in shared:
            A, B = ta[t], tb[t]
            u = len(A | B)
            if u:
                js.append(len(A & B) / u)
        if js and sum(js) / len(js) >= threshold:
            parent[find(a)] = find(b)
    comp = defaultdict(list)
    for r in ret_ids:
        comp[find(r)].append(r)
    # map each ret run to a cluster id
    cid = {}
    for i, c in enumerate(sorted((sorted(c) for c in comp.values()), key=len, reverse=True)):
        for r in c:
            cid[r] = (i, len(c))
    return cid
rcid = ret_clusters(0.9)

# ================= report =================
print("\n" + "=" * 74)
print("GENERATION CLUSTERS  ×  RETRIEVAL RUNS")
print("=" * 74)
print("\ncoverage = mean fraction of a gen run's cited shards found in the retriever's list")
print("cov@20   = same, but only the retriever's top-20 per topic\n")

print(f"--- Best-match retriever per generation run, grouped by cluster (Jaccard>=0.5) ---")
multi = [c for c in clusters if len(c) > 1]
singles = [c[0] for c in clusters if len(c) == 1]
for ci, c in enumerate(multi):
    print(f"\n  CLUSTER {ci} (size {len(c)}):")
    print(f"    {'gen':<8}{'best_retriever':<16}{'cov':>6}{'cov@20':>8}   {'2nd':<10}{'2ndcov':>7}  retClu")
    matched = []
    for g in c:
        r, cov, cov20, r2, cov2 = best[g]
        rc = rcid.get(r, ("?", 1))
        matched.append(r)
        print(f"    {g:<8}{r:<16}{cov:>6.2f}{cov20:>8.2f}   {r2:<10}{cov2:>7.2f}  #{rc[0]}(n={rc[1]})")
    # do they agree on retriever (or on retriever-cluster)?
    rset = set(matched)
    rcset = set(rcid.get(r, (r, 1))[0] for r in matched)
    verdict = "SAME retriever" if len(rset) == 1 else (
        "SAME retrieval-cluster" if len(rcset) == 1 else f"{len(rset)} distinct retrievers / {len(rcset)} retr-clusters")
    print(f"    -> {verdict}")

print(f"\n--- Singleton generation runs (no gen-cluster), best retriever match ---")
print(f"    {'gen':<8}{'best_retriever':<16}{'cov':>6}{'cov@20':>8}")
for g in sorted(singles, key=lambda g: -best[g][1]):
    r, cov, cov20, r2, cov2 = best[g]
    print(f"    {g:<8}{r:<16}{cov:>6.2f}{cov20:>8.2f}")

# overall coverage distribution
covs = [best[g][1] for g in gen_ids]
print(f"\n--- Best-match coverage distribution (all {len(gen_ids)} gen runs) ---")
print(f"    min={min(covs):.2f} median={statistics.median(covs):.2f} mean={statistics.mean(covs):.2f} max={max(covs):.2f}")
hi = sum(1 for c in covs if c >= 0.9); lo = sum(1 for c in covs if c < 0.5)
print(f"    gen runs whose citations are >=90% covered by some retrieval run: {hi}/{len(gen_ids)}")
print(f"    gen runs with <50% best coverage (retriever NOT among submitted runs): {lo}/{len(gen_ids)}")

# write csv
import csv
with open(f"{OUT}/gen_to_retrieval.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["gen_run", "gen_cluster", "best_retriever", "coverage_full", "coverage_top20",
                "second_retriever", "second_coverage", "retriever_cluster"])
    g2c = {}
    for ci, c in enumerate(clusters):
        for g in c:
            g2c[g] = ci if len(c) > 1 else ""
    for g in gen_ids:
        r, cov, cov20, r2, cov2 = best[g]
        w.writerow([g, g2c[g], r, f"{cov:.4f}", f"{cov20:.4f}", r2, f"{cov2:.4f}", rcid.get(r, ("", ""))[0]])
print(f"\nCSV: {OUT}/gen_to_retrieval.csv")
