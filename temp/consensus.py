#!/usr/bin/env python3
"""Cross-run document-consensus analysis for RAG26 generation runs.

For each topic, measure how many runs REFERENCED (retrieved) each shard vs.
how many actually CITED it. Separating the two avoids conflating "everyone had
the doc" with "everyone chose to use it" -- the citation rate among retrievers
is the real consensus signal.
"""
import json, os, sys, csv
from collections import defaultdict

GEN = "data/rag26/runs/generation"
OUT = "temp/rag26_consensus"
os.makedirs(OUT, exist_ok=True)

runs = sorted(os.listdir(GEN))

# topic -> shard -> set(run) that referenced / cited it
referenced = defaultdict(lambda: defaultdict(set))
cited      = defaultdict(lambda: defaultdict(set))
cite_wsum  = defaultdict(lambda: defaultdict(float))   # topic->shard->sum of weights
topics_runs = defaultdict(set)                          # topic -> set(run) present

for run in runs:
    path = os.path.join(GEN, run)
    if not os.path.isfile(path):
        continue
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            topic = d["metadata"]["topic_id"]
            rid   = d["metadata"]["run_id"]
            topics_runs[topic].add(rid)
            # references = ordered retrieved set
            for ref in d.get("references", []):
                shard = ref if isinstance(ref, str) else ref.get("id") or ref.get("document_id")
                referenced[topic][shard].add(rid)
            # citations across sentences
            ans = d.get("answer") or d.get("responses") or []
            seen_cited = set()
            for sent in ans:
                for shard, w in (sent.get("citations") or {}).items():
                    if w and w > 0:
                        cited[topic][shard].add(rid)
                        cite_wsum[topic][shard] += float(w)
    print(f"  parsed {run}", file=sys.stderr)

n_runs = len(runs)
n_topics = len(topics_runs)
print(f"\nParsed {n_runs} runs, {n_topics} topics", file=sys.stderr)

# ---- per (topic,shard) table ----
rows = []
for topic in referenced:
    shards = set(referenced[topic]) | set(cited[topic])
    R = len(topics_runs[topic])
    for shard in shards:
        nref = len(referenced[topic].get(shard, ()))
        ncit = len(cited[topic].get(shard, ()))
        rows.append({
            "topic": topic, "shard": shard,
            "n_runs_topic": R,
            "n_referenced": nref,
            "n_cited": ncit,
            "cite_rate_of_referencers": round(ncit / nref, 4) if nref else 0.0,
            "cite_rate_of_all_runs": round(ncit / R, 4) if R else 0.0,
            "sum_weight": round(cite_wsum[topic].get(shard, 0.0), 3),
        })

with open(f"{OUT}/consensus_by_topic_shard.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

# ---- consensus histograms: # shards cited by exactly k runs (across all topics) ----
from collections import Counter
cited_hist = Counter(r["n_cited"] for r in rows if r["n_cited"] > 0)
ref_hist   = Counter(r["n_referenced"] for r in rows if r["n_referenced"] > 0)
with open(f"{OUT}/consensus_histogram.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["k_runs", "n_shards_cited_by_k", "n_shards_referenced_by_k"])
    for k in range(1, n_runs + 1):
        w.writerow([k, cited_hist.get(k, 0), ref_hist.get(k, 0)])

# ---- per-topic summary ----
tsum = []
for topic in referenced:
    R = len(topics_runs[topic])
    tr = [r for r in rows if r["topic"] == topic]
    cited_shards = [r for r in tr if r["n_cited"] > 0]
    max_c = max((r["n_cited"] for r in tr), default=0)
    top = max(tr, key=lambda r: (r["n_cited"], r["sum_weight"]), default=None)
    tsum.append({
        "topic": topic, "n_runs": R,
        "n_shards_referenced": sum(1 for r in tr if r["n_referenced"] > 0),
        "n_shards_cited": len(cited_shards),
        "max_citation_consensus": max_c,
        "max_consensus_frac": round(max_c / R, 3) if R else 0,
        "top_shard": top["shard"] if top else "",
    })
tsum.sort(key=lambda x: -x["max_consensus_frac"])
with open(f"{OUT}/topic_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(tsum[0].keys()))
    w.writeheader(); w.writerows(tsum)

# ---- top consensus "gold" shards overall ----
gold = sorted(rows, key=lambda r: (-r["n_cited"], -r["sum_weight"]))[:30]

# ================= console report =================
def bar(n, mx, width=40):
    return "#" * int(round(width * n / mx)) if mx else ""

print("\n" + "=" * 70)
print("CROSS-RUN DOCUMENT CONSENSUS  (RAG26 generation, {} runs, {} topics)".format(n_runs, n_topics))
print("=" * 70)

print("\n--- Consensus distribution: # of (topic,shard) pairs cited by exactly k runs ---")
mx = max(cited_hist.values())
tot_cited_pairs = sum(cited_hist.values())
cum = 0
for k in range(1, n_runs + 1):
    c = cited_hist.get(k, 0)
    if c == 0 and k > max(cited_hist):
        break
    cum += c
    if c:
        print(f"  {k:2d} runs | {c:6d} {bar(c, mx)}")
print(f"\n  total distinct (topic,shard) cited pairs: {tot_cited_pairs}")
singletons = cited_hist.get(1, 0)
print(f"  cited by exactly 1 run (idiosyncratic): {singletons} ({100*singletons/tot_cited_pairs:.1f}%)")
ge_half = sum(v for k, v in cited_hist.items() if k >= n_runs / 2)
print(f"  cited by >= half the runs (strong consensus): {ge_half} ({100*ge_half/tot_cited_pairs:.1f}%)")

print("\n--- Top 20 highest-consensus shards (cited by the most runs, per topic) ---")
print(f"  {'topic':<12} {'shard':<20} {'#cited':>6} {'#ref':>5} {'cite/ref':>8} {'wsum':>7}")
for r in gold[:20]:
    print(f"  {r['topic']:<12} {r['shard']:<20} {r['n_cited']:>6} {r['n_referenced']:>5} "
          f"{r['cite_rate_of_referencers']:>8.2f} {r['sum_weight']:>7.1f}")

print("\n--- Topic summary: highest max-consensus topics (docs many runs agree on) ---")
print(f"  {'topic':<12} {'#runs':>5} {'#refShd':>7} {'#citShd':>7} {'maxCons':>7} {'frac':>5}")
for t in tsum[:12]:
    print(f"  {t['topic']:<12} {t['n_runs']:>5} {t['n_shards_referenced']:>7} {t['n_shards_cited']:>7} "
          f"{t['max_citation_consensus']:>7} {t['max_consensus_frac']:>5.2f}")
print("  ...")
print("  --- lowest max-consensus topics (runs disagree most on which docs matter) ---")
for t in tsum[-6:]:
    print(f"  {t['topic']:<12} {t['n_runs']:>5} {t['n_shards_referenced']:>7} {t['n_shards_cited']:>7} "
          f"{t['max_citation_consensus']:>7} {t['max_consensus_frac']:>5.2f}")

# overall cite-rate-of-referencers stats
import statistics
crr = [r["cite_rate_of_referencers"] for r in rows if r["n_referenced"] > 0]
print(f"\n--- Citation rate among runs that referenced a shard ---")
print(f"  mean={statistics.mean(crr):.3f}  median={statistics.median(crr):.3f}")
print(f"  shards referenced but NEVER cited by anyone: "
      f"{sum(1 for r in rows if r['n_referenced']>0 and r['n_cited']==0)} "
      f"/ {sum(1 for r in rows if r['n_referenced']>0)} referenced pairs")

print(f"\nCSVs written to {OUT}/")
print("  consensus_by_topic_shard.csv  topic_summary.csv  consensus_histogram.csv")
