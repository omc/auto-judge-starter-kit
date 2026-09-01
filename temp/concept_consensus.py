#!/usr/bin/env python3
"""Cross-run concept consensus on ANSWER text (RAG26 generation).

Treats each run's answer noun-lemma concepts as its "nuggets" for a topic, then
asks, per topic: which concepts are consensus (many runs state them) vs private
(one run only), and which runs/teams are concept outliers -- injecting concepts
few or no other runs share -- and by how much.

Concept = spaCy NOUN/PROPN lemma (lowercased), same extractor as concept-F1.
No documents are read; this is answer-only.
"""
import csv, html, json, os, re, statistics, sys
from collections import defaultdict, Counter
import spacy

GEN = "data/rag26/runs/generation"
OUT = "temp/rag26_consensus"
os.makedirs(OUT, exist_ok=True)

DISABLE = ["parser", "senter", "ner", "entity_ruler", "textcat",
           "morphologizer", "trainable_lemmatizer"]
TAG_RE = re.compile(r"<.*?>")


def clean(t):
    return html.unescape(TAG_RE.sub("", t or ""))


def main():
    nlp = spacy.load("en_core_web_lg", disable=DISABLE)
    nlp.max_length = 3_000_000

    # ---- collect answer texts ----
    keys = []            # (run, topic)
    texts = []
    run_team = {}
    for run in sorted(os.listdir(GEN)):
        p = os.path.join(GEN, run)
        if not os.path.isfile(p):
            continue
        for line in open(p):
            if not line.strip():
                continue
            d = json.loads(line)
            run_team[run] = d["metadata"].get("team_id", "?")
            topic = d["metadata"]["topic_id"]
            parts = [s.get("text", "") for s in (d.get("answer") or d.get("responses") or [])]
            keys.append((run, topic))
            texts.append(clean(" ".join(parts)))
    print(f"collected {len(keys)} answers", file=sys.stderr)

    # ---- tag -> concept sets ----
    concepts = {}        # (run,topic) -> frozenset(lemmas)
    done = 0
    for doc, key in nlp.pipe(((t, k) for t, k in zip(texts, keys)),
                             as_tuples=True, batch_size=256):
        concepts[key] = frozenset(t.lemma_.lower() for t in doc if t.pos_ in ("NOUN", "PROPN"))
        done += 1
        if done % 2000 == 0:
            print(f"  tagged {done}/{len(keys)}", file=sys.stderr)

    # ---- per topic: document-frequency of each concept across runs ----
    topic_runs = defaultdict(list)                 # topic -> [run,...]
    for (run, topic) in keys:
        topic_runs[topic].append(run)
    topics = sorted(topic_runs)

    # global consensus histogram: (topic,concept) present in exactly k runs
    consensus_hist = Counter()
    # per (run,topic) stats
    rt_rows = []
    for topic in topics:
        runs_here = topic_runs[topic]
        N = len(runs_here)
        df = Counter()
        for run in runs_here:
            for c in concepts[(run, topic)]:
                df[c] += 1
        for c, k in df.items():
            consensus_hist[k] += 1
        consensus_concepts = {c for c, k in df.items() if k >= N / 2}   # >=50% of runs
        for run in runs_here:
            cs = concepts[(run, topic)]
            n = len(cs)
            if n == 0:
                continue
            private = sum(1 for c in cs if df[c] == 1)
            rare = sum(1 for c in cs if df[c] / N < 0.10)              # <10% of runs
            mean_share = statistics.mean((df[c] - 1) / (N - 1) for c in cs) if N > 1 else 0.0
            cov_cons = (len(cs & consensus_concepts) / len(consensus_concepts)
                        if consensus_concepts else 0.0)
            rt_rows.append({
                "run": run, "team": run_team[run], "topic": topic, "n_concepts": n,
                "private": private, "private_rate": private / n,
                "rare": rare, "rare_rate": rare / n,
                "mean_share": mean_share,            # 0=always alone, 1=shared by all others
                "consensus_coverage": cov_cons,      # fraction of topic consensus concepts it includes
            })

    with open(f"{OUT}/answer_concept_by_run_topic.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rt_rows[0].keys()))
        w.writeheader(); w.writerows(rt_rows)

    # ---- aggregate per run ----
    per_run = defaultdict(list)
    for r in rt_rows:
        per_run[r["run"]].append(r)
    run_stats = []
    for run, rows in per_run.items():
        run_stats.append({
            "run": run, "team": run_team[run], "n_topics": len(rows),
            "avg_concepts": statistics.mean(x["n_concepts"] for x in rows),
            "private_rate": statistics.mean(x["private_rate"] for x in rows),
            "rare_rate": statistics.mean(x["rare_rate"] for x in rows),
            "mean_share": statistics.mean(x["mean_share"] for x in rows),
            "consensus_coverage": statistics.mean(x["consensus_coverage"] for x in rows),
            "total_private": sum(x["private"] for x in rows),
        })
    med_priv = statistics.median(x["private_rate"] for x in run_stats)
    for x in run_stats:
        x["priv_vs_median"] = x["private_rate"] / med_priv if med_priv else float("nan")
    run_stats.sort(key=lambda x: -x["private_rate"])
    with open(f"{OUT}/answer_concept_by_run.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(run_stats[0].keys()))
        w.writeheader(); w.writerows(run_stats)

    # ---- aggregate per team ----
    per_team = defaultdict(list)
    for x in run_stats:
        per_team[x["team"]].append(x)
    team_stats = []
    for team, xs in per_team.items():
        team_stats.append({
            "team": team, "n_runs": len(xs),
            "private_rate": statistics.mean(x["private_rate"] for x in xs),
            "mean_share": statistics.mean(x["mean_share"] for x in xs),
            "consensus_coverage": statistics.mean(x["consensus_coverage"] for x in xs),
            "avg_concepts": statistics.mean(x["avg_concepts"] for x in xs),
        })
    med_team = statistics.median(x["private_rate"] for x in team_stats)
    for x in team_stats:
        x["priv_vs_median"] = x["private_rate"] / med_team if med_team else float("nan")
    team_stats.sort(key=lambda x: -x["private_rate"])
    with open(f"{OUT}/answer_concept_by_team.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(team_stats[0].keys()))
        w.writeheader(); w.writerows(team_stats)

    # ================= report =================
    tot = sum(consensus_hist.values())
    print("\n" + "=" * 68)
    print(f"ANSWER-CONCEPT CONSENSUS ACROSS RUNS ({len(per_run)} runs, {len(topics)} topics)")
    print("=" * 68)
    print("\n--- Concept overlap: (topic,concept) pairs stated by exactly k runs ---")
    singles = consensus_hist.get(1, 0)
    print(f"  distinct (topic,concept) pairs : {tot}")
    print(f"  stated by exactly 1 run (private): {singles} ({100*singles/tot:.1f}%)")
    maxN = max(len(v) for v in topic_runs.values())
    ge = sum(v for k, v in consensus_hist.items() if k >= maxN/2)
    print(f"  stated by >= half the runs        : {ge} ({100*ge/tot:.1f}%)")
    for k in (1, 2, 3, 5, 10, 20, 40, 60, 80):
        c = consensus_hist.get(k, 0)
        print(f"    {k:3d} runs | {c:7d}")

    print("\n--- Most OUTLIER runs (highest private-concept rate) ---")
    print(f"  {'run':<9}{'team':<7}{'priv_rate':>10}{'xMed':>6}{'mean_share':>11}{'cons_cov':>9}{'avg_con':>8}")
    for x in run_stats[:12]:
        print(f"  {x['run']:<9}{x['team']:<7}{x['private_rate']:>10.3f}{x['priv_vs_median']:>6.1f}"
              f"{x['mean_share']:>11.3f}{x['consensus_coverage']:>9.3f}{x['avg_concepts']:>8.1f}")
    print("  --- most MAINSTREAM runs (lowest private rate) ---")
    for x in run_stats[-6:]:
        print(f"  {x['run']:<9}{x['team']:<7}{x['private_rate']:>10.3f}{x['priv_vs_median']:>6.1f}"
              f"{x['mean_share']:>11.3f}{x['consensus_coverage']:>9.3f}{x['avg_concepts']:>8.1f}")

    print(f"\n  median run private_rate = {med_priv:.3f}")

    print("\n--- Teams ranked by concept outlier-ness (mean private rate) ---")
    print(f"  {'team':<7}{'runs':>5}{'priv_rate':>10}{'xMed':>6}{'mean_share':>11}{'cons_cov':>9}{'avg_con':>8}")
    for x in team_stats:
        print(f"  {x['team']:<7}{x['n_runs']:>5}{x['private_rate']:>10.3f}{x['priv_vs_median']:>6.1f}"
              f"{x['mean_share']:>11.3f}{x['consensus_coverage']:>9.3f}{x['avg_concepts']:>8.1f}")

    print(f"\nCSVs: {OUT}/answer_concept_by_run.csv  answer_concept_by_team.csv  answer_concept_by_run_topic.csv")


if __name__ == "__main__":
    main()
