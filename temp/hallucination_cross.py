#!/usr/bin/env python3
"""Private-vs-cited-docs hallucination cross (RAG26 generation).

An answer concept is:
  * UNGROUNDED if it is NOT in the run's cited documents (precision false positive)
  * PRIVATE    if NO other run's answer states it for that topic (cross-run rarity)
The intersection -- ungrounded AND private -- is the strongest hallucination
candidate: fabricated content the source doesn't support and no peer independently
produced. Ungrounded-but-SHARED concepts are likely common-knowledge elaboration.

Cross-run rarity (answer df) is computed over ALL runs. Cited-document concepts
(the expensive full-doc tagging) are computed only for the selected runs, over the
first --topics topics, to keep the demo bounded. Use --runs all --topics 119 for
the full corpus (~2h).
"""
import argparse, csv, gc, html, json, os, pickle, re, statistics, sys, time
from collections import defaultdict, Counter
import spacy

GEN = "data/rag26/runs/generation"
OUT = "temp/rag26_consensus"
DISABLE = ["parser", "senter", "ner", "entity_ruler", "textcat",
           "morphologizer", "trainable_lemmatizer"]
TAG_RE = re.compile(r"<.*?>")


def clean(t):
    return html.unescape(TAG_RE.sub("", t or ""))


def tnum(tid):
    try:
        return int(tid.rsplit("-", 1)[-1])
    except ValueError:
        return 10**9


def tag_sets(items, label, chunk, batch_size=128):
    """items: list of (text, key). Tag in chunks, reloading spaCy per chunk to
    FLUSH the StringStore (which otherwise grows unbounded over millions of unique
    tokens and drives the process into swap). Keeps only NOUN/PROPN lemmas, interned
    so duplicates across items share one string object. Returns {key: frozenset}.
    Prints throughput + ETA per chunk so the long run is never invisible."""
    out = {}
    total = len(items)
    t0 = time.time()
    done = 0
    for start in range(0, total, chunk):
        sub = items[start:start + chunk]
        nlp = spacy.load("en_core_web_lg", disable=DISABLE)
        nlp.max_length = 3_000_000
        for doc, key in nlp.pipe(((t, k) for t, k in sub), as_tuples=True, batch_size=batch_size):
            out[key] = frozenset(sys.intern(t.lemma_.lower())
                                 for t in doc if t.pos_ in ("NOUN", "PROPN"))
            done += 1
        # release this chunk's source text and the pipeline (with its grown vocab)
        for j in range(start, min(start + chunk, total)):
            items[j] = None
        del nlp, sub
        gc.collect()
        el = time.time() - t0
        rate = done / el if el else 0
        eta = (total - done) / rate / 60 if rate else 0
        print(f"  [{label}] {done}/{total}  {rate:.0f}/s  elapsed {el/60:.1f}m  eta {eta:.1f}m",
              file=sys.stderr, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="brad,june,rene,april,hana,nina,paula",
                    help="comma list, or 'all'")
    ap.add_argument("--topics", type=int, default=40, help="first N topics by index")
    ap.add_argument("--out-csv", action="store_true",
                    help="persist per-(run,topic) detail + per-run summary CSVs")
    ap.add_argument("--cache-dir", default=os.path.join(OUT, "shard_cache"),
                    help="per-run pickle cache of cited-doc concept sets (resume support)")
    args = ap.parse_args()

    all_runs = [r for r in sorted(os.listdir(GEN)) if os.path.isfile(os.path.join(GEN, r))]
    selected = set(all_runs) if args.runs == "all" else set(args.runs.split(","))
    os.makedirs(args.cache_dir, exist_ok=True)

    def cache_path(run):
        # cache is keyed by run AND topic scope, so a wider --topics run re-tags cleanly
        return os.path.join(args.cache_dir, f"{run}.t{args.topics}.pkl")

    cached_runs = {r for r in selected if os.path.exists(cache_path(r))}
    if cached_runs:
        print(f"cache: {len(cached_runs)}/{len(selected)} selected runs already pickled "
              f"(spaCy inference will be skipped for them)", file=sys.stderr)

    # ---- pass 1: collect answer texts (all runs) + cited-doc texts (uncached selected) ----
    ans_keys, ans_texts = [], []
    cited = {}                       # (run,topic)->set(shards) for selected runs
    run_shard_ids = defaultdict(set)  # run -> all shards it cites (for pickling)
    shard_text = {}                  # shard->text (only for uncached selected runs)
    run_team = {}
    for run in all_runs:
        for line in open(os.path.join(GEN, run)):
            if not line.strip():
                continue
            d = json.loads(line)
            topic = d["metadata"]["topic_id"]
            if tnum(topic) >= args.topics:
                continue
            run_team[run] = d["metadata"].get("team_id", "?")
            parts, cset = [], set()
            for s in d.get("answer") or d.get("responses") or []:
                parts.append(s.get("text", ""))
                for shard in (s.get("citations") or {}):
                    cset.add(shard)
            ans_keys.append((run, topic))
            ans_texts.append(clean(" ".join(parts)))
            if run in selected:
                cited[(run, topic)] = frozenset(cset)
                run_shard_ids[run] |= cset
                if run not in cached_runs:      # cached runs load concepts from pickle
                    docs = d.get("documents", {})
                    for shard in cset:
                        if shard not in shard_text:
                            doc = docs.get(shard)
                            if isinstance(doc, dict):
                                shard_text[shard] = clean(doc.get("text", ""))
    print(f"answers={len(ans_keys)} (all runs), selected={sorted(selected)}, "
          f"uncached docs to tag={len(shard_text)}", file=sys.stderr)

    # ---- tag answers -> concept sets + cross-run df per topic ----
    ans_items = list(zip(ans_texts, ans_keys))
    del ans_texts
    ans_concepts = tag_sets(ans_items, "answers", chunk=3000)
    del ans_items
    df = defaultdict(Counter)        # topic -> concept -> #runs stating it
    nruns_topic = Counter()
    for (run, topic), cs in ans_concepts.items():
        nruns_topic[topic] += 1
        for c in cs:
            df[topic][c] += 1

    # ---- cited-doc concepts: load per-run pickle if present, else tag & save ----
    # shard_nouns is a global in-execution cache so a doc cited by several runs is
    # tagged once; each run's pickle is self-contained (all its cited docs' concepts),
    # so a re-run skips spaCy entirely for any run already pickled.
    shard_nouns = {}
    for run in sorted(cached_runs):
        with open(cache_path(run), "rb") as fh:
            shard_nouns.update(pickle.load(fh))
    if cached_runs:
        print(f"  [cache] loaded {len(cached_runs)} run pickles "
              f"({len(shard_nouns)} doc concept sets)", file=sys.stderr, flush=True)

    to_tag = sorted(selected - cached_runs)
    for i, run in enumerate(to_tag, 1):
        todo = [(shard_text[s], s) for s in run_shard_ids[run]
                if s in shard_text and s not in shard_nouns]
        if todo:
            shard_nouns.update(tag_sets(todo, f"docs {run} ({i}/{len(to_tag)})", chunk=8000))
        # persist THIS run's full cited-doc concept map (incl. docs tagged for other runs)
        run_map = {s: shard_nouns[s] for s in run_shard_ids[run] if s in shard_nouns}
        tmp = cache_path(run) + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(run_map, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cache_path(run))   # atomic: a crash never leaves a partial pickle
        for s in run_shard_ids[run]:       # free text as each run completes
            shard_text.pop(s, None)
    shard_text.clear()

    # ---- cross ----
    per_run = defaultdict(lambda: {"prec": [], "ung": [], "pu": [], "pu_of_ung": [],
                                   "redundancy": [], "eff_docs": [], "cites": [],
                                   "examples": Counter()})
    detail_rows = []
    for (run, topic), cset in cited.items():
        sn = ans_concepts[(run, topic)]
        if not sn:
            continue
        hn = set()
        doc_sizes = []                             # concept counts of each cited doc
        for shard in cset:
            dn = shard_nouns.get(shard)
            if dn:
                hn |= dn
                doc_sizes.append(len(dn))
        if not hn:
            continue
        # source diversity of the cited set (from cached concept sets; no extra inference):
        #   redundancy    = 1 - |union| / sum(|doc|)   0=disjoint docs, ->1=near-duplicates
        #   effective_docs= |union| / mean(|doc|)       "doc-equivalents" of unique concept mass
        sum_sizes = sum(doc_sizes)
        hn_size = len(hn)
        redundancy = 1.0 - hn_size / sum_sizes if sum_sizes else 0.0
        effective_docs = (hn_size / (sum_sizes / len(doc_sizes))
                          if doc_sizes and sum_sizes else 0.0)
        ungrounded = sn - hn                       # precision false positives
        private_ung = {c for c in ungrounded if df[topic][c] == 1}
        s = per_run[run]
        s["prec"].append(len(sn & hn) / len(sn))
        s["ung"].append(len(ungrounded) / len(sn))
        s["pu"].append(len(private_ung) / len(sn))
        s["pu_of_ung"].append(len(private_ung) / len(ungrounded) if ungrounded else 0.0)
        s["redundancy"].append(redundancy)
        s["eff_docs"].append(effective_docs)
        s["cites"].append(len(cset))
        for c in private_ung:
            s["examples"][c] += 1
        if args.out_csv:
            detail_rows.append({
                "run": run, "team": run_team[run], "topic": topic,
                "n_answer_concepts": len(sn), "n_cited_docs": len(cset),
                # source diversity of the cited set (see above)
                "hn_size": hn_size, "redundancy": round(redundancy, 4),
                "effective_docs": round(effective_docs, 4),
                "precision": round(len(sn & hn) / len(sn), 4),
                "ungrounded_rate": round(len(ungrounded) / len(sn), 4),
                "private_ung_rate": round(len(private_ung) / len(sn), 4),
                "n_ungrounded": len(ungrounded), "n_private_ung": len(private_ung),
                # the actual hallucination-candidate concepts (ungrounded AND private)
                "private_ung_concepts": "|".join(sorted(private_ung)),
            })

    # ================= report =================
    print("\n" + "=" * 74)
    print(f"HALLUCINATION CROSS  (first {args.topics} topics; df over all "
          f"{max(nruns_topic.values())} runs)")
    print("=" * 74)
    print("  precision   = answer concepts grounded in cited docs")
    print("  ungrounded  = 1 - precision  (concept in answer, absent from cited docs)")
    print("  private_ung = ungrounded AND stated by NO other run  <- hallucination candidate")
    print("  %ung_priv   = share of a run's ungrounded concepts that are private\n")
    print(f"  {'run':<8}{'team':<7}{'prec':>7}{'ungrnd':>8}{'priv_ung':>9}{'%ung_priv':>10}"
          f"{'cites':>7}{'redund':>8}{'eff_dc':>8}{'topics':>7}")
    rows = sorted(per_run.items(), key=lambda kv: -statistics.mean(kv[1]["pu"]))
    for run, s in rows:
        print(f"  {run:<8}{run_team[run]:<7}"
              f"{statistics.mean(s['prec']):>7.3f}{statistics.mean(s['ung']):>8.3f}"
              f"{statistics.mean(s['pu']):>9.3f}{statistics.mean(s['pu_of_ung']):>10.3f}"
              f"{statistics.mean(s['cites']):>7.1f}{statistics.mean(s['redundancy']):>8.3f}"
              f"{statistics.mean(s['eff_docs']):>8.2f}{len(s['prec']):>7}")

    def ranked_examples(counter, n):
        # deterministic: count desc, then concept asc (most_common ties are insertion-order)
        return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:n]

    print("\n--- Top private-ungrounded concepts per run (freq across its topics) ---")
    for run, s in rows:
        top = ", ".join(f"{c}({n})" for c, n in ranked_examples(s["examples"], 12))
        print(f"  {run:<8}[{run_team[run]}]: {top}")

    # ---- persist CSVs ----
    if args.out_csv:
        os.makedirs(OUT, exist_ok=True)
        detail_rows.sort(key=lambda r: (r["run"], tnum(r["topic"])))
        with open(f"{OUT}/hallucination_by_run_topic.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
            w.writeheader(); w.writerows(detail_rows)
        with open(f"{OUT}/hallucination_by_run.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run", "team", "precision", "ungrounded_rate", "private_ung_rate",
                        "pct_ungrounded_private", "avg_cites", "redundancy", "effective_docs",
                        "n_topics", "top_private_ung_concepts"])
            for run, s in rows:
                w.writerow([run, run_team[run],
                            round(statistics.mean(s["prec"]), 4),
                            round(statistics.mean(s["ung"]), 4),
                            round(statistics.mean(s["pu"]), 4),
                            round(statistics.mean(s["pu_of_ung"]), 4),
                            round(statistics.mean(s["cites"]), 2),
                            round(statistics.mean(s["redundancy"]), 4),
                            round(statistics.mean(s["eff_docs"]), 4),
                            len(s["prec"]),
                            "|".join(c for c, _ in ranked_examples(s["examples"], 25))])
        print(f"\nCSVs: {OUT}/hallucination_by_run_topic.csv  hallucination_by_run.csv")


if __name__ == "__main__":
    main()
