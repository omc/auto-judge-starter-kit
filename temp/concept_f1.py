#!/usr/bin/env python3
"""Concept-F1 for RAG26 generation runs.

Faithful port of temp/concept-f1-reference.py (maxirwin.com/articles/llm-rag/):
  * spaCy en_core_web_lg, same disabled pipes -> keeps tagger + attribute_ruler + lemmatizer
  * concept = token.lemma_.lower() where token.pos_ in (NOUN, PROPN), as a SET
  * text cleaned with html.unescape(re.sub(r"<.*?>","",text)) before parsing
  * reference (hn) = lemma-nouns of the CITED docs; candidate (sn) = lemma-nouns of the summary
  * precision = 1 - |sn-hn|/|sn|  ;  recall = 1 - |hn-sn|/|hn|  ;  f1 = 2PR/(P+R)
  * records with empty hn or empty sn are SKIPPED (as in the reference), not scored 0

RAG26 mapping: a "cited search result" = a shard cited with weight>0 (reference used [n]
markers into hit snippets; here citations are explicit shard->weight). Note the blog cited
short title+description snippets, whereas RAG26 cites full documents -> far larger hn, so
recall runs lower here. That is a corpus difference, not a formula change.
"""
import argparse, csv, html, json, os, re, statistics, sys
from collections import defaultdict
from sys import intern
import spacy

MODEL = os.environ.get("CF1_MODEL", "en_core_web_lg")
DISABLE = ["parser", "senter", "ner", "entity_ruler", "textcat",
           "morphologizer", "trainable_lemmatizer"]
GEN = "data/rag26/runs/generation"
OUT = "temp/rag26_consensus"
TAG_RE = re.compile(r"<.*?>")


def clean(text):
    return html.unescape(TAG_RE.sub("", text or ""))


def doc_nouns(doc):
    return frozenset(intern(t.lemma_.lower()) for t in doc
                     if t.pos_ in ("NOUN", "PROPN"))


def prf(sn, hn):
    """precision, recall, f1 on two non-empty lemma-noun sets (reference formulas)."""
    precision = 1 - len(sn - hn) / len(sn)
    recall = 1 - len(hn - sn) / len(hn)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _selftest():
    # workforces/hospitals lemmatize to match singular forms in the reference
    sn = {"nurse", "boston", "diversity", "unicorn"}   # unicorn = hallucination (FP)
    hn = {"nurse", "boston", "diversity", "workforce"}  # workforce uncovered (FN)
    p, r, f1 = prf(sn, hn)
    assert abs(p - 0.75) < 1e-9 and abs(r - 0.75) < 1e-9 and abs(f1 - 0.75) < 1e-9, (p, r, f1)
    print("selftest OK", file=sys.stderr)


def iter_records(runs):
    for run in runs:
        with open(os.path.join(GEN, run)) as fh:
            for line in fh:
                if line.strip():
                    yield run, json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="first N runs only (validation)")
    ap.add_argument("--nproc", type=int, default=min(8, (os.cpu_count() or 2) - 2))
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--tag", default="concept", help="output filename tag")
    args = ap.parse_args()
    _selftest()

    nlp = spacy.load(MODEL, disable=DISABLE)
    nlp.max_length = 3_000_000
    print(f"model={MODEL} pipes={nlp.pipe_names} nproc={args.nproc}", file=sys.stderr)

    runs = [r for r in sorted(os.listdir(GEN)) if os.path.isfile(os.path.join(GEN, r))]
    if args.limit:
        runs = runs[:args.limit]

    # ---- collect cited sets + summaries; find unique shard texts (streamed) ----
    cited = {}          # (run,topic) -> frozenset(shards)
    summaries = {}      # (run,topic) -> cleaned summary text
    shard_first_text = {}  # shard_id -> cleaned text (first occurrence)
    for run, d in iter_records(runs):
        topic = d["metadata"]["topic_id"]
        key = (run, topic)
        ans = d.get("answer") or d.get("responses") or []
        parts, cset = [], set()
        for sent in ans:
            parts.append(sent.get("text", ""))
            for shard, w in (sent.get("citations") or {}).items():
                if w and w > 0:
                    cset.add(shard)
        summaries[key] = clean(" ".join(parts))
        cited[key] = frozenset(cset)
        docs = d.get("documents", {})
        for shard in cset:
            if shard not in shard_first_text:
                doc = docs.get(shard)
                if isinstance(doc, dict):
                    shard_first_text[shard] = clean(doc.get("text", ""))
    print(f"records={len(summaries)} unique_cited_shards={len(shard_first_text)}", file=sys.stderr)

    # ---- tag unique shards (as_tuples keeps the id alongside each doc) ----
    shard_nouns = {}
    shard_items = list(shard_first_text.items())          # [(shard, text)]
    stream = ((text, sid) for sid, text in shard_items)
    done = 0
    for doc, sid in nlp.pipe(stream, as_tuples=True, batch_size=args.batch, n_process=args.nproc):
        shard_nouns[sid] = doc_nouns(doc)
        done += 1
        if done % 5000 == 0:
            print(f"  shards tagged {done}/{len(shard_items)}", file=sys.stderr)
    shard_first_text.clear()

    # ---- tag summaries ----
    summary_nouns = {}
    s_items = list(summaries.items())                     # [((run,topic), text)]
    stream = ((text, key) for key, text in s_items)
    done = 0
    for doc, key in nlp.pipe(stream, as_tuples=True, batch_size=args.batch, n_process=args.nproc):
        summary_nouns[key] = doc_nouns(doc)
        done += 1
        if done % 3000 == 0:
            print(f"  summaries tagged {done}/{len(s_items)}", file=sys.stderr)

    # ---- compute metric per (run,topic) ----
    rows = []
    per_run = defaultdict(lambda: {"p": [], "r": [], "f1": []})
    skipped = defaultdict(int)
    for key, cset in cited.items():
        run, topic = key
        hn = set()
        for shard in cset:
            hn |= shard_nouns.get(shard, frozenset())
        sn = summary_nouns.get(key, frozenset())
        if not hn or not sn:
            skipped[run] += 1
            continue
        p, r, f1 = prf(sn, hn)
        rows.append([run, topic, round(p, 4), round(r, 4), round(f1, 4),
                     len(sn & hn), len(sn - hn), len(hn - sn), len(cset)])
        per_run[run]["p"].append(p); per_run[run]["r"].append(r); per_run[run]["f1"].append(f1)

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{args.tag}_f1_by_topic.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "topic", "precision", "recall", "f1", "tp", "fp", "fn", "n_cited_docs"])
        w.writerows(rows)

    # per-run means: (run, precision, recall, f1, n_topics, n_skipped)
    stats = [(run, statistics.mean(s["p"]), statistics.mean(s["r"]),
              statistics.mean(s["f1"]), len(s["f1"]), skipped[run])
             for run, s in per_run.items()]
    by_prec = sorted(stats, key=lambda x: -x[1])
    with open(f"{OUT}/{args.tag}_f1_leaderboard.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "mean_precision", "mean_recall", "mean_f1", "n_topics", "n_skipped"])
        for run, p, r, f1, n, sk in by_prec:
            w.writerow([run, f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}", n, sk])

    # ---- report: precision and recall as SEPARATE axes (not collapsed to F1) ----
    def stat(vals):
        return f"min={min(vals):.3f} median={statistics.median(vals):.3f} max={max(vals):.3f}"
    allp = [s[1] for s in stats]; allr = [s[2] for s in stats]; allf1 = [s[3] for s in stats]
    print("\n" + "=" * 66)
    print(f"CONCEPT-F1  ({MODEL}, lemmatized NOUN+PROPN, {len(runs)} runs)")
    print("=" * 66)
    print("\nPrecision and recall reported separately (see AskUserQuestion decision):")
    print("  precision = grounding: fraction of summary concepts present in cited docs")
    print("  recall    = coverage:  fraction of cited-doc concepts present in summary")
    print("             (length-driven with full-doc references -- read cautiously)")
    print(f"\nPer-run means over topics:")
    print(f"  PRECISION : {stat(allp)}")
    print(f"  RECALL    : {stat(allr)}")
    print(f"  F1 (ref.) : {stat(allf1)}")

    def table(title, key, rev=True, k=15):
        ranked = sorted(stats, key=lambda x: (-x[key] if rev else x[key]))
        print(f"\n--- {title} ---")
        print(f"  {'run':<10}{'prec':>7}{'rec':>7}{'F1':>7}{'topics':>7}{'skip':>6}")
        for run, p, r, f1, n, sk in ranked[:k]:
            print(f"  {run:<10}{p:>7.3f}{r:>7.3f}{f1:>7.3f}{n:>7}{sk:>6}")

    table("Top 15 by PRECISION (best-grounded / least hallucinatory)", 1)
    table("Top 15 by RECALL (most complete coverage of cited docs)", 2)
    print(f"\nCSVs: {OUT}/{args.tag}_f1_leaderboard.csv  {args.tag}_f1_by_topic.csv")


if __name__ == "__main__":
    main()
