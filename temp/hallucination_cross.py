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
import argparse, html, json, os, re, statistics, sys
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="brad,june,rene,april,hana,nina,paula",
                    help="comma list, or 'all'")
    ap.add_argument("--topics", type=int, default=40, help="first N topics by index")
    args = ap.parse_args()

    nlp = spacy.load("en_core_web_lg", disable=DISABLE)
    nlp.max_length = 3_000_000

    all_runs = [r for r in sorted(os.listdir(GEN)) if os.path.isfile(os.path.join(GEN, r))]
    selected = set(all_runs) if args.runs == "all" else set(args.runs.split(","))

    # ---- pass 1: collect answer texts (all runs) + cited-doc texts (selected) ----
    ans_keys, ans_texts = [], []
    cited = {}                       # (run,topic)->set(shards) for selected runs
    shard_text = {}                  # shard->text (selected runs' cites only)
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
                docs = d.get("documents", {})
                for shard in cset:
                    if shard not in shard_text:
                        doc = docs.get(shard)
                        if isinstance(doc, dict):
                            shard_text[shard] = clean(doc.get("text", ""))
    print(f"answers={len(ans_keys)} (all runs), selected={sorted(selected)}, "
          f"unique cited docs to tag={len(shard_text)}", file=sys.stderr)

    # ---- tag answers -> concept sets + cross-run df per topic ----
    ans_concepts = {}
    for doc, key in nlp.pipe(zip(ans_texts, ans_keys), as_tuples=True, batch_size=256):
        ans_concepts[key] = frozenset(t.lemma_.lower() for t in doc if t.pos_ in ("NOUN", "PROPN"))
    df = defaultdict(Counter)        # topic -> concept -> #runs stating it
    nruns_topic = Counter()
    for (run, topic), cs in ans_concepts.items():
        nruns_topic[topic] += 1
        for c in cs:
            df[topic][c] += 1
    print("  answers tagged", file=sys.stderr)

    # ---- tag selected runs' cited docs -> concept sets ----
    shard_nouns = {}
    for doc, sid in nlp.pipe(((t, sid) for sid, t in shard_text.items()),
                             as_tuples=True, batch_size=128):
        shard_nouns[sid] = frozenset(t.lemma_.lower() for t in doc if t.pos_ in ("NOUN", "PROPN"))
    print("  docs tagged", file=sys.stderr)

    # ---- cross ----
    per_run = defaultdict(lambda: {"prec": [], "ung": [], "pu": [], "pu_of_ung": [],
                                   "examples": Counter()})
    for (run, topic), cset in cited.items():
        sn = ans_concepts[(run, topic)]
        if not sn:
            continue
        hn = set()
        for shard in cset:
            hn |= shard_nouns.get(shard, frozenset())
        if not hn:
            continue
        ungrounded = sn - hn                       # precision false positives
        private_ung = {c for c in ungrounded if df[topic][c] == 1}
        s = per_run[run]
        s["prec"].append(len(sn & hn) / len(sn))
        s["ung"].append(len(ungrounded) / len(sn))
        s["pu"].append(len(private_ung) / len(sn))
        s["pu_of_ung"].append(len(private_ung) / len(ungrounded) if ungrounded else 0.0)
        for c in private_ung:
            s["examples"][c] += 1

    # ================= report =================
    print("\n" + "=" * 74)
    print(f"HALLUCINATION CROSS  (first {args.topics} topics; df over all "
          f"{max(nruns_topic.values())} runs)")
    print("=" * 74)
    print("  precision   = answer concepts grounded in cited docs")
    print("  ungrounded  = 1 - precision  (concept in answer, absent from cited docs)")
    print("  private_ung = ungrounded AND stated by NO other run  <- hallucination candidate")
    print("  %ung_priv   = share of a run's ungrounded concepts that are private\n")
    print(f"  {'run':<8}{'team':<7}{'prec':>7}{'ungrnd':>8}{'priv_ung':>9}{'%ung_priv':>10}{'topics':>7}")
    rows = sorted(per_run.items(), key=lambda kv: -statistics.mean(kv[1]["pu"]))
    for run, s in rows:
        print(f"  {run:<8}{run_team[run]:<7}"
              f"{statistics.mean(s['prec']):>7.3f}{statistics.mean(s['ung']):>8.3f}"
              f"{statistics.mean(s['pu']):>9.3f}{statistics.mean(s['pu_of_ung']):>10.3f}"
              f"{len(s['prec']):>7}")

    print("\n--- Top private-ungrounded concepts per run (freq across its topics) ---")
    for run, s in rows:
        top = ", ".join(f"{c}({n})" for c, n in s["examples"].most_common(12))
        print(f"  {run:<8}[{run_team[run]}]: {top}")


if __name__ == "__main__":
    main()
