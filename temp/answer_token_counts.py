#!/usr/bin/env python3
"""Per-answer token counts for the GENERATED SUMMARY (the `answer` array) of every
response in data/rag26/runs/generation, for gpt-6 and qwen-3.8.

Per the tokenlen.py routing these two versions have no published tokenizer, so:
  gpt-6    -> tiktoken o200k_base   (family fallback, exact-lib but approx for v6)
  qwen-3.8 -> Qwen/Qwen2.5-7B       (family fallback HF tokenizer)
Tokenizers are loaded ONCE (loading the Qwen tokenizer per row would be fatal).

Text = space-join of every object's `text` in the `answer` list, matching the
concept pipeline. One CSV row per (run, topic).

  python temp/answer_token_counts.py [--out temp/rag26_consensus/answer_token_counts.csv]
"""
import argparse, csv, json, os, sys

GEN = "data/rag26/runs/generation"
DEFAULT_OUT = "temp/rag26_consensus/answer_token_counts.csv"

OPENAI_ENCODING = "o200k_base"        # gpt-6 family fallback
QWEN_REPO = "Qwen/Qwen2.5-7B"         # qwen-3.8 family fallback


def summary_text(rec):
    """Concatenate all text values across the summary objects. Prefer `answer`;
    fall back to `responses` (a duplicate in this dataset, kept for robustness)."""
    ans = rec.get("answer") or rec.get("responses") or []
    return " ".join(o.get("text", "") for o in ans), len(ans)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--gen-dir", default=GEN)
    args = ap.parse_args(argv)

    import tiktoken
    from transformers import AutoTokenizer
    enc = tiktoken.get_encoding(OPENAI_ENCODING)
    qwen = AutoTokenizer.from_pretrained(QWEN_REPO, trust_remote_code=False)

    # ---- collect (metadata, text) across all runs; drop empty summaries ----
    meta, texts, n_items = [], [], []
    n_dropped = 0
    for run in sorted(os.listdir(args.gen_dir)):
        p = os.path.join(args.gen_dir, run)
        if not os.path.isfile(p):
            continue
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                m = d.get("metadata", {})
                txt, nit = summary_text(d)
                if not txt.strip():          # no generated summary -> drop
                    n_dropped += 1
                    continue
                meta.append((m.get("team_id", ""), m.get("run_id", run), m.get("topic_id", "")))
                texts.append(txt)
                n_items.append(nit)
    print(f"collected {len(texts)} answers from {args.gen_dir} "
          f"({n_dropped} empty summaries dropped)", file=sys.stderr)

    # ---- tokenize once, in batch ----
    gpt6 = [len(ids) for ids in enc.encode_batch(texts, disallowed_special=())]
    qids = qwen(texts, add_special_tokens=False)["input_ids"]
    qtok = [len(x) for x in qids]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["team_id", "run_id", "topic_id",
                    "n_answer_items", "n_chars", "gpt6_tokens", "qwen_tokens"])
        for (team, rid, topic), txt, nit, g, q in zip(meta, texts, n_items, gpt6, qtok):
            w.writerow([team, rid, topic, nit, len(txt), g, q])

    print(f"wrote {len(texts)} rows -> {args.out}", file=sys.stderr)
    print(f"  gpt6_tokens : total={sum(gpt6):,}  mean={sum(gpt6)/len(gpt6):.1f}", file=sys.stderr)
    print(f"  qwen_tokens : total={sum(qtok):,}  mean={sum(qtok)/len(qtok):.1f}", file=sys.stderr)


if __name__ == "__main__":
    main()
