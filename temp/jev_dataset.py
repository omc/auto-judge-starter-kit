"""Dataset switch for the Jev study scripts: JEV_DATASET=rag26 (default) | ragtime26.

DATA POLICY (CLAUDE.md): development work -- pilots, probes, prompt tests and every
statistic computed from reports or from the judge's per-comparison outputs -- may use
ONLY the permitted window: the first 10 topics of each topics file, in file order
(rag26: rag2026-0..9, ragtime26: 2000..2009). Everything here enforces that:

  - every configured topic must lie in the window (checked on import of DS)
  - window()            the permitted topic ids, read from the topics file's order
  - assert_window(ids)  raises if any id is outside the window
  - window_records(p)   reads a comparisons/probe jsonl, keeping ONLY window topics
                        (other lines are skipped after reading the topic id only)
  - word_lengths(ts)    report lengths, window topics only

Outputs go to fresh window-only locations (output-win-*, temp/jev_probes_win_*), separate
from earlier runs that included restricted topics.
"""
import json
import os
from pathlib import Path

from autojudge_base import Request

from judges.bonsai_judge.pairwise import topic_fields

WINDOW_SIZE = 10

DATASETS = {
    "rag26": {
        "runs": "data/rag26/runs/generation",
        "topics": "data/rag26/topics/trec_rag_2026_queries.jsonl",
        "window": [f"rag2026-{i}" for i in range(10)],            # first 10 in file order
        "det_topic": "rag2026-3",                                  # determinism (1 topic)
        "probes": "temp/jev_probes_win_rag26",
        "out": "output-win-rag26",                                 # out-dir prefix for runs
        # Gemini baseline on all 10 permitted topics: workflow.pairwise.yml `dev` variant
        # (rag2026-0..6 served from the Sept partial run's cache). Read via window_records.
        "gemini": "output-win-rag26-gemini/bonsai_pairwise.pairwise/comparisons.jsonl",
        "tag": "_win_rag26",
    },
    "ragtime26": {
        "runs": "data/ragtime26/runs/repgen",
        "topics": "data/ragtime26/topics/topics.all.2026.v0625-fix.request.jsonl",
        "window": [str(2000 + i) for i in range(10)],
        "det_topic": "2003",
        "probes": "temp/jev_probes_win_ragtime26",
        "out": "output-win-ragtime26",
        # existing Gemini run on 2000-2002 (all permitted); read via window_records
        "gemini": "output-ragtime26-gemini/bonsai_pairwise.pairwise/comparisons.jsonl",
        "tag": "_win_ragtime26",
    },
}
for _d in DATASETS.values():
    _d["probe_topics"] = list(_d["window"])         # probes / padding: all 10 permitted topics
    _d["narrow_topics"] = list(_d["window"])        # broad-vs-narrow test: all 10
    _d["gemini_padding_topics"] = _d["window"][:3]  # Gemini padding is costly: first 3 only

OFFTOPIC_SHIFT = 5   # padding: topic i takes off-topic sentences from window topic i+5 (mod 10)

NAME = os.environ.get("JEV_DATASET", "rag26")
DS = {**DATASETS[NAME], "name": NAME}
DS["prior"] = f"{DS['out']}-jev-prompts/bonsai_pairwise_jev.pairwise/comparisons.jsonl"

_configured = (DS["probe_topics"] + [DS["det_topic"]] + DS["narrow_topics"]
               + DS["gemini_padding_topics"])
assert set(_configured) <= set(DS["window"]), f"configured topics outside the window: {_configured}"


def offtopic_for(topic_id: str) -> str:
    """Off-topic padding source for a window topic: a different window topic (fixed shift)."""
    w = DS["window"]
    return w[(w.index(topic_id) + OFFTOPIC_SHIFT) % len(w)]


def runs_dir() -> Path:
    return Path(DS["runs"])


def window() -> list:
    """Permitted topic ids: the first WINDOW_SIZE topics of the topics file, in file order.
    Cross-checked against the configured list so a reordered file can't widen it."""
    ids = []
    for line in open(DS["topics"]):
        ids.append(json.loads(line)["request_id"])
        if len(ids) == WINDOW_SIZE:
            break
    assert ids == DS["window"], f"topics-file window {ids} != configured {DS['window']}"
    return ids


def in_window(topic_id: str) -> bool:
    return topic_id in DS["window"]


def assert_window(topic_ids) -> None:
    bad = sorted(set(topic_ids) - set(DS["window"]))
    if bad:
        raise ValueError(f"restricted topics requested (outside the permitted window): {bad}")


def window_records(path, stats: dict = None):
    """Records of a jsonl file whose topic_id is in the window; other lines are skipped
    after reading only their topic id. stats (optional) receives kept/skipped counts."""
    kept = skipped = 0
    out = []
    for line in open(path):
        r = json.loads(line)
        if in_window(r.get("topic_id", "")):
            out.append(r); kept += 1
        else:
            skipped += 1
    if stats is not None:
        stats.update(kept=kept, skipped=skipped)
    if skipped:
        print(f"[window] {Path(path).name}: kept {kept:,} records on permitted topics, "
              f"skipped {skipped:,} on restricted topics (not read)")
    return out


def topic_info() -> dict:
    """request_id -> topic_fields dict ({query, [problem_statement], [background]}).
    Topics are not restricted; all are returned."""
    out = {}
    for line in open(DS["topics"]):
        t = Request.model_validate(json.loads(line))
        out[t.request_id] = topic_fields(t)
    return out


def word_lengths(topics=None) -> dict:
    """{topic: {run: words}} for WINDOW topics only (default: the whole window)."""
    from collections import defaultdict
    topics = set(DS["window"] if topics is None else topics)
    assert_window(topics)
    L = defaultdict(dict)
    for f in runs_dir().iterdir():
        for line in open(f):
            r = json.loads(line)
            t = r["metadata"]["topic_id"]
            if t in topics:
                L[t][r["metadata"]["run_id"]] = len(" ".join(s["text"] for s in r["responses"]).split())
    return L
