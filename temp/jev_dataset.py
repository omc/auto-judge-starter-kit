"""Dataset switch for the Jev study scripts: JEV_DATASET=rag26 (default) | ragtime26.

Every script reads paths, probe topics and output locations from DS, so the same
experiment can be replicated on another dataset without overwriting rag26 results.
"""
import json
import os
from pathlib import Path

from autojudge_base import Request

from judges.bonsai_judge.pairwise import topic_fields

DATASETS = {
    "rag26": {
        "runs": "data/rag26/runs/generation",
        "topics": "data/rag26/topics/trec_rag_2026_queries.jsonl",
        "probe_topics": ["rag2026-0", "rag2026-1", "rag2026-10"],   # 3-topic experiments
        "det_topic": "rag2026-100",                                 # uncached, determinism
        "offtopic": "rag2026-50",                                   # padding irrelevant source
        "probes": "temp/jev_probes",
        "prior": "output-pairwise-jev-prompts/bonsai_pairwise_jev.pairwise/comparisons.jsonl",
        "gemini": "temp/rag26_pairwise/bonsai_pairwise.pairwise/comparisons.jsonl",
        "tag": "",
    },
    "ragtime26": {
        "runs": "data/ragtime26/runs/repgen",
        "topics": "data/ragtime26/topics/topics.all.2026.v0625-fix.request.jsonl",
        "probe_topics": ["2000", "2001", "2002"],
        "det_topic": "2003",
        "offtopic": "2050",
        "probes": "temp/jev_probes_ragtime26",
        "prior": "output-ragtime26-jev-prompts/bonsai_pairwise_jev.pairwise/comparisons.jsonl",
        "gemini": "output-ragtime26-gemini/bonsai_pairwise.pairwise/comparisons.jsonl",
        "tag": "_ragtime26",
    },
}
NAME = os.environ.get("JEV_DATASET", "rag26")
DS = {**DATASETS[NAME], "name": NAME}


def runs_dir() -> Path:
    return Path(DS["runs"])


def topic_info() -> dict:
    """request_id -> topic_fields dict ({query, [problem_statement], [background]})."""
    out = {}
    for line in open(DS["topics"]):
        t = Request.model_validate(json.loads(line))
        out[t.request_id] = topic_fields(t)
    return out


def word_lengths(topics=None) -> dict:
    """{topic: {run: words}} from the run files (same text the judge sees, uncleaned)."""
    from collections import defaultdict
    L = defaultdict(dict)
    for f in runs_dir().iterdir():
        for line in open(f):
            r = json.loads(line)
            t = r["metadata"]["topic_id"]
            if topics is None or t in topics:
                L[t][r["metadata"]["run_id"]] = len(" ".join(s["text"] for s in r["responses"]).split())
    return L
