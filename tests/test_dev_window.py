"""Development work must stay on PERMITTED topics (data policy, CLAUDE.md).

The permitted window is the first 10 topics of each evaluation topics file
(rag26: rag2026-0..9, ragtime26: 2000..2009); kiddie is unrestricted. Pilots, probes,
prompt tests and their analyses may only touch these. Full runs (variants named
full*) judge everything -- that is the task -- and are exempt.

This test needs no data files, so it also runs inside the TIRA code-submission check.
"""
import json
import re
from pathlib import Path

import yaml

REPO = Path(__file__).parent.parent
PERMITTED = ({f"rag2026-{i}" for i in range(10)}
             | {str(2000 + i) for i in range(10)}
             | {"leaf", "cloud", "bee", "earthworms", "hibernation"})          # kiddie
# topic ids of the 2026 evaluation sets outside the window
RESTRICTED_RE = re.compile(r"\brag2026-(?:[1-9]\d+)\b|(?<![\w.])\b20(?:1\d|[2-9]\d|10\d)\b(?![\w.])")


def _workflows():
    return sorted((REPO / "judges").glob("*/workflow*.yml"))


def test_dev_variants_are_window_only():
    checked = 0
    for wf in _workflows():
        variants = (yaml.safe_load(wf.read_text()) or {}).get("variants") or {}
        for name, v in variants.items():
            js = (v or {}).get("judge_settings") or {}
            if name.startswith("full"):
                continue
            uses_topic_limits = "max_topics" in js or "dev_topics" in js
            if not uses_topic_limits and "pairwise" not in wf.name:
                continue                       # non-pairwise judges have no dev variants
            assert "dev_topics" in js, f"{wf.name}:{name} is a development variant without dev_topics"
            bad = set(map(str, js["dev_topics"])) - PERMITTED
            assert not bad, f"{wf.name}:{name} dev_topics include restricted topics {sorted(bad)}"
            checked += 1
    assert checked, "no development variants found"


def test_study_config_is_window_only():
    import importlib.util
    spec = importlib.util.spec_from_file_location("jev_dataset", REPO / "temp" / "jev_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)               # asserts configured topics lie in DS window
    for name, d in mod.DATASETS.items():
        used = (set(d["probe_topics"]) | {d["det_topic"]} | set(d["narrow_topics"])
                | set(d["gemini_padding_topics"]))
        assert set(d["window"]) <= PERMITTED, name
        assert used <= set(d["window"]), f"{name}: restricted topics configured {sorted(used - set(d['window']))}"
    mod_ds = mod.DS
    for t in mod_ds["window"]:                  # off-topic padding sources stay in the window
        assert mod.offtopic_for(t) in mod_ds["window"] and mod.offtopic_for(t) != t


def test_narrow_topic_files_are_window_only():
    for f in (REPO / "temp").glob("narrow_topics_*.jsonl"):
        ids = {json.loads(l)["request_id"] for l in f.read_text().splitlines() if l.strip()}
        assert ids <= PERMITTED, f"{f.name}: restricted topics {sorted(ids - PERMITTED)}"


def test_no_hardcoded_restricted_topics_in_study_code():
    files = list((REPO / "temp").glob("jev_*.py")) + list((REPO / "temp").glob("run_*.sh"))
    for f in files:
        for i, line in enumerate(f.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            m = RESTRICTED_RE.search(code)
            assert not m, f"{f.name}:{i} references restricted topic {m.group(0)!r}: {line.strip()}"
