"""Padding test: does Jev reward length, or content?

Each mid-ranked TARGET summary is rewritten into variants and judged against fixed
OPPONENTS in BOTH orientations (Jev has a slot-A bias on close pairs, so we average
P(target better) over the two orders). Effect = mean change in P(target beats
opponent) vs the unmodified original, bootstrapped over targets.

  original      unchanged
  pad_repeat    +~50% words: its own sentences duplicated        (no new info)
  pad_generic   +~50% words: generic, topic-agnostic filler      (no info)
  pad_offtopic  +~50% words: sentences from another topic        (irrelevant)
  pad_relevant  +~50% words: unseen sentences from a strong run,
                same topic                                       (positive control)
  truncate      last third of sentences removed                  (less info, shorter)

Padding sentences are interleaved at seeded random positions, not appended.

Usage:
  set -a; source ./.env; set +a
  python temp/jev_padding.py run    [--judge jev|gemini]
  python temp/jev_padding.py report [--judge jev|gemini]

--judge gemini replays the original BonsaiPairwiseJudge setup (pairwise_summary_1.md,
OPENAI_MODEL without ':batch', max_tokens 8, temperature 0, realtime) on the same
items; its binary A/B answer is recorded as P(A) in {0, 1}.
"""
import argparse
import asyncio
import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

from autojudge_base import load_report

from judges.bonsai_judge.bonsai_judge import _clean
from judges.bonsai_judge.pairwise_jev import load_questions
from jev_probes import ask_all  # noqa: E402

import sys
sys.path.insert(0, str(Path(__file__).parent))
from jev_dataset import DS, runs_dir, topic_info  # noqa: E402  JEV_DATASET=rag26|ragtime26

TOPICS = DS["probe_topics"]
OFFTOPIC_SOURCE = DS["offtopic"]    # far-away topic supplying irrelevant sentences
PRIOR = Path(DS["prior"])           # prompt-variant run: control win-rates pick targets
RUNS = runs_dir()
OUTS = {"jev": Path(DS["probes"]) / "padding.jsonl",
        "noul": Path(DS["probes"]) / "padding_noul.jsonl",
        "gemini": Path(DS["probes"]) / "padding_gemini.jsonl"}
N_TARGETS, N_OPPONENTS, PAD = 16, 8, 0.5
VARIANTS = ["original", "pad_repeat", "pad_generic", "pad_offtopic", "pad_relevant", "truncate"]

GENERIC = [
    "It is important to consider multiple perspectives when approaching this question.",
    "Every situation is different, so what works in one context may not work in another.",
    "Careful planning and thoughtful consideration are essential to achieving good outcomes.",
    "Stakeholders should communicate openly and regularly throughout the process.",
    "Research in this area continues to evolve, and new findings may change best practices.",
    "Balancing short-term needs with long-term goals is often a key challenge.",
    "Consulting with qualified professionals can provide additional guidance.",
    "There is no one-size-fits-all solution, and flexibility is important.",
    "Ongoing evaluation helps ensure that efforts remain effective over time.",
    "Clear goals and measurable objectives can help guide decision-making.",
    "Resources should be allocated in a way that reflects priorities.",
    "Building trust takes time and requires consistent effort.",
    "Small, incremental changes can add up to meaningful progress.",
    "It is helpful to learn from the experiences of others who have faced similar issues.",
    "Being aware of potential challenges in advance can make them easier to address.",
    "Documentation and record-keeping support accountability and transparency.",
    "Feedback from those affected should be gathered and taken seriously.",
    "Patience and persistence are often necessary to see results.",
    "Different approaches have different strengths and weaknesses.",
    "Ultimately, the right choice depends on individual circumstances and goals.",
]


def load_sentences(topics):
    """{topic: {run: (team, [sentences])}} with the judge's cleaning."""
    out = defaultdict(dict)
    for f in sorted(RUNS.iterdir()):
        for r in load_report(f):
            t = r.metadata.topic_id
            if t in topics:
                sents = [s for s in (_clean(x.text).strip() for x in r.responses) if s]
                if sents:
                    out[t][r.metadata.run_id] = (r.metadata.team_id, sents)
    return out


def control_winrates():
    """Per-topic win-rate under better_summary from the prompt pilot (both slots)."""
    w, g = defaultdict(float), defaultdict(int)
    for line in open(PRIOR):
        r = json.loads(line)
        if not r["valid"]:
            continue
        p, t = r["p"]["better_summary"], r["topic_id"]
        w[(t, r["a_run"])] += p; w[(t, r["b_run"])] += 1 - p
        g[(t, r["a_run"])] += 1; g[(t, r["b_run"])] += 1
    return {k: w[k] / g[k] for k in g}


def interleave(base, extra, rng):
    out = list(base)
    for s in extra:
        out.insert(rng.randint(0, len(out)), s)
    return out


def take_words(pool, n_words, rng, shuffle=True):
    pool = list(pool)
    if shuffle:
        rng.shuffle(pool)
    out, w = [], 0
    while pool and w < n_words:
        s = pool.pop(0)
        out.append(s); w += len(s.split())
    return out


def variants(sents, rng, generic_pool, offtopic_pool, relevant_pool):
    words = sum(len(s.split()) for s in sents)
    need = int(words * PAD)
    rep = []
    while sum(len(s.split()) for s in rep) < need:     # cycle own sentences if short
        rep += take_words(sents, need - sum(len(s.split()) for s in rep), rng)
    gen = []
    while sum(len(s.split()) for s in gen) < need:
        gen += take_words(generic_pool, need - sum(len(s.split()) for s in gen), rng)
    own = set(sents)
    return {
        "original": sents,
        "pad_repeat": interleave(sents, rep, rng),
        "pad_generic": interleave(sents, gen, rng),
        "pad_offtopic": interleave(sents, take_words(offtopic_pool, need, rng), rng),
        "pad_relevant": interleave(sents, take_words([s for s in relevant_pool if s not in own], need, rng), rng),
        "truncate": sents[: max(1, round(len(sents) * 2 / 3))],
    }


def build():
    data = load_sentences(set(TOPICS) | {OFFTOPIC_SOURCE})
    titles = topic_info()   # {query, [problem_statement], [background]} per topic
    wr = control_winrates()
    offtopic_pool = [s for _team, ss in data[OFFTOPIC_SOURCE].values() for s in ss]
    items, meta = [], []
    for t in TOPICS:
        ranked = sorted((r for r in data[t] if (t, r) in wr), key=lambda r: -wr[(t, r)])
        mid = [r for r in ranked if 0.25 <= wr[(t, r)] <= 0.75]
        targets = [mid[round(i * (len(mid) - 1) / (N_TARGETS - 1))] for i in range(N_TARGETS)]
        targets = list(dict.fromkeys(targets))
        # opponent pool: evenly spread over the whole ranking, excluding targets
        rest = [r for r in ranked if r not in targets]
        pool = [rest[round(i * (len(rest) - 1) / (N_OPPONENTS + 3))] for i in range(N_OPPONENTS + 4)]
        pool = list(dict.fromkeys(pool))
        strong = ranked[0]                     # source of relevant padding
        for tgt in targets:
            team, sents = data[t][tgt]
            src = strong if data[t][strong][0] != team else ranked[1]
            rng = random.Random(f"{t}|{tgt}")
            vs = variants(sents, rng, GENERIC, offtopic_pool, data[t][src][1])
            opps = [o for o in pool if data[t][o][0] != team][:N_OPPONENTS]
            for v, vs_sents in vs.items():
                text = " ".join(vs_sents)
                for o in opps:
                    otext = " ".join(data[t][o][1])
                    for slot in ("A", "B"):
                        a, b = (text, otext) if slot == "A" else (otext, text)
                        tag = {"topic_id": t, "target": tgt, "opponent": o, "variant": v,
                               "target_slot": slot, "target_words": len(text.split()),
                               "orig_words": len(" ".join(sents).split()),
                               "opp_words": len(otext.split()), "target_wr": wr[(t, tgt)]}
                        items.append((tag, {**titles[t], "summary_a": a, "summary_b": b}))
    return items


async def ask_gemini(items):
    """Original pairwise judge path: chat prompt -> single 'A'/'B' token. Cached."""
    from types import SimpleNamespace
    from minima_llm import MinimaLlmRequest
    from judges.bonsai_judge.pairwise import _PROMPT_PATH, _render, BonsaiPairwiseJudge, topic_query_text
    judge = BonsaiPairwiseJudge()
    backend, cfg = judge._make_backend(SimpleNamespace(raw=None), "live", 48, 0)
    template = _PROMPT_PATH.read_text()
    reqs = [MinimaLlmRequest(
                request_id=f"pad{i}",
                messages=[{"role": "user", "content": _render(
                    template, topic_query=topic_query_text(st_),
                    summary_a=st_["summary_a"], summary_b=st_["summary_b"])}],
                temperature=0.0, max_tokens=8)
            for i, (_tag, st_) in enumerate(items)]
    print(f"[padding] gemini model={cfg.model}")
    await backend.run_batched(reqs)
    cache = backend._ensure_cache()
    rows = []
    for (tag, _s), req in zip(items, reqs):
        hit = cache.get(backend._make_cache_key(req))
        choice, ok = judge._parse_choice(hit[0] if hit else None)
        rows.append({**tag, "p": {"better_summary": (1.0 if choice == "A" else 0.0) if ok else None},
                     "raw_text": (hit[0] if hit else None), "err": "" if ok else "missing/unparseable",
                     "served_model": ((hit[1] or {}).get("model") if hit else None),
                     "cost": (((hit[1] or {}).get("usage") or {}).get("cost") if hit else None)})
    await backend.aclose()
    return rows


def run(judge="jev"):
    items = build()
    n_t = len({(i[0]["topic_id"], i[0]["target"]) for i in items})
    print(f"[padding] {len(items)} calls: {n_t} targets x {len(VARIANTS)} variants x "
          f"<= {N_OPPONENTS} opponents x 2 slots")
    if judge == "gemini":
        rows = asyncio.run(ask_gemini(items))
    elif judge == "noul":
        # mirrored Noul: score = mean of P(yes A better) and 1 - P(yes B better)
        rows = asyncio.run(ask_all(items, load_questions(["noul_a_better", "noul_b_better"])))
        for r in rows:
            ya, yb = r["p"]["noul_a_better"], r["p"]["noul_b_better"]
            r["score"] = None if ya is None or yb is None else (ya + 1 - yb) / 2
            r["tie"] = ya is not None and yb is not None and ya < .5 and yb < .5
    else:
        rows = asyncio.run(ask_all(items, load_questions(["better_summary"])))
    OUT = OUTS[judge]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[padding] wrote {OUT}: {sum(bool(r['err']) for r in rows)} errors, "
          f"${sum(r['cost'] or 0 for r in rows):.4f}")


def report(judge="jev"):
    OUT = OUTS[judge]
    rows = [json.loads(l) for l in open(OUT)]
    for r in rows:   # 'score' = P(A better) for this judge; older files used better_summary
        r.setdefault("score", r["p"].get("better_summary"))
    print(f"judge={judge} invalid={sum(r['score'] is None for r in rows)}")
    rows = [r for r in rows if r["score"] is not None]
    if judge == "noul":
        print(f"tie flags (both 'no'): {sum(r['tie'] for r in rows)} / {len(rows)}")
    # P(target better), orientation-averaged per (target, opponent, variant)
    cell = defaultdict(list)
    words = {}
    slot_bias = []
    for r in rows:
        p = r["score"]
        pt = p if r["target_slot"] == "A" else 1 - p
        k = (r["topic_id"], r["target"], r["variant"])
        cell[k + (r["opponent"],)].append(pt)
        words[k] = (r["target_words"], r["orig_words"])
    avg = {k: st.mean(v) for k, v in cell.items() if len(v) == 2}
    # slot bias in this data: P(slot A wins) per cell
    for k, v in cell.items():
        if len(v) == 2:
            slot_bias.append((v[0] + (1 - v[1])) / 2)   # rows written A then B per cell
    tgt_mean = defaultdict(list)
    for (t, tgt, v, _o), p in avg.items():
        tgt_mean[(t, tgt, v)].append(p)
    targets = sorted({(t, tgt) for t, tgt, _v in tgt_mean})
    base = {(t, tgt): st.mean(tgt_mean[(t, tgt, "original")]) for t, tgt in targets}
    rng = random.Random(0)
    print(f"targets={len(targets)} calls={len(rows)} | mean P(slot A wins)={st.mean(slot_bias):.3f}\n")
    print(f"{'variant':14} {'words x':>8} {'P(win)':>7} {'dP':>7} {'95% CI':>17} {'targets dP>0':>13} {'dP>+.05':>8} {'dP<-.05':>8}")
    for v in VARIANTS:
        d = [st.mean(tgt_mean[(t, tgt, v)]) - base[(t, tgt)] for t, tgt in targets]
        boots = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(5000))
        lo, hi = boots[125], boots[4874]
        ratio = st.mean(words[(t, tgt, v)][0] / words[(t, tgt, v)][1] for t, tgt in targets)
        pw = st.mean(st.mean(tgt_mean[(t, tgt, v)]) for t, tgt in targets)
        print(f"{v:14} {ratio:8.2f} {pw:7.3f} {st.mean(d):+7.3f} [{lo:+.3f},{hi:+.3f}] "
              f"{sum(x > 0 for x in d):>7}/{len(d):<5} {sum(x > .05 for x in d):8} {sum(x < -.05 for x in d):8}")
    print("\nper topic dP:")
    for t in TOPICS:
        ts = [x for x in targets if x[0] == t]
        print(f"  {t}: " + "  ".join(
            f"{v}={st.mean(st.mean(tgt_mean[(t, tgt, v)]) - base[(t, tgt)] for _t, tgt in ts):+.3f}"
            for v in VARIANTS[1:]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report", "dry"])
    ap.add_argument("--judge", choices=["jev", "gemini", "noul"], default="jev")
    a = ap.parse_args()
    if a.cmd == "dry":
        items = build()
        print(len(items), "calls")
        ex = [i for i in items if i[0]["variant"] == "pad_generic"][0]
        print(ex[0]); print(ex[1]["summary_a"][:1200])
    else:
        {"run": run, "report": report}[a.cmd](a.judge)
