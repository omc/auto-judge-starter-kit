"""Window-only check of the BOD x Hallucination Risk criteria (temp/BOD_RISK_PLAN.md).
Usage: JEV_DATASET=rag26|ragtime26 python temp/bod_risk_analysis.py <out-dir of the pilot variant>

Reads <out>/bonsai_bod_risk.eval.txt, .bod/risk.jsonl, .bod/run_manifest.json and the
window reports (for the pool-stability recomputation and word counts). Window topics only.
Prints aggregates only, never concepts or text.
"""
import json
import random
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from jev_dataset import DS, in_window, runs_dir, word_lengths  # noqa: E402

from autojudge_base import load_report  # noqa: E402
from judges.bonsai_judge.bod_risk import _answer_and_citations, _tag  # noqa: E402

out = Path(sys.argv[1])
fb = out / "bonsai_bod_risk"
art = out / "bonsai_bod_risk.bod"

# ---- per-topic measures from the leaderboard (window topics only) ----
V = defaultdict(dict)                       # measure -> {(topic, run): value}
for line in open(f"{fb}.eval.txt"):
    p = line.split()
    if len(p) != 4 or p[1] == "all" or not in_window(p[1]):
        continue
    V[p[2]][(p[1], p[0])] = float(p[3])
topics = [t for t in DS["window"] if any(k[0] == t for k in V["BOD_MEAN"])]
runs = sorted({r for _, r in V["BOD_MEAN"]})


def board(m, ts):
    s = defaultdict(list)
    for (t, r), v in m.items():
        if t in ts:
            s[r].append(v)
    return {r: st.mean(v) for r, v in s.items()}


def rho(a, b):
    xs = [x for x in a if x in b]
    return spearmanr([a[x] for x in xs], [b[x] for x in xs])[0]


def split_half(m):
    sh = []
    for i in range(200):
        rr = random.Random(i); tt = topics[:]; rr.shuffle(tt)
        sh.append(rho(board(m, set(tt[:len(tt) // 2])), board(m, set(tt[len(tt) // 2:]))))
    return st.mean(sh)


def length_rho(m, L):
    return st.median(spearmanr([m[(t, r)] for r in runs if (t, r) in m and r in L[t]],
                               [L[t][r] for r in runs if (t, r) in m and r in L[t]])[0]
                     for t in topics)


G, B = V["BOD_GROUNDED"], V["BOD_MEAN"]
BG, BB = board(G, set(topics)), board(B, set(topics))
man = json.loads((art / "run_manifest.json").read_text())
risk = [json.loads(l) for l in open(art / "risk.jsonl")]
risk = [r for r in risk if in_window(r["topic_id"])]
L = word_lengths(set(topics))

print(f"{DS['name']} | {len(topics)} window topics, {len(runs)} runs, {len(risk)} non-empty answers")
res = {}
res[1] = (man["new_calls"] == 0 and st.mean(r["defined"] for r in risk) >= .99)
print(f"C1 pipeline: new Jev calls {man['new_calls']}, risk defined "
      f"{sum(r['defined'] for r in risk)}/{len(risk)} -> {'PASS' if res[1] else 'FAIL'}")
r_gb = rho(BG, BB)
res[2], res[3] = r_gb < .99, r_gb >= .70
print(f"C2 not redundant: board rho(GROUNDED, MEAN) {r_gb:.3f} < 0.99 -> {'PASS' if res[2] else 'FAIL'}")
print(f"C3 not dominant:  board rho(GROUNDED, MEAN) {r_gb:.3f} >= 0.70 -> {'PASS' if res[3] else 'FAIL'}")
sg, sb = split_half(G), split_half(B)
res[4] = sg >= sb - .05
print(f"C4 reliability: split-half GROUNDED {sg:.3f} vs MEAN {sb:.3f} (>= {sb - .05:.3f}) "
      f"-> {'PASS' if res[4] else 'FAIL'}")
lg, lb = length_rho(G, L), length_rho(B, L)
res[5] = abs(lg) <= abs(lb) + .15
print(f"C5 length: median per-topic rho(score, words) GROUNDED {lg:+.3f} vs MEAN {lb:+.3f} "
      f"(|.| <= {abs(lb) + .15:.3f}) -> {'PASS' if res[5] else 'FAIL'}")

# ---- C6: pool stability (recompute df from random halves of the other runs) ----
reports = [r for f in sorted(runs_dir().iterdir()) for r in load_report(f)
           if in_window(r.metadata.topic_id)]
items, cited = [], {}
for r in reports:
    text, cset = _answer_and_citations(r)
    if text.strip():
        k = (r.metadata.run_id, r.metadata.topic_id)
        items.append((text, k)); cited[k] = cset
import pickle  # noqa: E402
docs = pickle.loads((art / "doc_concepts.pkl").read_bytes())["docs"]
sn = _tag(items, "en_core_web_lg", True, 5000, 128, "answers")
hn = {k: frozenset().union(*(docs.get(d, frozenset()) for d in cset)) for k, cset in cited.items()}


def grounded_board(pool):
    df = defaultdict(Counter)
    for (run, t), s in sn.items():
        if run in pool:
            df[t].update(s)
    per = {}
    for (t, run), b in B.items():
        k = (run, t)
        s, h = sn.get(k), hn.get(k)
        if not s or not h:              # undefined risk (= 1) or empty report: 0 either way
            per[(t, run)] = 0.0
            continue
        own = 0 if run in pool else 1   # a run outside the pool still counts its own answer
        priv = sum(df[t][c] + own == 1 for c in s - h)
        per[(t, run)] = b * (1 - priv / len(s))
    return board(per, set(topics))


full = grounded_board(set(runs))
check = rho(full, BG)
draws = []
for i in range(20):
    rr = random.Random(1000 + i)
    draws.append(rho(grounded_board(set(rr.sample(runs, len(runs) // 2))), full))
res[6] = min(draws) >= .90
print(f"C6 pool stability: recomputed full-pool board vs judge rho {check:.4f} (sanity); "
      f"half-pool rho min {min(draws):.3f} mean {st.mean(draws):.3f} -> {'PASS' if res[6] else 'FAIL'}")

# ---- diagnostics (no pass/fail) ----
R = {(r["topic_id"], r["run_id"]): r for r in risk}
d_cites = st.median(spearmanr([R[(t, x)]["risk"] for x in runs if (t, x) in R],
                              [R[(t, x)]["n_cited"] for x in runs if (t, x) in R])[0] for t in topics)
d_bod = st.median(spearmanr([R[(t, x)]["risk"] for x in runs if (t, x) in R],
                            [B[(t, x)] for x in runs if (t, x) in R])[0] for t in topics)
rk = sorted(r["risk"] for r in risk)
q = lambda f: rk[int(f * (len(rk) - 1))]
print(f"diag: median per-topic rho(risk, n cited docs) {d_cites:+.3f} | rho(risk, BOD_MEAN) {d_bod:+.3f}")
print(f"diag: risk mean {st.mean(rk):.3f} p25 {q(.25):.3f} p50 {q(.5):.3f} p75 {q(.75):.3f} max {rk[-1]:.3f} | "
      f"ungrounded mean {st.mean(r['ungrounded'] for r in risk):.3f}")
print(f"diag: mean BOD_GROUNDED / BOD_MEAN {st.mean(BG.values()):.3f} / {st.mean(BB.values()):.3f}")
print(f"ALL CRITERIA {'PASS' if all(res.values()) else 'FAIL: ' + str([k for k, v in res.items() if not v])}")
