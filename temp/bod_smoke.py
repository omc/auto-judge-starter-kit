"""BOD smoke test (kiddie only, unrestricted): does Jev accept a 4-option graded Choice,
and do its probabilities behave like a grade?

Checks, on the kiddie topic 'leaf' with hand-written questions taken from its problem statement:
  1. API accepts a Choice with 4 options (labels "0".."3") + returns probabilities for all four
  2. on-topic answer (run1/leaf) scores higher than an off-topic answer (run1/cloud judged
     against the leaf questions)
  3. option-order invariance: the same scale with criteria listed 3..0 gives similar grades
  4. a Noul padding question alongside the graded ones
Usage: set -a; source .env; set +a; python temp/bod_smoke.py
"""
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from judges.bonsai_judge.pairwise_jev import decisions_url  # noqa: E402

MODEL = "typesafe/jev-1.13"
URL = decisions_url(os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"))
KEY = os.environ["OPENAI_API_KEY"]
DATA = Path("data/kiddie")

SCALE = {
    "0": "The answer does not address this question at all.",
    "1": "The answer mentions this topic but gives no real substance or explanation.",
    "2": "The answer partially answers this question; important parts are missing or vague.",
    "3": "The answer fully answers this question with specific, accurate detail.",
}
QUESTIONS = [
    "Does the answer explain why tree leaves change color in the fall?",
    "Does the answer say whether the leaves are dying when they change color?",
    "Does the answer say whether the tree will be okay after its leaves change and fall?",
    "Does the answer explain why some trees stay green all year?",
    "Does the answer say whether cold weather causes the color change?",
]
PADDING = ("Does the answer contain substantial content that is off-topic, repeated, or generic "
           "filler that does not help with the user's question?")


def topic(tid):
    for line in open(DATA / "topics/kiddie-topics.jsonl"):
        t = json.loads(line)
        if t["request_id"] == tid:
            return {k: t[k] for k in ("title", "problem_statement", "background") if t.get(k)}


def answer(run, tid):
    for line in open(DATA / f"runs/repgen/{run}.jsonl"):
        r = json.loads(line)
        if r["metadata"]["topic_id"] == tid:
            return " ".join(s["text"] for s in r["responses"])


def questions(reverse=False):
    crit = dict(reversed(list(SCALE.items()))) if reverse else SCALE
    qs = {f"q{i+1}": {"type": "choice", "instructions": f"Grade the answer on this question: {q}",
                      "criteria": crit} for i, q in enumerate(QUESTIONS)}
    qs["padding"] = {"type": "noul", "instructions": PADDING}
    return qs


def decide(state, qs):
    body = json.dumps({"model": MODEL, "state": state, "questions": qs}).encode()
    req = urllib.request.Request(URL, body, {"Authorization": f"Bearer {KEY}",
                                             "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read()[:500].decode('utf-8', 'replace')}")
        sys.exit(1)


def expected(a):
    p = {k: float(v) for k, v in (a.get("probabilities") or {}).items()}
    s = sum(p.values())
    return sum(int(k) * v for k, v in p.items()) / s if s else None, p


def show(label, resp):
    print(f"\n== {label}")
    grades = []
    for qid, a in resp["answers"].items():
        if a.get("type") == "choice":
            e, p = expected(a)
            grades.append(e)
            print(f"  {qid}: E[g]={e:.2f}  choice={a.get('choice')}  "
                  + " ".join(f"{k}:{p[k]:.2f}" for k in sorted(p)))
        else:
            print(f"  {qid}: {a.get('type')} {a.get('noul')}")
    m = sum(grades) / len(grades) / 3
    print(f"  BOD_MEAN={m:.3f}")
    return m


leaf = topic("leaf")
r1 = decide({**leaf, "answer": answer("run1", "leaf")}, questions())
print("raw answer shape (q1):", json.dumps(r1["answers"]["q1"])[:400])
print("other top-level keys:", {k: v for k, v in r1.items() if k != "answers"})
on = show("run1/leaf (on-topic)", r1)
off = show("run1/cloud judged against leaf questions (off-topic)",
           decide({**leaf, "answer": answer("run1", "cloud")}, questions()))
rev = show("run1/leaf, criteria listed 3..0", decide({**leaf, "answer": answer("run1", "leaf")},
                                                      questions(reverse=True)))
r2 = show("run2/leaf (shorter run)", decide({**leaf, "answer": answer("run2", "leaf")}, questions()))
print(f"\nsummary: on {on:.3f} | off {off:.3f} | reversed-order {rev:.3f} | run2 {r2:.3f}")
