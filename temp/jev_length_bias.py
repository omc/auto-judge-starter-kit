"""Length-bias diagnostics for pairwise judges on one rag26 topic.
Usage: python temp/jev_length_bias.py <comparisons.jsonl> [topic]"""
import json, sys, statistics as st
from collections import defaultdict
from pathlib import Path
from scipy.stats import spearmanr
from jev_dataset import runs_dir  # JEV_DATASET=rag26|ragtime26

def lengths(topic):
    L = {}
    for f in runs_dir().iterdir():
        for line in open(f):
            r = json.loads(line)
            if r["metadata"]["topic_id"] == topic:
                L[r["metadata"]["run_id"]] = len(" ".join(s["text"] for s in r["responses"]).split())
    return L

def analyze(path, topic=None, label=None):
    recs = [json.loads(l) for l in open(path)]
    topic = topic or recs[0]["topic_id"]
    recs = [r for r in recs if r["topic_id"] == topic and r.get("valid")]
    L = lengths(topic)
    def pa(r):  # P(A wins): soft for jev, hard for binary judges
        return r["p_a"] if "p_a" in r else float(r["winner_run"] == r["a_run"])
    longer_win, bins = [], defaultdict(list)
    w = defaultdict(float); g = defaultdict(int)
    for r in recs:
        la, lb, p = L[r["a_run"]], L[r["b_run"]], pa(r)
        if la != lb:
            pl = p if la > lb else 1 - p
            longer_win.append(pl)
            ratio = max(la, lb) / max(1, min(la, lb))
            bins["<1.25" if ratio < 1.25 else "1.25-2" if ratio < 2 else ">=2"].append(pl)
        w[r["a_run"]] += p; w[r["b_run"]] += 1 - p; g[r["a_run"]] += 1; g[r["b_run"]] += 1
    runs = sorted(g)
    rho = spearmanr([w[x] / g[x] for x in runs], [L[x] for x in runs])[0]
    out = {"label": label or path, "topic": topic, "n": len(recs),
           "P(longer wins)": round(st.mean(longer_win), 3),
           **{f"P(longer) ratio {k}": round(st.mean(v), 3) for k, v in sorted(bins.items())},
           "spearman(winrate, words)": round(rho, 3),
           "words median/min/max": (st.median(L.values()), min(L.values()), max(L.values()))}
    return out, {x: w[x] / g[x] for x in runs}

if __name__ == "__main__":
    o, _ = analyze(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    print(json.dumps(o, indent=1))
