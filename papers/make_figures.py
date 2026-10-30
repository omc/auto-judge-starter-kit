"""Figures for the paper (seaborn), from the permitted window and the scored leaderboards only.

    source .venv/bin/activate
    python papers/make_figures.py            # writes papers/figures/*.pdf and *.png

Data policy: every per-comparison / per-answer input is read through temp/jev_dataset.py's
window filter (first 10 topics of each topics file). Over all topics only the scored
leaderboards' "all" rows (.eval.txt) are read (figure 11). No figure labels a run.
Each figure also prints the aggregate it plots, to check against the findings doc.
"""
from __future__ import annotations

import importlib
import json
import os
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "papers" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "temp"))
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

DATASETS = ("rag26", "ragtime26")
JUDGE_ORDER = ["Jev Choice", "Jev mirror", "Gemini", "BOD"]
PALETTE = dict(zip(JUDGE_ORDER, sns.color_palette("colorblind", 4)))
PJ = "bonsai_pairwise_jev.pairwise/comparisons.jsonl"
PG = "bonsai_pairwise.pairwise/comparisons.jsonl"
SRC = {
    "rag26": {"choice": f"output-win-rag26-jev/{PJ}", "prompts": f"output-win-rag26-jev-prompts/{PJ}",
              "mirror": f"output-pairwise-jev-full-prop/{PJ}", "gemini": f"output-win-rag26-gemini/{PG}",
              "broad": f"output-win-rag26-prop-broad/{PJ}", "narrow": f"output-win-rag26-prop-narrow/{PJ}",
              "bod": "output-win-rag26-bod/bonsai_bod.bod/grades.jsonl",
              "risk": "output-win-rag26-bodrisk", "probes": "temp/jev_probes_win_rag26",
              "determinism": "rag2026-3"},
    "ragtime26": {"choice": f"output-win-ragtime26-jev/{PJ}", "prompts": f"output-win-ragtime26-jev-prompts/{PJ}",
                  "mirror": f"output-ragtime26-jev-full-prop/{PJ}", "gemini": f"output-ragtime26-gemini/{PG}",
                  "broad": f"output-win-ragtime26-prop-broad/{PJ}", "narrow": f"output-win-ragtime26-prop-narrow/{PJ}",
                  "bod": "output-win-ragtime26-bod/bonsai_bod.bod/grades.jsonl",
                  "risk": "output-win-ragtime26-bodrisk", "probes": "temp/jev_probes_win_ragtime26",
                  "determinism": "2003"},
}
SUBMISSIONS = {
    "pairwise": "submissions/bonsai-jev-pairwise-noul-mirror-proportionate-20261009/{ds}/bonsai_pairwise_jev.eval.txt",
    "bod": "submissions/bonsai-bod-jev-graded-20261009/{ds}/bonsai_bod.eval.txt",
    "bodrisk": "submissions/bonsai-bod-risk-jev-20261013/{ds}/bonsai_bod_risk.eval.txt",
}

sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42})


# ---------------------------------------------------------------------------- data access

def use(ds):
    """Switch temp/jev_dataset.py to `ds` (it reads JEV_DATASET at import)."""
    os.environ["JEV_DATASET"] = ds
    import jev_dataset
    return importlib.reload(jev_dataset)


def jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def window_jsonl(D, path):
    """Probe/grade files carry topic_id: keep window topics only."""
    return [r for r in jsonl(path) if D.in_window(r["topic_id"])]


def mirror(p, a="noul_a_proportionate", b="noul_b_proportionate"):
    return (p[a] + 1 - p[b]) / 2


def win_rates(recs, P):
    """{(topic, run): mean P(win)} over all comparisons the run took part in."""
    w, g = defaultdict(float), defaultdict(int)
    for r in recs:
        p, t = P(r), r["topic_id"]
        w[(t, r["a_run"])] += p; w[(t, r["b_run"])] += 1 - p
        g[(t, r["a_run"])] += 1; g[(t, r["b_run"])] += 1
    return {k: w[k] / g[k] for k in g}


def board(per_topic, topics=None):
    s = defaultdict(list)
    for (t, x), v in per_topic.items():
        if topics is None or t in topics:
            s[x].append(v)
    return {x: st.mean(v) for x, v in s.items()}


def boot_ci(xs, n=4000, seed=0):
    rng = random.Random(seed)
    m = sorted(st.mean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return m[int(.025 * n)], m[int(.975 * n)]


def save(fig, name, note):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"[{name}] {note}")


_cache = {}


def comparisons(ds, key):
    if (ds, key) not in _cache:
        D = use(ds)
        _cache[(ds, key)] = [r for r in D.window_records(SRC[ds][key]) if r.get("valid")]
    return _cache[(ds, key)]


def lengths(ds):
    if ("L", ds) not in _cache:
        _cache[("L", ds)] = use(ds).word_lengths()
    return _cache[("L", ds)]


P_CHOICE = lambda r: r["p_a"]
P_MIRROR = lambda r: mirror(r["raw_p"])
P_GEMINI = lambda r: 1.0 if r["winner_run"] == r["a_run"] else 0.0


# ---------------------------------------------------------------------------- figures

def fig_distribution():
    """F1: Choice saturates at 0/1; the mirror never does."""
    rows = []
    for ds in DATASETS:
        rows += [{"dataset": ds, "judge": "Jev Choice", "P(A better)": r["p_a"]} for r in comparisons(ds, "choice")]
        rows += [{"dataset": ds, "judge": "Jev mirror", "P(A better)": P_MIRROR(r)} for r in comparisons(ds, "mirror")]
    df = pd.DataFrame(rows)
    g = sns.displot(df, x="P(A better)", hue="judge", col="dataset", stat="probability", common_norm=False,
                    bins=40, element="step", palette=PALETTE, height=2.6, aspect=1.4,
                    facet_kws={"sharey": False})
    g.set_titles("{col_name}")
    sat = df.assign(ex=df["P(A better)"].isin([0, 1])).groupby(["dataset", "judge"])["ex"].mean().round(3)
    save(g.figure, "f01_probability_distribution", f"share exactly 0/1: {sat.to_dict()}")


def fig_position():
    """F2: P(slot A wins) by closeness, pairs judged in both orientations."""
    bins = [(0, .05), (.05, .25), (.25, .4), (.4, .6), (.6, .75), (.75, .95), (.95, 1.01)]
    rows = []
    for ds in DATASETS:
        for judge, key, P in (("Jev Choice", "choice", P_CHOICE), ("Jev mirror", "mirror", P_MIRROR)):
            pairs = defaultdict(dict)
            for r in comparisons(ds, key):
                x, y = sorted((r["a_run"], r["b_run"]))
                p = P(r)
                pairs[(r["topic_id"], x, y)]["fwd" if r["a_run"] == x else "rev"] = p if r["a_run"] == x else 1 - p
            for v in pairs.values():
                if len(v) == 2:
                    avg = (v["fwd"] + v["rev"]) / 2
                    b = next(i for i, (lo, hi) in enumerate(bins) if lo <= avg < hi)
                    rows.append({"dataset": ds, "judge": judge, "bin": b,
                                 "P(slot A wins)": (v["fwd"] + 1 - v["rev"]) / 2})
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.8), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        sns.pointplot(df[df.dataset == ds], x="bin", y="P(slot A wins)", hue="judge", palette=PALETTE,
                      errorbar=("ci", 95), dodge=.3, markers=["o", "s"], linestyles=["-", "--"], ax=ax,
                      legend=ax is axes[0])
        ax.axhline(.5, color="grey", lw=.8, ls=":")
        ax.set_xticks(range(len(bins)), [f"{lo:.2f}–{min(hi, 1):.2f}" for lo, hi in bins], rotation=40)
        ax.set_xlabel("orientation-averaged P(x better)")
        ax.set_title(ds)
    mid = df[df.bin == 3].groupby(["dataset", "judge"])["P(slot A wins)"].mean().round(3)
    save(fig, "f02_position_bias", f"close pairs (0.4-0.6) P(slot A wins): {mid.to_dict()}")


def fig_identical():
    """F3: identical summaries in both slots: Choice vs mirror."""
    rows = []
    for ds in DATASETS:
        D = use(ds)
        d = SRC[ds]["probes"]
        for r in window_jsonl(D, f"{d}/identical.jsonl"):
            if r["p"].get("better_summary") is not None:
                rows.append({"dataset": ds, "judge": "Jev Choice", "P(A better)": r["p"]["better_summary"]})
        a = {(r["topic_id"], r["run_id"]): r["p"].get("noul_a_proportionate")
             for r in window_jsonl(D, f"{d}/identical_noul_a_proportionate.jsonl")}
        b = {(r["topic_id"], r["run_id"]): r["p"].get("noul_b_proportionate")
             for r in window_jsonl(D, f"{d}/identical_noul_b_proportionate.jsonl")}
        for k in a.keys() & b.keys():
            if a[k] is not None and b[k] is not None:
                rows.append({"dataset": ds, "judge": "Jev mirror", "P(A better)": (a[k] + 1 - b[k]) / 2})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    sns.boxplot(df, x="dataset", y="P(A better)", hue="judge", palette=PALETTE, fliersize=1.5, ax=ax)
    ax.axhline(.5, color="grey", lw=.8, ls=":")
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    m = df.groupby(["dataset", "judge"])["P(A better)"].agg(["mean", "count"]).round(3)
    save(fig, "f03_identical_summaries", f"means: {m['mean'].to_dict()} n: {m['count'].to_dict()}")


def fig_length():
    """F4: P(longer wins) by length ratio (left); per-topic Spearman(score, words) (right)."""
    ratio_bins = [(1, 1.25, "<1.25×"), (1.25, 2, "1.25–2×"), (2, 1e9, "≥2×")]
    rows, rho_rows = [], []
    for ds in DATASETS:
        L = lengths(ds)
        for judge, key, P in (("Jev Choice", "choice", P_CHOICE), ("Jev mirror", "mirror", P_MIRROR),
                              ("Gemini", "gemini", P_GEMINI)):
            recs = comparisons(ds, key)
            for r in recs:
                la, lb = L[r["topic_id"]][r["a_run"]], L[r["topic_id"]][r["b_run"]]
                if la == lb or min(la, lb) == 0:
                    continue
                p = P(r)
                ratio = max(la, lb) / min(la, lb)
                lab = next(l for lo, hi, l in ratio_bins if lo <= ratio < hi)
                rows.append({"dataset": ds, "judge": judge, "length ratio": lab,
                             "P(longer wins)": p if la > lb else 1 - p})
            wr = win_rates(recs, P)
            for t in sorted({t for t, _ in wr}):
                xs = [x for (tt, x) in wr if tt == t]
                rho_rows.append({"dataset": ds, "judge": judge,
                                 "Spearman(score, words)": spearmanr([wr[(t, x)] for x in xs],
                                                                     [L[t][x] for x in xs])[0]})
        bs = bod_scores(ds)
        for t in sorted({t for t, _ in bs}):
            xs = [x for (tt, x) in bs if tt == t and x in L[t]]
            rho_rows.append({"dataset": ds, "judge": "BOD", "Spearman(score, words)":
                             spearmanr([bs[(t, x)] for x in xs], [L[t][x] for x in xs])[0]})
    df, rho = pd.DataFrame(rows), pd.DataFrame(rho_rows)
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.9), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    for ax, ds in zip(axes[:2], DATASETS):
        sns.pointplot(df[df.dataset == ds], x="length ratio", y="P(longer wins)", hue="judge",
                      order=[l for *_x, l in ratio_bins], palette=PALETTE, hue_order=JUDGE_ORDER[:3], errorbar=("ci", 95), dodge=.3,
                      ax=ax, legend=ax is axes[0])
        ax.axhline(.5, color="grey", lw=.8, ls=":")
        ax.set_ylim(.3, .9)
        ax.set_title(ds)
    sns.stripplot(rho, x="dataset", y="Spearman(score, words)", hue="judge", dodge=True, palette=PALETTE,
                  hue_order=JUDGE_ORDER, size=3.5, alpha=.8, ax=axes[2], legend=True)
    sns.move_legend(axes[2], "lower left", title=None, fontsize=7, markerscale=1.5)
    sns.pointplot(rho, x="dataset", y="Spearman(score, words)", hue="judge", dodge=.6, palette=PALETTE,
                  hue_order=JUDGE_ORDER,
                  estimator="median", errorbar=None, markers="_", markersize=14, linestyles="none",
                  ax=axes[2], legend=False)
    axes[2].axhline(0, color="grey", lw=.8, ls=":")
    axes[2].set_title("per window topic (bar = median)")
    axes[1].set_title("ragtime26 (Gemini: 3 topics)")
    axes[2].set_xlabel("")
    pl = df.groupby(["dataset", "judge"])["P(longer wins)"].mean().round(3)
    med = rho.groupby(["dataset", "judge"])["Spearman(score, words)"].median().round(3)
    save(fig, "f04_length", f"P(longer): {pl.to_dict()} | median rho: {med.to_dict()}")


def fig_prompts():
    """F5: ranking agreement between question wordings (mean per-topic Spearman)."""
    qs = ["better_summary", "length_neutral", "needs_coverage", "precision"]
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.6), gridspec_kw={"width_ratios": [1, 1, .06]})
    notes = {}
    for ax, ds in zip(axes[:2], DATASETS):
        recs = [r for r in comparisons(ds, "prompts") if all(r["p"].get(q) is not None for q in qs)]
        wr = {q: win_rates(recs, lambda r, q=q: r["p"][q]) for q in qs}
        topics = sorted({t for t, _ in wr[qs[0]]})
        M = pd.DataFrame(index=qs, columns=qs, dtype=float)
        for a in qs:
            for b in qs:
                vals = []
                for t in topics:
                    xs = [x for (tt, x) in wr[a] if tt == t]
                    vals.append(spearmanr([wr[a][(t, x)] for x in xs], [wr[b][(t, x)] for x in xs])[0])
                M.loc[a, b] = st.mean(vals)
        lab = [q.replace("_", " ") for q in qs]
        sns.heatmap(M.values, annot=True, fmt=".3f", vmin=.8, vmax=1, cmap="crest", square=True,
                    cbar=ax is axes[1], cbar_ax=axes[2] if ax is axes[1] else None,
                    xticklabels=lab, yticklabels=lab if ax is axes[0] else False, ax=ax)
        ax.set_title(ds)
        notes[ds] = round(float(M.loc["better_summary", "length_neutral"]), 3)
    save(fig, "f05_prompt_agreement", f"better~length_neutral: {notes}")


PAD_ORDER = ["pad_repeat", "pad_generic", "pad_offtopic", "pad_relevant", "truncate"]


def padding_deltas(D, path, score):
    """ΔP(target wins) vs the original, per target (orientation-averaged, mean over opponents)."""
    cell = defaultdict(list)
    for r in window_jsonl(D, path):
        p = score(r)
        if p is None:
            continue
        cell[(r["topic_id"], r["target"], r["variant"], r["opponent"])].append(
            p if r["target_slot"] == "A" else 1 - p)
    tv = defaultdict(list)
    for (t, tgt, v, _o), ps in cell.items():
        if len(ps) == 2:
            tv[(t, tgt, v)].append(st.mean(ps))
    out = []
    for (t, tgt, v), ps in tv.items():
        if v != "original" and (t, tgt, "original") in tv:
            out.append((t, v, st.mean(ps) - st.mean(tv[(t, tgt, "original")])))
    return out


def fig_padding():
    """F6: controlled padding. Pairwise judges (ΔP win) and BOD (ΔBOD_MEAN, ΔPADDING)."""
    rows = []
    for ds in DATASETS:
        D = use(ds)
        d = SRC[ds]["probes"]
        noul = lambda r: r.get("score") if r.get("score") is not None else (
            mirror(r["p"]) if r["p"].get("noul_a_proportionate") is not None else None)
        gem = lambda r: r["p"].get("better_summary")
        for judge, f, sc in (("Jev mirror", "padding_noul.jsonl", noul), ("Gemini", "padding_gemini.jsonl", gem)):
            for t, v, dlt in padding_deltas(D, f"{d}/{f}", sc):
                rows.append({"dataset": ds, "judge": judge, "variant": v, "Δ": dlt, "topic": t})
    df = pd.DataFrame(rows)
    D = use("rag26")
    bp = window_jsonl(D, f"{SRC['rag26']['probes']}/bod_padding.jsonl")
    base = {(r["topic_id"], r["target"]): r for r in bp if r["variant"] == "original"}
    brow = []
    for r in bp:
        o = base.get((r["topic_id"], r["target"]))
        if r["variant"] != "original" and o and r["bod_mean"] is not None and o["bod_mean"] is not None:
            brow.append({"measure": "ΔBOD_MEAN", "variant": r["variant"], "Δ": r["bod_mean"] - o["bod_mean"]})
            if r.get("padding") is not None and o.get("padding") is not None:
                brow.append({"measure": "ΔBOD_PADDING", "variant": r["variant"], "Δ": r["padding"] - o["padding"]})
    bdf = pd.DataFrame(brow)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1), sharey=False)
    for ax, ds in zip(axes[:2], DATASETS):
        sns.pointplot(df[df.dataset == ds], x="variant", y="Δ", hue="judge", order=PAD_ORDER, palette=PALETTE,
                      errorbar=("ci", 95), dodge=.3, linestyles="none", ax=ax, legend=ax is axes[0])
        ax.axhline(0, color="grey", lw=.8, ls=":")
        ax.set_ylabel("ΔP(target wins) vs original")
        ax.set_title(f"{ds}: Jev 10 topics, Gemini 3")
        ax.tick_params(axis="x", rotation=35)
    sns.pointplot(bdf, x="variant", y="Δ", hue="measure", order=PAD_ORDER, palette="dark",
                  errorbar=("ci", 95), dodge=.3, linestyles="none", ax=axes[2])
    axes[2].axhline(0, color="grey", lw=.8, ls=":")
    axes[2].set_title("rag26: BOD (graded rubric)")
    axes[2].tick_params(axis="x", rotation=35)
    m = df.groupby(["dataset", "judge", "variant"])["Δ"].mean().round(3)
    mb = bdf.groupby(["measure", "variant"])["Δ"].mean().round(3)
    save(fig, "f06_padding", f"pairwise: {m.to_dict()} | BOD: {mb.to_dict()}")


def fig_broad_narrow():
    """F7: depth-proportionate prompt. Per-topic difference-in-differences vs `better`."""
    pairs = {"better": ("noul_a_better", "noul_b_better"),
             "proportionate": ("noul_a_proportionate", "noul_b_proportionate")}
    rows = []
    for ds in DATASETS:
        L = lengths(ds)
        B = {r["comp_id"]: r for r in comparisons(ds, "broad") if r.get("result") == "ok"}
        N = {r["comp_id"]: r for r in comparisons(ds, "narrow") if r.get("result") == "ok"}

        def lp(r, k):
            la, lb = L[r["topic_id"]][r["a_run"]], L[r["topic_id"]][r["b_run"]]
            if la == lb:
                return None
            p = mirror(r["raw_p"], *pairs[k])
            return p if la > lb else 1 - p
        per = defaultdict(list)
        for i in B.keys() & N.keys():
            x = [lp(B[i], "proportionate"), lp(N[i], "proportionate"), lp(B[i], "better"), lp(N[i], "better")]
            if None not in x:
                per[B[i]["topic_id"]].append((x[1] - x[0]) - (x[3] - x[2]))
        D = use(ds)
        for t in D.DS["window"]:
            if per.get(t):
                rows.append({"dataset": ds, "topic": D.DS["window"].index(t), "DiD": st.mean(per[t])})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    sns.stripplot(df, x="dataset", y="DiD", color=".35", size=5, jitter=.12, ax=ax)
    note = {}
    for i, ds in enumerate(DATASETS):
        v = df[df.dataset == ds]["DiD"].tolist()
        lo, hi = boot_ci(v)
        ax.errorbar(i + .28, st.mean(v), yerr=[[st.mean(v) - lo], [hi - st.mean(v)]], fmt="D", color="C3",
                    capsize=4, ms=5)
        note[ds] = (round(st.mean(v), 4), round(lo, 4), round(hi, 4), sum(x < 0 for x in v))
    ax.axhline(0, color="grey", lw=.8, ls=":")
    ax.set_ylabel("Δ shift toward shorter\n(proportionate − better)")
    ax.set_xlabel("")
    ax.set_title("narrow vs broad request: per window topic\n(red = mean over topics, topic-bootstrap 95% CI)")
    save(fig, "f07_broad_narrow", f"mean, CI, topics<0: {note}")


def bod_scores(ds):
    """{(topic, run): BOD_MEAN} from window grade rows (complete answers only)."""
    if ("bod", ds) not in _cache:
        D = use(ds)
        by = defaultdict(list)
        for r in window_jsonl(D, SRC[ds]["bod"]):
            if r["qid"] != "padding":
                by[(r["topic_id"], r["run_id"])].append(r.get("expected"))
        _cache[("bod", ds)] = {k: sum(v) / (3 * len(v)) for k, v in by.items() if None not in v}
    return _cache[("bod", ds)]


def fig_agreement():
    """F8: run boards on the window: Jev mirror vs Gemini, BOD vs Jev mirror (no labels)."""
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.9))
    note = {}
    for i, ds in enumerate(DATASETS):
        jm = win_rates(comparisons(ds, "mirror"), P_MIRROR)
        gm = win_rates(comparisons(ds, "gemini"), P_GEMINI)
        shared = {t for t, _ in gm}
        a, b = board(jm, shared), board(gm, shared)
        xs = sorted(a.keys() & b.keys())
        ax = axes[2 * i]
        sns.scatterplot(x=[a[x] for x in xs], y=[b[x] for x in xs], s=14, color=PALETTE["Gemini"], ax=ax)
        r1 = spearmanr([a[x] for x in xs], [b[x] for x in xs])[0]
        ax.set(xlabel="Jev mirror win rate", ylabel="Gemini win rate",
               title=f"{ds}: {len(shared)} topics, ρ={r1:.3f}")
        bo = board(bod_scores(ds))
        jb = board(jm)
        xs = sorted(bo.keys() & jb.keys())
        ax = axes[2 * i + 1]
        sns.scatterplot(x=[jb[x] for x in xs], y=[bo[x] for x in xs], s=14, color=PALETTE["BOD"], ax=ax)
        r2 = spearmanr([jb[x] for x in xs], [bo[x] for x in xs])[0]
        ax.set(xlabel="Jev mirror win rate", ylabel="BOD_MEAN", title=f"{ds}: 10 topics, ρ={r2:.3f}")
        note[ds] = (round(r1, 3), round(r2, 3))
    save(fig, "f08_judge_agreement", f"(Jev~Gemini, BOD~Jev): {note}")


def fig_bod():
    """F9: BOD grade distributions on the window."""
    rows, ans = [], []
    for ds in DATASETS:
        D = use(ds)
        for r in window_jsonl(D, SRC[ds]["bod"]):
            if r["qid"] != "padding" and r.get("probs"):
                rows.append({"dataset": ds, "most likely grade": max(r["probs"], key=r["probs"].get),
                             "E[g] − most likely": r["expected"] - int(max(r["probs"], key=r["probs"].get))})
        ans += [{"dataset": ds, "BOD_MEAN": v} for v in bod_scores(ds).values()]
    df, adf = pd.DataFrame(rows), pd.DataFrame(ans)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.8))
    share = (df.groupby("dataset")["most likely grade"].value_counts(normalize=True)
             .rename("share").reset_index())
    sns.barplot(share, x="most likely grade", y="share", hue="dataset", order=list("0123"), ax=axes[0])
    axes[0].set_title("most likely grade per question")
    sns.histplot(df, x="E[g] − most likely", hue="dataset", stat="probability", common_norm=False, bins=30,
                 element="step", ax=axes[1])
    axes[1].set_title("expected vs most likely grade")
    sns.histplot(adf, x="BOD_MEAN", hue="dataset", stat="probability", common_norm=False, bins=30,
                 element="step", ax=axes[2])
    axes[2].set_title("BOD_MEAN per answer")
    big = df.assign(b=df["E[g] − most likely"].abs() > .25).groupby("dataset")["b"].mean().round(3)
    save(fig, "f09_bod_grades", f"|E[g]-argmax|>0.25: {big.to_dict()} | grade shares: "
         f"{share.round(3).set_index(['dataset', 'most likely grade'])['share'].to_dict()}")


def fig_risk():
    """F10: Hallucination Risk on the window: distribution, and BOD_GROUNDED vs BOD_MEAN."""
    rows, sc = [], []
    for ds in DATASETS:
        D = use(ds)
        base = SRC[ds]["risk"]
        for r in window_jsonl(D, f"{base}/bonsai_bod_risk.bod/risk.jsonl"):
            rows.append({"dataset": ds, "HALLUC_RISK": r["risk"], "UNGROUNDED_RATE": r["ungrounded"],
                         "defined": r["defined"]})
        vals = defaultdict(dict)
        for line in open(f"{base}/bonsai_bod_risk.eval.txt"):
            p = line.split()
            if len(p) == 4 and p[1] != "all" and D.in_window(p[1]):
                vals[(p[1], p[0])][p[2]] = float(p[3])
        sc += [{"dataset": ds, "BOD_MEAN": v["BOD_MEAN"], "BOD_GROUNDED": v["BOD_GROUNDED"]}
               for v in vals.values() if "BOD_MEAN" in v]
    df, sdf = pd.DataFrame(rows), pd.DataFrame(sc)
    long = df[df.defined].melt(id_vars="dataset", value_vars=["HALLUC_RISK", "UNGROUNDED_RATE"],
                               var_name="measure", value_name="share of answer concepts")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.9))
    sns.boxplot(long, x="measure", y="share of answer concepts", hue="dataset", fliersize=1, ax=axes[0])
    axes[0].set_xlabel("")
    axes[0].set_title("per answer (risk defined)")
    for ax, ds in zip(axes[1:], DATASETS):
        d = sdf[sdf.dataset == ds]
        sns.scatterplot(d, x="BOD_MEAN", y="BOD_GROUNDED", s=6, alpha=.5, ax=ax)
        ax.plot([0, 1], [0, 1], color="grey", lw=.8, ls=":")
        ax.set_title(f"{ds}: per answer")
    med = df[df.defined].groupby("dataset")[["HALLUC_RISK", "UNGROUNDED_RATE"]].median().round(3)
    save(fig, "f10_hallucination_risk", f"medians: {med.to_dict()} | answers: {len(sdf)}")


def eval_all(path):
    out = defaultdict(dict)
    for line in open(path):
        p = line.split()
        if len(p) == 4 and p[1] == "all":
            out[p[2]][p[0]] = float(p[3])
    return out


def fig_full_boards():
    """F11: submitted full-run boards ("all" rows), unlabeled scatter."""
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6))
    note = {}
    for i, ds in enumerate(DATASETS):
        pw = eval_all(SUBMISSIONS["pairwise"].format(ds=ds))["PAIRWISE_WINRATE"]
        bod = eval_all(SUBMISSIONS["bod"].format(ds=ds))["BOD_MEAN"]
        gr = eval_all(SUBMISSIONS["bodrisk"].format(ds=ds))["BOD_GROUNDED"]
        for j, (xa, xn, ya, yn) in enumerate(((pw, "pairwise win rate", bod, "BOD_MEAN"),
                                              (bod, "BOD_MEAN", gr, "BOD_GROUNDED"),
                                              (pw, "pairwise win rate", gr, "BOD_GROUNDED"))):
            xs = sorted(xa.keys() & ya.keys())
            r = spearmanr([xa[x] for x in xs], [ya[x] for x in xs])[0]
            ax = axes[i, j]
            sns.scatterplot(x=[xa[x] for x in xs], y=[ya[x] for x in xs], s=14, ax=ax)
            ax.set(xlabel=xn, ylabel=yn, title=f"{ds} (all topics, {len(xs)} runs): ρ={r:.3f}")
            note[(ds, yn, xn)] = round(r, 3)
    fig.tight_layout()
    save(fig, "f11_full_leaderboards", str(note))


def fig_split_half():
    """F12: split-half board reliability on the window (200 random 5/5 topic splits)."""
    rows = []
    for ds in DATASETS:
        D = use(ds)
        risk_vals = defaultdict(dict)
        for line in open(f"{SRC[ds]['risk']}/bonsai_bod_risk.eval.txt"):
            p = line.split()
            if len(p) == 4 and p[1] != "all" and D.in_window(p[1]) and p[2] == "BOD_GROUNDED":
                risk_vals[(p[1], p[0])] = float(p[3])
        scores = {"Jev mirror": win_rates(comparisons(ds, "mirror"), P_MIRROR),
                  "BOD": bod_scores(ds), "BOD_GROUNDED": dict(risk_vals)}
        topics = [t for t in D.DS["window"]]
        for name, s in scores.items():
            for i in range(200):
                rr = random.Random(i)
                tt = topics[:]
                rr.shuffle(tt)
                a, b = board(s, set(tt[:5])), board(s, set(tt[5:]))
                xs = sorted(a.keys() & b.keys())
                rows.append({"dataset": ds, "judge": name,
                             "split-half Spearman": spearmanr([a[x] for x in xs], [b[x] for x in xs])[0]})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(5.5, 2.9))
    pal = {**PALETTE, "BOD_GROUNDED": sns.color_palette("colorblind")[4]}
    sns.boxplot(df, x="dataset", y="split-half Spearman", hue="judge", palette=pal, fliersize=1, ax=ax)
    ax.set_xlabel("")
    m = df.groupby(["dataset", "judge"])["split-half Spearman"].mean().round(3)
    save(fig, "f12_split_half", f"means: {m.to_dict()}")


def fig_determinism():
    """F13: identical mirror requests sent twice (one window topic per dataset)."""
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3))
    note = {}
    for ax, ds in zip(axes, DATASETS):
        t = SRC[ds]["determinism"]
        use(ds).assert_window([t])
        d = SRC[ds]["probes"]
        p1 = {r["comp_id"]: r["p"] for r in jsonl(f"{d}/determinism_{t}_mirror_pass1.jsonl")}
        p2 = {r["comp_id"]: r["p"] for r in jsonl(f"{d}/determinism_{t}_mirror_pass2.jsonl")}
        ks = [k for k in p1.keys() & p2.keys() if p1[k].get("noul_a_proportionate") is not None
              and p2[k].get("noul_a_proportionate") is not None]
        x, y = [mirror(p1[k]) for k in ks], [mirror(p2[k]) for k in ks]
        sns.scatterplot(x=x, y=y, s=6, alpha=.4, ax=ax)
        ax.plot([0, 1], [0, 1], color="grey", lw=.8, ls=":")
        flips = st.mean((a - .5) * (b - .5) < 0 for a, b in zip(x, y))
        ax.set(xlabel="mirror P, request 1", ylabel="mirror P, request 2",
               title=f"{ds} (1 topic, n={len(ks):,}): flips {flips:.2%}")
        note[ds] = (len(ks), round(flips, 4))
    save(fig, "f13_determinism", f"(n, flip rate): {note}")


FIGS = [fig_distribution, fig_position, fig_identical, fig_length, fig_prompts, fig_padding,
        fig_broad_narrow, fig_agreement, fig_bod, fig_risk, fig_full_boards, fig_split_half, fig_determinism]

if __name__ == "__main__":
    only = set(sys.argv[1:])
    for f in FIGS:
        if not only or f.__name__ in only:
            f()
