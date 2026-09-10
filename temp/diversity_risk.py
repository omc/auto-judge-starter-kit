#!/usr/bin/env python3
"""Hallucination Risk vs citation count, source diversity, and topic complexity.

Reads hallucination_by_run_topic.csv (produced by hallucination_cross.py --out-csv)
and answers, WITHOUT any spaCy inference:

  1. Zero-order Spearman correlations among the per-(run,topic) signals.
  2. PARTIAL correlations (precision-matrix method) so we can strip the mechanical
     confound: more cited docs -> bigger grounding set hn -> lower ungrounded by
     construction. The partial redundancy<->ungrounded controlling hn & sn is the
     honest "does diversity matter beyond raw budget" test.
  3. WITHIN-TOPIC z-score correlations: difference out topic complexity by comparing
     runs that answered the SAME topic. This is the confounder-robust view.
  4. The "busted RAG" quadrant: high citation count x high redundancy.
  5. Topic-level proxies for complexity vs mean risk.

Numpy only (already present via spaCy). No scipy dependency.

  python3 temp/diversity_risk.py [temp/rag26_consensus/hallucination_by_run_topic.csv]
"""
import csv, sys
from collections import defaultdict
import numpy as np

CSV = sys.argv[1] if len(sys.argv) > 1 else "temp/rag26_consensus/hallucination_by_run_topic.csv"

# friendly name -> CSV column
COLS = {
    "cites":  "n_cited_docs",
    "redund": "redundancy",
    "effdoc": "effective_docs",
    "hn":     "hn_size",
    "sn":     "n_answer_concepts",
    "prec":   "precision",
    "ung":    "ungrounded_rate",
    "priv":   "private_ung_rate",
}


def rankdata(x):
    """Average ranks (ties averaged), like scipy.stats.rankdata."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)
    ranks[order] = np.arange(1, len(x) + 1)
    # average tied groups
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def spearman_matrix(mat):
    r = np.column_stack([rankdata(mat[:, j]) for j in range(mat.shape[1])])
    return np.corrcoef(r, rowvar=False)


def partial_matrix(mat):
    """Full pairwise partial correlations, each controlling for ALL other columns,
    via the precision matrix of the rank-transformed data."""
    r = np.column_stack([rankdata(mat[:, j]) for j in range(mat.shape[1])])
    C = np.corrcoef(r, rowvar=False)
    P = np.linalg.pinv(C)
    d = np.sqrt(np.diag(P))
    part = -P / np.outer(d, d)
    np.fill_diagonal(part, 1.0)
    return part


def print_matrix(names, M):
    print("            " + "".join(f"{n:>8}" for n in names))
    for i, n in enumerate(names):
        print(f"  {n:<10}" + "".join(f"{M[i, j]:>8.2f}" for j in range(len(names))))


def main():
    rows = list(csv.DictReader(open(CSV)))
    if not rows:
        sys.exit(f"no rows in {CSV}")
    names = list(COLS)
    data = {k: np.array([float(r[c]) for r in rows]) for k, c in COLS.items()}
    topics = np.array([r["topic"] for r in rows])
    n = len(rows)
    print(f"loaded {n} (run,topic) rows from {CSV}\n")

    mat = np.column_stack([data[k] for k in names])

    print("=" * 74)
    print("1. ZERO-ORDER Spearman correlations")
    print("=" * 74)
    print_matrix(names, spearman_matrix(mat))
    print("\n  read: cites->ung and hn->ung are the mechanical channel (bigger target).")

    print("\n" + "=" * 74)
    print("2. PARTIAL correlations (each pair controls for all other columns)")
    print("=" * 74)
    print_matrix(names, partial_matrix(mat))
    print("\n  key cell: redund x ung, controlling hn & sn -> does diversity add signal")
    print("  beyond raw grounding budget? Near 0 => redundancy is a recall story, not")
    print("  a grounding one (expected given precision saturation).")

    # ---- 3. within-topic z-scores (difference out topic complexity) ----
    print("\n" + "=" * 74)
    print("3. WITHIN-TOPIC correlations (z-scored per topic; topic complexity removed)")
    print("=" * 74)
    by_topic = defaultdict(list)
    for i, t in enumerate(topics):
        by_topic[t].append(i)

    def within_z(col):
        out = np.full(n, np.nan)
        for t, idx in by_topic.items():
            if len(idx) < 2:
                continue
            v = data[col][idx]
            sd = v.std()
            if sd == 0:
                continue
            out[idx] = (v - v.mean()) / sd
        return out

    zcols = {c: within_z(c) for c in names}
    pairs = [("redund", "ung"), ("redund", "priv"), ("effdoc", "ung"),
             ("effdoc", "priv"), ("cites", "ung"), ("cites", "priv")]
    n_topics_ge2 = sum(1 for idx in by_topic.values() if len(idx) >= 2)
    print(f"  ({n_topics_ge2} topics with >=2 runs)\n")
    print(f"  {'pair':<20}{'r':>8}{'n':>8}")
    for a, b in pairs:
        m = ~(np.isnan(zcols[a]) | np.isnan(zcols[b]))
        if m.sum() < 3:
            print(f"  {a+' x '+b:<20}{'n/a':>8}{int(m.sum()):>8}")
            continue
        r = np.corrcoef(zcols[a][m], zcols[b][m])[0, 1]
        print(f"  {a+' x '+b:<20}{r:>8.3f}{int(m.sum()):>8}")
    print("\n  this is the causal-ish view: among runs answering the SAME topic, do the")
    print("  more-redundant / lower-diversity ones hallucinate more?")

    # ---- 3b. within-topic PARTIALs controlling answer length (sn), and sn+volume ----
    print("\n" + "=" * 74)
    print("3b. WITHIN-TOPIC PARTIAL correlations (topic removed via z; then control sn)")
    print("=" * 74)

    def partial_corr(a, b, ctrl):
        """corr(a,b) after regressing both on the control columns, on pooled
        within-topic z-scores. ctrl = list of column names."""
        cols = [a, b] + ctrl
        m = ~np.any([np.isnan(zcols[c]) for c in cols], axis=0)
        if m.sum() < len(ctrl) + 3:
            return None, int(m.sum())
        Z = np.column_stack([np.ones(m.sum())] + [zcols[c][m] for c in ctrl])
        ra = zcols[a][m] - Z @ np.linalg.lstsq(Z, zcols[a][m], rcond=None)[0]
        rb = zcols[b][m] - Z @ np.linalg.lstsq(Z, zcols[b][m], rcond=None)[0]
        if ra.std() == 0 or rb.std() == 0:
            return None, int(m.sum())
        return float(np.corrcoef(ra, rb)[0, 1]), int(m.sum())

    specs = [
        ("redund", "priv", ["sn"]),        # redundancy vs fabrication, net of answer length
        ("redund", "ung",  ["sn"]),
        ("effdoc", "priv", ["sn"]),
        ("cites",  "priv", ["sn"]),
        ("redund", "priv", ["sn", "cites"]),  # redundancy net of length AND raw volume
        ("cites",  "priv", ["sn", "redund"]),  # volume net of length AND redundancy
    ]
    print(f"  {'pair':<16}{'controls':<16}{'partial_r':>10}{'n':>7}")
    for a, b, ctrl in specs:
        r, nn = partial_corr(a, b, ctrl)
        rs = "n/a" if r is None else f"{r:>10.3f}"
        print(f"  {a+' x '+b:<16}{'|'.join(ctrl):<16}{rs:>10}{nn:>7}")
    print("\n  the last two rows disentangle the collinear pair: does redundancy predict")
    print("  private-hallucination independent of raw citation volume, and vice-versa?")

    # ---- 4. busted-RAG quadrant ----
    print("\n" + "=" * 74)
    print("4. QUADRANT: citation count x source redundancy (median splits)")
    print("=" * 74)
    cmed, rmed = np.median(data["cites"]), np.median(data["redund"])
    print(f"  cites median={cmed:.1f}   redundancy median={rmed:.3f}\n")
    print(f"  {'cell':<22}{'n':>6}{'ung':>8}{'priv':>8}{'prec':>8}{'effdoc':>8}")
    for clab, cmask in (("cites<=med", data["cites"] <= cmed), ("cites>med", data["cites"] > cmed)):
        for rlab, rmask in (("redund<=med", data["redund"] <= rmed), ("redund>med", data["redund"] > rmed)):
            m = cmask & rmask
            if not m.any():
                continue
            print(f"  {clab+' '+rlab:<22}{int(m.sum()):>6}"
                  f"{data['ung'][m].mean():>8.3f}{data['priv'][m].mean():>8.3f}"
                  f"{data['prec'][m].mean():>8.3f}{data['effdoc'][m].mean():>8.2f}")
    print("\n  'busted RAG' = cites>med & redund>med: many docs, little unique material.")

    # ---- 5. topic-level: complexity proxies vs risk ----
    print("\n" + "=" * 74)
    print("5. TOPIC-LEVEL: complexity proxies vs mean risk (Spearman)")
    print("=" * 74)
    tnames = ["n_runs", "mean_sn", "mean_cites", "mean_redund", "mean_ung", "mean_priv"]
    trows = []
    for t, idx in by_topic.items():
        trows.append([
            len(idx),
            data["sn"][idx].mean(),      # answer breadth: proxy for topic complexity
            data["cites"][idx].mean(),
            data["redund"][idx].mean(),
            data["ung"][idx].mean(),
            data["priv"][idx].mean(),
        ])
    tmat = np.array(trows, float)
    print(f"  ({len(trows)} topics)  complexity proxies: mean_sn (answer breadth), mean_cites\n")
    print_matrix(tnames, spearman_matrix(tmat))
    print("\n  if mean_sn/mean_cites correlate with mean_ung/mean_priv, complexity is a")
    print("  real driver and MUST be conditioned on (see section 3, which already does).")


if __name__ == "__main__":
    main()
