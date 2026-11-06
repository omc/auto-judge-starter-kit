"""Build papers/notebook-methods.{tex,pdf} from papers/notebook-methods.md.

    python papers/build_notebook.py

Inserts the figures from papers/figures/ (captions from figures/CAPTIONS.md) at the
sections that discuss them, turns "Figure n" into LaTeX cross-references, converts
with pandoc (via nix), and compiles with write_paper.compile_pdf (tectonic).
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from write_paper import compile_pdf  # noqa: E402

SRC = HERE / "notebook-methods.md"
FIG_DIR = "figures"            # or figures_ggplot: python papers/build_notebook.py figures_ggplot
if len(sys.argv) > 1:
    FIG_DIR = sys.argv[1].strip("/")
TEX = HERE / ("notebook-methods.tex" if FIG_DIR == "figures" else f"notebook-methods-{FIG_DIR.split('_')[-1]}.tex")
PANDOC = ["nix", "shell", "nixpkgs#pandoc", "-c", "pandoc"]

# figure file number -> (stem, width as a fraction of the text width)
FIGS = {
    1: ("f01_probability_distribution", .9), 2: ("f02_position_bias", .8),
    3: ("f03_identical_summaries", .55), 4: ("f04_length", 1), 5: ("f05_prompt_agreement", .85),
    6: ("f06_padding", 1), 7: ("f07_broad_narrow", .6), 8: ("f08_judge_agreement", 1),
    9: ("f09_bod_grades", 1), 10: ("f10_hallucination_risk", 1), 11: ("f11_full_leaderboards", 1),
    12: ("f12_split_half", .6), 13: ("f13_determinism", .7),
}
# insert these figures just before the given heading
PLACE = {
    "### 4.5 ": [1, 3, 2, 13], "### 4.6 ": [5, 7], "### 4.7 ": [4, 6],
    "### 5.3 ": [8, 9], "### 6.3 ": [10], "## 8. ": [11, 12],
}
MATH = {"ρ": r"\rho", "τ": r"\tau", "Δ": r"\Delta", "Σ": r"\Sigma", "≥": r"\geq", "≤": r"\leq",
        "≈": r"\approx", "×": r"\times", "→": r"\rightarrow", "·": r"\cdot"}

META = """---
title: "OMC at TREC 2026 AutoJudge: Probabilistic Pairwise, Graded-Rubric and Grounding-Discounted Judges with Jev"
subtitle: "Methods and results (notebook paper draft)"
author: "Team omc26"
date: "Draft, 30 October 2026"
documentclass: article
fontsize: 10pt
geometry: margin=2.2cm
colorlinks: true
header-includes: |
  \\usepackage{etoolbox}
  \\AtBeginEnvironment{longtable}{\\footnotesize}
  \\lstset{basicstyle=\\ttfamily\\footnotesize,breaklines=true,columns=fullflexible,frame=single}
  \\setlength{\\emergencystretch}{3em}
  \\renewcommand{\\topfraction}{0.9}
  \\renewcommand{\\textfraction}{0.07}
  \\renewcommand{\\floatpagefraction}{0.8}
---

"""


def captions() -> dict:
    out = {}
    for line in (HERE / "figures" / "CAPTIONS.md").read_text().splitlines():
        m = re.match(r"\| `f(\d\d)_[^`]+` \| (.+?) \| [^|]+ \|$", line)
        if m:
            out[int(m.group(1))] = m.group(2)
    return out


def math_fix(text: str) -> str:
    """Escape money dollars, then turn Unicode math symbols into inline math. Code is
    left alone (fenced blocks and `inline code`)."""
    out, fence = [], False
    for line in text.split("\n"):
        if line.startswith("```"):
            fence = not fence
        if not fence:
            parts = re.split(r"(`[^`]*`)", line)
            for i in range(1, len(parts), 2):      # code spans: plain ASCII for listings
                parts[i] = parts[i].replace("−", "-").replace("≈", "~").replace("×", "x").replace("…", "...")
            for i in range(0, len(parts), 2):
                p = re.sub(r"(?<!\\)\$(?=\d)", r"\\$", parts[i])
                p = p.replace("Σ_g P(g)·g", r"\ensuremath{\sum_g P(g)\cdot g}")
                p = p.replace("−", "-")
                for ch, cmd in MATH.items():
                    p = p.replace(ch, f"\\ensuremath{{{cmd}}}")
                parts[i] = p
            line = "".join(parts)
        out.append(line)
    return "\n".join(out)


def figure_block(n: int, cap: dict) -> str:
    stem, w = FIGS[n]
    text = math_fix(cap[n])
    return f"![{text}]({FIG_DIR}/{stem}.pdf){{#fig:f{n:02d} width={int(w * 100)}%}}\n\n"


def main() -> None:
    md = SRC.read_text()
    cap = captions()
    # drop the H1 and the editorial status block (title comes from META)
    md = md[md.index("## 1. Overview"):]
    # figure index table: rows in the order the figures appear, first column -> \ref
    order = [n for nums in PLACE.values() for n in nums]
    rows = re.findall(r"(?m)^\| \d{1,2} \| `f\d\d_[^\n]+\n", md)
    if rows:
        start = md.index(rows[0])
        end = md.index(rows[-1]) + len(rows[-1])
        key = lambda r: order.index(int(re.match(r"\| (\d+)", r).group(1)))
        md = md[:start] + "".join(sorted(rows, key=key)) + md[end:]
    md = re.sub(r"(?m)^\| (\d{1,2}) \| `(f\d\d_[^`]+)` \|",
                lambda m: f"| \\ref{{fig:f{int(m.group(1)):02d}}} | {m.group(2).replace('_', chr(92) + '_')} |", md)
    # "Figure 6, right" / "Figures 2, 4, 12" -> \ref
    md = re.sub(r"Figures? (\d{1,2}(?:, \d{1,2})*)",
                lambda m: ("Figures " if "," in m.group(1) else "Figure~") +
                ", ".join(f"\\ref{{fig:f{int(x):02d}}}" for x in m.group(1).split(", ")), md)
    # pandoc sizes columns by separator length; give the label column more room
    md = re.sub(r"(?m)^\| (-{3,}) \|", lambda m: f"| {'-' * int(len(m.group(1)) * 1.5)} |", md)
    md = math_fix(md)
    for head, nums in PLACE.items():
        i = md.index("\n" + head) + 1
        md = md[:i] + "".join(figure_block(n, cap) for n in nums) + md[i:]
    tmp = HERE / ".notebook-build.md"
    tmp.write_text(META + md)
    subprocess.run(PANDOC + [str(tmp), "-s", "--listings", "-o", str(TEX)], check=True)
    tmp.unlink()
    print(f"wrote {TEX.name}")
    pdf = compile_pdf(TEX)
    print(f"pdf: {pdf}" if pdf else "compile FAILED (see notebook-methods.log)")


if __name__ == "__main__":
    main()
