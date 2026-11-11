"""Build papers/notebook-methods-human.{tex,pdf} from papers/notebook-methods-human.md.

    python papers/build_notebook_human.py

Title, authors and abstract come from the Markdown itself: the H1, the **Authors:** line
and the ## Abstract section. Section levels follow the number in each heading ("2."
section, "2.1" subsection, "2.2.1" subsubsection), whatever count of #s it was typed
with. Figures are ordinary Markdown images, referenced with \\ref{fig:length}:

    ![Caption.](figures_ggplot/f04_length.pdf){#fig:length width=80%}

Citations are pandoc's [@key], resolved from papers/references.bib; the reference list
goes where the Markdown has a ::: {#refs} div.

Converts with pandoc (via nix) and compiles with write_paper.compile_pdf (tectonic).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from write_paper import compile_pdf  # noqa: E402

SRC = HERE / "notebook-methods-human.md"
TEX = SRC.with_suffix(".tex")
BIB = HERE / "references.bib"
PANDOC = ["nix", "shell", "nixpkgs#pandoc", "-c", "pandoc"]

HEADER = r"""documentclass: article
fontsize: 10pt
geometry: margin=2.2cm
colorlinks: true
header-includes: |
  \usepackage{etoolbox}
  \AtBeginEnvironment{longtable}{\footnotesize}
  \lstset{basicstyle=\ttfamily\footnotesize,breaklines=true,breakatwhitespace=true,columns=fullflexible,frame=single}
  \setlength{\emergencystretch}{3em}
  \usepackage{float}
  \makeatletter\def\fps@figure{H}\makeatother
"""


def need(m: re.Match | None, what: str) -> re.Match:
    if m is None:
        sys.exit(f"{SRC.name}: no {what} found")
    return m


def map_prose(text: str, fn) -> str:
    """Apply fn to each line outside fenced code blocks."""
    out, fence = [], False
    for line in text.split("\n"):
        if line.startswith("```"):
            fence = not fence
        elif not fence:
            line = fn(line)
        out.append(line)
    return "\n".join(out)


def escape_dollars(line: str) -> str:
    """$93.84 is money, not math. Inline code is left alone."""
    parts = re.split(r"(`[^`]*`)", line)
    parts[::2] = [re.sub(r"(?<!\\)\$", r"\\$", p) for p in parts[::2]]
    return "".join(parts)


def heading_level(line: str) -> str:
    """Numbered: depth of the number. Unnumbered: one up, since the Markdown's # is the title."""
    m = re.match(r"(#+) (\d+(?:\.\d+)*)\.?(?=\s|$)", line)
    if m:
        return "#" * len(m.group(2).split(".")) + line[len(m.group(1)) :]
    m = re.match(r"(#+) ", line)
    if m:
        return "#" * max(1, len(m.group(1)) - 1) + line[len(m.group(1)) :]
    return line


def main() -> None:
    md = SRC.read_text()
    title = need(re.match(r"# (.+)\n", md), "H1 title").group(1)
    authors = need(re.search(r"(?m)^\*\*Authors:\*\* (.+)$", md), "**Authors:** line")
    abstract = need(re.search(r"(?ms)^## Abstract\n(.*?)(?=^#)", md), "## Abstract")
    body = md[abstract.end() :]

    meta = {
        "title": title,
        "author": [
            a.strip() for a in authors.group(1).split(",")
        ],  # <email> -> mailto link
        "date": f"{date.today():%-d %B %Y}",
        "abstract": escape_dollars(abstract.group(1).strip()),
        "bibliography": str(BIB),
        "link-citations": True,
    }
    front = "---\n" + "".join(f"{k}: {json.dumps(v)}\n" for k, v in meta.items())
    body = map_prose(body, lambda line: heading_level(escape_dollars(line)))

    tmp = HERE / ".notebook-human-build.md"
    tmp.write_text(front + HEADER + "---\n\n" + body)
    try:
        subprocess.run(
            PANDOC + [str(tmp), "-s", "--listings", "--citeproc", "-o", str(TEX)],
            check=True,
        )
    finally:
        tmp.unlink()
    print(f"wrote {TEX.name}")
    pdf = compile_pdf(TEX)
    print(
        f"pdf: {pdf}" if pdf else f"compile FAILED (see {TEX.with_suffix('.log').name})"
    )


if __name__ == "__main__":
    main()
