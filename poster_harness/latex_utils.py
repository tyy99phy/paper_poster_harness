from __future__ import annotations

import re


LATEX_INLINE_REPLACEMENTS = {
    r"\hT": "HT",
    r"\pT": "pT",
    r"\sqrt{s}": "sqrt(s)",
    r"\fbinv": "fb^-1",
    r"\PGm": "mu",
    r"\Pgm": "mu",
    r"\hmn": "heavy Majorana neutrino",
    r"\WO": "Weinberg operator",
    r"\mN": "mN",
    r"\Vmn": "VmuN",
    r"\GeV": "GeV",
    r"\TeV": "TeV",
    r"\CL": "CL",
}


def extract_latex_braced(text: str, command: str) -> str:
    """Extract the first braced argument of a LaTeX command.

    This is intentionally small and dependency-free.  It handles nested braces
    and escaped braces well enough for titles, abstracts, captions, and common
    CMS macro-heavy source files.
    """

    marker = f"\\{command}"
    index = text.find(marker)
    if index < 0:
        return ""
    brace = text.find("{", index + len(marker))
    if brace < 0:
        return ""
    depth = 0
    start = brace + 1
    for pos in range(brace, len(text)):
        char = text[pos]
        if char == "{" and (pos == 0 or text[pos - 1] != "\\"):
            depth += 1
        elif char == "}" and (pos == 0 or text[pos - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return text[start:pos]
    return ""


def clean_latex_inline(text: str) -> str:
    """Convert lightweight LaTeX markup/macros into public plain text."""

    text = re.sub(r"\\texorpdfstring\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\{([^{}]*)\}", r"\2", text)
    for old, new in LATEX_INLINE_REPLACEMENTS.items():
        text = text.replace(old, new)
    text = re.sub(r"~?\\cite\{[^{}]*\}", "", text)
    text = re.sub(r"~+", " ", text)
    text = re.sub(r"\$+", "", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", "", text)
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\s+", " ", text).strip(" .\n\t")
    return text.strip()
