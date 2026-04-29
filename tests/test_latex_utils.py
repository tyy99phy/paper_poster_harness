from poster_harness.latex_utils import clean_latex_inline, extract_latex_braced


def test_extract_latex_braced_handles_nested_braces():
    text = r"\caption{Observed \textbf{limits} in the \PGm\PGm channel}"
    assert extract_latex_braced(text, "caption") == r"Observed \textbf{limits} in the \PGm\PGm channel"


def test_clean_latex_inline_removes_markup_and_preserves_public_terms():
    cleaned = clean_latex_inline(r"Observed 95\% \CL limits at $\sqrt{s}=13\TeV$ with \PGm\PGm events~\cite{x}.")
    assert "CL" in cleaned
    assert "TeV" in cleaned
    assert "mu" in cleaned
    assert "cite" not in cleaned
