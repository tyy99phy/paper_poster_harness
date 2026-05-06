import pytest

from poster_harness.arxiv import _validate_download_url, _validate_pdf_file, extract_arxiv_source, find_main_tex, find_source_asset_roots


def test_validate_download_url_allows_arxiv_https():
    _validate_download_url("https://arxiv.org/pdf/2206.08956", allowed_hosts=("arxiv.org", "export.arxiv.org"))
    _validate_download_url("https://export.arxiv.org/e-print/2206.08956", allowed_hosts=("arxiv.org", "export.arxiv.org"))


def test_validate_download_url_rejects_non_arxiv_and_non_https():
    with pytest.raises(RuntimeError):
        _validate_download_url("https://arxiv.org.evil.test/pdf/2206.08956", allowed_hosts=("arxiv.org",))
    with pytest.raises(RuntimeError):
        _validate_download_url("http://arxiv.org/pdf/2206.08956", allowed_hosts=("arxiv.org",))


def test_extract_arxiv_source_accepts_pdf_only_eprint(tmp_path):
    eprint = tmp_path / "paper.eprint"
    eprint.write_bytes(b"%PDF-1.5\nfake pdf bytes")
    source = extract_arxiv_source(eprint, tmp_path / "source")
    assert (source / "SOURCE_IS_PDF.txt").exists()
    assert (source / "source.pdf").exists()
    assert find_main_tex(source) is None
    assert find_source_asset_roots(source, []) == [source]


def test_validate_pdf_file_rejects_truncated_pdf(tmp_path):
    pdf = tmp_path / "truncated.pdf"
    pdf.write_bytes(b"%PDF-1.7\n1 0 obj\n<< /Type /Catalog >>\n")
    with pytest.raises(RuntimeError, match="not readable|no pages"):
        _validate_pdf_file(pdf)
