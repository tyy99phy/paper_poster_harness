import pytest

from poster_harness.arxiv import _validate_download_url


def test_validate_download_url_allows_arxiv_https():
    _validate_download_url("https://arxiv.org/pdf/2206.08956", allowed_hosts=("arxiv.org", "export.arxiv.org"))
    _validate_download_url("https://export.arxiv.org/e-print/2206.08956", allowed_hosts=("arxiv.org", "export.arxiv.org"))


def test_validate_download_url_rejects_non_arxiv_and_non_https():
    with pytest.raises(RuntimeError):
        _validate_download_url("https://arxiv.org.evil.test/pdf/2206.08956", allowed_hosts=("arxiv.org",))
    with pytest.raises(RuntimeError):
        _validate_download_url("http://arxiv.org/pdf/2206.08956", allowed_hosts=("arxiv.org",))
