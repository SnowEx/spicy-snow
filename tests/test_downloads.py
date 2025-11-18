import pytest
import gzip
from pathlib import Path
from unittest.mock import patch, MagicMock

from spicy_snow.utils.download import (
    download_urls,
    download_urls_parallel,
    decompress,
    download_proba_v
)

# ----------------------------
# Fixtures
# ----------------------------

@pytest.fixture
def tmp_out_dir(tmp_path):
    return tmp_path

@pytest.fixture
def sample_urls(tmp_path):
    file1 = tmp_path / "file1.txt"
    file1.write_text("hello")
    file2 = tmp_path / "file2.txt"
    file2.write_text("world")
    return [str(file1), str(file2)]

# ----------------------------
# decompress tests
# ----------------------------

def test_decompress(tmp_path):
    in_fp = tmp_path / "test.txt.gz"
    out_fp = tmp_path / "test.txt"

    # compress some data
    in_fp.write_bytes(gzip.compress(b"hello world"))

    result_fp = decompress(in_fp, out_fp)
    assert result_fp == out_fp
    assert out_fp.read_text() == "hello world"

# ----------------------------
# download_urls tests
# ----------------------------

@patch("spicy_snow.utils.download.download_url")
@patch("spicy_snow.utils.download.validate_urls")
@patch("asf_search.ASFSession")
def test_download_urls_basic(mock_session, mock_validate, mock_download, tmp_out_dir):
    # mock validate_urls to just return urls
    urls = ["http://example.com/file1", "http://example.com/file2"]
    mock_validate.return_value = urls

    mock_download.side_effect = lambda url, out_dir, session=None: Path(out_dir) / Path(url).name

    results = download_urls(urls, tmp_out_dir)
    assert all(isinstance(fp, Path) for fp in results)
    assert [fp.name for fp in results] == ["file1", "file2"]

@patch("spicy_snow.utils.download.download_url")
@patch("spicy_snow.utils.download.validate_urls")
@patch("asf_search.ASFSession")
def test_download_urls_reprocess(mock_session, mock_validate, mock_download, tmp_out_dir):
    urls = ["http://example.com/file1"]
    mock_validate.return_value = urls
    mock_download.side_effect = lambda url, out_dir, session=None: Path(out_dir) / Path(url).name

    # Create file manually to simulate existing download
    existing_fp = tmp_out_dir / "file1"
    existing_fp.write_text("existing content")

    # reprocess=False should skip download
    results = download_urls(urls, tmp_out_dir, reprocess=False)
    assert results[0] == existing_fp

    # reprocess=True should call download
    results = download_urls(urls, tmp_out_dir, reprocess=True)
    assert results[0].name == "file1"

# ----------------------------
# download_urls_parallel tests
# ----------------------------

@patch("spicy_snow.utils.download.requests.Session")
@patch("spicy_snow.utils.download.validate_urls")
def test_download_urls_parallel_basic(mock_validate, mock_session_cls, tmp_out_dir):
    urls = ["http://example.com/file1", "http://example.com/file2"]
    mock_validate.return_value = urls

    # mock session.get to return dummy response
    mock_session = MagicMock()
    mock_session.get.return_value.__enter__.return_value.iter_content.return_value = [b"data"]
    mock_session.get.return_value.__enter__.return_value.raise_for_status.return_value = None
    mock_session_cls.return_value = mock_session

    results = download_urls_parallel(urls, tmp_out_dir, max_workers=2)
    assert all(isinstance(fp, Path) for fp in results)
    assert sorted([fp.name for fp in results]) == ["file1", "file2"]


# ----------------------------
# download_proba_v test
# ----------------------------

@patch("spicy_snow.utils.download.download_url")
def test_download_proba_v(mock_download, tmp_out_dir):
    out_fp = tmp_out_dir / "proba.tif"
    mock_download.return_value = out_fp

    result = download_proba_v(out_fp)
    assert result == out_fp
