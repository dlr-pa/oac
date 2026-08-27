"""
Provides tests for module openairclim.utils.download_zenodo
"""

# accessing the private _extract_record_id directly is the point of these
# tests, since it's the shared parsing logic behind every public function
# pylint: disable=protected-access

import hashlib

import pytest
import requests

from openairclim.utils import download_zenodo


class TestExtractRecordId:
    """Tests function _extract_record_id(record_or_doi)"""

    def test_bare_id(self):
        """A bare numeric ID is returned unchanged."""
        assert download_zenodo._extract_record_id("11442322") == "11442322"

    def test_doi_url(self):
        """A full DOI URL is parsed down to the trailing record ID."""
        doi = "https://doi.org/10.5281/zenodo.11442322"
        assert download_zenodo._extract_record_id(doi) == "11442322"

    def test_no_digits_raises(self):
        """A string with no digits raises ValueError."""
        with pytest.raises(ValueError):
            download_zenodo._extract_record_id("not-a-record")


class _FakeResponse:
    """Stand-in for requests.Response, used across this file's tests."""

    def __init__(self, json_body=None, content_chunks=None, status_ok=True):
        self._json_body = json_body
        self._content_chunks = content_chunks or []
        self._status_ok = status_ok

    def raise_for_status(self):
        """Raise like requests.Response.raise_for_status() would."""
        if not self._status_ok:
            raise requests.HTTPError("simulated HTTP error")

    def json(self):
        """Return the canned JSON body."""
        return self._json_body

    def iter_content(self, chunk_size=None):  # pylint: disable=unused-argument
        """Yield the canned content chunks."""
        yield from self._content_chunks


class TestFetchJson:
    """Tests function fetch_json(url)"""

    def test_parses_response_body(self, monkeypatch):
        """The URL's JSON body is parsed and returned as a dict."""
        monkeypatch.setattr(
            download_zenodo.requests,
            "get",
            lambda _url, timeout=None: _FakeResponse(json_body={"key": "value"}),
        )
        assert download_zenodo.fetch_json("https://example.org") == {"key": "value"}

    def test_error_status_raises(self, monkeypatch):
        """A non-2xx response raises via raise_for_status()."""
        monkeypatch.setattr(
            download_zenodo.requests,
            "get",
            lambda _url, timeout=None: _FakeResponse(status_ok=False),
        )
        with pytest.raises(requests.HTTPError):
            download_zenodo.fetch_json("https://example.org")


class TestFetchRecordJson:
    """Tests function fetch_record_json(record_or_doi)"""

    def test_builds_records_url(self, monkeypatch):
        """The record ID is extracted and used to build the API URL."""
        seen_urls = []
        monkeypatch.setattr(
            download_zenodo,
            "fetch_json",
            lambda url: seen_urls.append(url) or {"id": "123"},
        )
        result = download_zenodo.fetch_record_json("10.5281/zenodo.123")
        assert seen_urls == ["https://zenodo.org/api/records/123"]
        assert result == {"id": "123"}


class TestFetchRecordVersions:
    """Tests function fetch_record_versions(record_or_doi)"""

    def test_builds_versions_url_and_unwraps_hits(self, monkeypatch):
        """The /versions endpoint is queried and hits.hits is returned."""
        seen_urls = []

        def _fake_fetch_json(url):
            seen_urls.append(url)
            return {"hits": {"hits": [{"id": "1"}, {"id": "2"}]}}

        monkeypatch.setattr(download_zenodo, "fetch_json", _fake_fetch_json)
        result = download_zenodo.fetch_record_versions("123")
        assert seen_urls == ["https://zenodo.org/api/records/123/versions"]
        assert result == [{"id": "1"}, {"id": "2"}]

    def test_missing_hits_returns_empty_list(self, monkeypatch):
        """A malformed/empty response yields an empty list, not a KeyError."""
        monkeypatch.setattr(download_zenodo, "fetch_json", lambda _url: {})
        assert download_zenodo.fetch_record_versions("123") == []


class TestDownloadFile:
    """Tests function download_file(url, dest)"""

    def test_writes_streamed_content(self, tmp_path, monkeypatch):
        """The response's content chunks are written to dest in order."""
        dest = tmp_path / "data.nc"
        monkeypatch.setattr(
            download_zenodo.requests,
            "get",
            lambda _url, stream=None, timeout=None: _FakeResponse(
                content_chunks=[b"hello ", b"world"]
            ),
        )
        download_zenodo.download_file("https://example.org/data.nc", dest)
        assert dest.read_bytes() == b"hello world"

    def test_error_status_raises(self, tmp_path, monkeypatch):
        """A non-2xx response raises via raise_for_status(), nothing written."""
        dest = tmp_path / "data.nc"
        monkeypatch.setattr(
            download_zenodo.requests,
            "get",
            lambda _url, stream=None, timeout=None: _FakeResponse(status_ok=False),
        )
        with pytest.raises(requests.HTTPError):
            download_zenodo.download_file("https://example.org/data.nc", dest)
        assert not dest.exists()


class TestVerifyChecksum:
    """Tests function verify_checksum(path, expected)"""

    def test_matching_checksum_true(self, tmp_path):
        """A file whose md5 matches the expected checksum verifies True."""
        content = b"hello world"
        file_path = tmp_path / "data.nc"
        file_path.write_bytes(content)
        expected = "md5:" + hashlib.md5(content).hexdigest()
        assert download_zenodo.verify_checksum(file_path, expected) is True

    def test_mismatched_checksum_false(self, tmp_path):
        """A file whose content doesn't match the checksum verifies False."""
        file_path = tmp_path / "data.nc"
        file_path.write_bytes(b"hello world")
        expected = "md5:" + hashlib.md5(b"different content").hexdigest()
        assert download_zenodo.verify_checksum(file_path, expected) is False

    def test_missing_file_false(self, tmp_path):
        """A non-existent file verifies False rather than raising."""
        missing = tmp_path / "missing.nc"
        assert download_zenodo.verify_checksum(missing, "md5:whatever") is False

    def test_blank_expected_false(self, tmp_path):
        """A blank expected checksum (e.g. record has none) verifies False."""
        file_path = tmp_path / "data.nc"
        file_path.write_bytes(b"hello world")
        assert download_zenodo.verify_checksum(file_path, "") is False
