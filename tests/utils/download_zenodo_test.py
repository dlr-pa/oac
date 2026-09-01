"""
Provides tests for module openairclim.utils.download_zenodo
"""

# accessing the private _extract_record_id directly is the point of these
# tests, since it's the shared parsing logic behind every public function
# pylint: disable=protected-access

import hashlib
from pathlib import Path

import pytest
import requests

from openairclim.utils import download_zenodo


def _md5(data: bytes) -> str:
    """Build a Zenodo-format md5 checksum string for the given bytes."""
    return "md5:" + hashlib.md5(data).hexdigest()


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


class TestFetchJsonRetry:
    """Tests fetch_json's retry/backoff behaviour."""

    def test_invalid_max_attempts_raises(self):
        """max_attempts < 1 raises ValueError immediately."""
        with pytest.raises(ValueError):
            download_zenodo.fetch_json("https://example.org", max_attempts=0)

    def test_retries_on_timeout_then_succeeds(self, monkeypatch):
        """A transient Timeout is retried and a later attempt can succeed."""
        monkeypatch.setattr(download_zenodo.time, "sleep", lambda _s: None)
        calls = {"n": 0}

        def _get(_url, timeout=None):  # pylint: disable=unused-argument
            calls["n"] += 1
            if calls["n"] < 2:
                raise requests.exceptions.Timeout("simulated timeout")
            return _FakeResponse(json_body={"ok": True})

        monkeypatch.setattr(download_zenodo.requests, "get", _get)
        result = download_zenodo.fetch_json("https://example.org", max_attempts=3)
        assert result == {"ok": True}
        assert calls["n"] == 2

    def test_gives_up_after_max_attempts(self, monkeypatch):
        """The original exception is re-raised once attempts are exhausted."""
        monkeypatch.setattr(download_zenodo.time, "sleep", lambda _s: None)
        monkeypatch.setattr(
            download_zenodo.requests,
            "get",
            lambda _url, timeout=None: (_ for _ in ()).throw(
                requests.exceptions.ConnectionError("simulated")
            ),
        )
        with pytest.raises(requests.exceptions.ConnectionError):
            download_zenodo.fetch_json("https://example.org", max_attempts=2)

    def test_http_error_not_retried(self, monkeypatch):
        """A non-2xx status raises immediately, without consuming retries."""
        calls = {"n": 0}

        def _get(_url, timeout=None):  # pylint: disable=unused-argument
            calls["n"] += 1
            return _FakeResponse(status_ok=False)

        monkeypatch.setattr(download_zenodo.requests, "get", _get)
        with pytest.raises(requests.HTTPError):
            download_zenodo.fetch_json("https://example.org", max_attempts=3)
        assert calls["n"] == 1


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


class TestDownload:
    """Tests function download(record_or_doi, output_dir, file_glob, force, ...)"""

    def test_downloads_matching_files_only(self, tmp_path, monkeypatch):
        """Only files matching file_glob are downloaded."""
        record = {
            "files": [
                {"key": "a.nc", "checksum": "", "links": {"self": "https://x/a.nc"}},
                {"key": "b.txt", "checksum": "", "links": {"self": "https://x/b.txt"}},
            ]
        }
        monkeypatch.setattr(download_zenodo, "fetch_record_json", lambda _r: record)
        calls = []
        monkeypatch.setattr(
            download_zenodo, "download_file",
            lambda url, dest, *a, **kw: calls.append(url),
        )

        download_zenodo.download("123", tmp_path, file_glob="*.nc")

        assert calls == ["https://x/a.nc"]

    def test_skips_already_valid_files(self, tmp_path, monkeypatch):
        """A present, checksum-valid file is not re-downloaded."""
        content = b"aaa"
        (tmp_path / "a.nc").write_bytes(content)
        record = {
            "files": [{
                "key": "a.nc",
                "checksum": _md5(content),
                "links": {"self": "https://x/a.nc"},
            }]
        }
        monkeypatch.setattr(download_zenodo, "fetch_record_json", lambda _r: record)
        calls = []
        monkeypatch.setattr(
            download_zenodo, "download_file",
            lambda url, dest, *a, **kw: calls.append(url),
        )

        download_zenodo.download("123", tmp_path)

        assert not calls

    def test_force_redownloads_valid_files(self, tmp_path, monkeypatch):
        """force=True re-downloads even a valid, already-present file."""
        content = b"aaa"
        (tmp_path / "a.nc").write_bytes(content)
        record = {
            "files": [{
                "key": "a.nc",
                "checksum": _md5(content),
                "links": {"self": "https://x/a.nc"},
            }]
        }
        monkeypatch.setattr(download_zenodo, "fetch_record_json", lambda _r: record)
        calls = []

        def _download_file(url, dest, *_a, **_kw):
            calls.append(url)
            Path(dest).write_bytes(content)

        monkeypatch.setattr(download_zenodo, "download_file", _download_file)

        download_zenodo.download("123", tmp_path, force=True)

        assert calls == ["https://x/a.nc"]

    def test_checksum_mismatch_after_download_raises(self, tmp_path, monkeypatch):
        """A downloaded file that doesn't match its checksum raises RuntimeError."""
        record = {
            "files": [{
                "key": "a.nc",
                "checksum": _md5(b"expected"),
                "links": {"self": "https://x/a.nc"},
            }]
        }
        monkeypatch.setattr(download_zenodo, "fetch_record_json", lambda _r: record)
        monkeypatch.setattr(
            download_zenodo, "download_file",
            lambda url, dest, *a, **kw: Path(dest).write_bytes(b"corrupted"),
        )

        with pytest.raises(RuntimeError):
            download_zenodo.download("123", tmp_path)

    def test_creates_output_dir(self, tmp_path, monkeypatch):
        """A missing output_dir is created."""
        target = tmp_path / "nested" / "dir"
        monkeypatch.setattr(
            download_zenodo, "fetch_record_json", lambda _r: {"files": []}
        )

        download_zenodo.download("123", target)

        assert target.is_dir()
