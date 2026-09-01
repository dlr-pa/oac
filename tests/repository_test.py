"""
Provides tests for module openairclim.repository
"""

import hashlib

import pytest

from openairclim import repository


def _md5(data: bytes) -> str:
    """Build a Zenodo-format md5 checksum string for the given bytes."""
    return "md5:" + hashlib.md5(data).hexdigest()


def _fake_record(files: dict, record_id: str = "12345") -> dict:
    """Build a minimal fake Zenodo record dict matching the API shape.

    Args:
        files (dict[str, bytes]): filename -> file content.
        record_id (str): The record's "id" field.

    Returns:
        dict: A record with a "files" list.
    """
    return {
        "id": record_id,
        "metadata": {"version": repository.DEFAULT_REPOSITORY_DATA_VERSION},
        "files": [
            {
                "key": name,
                "checksum": _md5(content),
                "links": {
                    "self": f"https://zenodo.org/records/{record_id}/files/{name}"
                },
            }
            for name, content in files.items()
        ],
    }


class TestGetCacheDir:
    """Tests function get_cache_dir(data_version)"""

    def test_env_override_used_as_is(self, monkeypatch, tmp_path):
        """ENV_CACHE_DIR, when set, is returned unchanged (no version suffix)."""
        monkeypatch.setenv(repository.ENV_CACHE_DIR, str(tmp_path))
        assert repository.get_cache_dir() == tmp_path
        assert repository.get_cache_dir("9.9.9") == tmp_path

    def test_default_version_namespacing(self, monkeypatch):
        """Without an override, the path is namespaced by
        DEFAULT_REPOSITORY_DATA_VERSION."""
        monkeypatch.delenv(repository.ENV_CACHE_DIR, raising=False)
        path = repository.get_cache_dir()
        assert repository.DEFAULT_REPOSITORY_DATA_VERSION in path.parts

    def test_custom_data_version_namespacing(self, monkeypatch):
        """An explicit data_version overrides the default in the path."""
        monkeypatch.delenv(repository.ENV_CACHE_DIR, raising=False)
        path = repository.get_cache_dir("2.5.0")
        assert "2.5.0" in path.parts
        assert repository.DEFAULT_REPOSITORY_DATA_VERSION not in path.parts


class TestResolveRecordId:
    """Tests function resolve_record_id(data_version)"""

    def test_matches_default_version(self, monkeypatch):
        """Resolves to the record whose metadata.version matches the default."""
        versions = [
            {"id": 1, "metadata": {"version": "0.0.9"}},
            {
                "id": 2,
                "metadata": {"version": repository.DEFAULT_REPOSITORY_DATA_VERSION},
            },
        ]
        monkeypatch.setattr(
            repository, "fetch_record_versions", lambda _doi: versions
        )
        assert repository.resolve_record_id() == "2"

    def test_matches_explicit_version(self, monkeypatch):
        """An explicit data_version is matched instead of the default."""
        versions = [
            {"id": 1, "metadata": {"version": "0.0.9"}},
            {"id": 2, "metadata": {"version": "1.0.0"}},
        ]
        monkeypatch.setattr(
            repository, "fetch_record_versions", lambda _doi: versions
        )
        assert repository.resolve_record_id("1.0.0") == "2"

    def test_no_match_raises(self, monkeypatch):
        """No matching version raises ValueError - no silent 'latest' fallback."""
        versions = [{"id": 1, "metadata": {"version": "0.0.9"}}]
        monkeypatch.setattr(
            repository, "fetch_record_versions", lambda _doi: versions
        )
        with pytest.raises(ValueError):
            repository.resolve_record_id("9.9.9")

    def test_empty_versions_raises(self, monkeypatch):
        """An empty version list also raises ValueError."""
        monkeypatch.setattr(repository, "fetch_record_versions", lambda _doi: [])
        with pytest.raises(ValueError):
            repository.resolve_record_id()


class TestCheckData:
    """Tests function check_data(cache_dir)"""

    def test_all_present_returns_empty(self, tmp_path):
        """No missing files if every REQUIRED_FILES entry exists."""
        for name in repository.REQUIRED_FILES:
            (tmp_path / name).write_bytes(b"x")
        assert repository.check_data(tmp_path) == []

    def test_missing_files_reported(self, tmp_path):
        """Missing files are returned by name."""
        (tmp_path / repository.REQUIRED_FILES[0]).write_bytes(b"x")
        missing = repository.check_data(tmp_path)
        assert repository.REQUIRED_FILES[0] not in missing
        assert repository.REQUIRED_FILES[1] in missing


class TestIsDataPresent:
    """Tests function is_data_present(cache_dir, record, verify_checksums)"""

    def test_true_when_all_present_no_checksum(self, tmp_path):
        """Existence-only check succeeds without a record."""
        for name in repository.REQUIRED_FILES:
            (tmp_path / name).write_bytes(b"x")
        assert repository.is_data_present(tmp_path) is True

    def test_false_when_missing(self, tmp_path):
        """False if any required file is absent."""
        assert repository.is_data_present(tmp_path) is False

    def test_checksum_verification_success(self, tmp_path):
        """With verify_checksums=True and matching checksums, returns True."""
        files = {name: name.encode() for name in repository.REQUIRED_FILES}
        for name, content in files.items():
            (tmp_path / name).write_bytes(content)
        record = _fake_record(files)
        assert (
            repository.is_data_present(tmp_path, record=record, verify_checksums=True)
            is True
        )

    def test_checksum_verification_failure(self, tmp_path):
        """A corrupted file fails checksum verification even though present."""
        files = {name: name.encode() for name in repository.REQUIRED_FILES}
        for name, content in files.items():
            (tmp_path / name).write_bytes(content)
        record = _fake_record(files)
        (tmp_path / repository.REQUIRED_FILES[0]).write_bytes(b"corrupted")
        assert (
            repository.is_data_present(tmp_path, record=record, verify_checksums=True)
            is False
        )


class TestDownloadData:
    """Tests function download_data(record_or_doi, output_dir, data_version, force)"""

    def test_uses_given_record_id(self, tmp_path, monkeypatch):
        """An explicit record_or_doi is passed straight through to download()."""
        calls = []
        monkeypatch.setattr(
            repository, "download",
            lambda record_id, target_dir, force=False: calls.append(
                (record_id, target_dir, force)
            ),
        )

        result_dir = repository.download_data(
            record_or_doi="12345", output_dir=tmp_path
        )

        assert result_dir == tmp_path
        assert calls == [("12345", tmp_path, False)]

    def test_resolves_record_id_when_not_given(self, tmp_path, monkeypatch):
        """Without record_or_doi, resolve_record_id() supplies the record id."""
        monkeypatch.setattr(repository, "resolve_record_id", lambda _v: "99999")
        calls = []
        monkeypatch.setattr(
            repository, "download",
            lambda record_id, target_dir, force=False: calls.append(record_id),
        )

        repository.download_data(output_dir=tmp_path)

        assert calls == ["99999"]

    def test_force_passed_through(self, tmp_path, monkeypatch):
        """force=True is forwarded to download()."""
        calls = []
        monkeypatch.setattr(
            repository, "download",
            lambda record_id, target_dir, force=False: calls.append(force),
        )

        repository.download_data(
            record_or_doi="12345", output_dir=tmp_path, force=True
        )

        assert calls == [True]

    def test_output_dir_defaults_to_cache_dir(self, tmp_path, monkeypatch):
        """Without output_dir, the cache dir resolver is used."""
        monkeypatch.setattr(repository, "get_cache_dir", lambda v=None: tmp_path)
        monkeypatch.setattr(
            repository, "download", lambda record_id, target_dir, force=False: None
        )

        result_dir = repository.download_data(record_or_doi="12345")

        assert result_dir == tmp_path
