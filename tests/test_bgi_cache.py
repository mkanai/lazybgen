"""Unit tests for the BGI index cache and remote .bgi download.

These cover ``lazybgen.remote.ensure_local_bgi`` and the cache-key helpers with
mocked filesystems. They are network-free; the live remote read path is covered
by the integration tests in test_gcs.py.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from lazybgen.remote import (
    _local_bgi_cache_path,
    _storage_options_discriminator,
    clear_bgi_cache,
    ensure_local_bgi,
    get_bgi_cache_info,
)

# ==================== Memory cache ====================


class TestBGIMemoryCache:
    """Test BGI memory cache functionality."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_bgi_cache()

    def test_cache_starts_empty(self):
        """Test that cache starts empty."""
        cache_info = get_bgi_cache_info()
        assert len(cache_info) == 0

    def test_local_path_not_cached(self):
        """Test that local paths are not cached."""
        local_path = "/tmp/test.bgi"
        result = ensure_local_bgi(local_path)
        assert result == local_path
        assert len(get_bgi_cache_info()) == 0

    def test_gcs_path_cached_after_download(self, tmp_path, monkeypatch):
        """Test that GCS paths are cached after successful download."""
        gcs_path = "gs://bucket/test.bgi"
        monkeypatch.setenv("LAZYBGEN_BGI_CACHE_DIR", str(tmp_path))
        expected_local = _local_bgi_cache_path(gcs_path)

        # Mock fsspec to simulate download
        mock_fs = MagicMock()
        mock_fs.get = MagicMock()

        with patch("fsspec.filesystem", return_value=mock_fs):
            ensure_local_bgi(gcs_path)

            # Download is targeted at a temp file in the hashed cache dir (atomic
            # download), then os.replace-d into place; never the CWD or final path.
            mock_fs.get.assert_called_once()
            called_src, called_dst = mock_fs.get.call_args.args
            assert called_src == gcs_path
            assert os.path.dirname(called_dst) == str(tmp_path)
            assert os.path.basename(called_dst).startswith(".tmp-")
            assert called_dst != expected_local
            assert os.path.dirname(expected_local) == str(tmp_path)

            # Check cache: keyed by (URL, storage_options discriminator); value is
            # the final atomic path.
            cache_info = get_bgi_cache_info()
            assert len(cache_info) == 1
            (cache_key,) = list(cache_info.keys())
            assert cache_key.startswith(gcs_path)
            assert cache_info[cache_key] == expected_local

    def test_cache_reuse(self, tmp_path, monkeypatch):
        """Test that cached paths are reused without re-downloading."""
        gcs_path = "gs://bucket/test.bgi"
        monkeypatch.setenv("LAZYBGEN_BGI_CACHE_DIR", str(tmp_path))
        expected_local = _local_bgi_cache_path(gcs_path)

        # Create existing cached BGI file
        os.makedirs(os.path.dirname(expected_local), exist_ok=True)
        with open(expected_local, "w") as f:
            f.write("dummy content")

        # First call - should reuse the cached file, no download
        result1 = ensure_local_bgi(gcs_path)
        assert result1 == expected_local

        # Second call - should use memory cache, no download
        with patch("fsspec.filesystem") as mock_fs_factory:
            result2 = ensure_local_bgi(gcs_path)
            assert result2 == result1
            mock_fs_factory.assert_not_called()

    def test_cache_clear(self):
        """Test cache clearing functionality."""
        # Add something to cache by checking a GCS path
        gcs_path = "gs://bucket/test.bgi"

        # Mock the download to populate cache
        with patch("fsspec.filesystem") as mock_fs_factory:
            mock_fs = MagicMock()
            mock_fs.get = MagicMock()
            mock_fs_factory.return_value = mock_fs

            with patch("os.path.exists", return_value=True):
                ensure_local_bgi(gcs_path)

        # Verify cache has content
        assert len(get_bgi_cache_info()) > 0

        # Clear cache
        clear_bgi_cache()
        assert len(get_bgi_cache_info()) == 0


# ==================== Download ====================


class TestBGIDownload:
    """Test BGI download functionality."""

    def setup_method(self):
        """Clear the memory cache before each test for isolation."""
        clear_bgi_cache()

    def test_download_with_retry(self, tmp_path, monkeypatch):
        """Test that download retries on failure."""
        gcs_path = "gs://bucket/test.bgi"
        monkeypatch.setenv("LAZYBGEN_BGI_CACHE_DIR", str(tmp_path))
        expected_local = _local_bgi_cache_path(gcs_path)

        # Mock fsspec.filesystem to fail twice then succeed
        mock_fs = MagicMock()
        mock_fs.get = MagicMock(
            side_effect=[
                Exception("Network error"),
                Exception("Timeout"),
                None,  # Success on third try
            ]
        )

        with patch("fsspec.filesystem", return_value=mock_fs):
            with patch("time.sleep"):  # Speed up test
                result = ensure_local_bgi(gcs_path)
                assert result == expected_local
                assert mock_fs.get.call_count == 3

    def test_download_failure_after_retries(self, tmp_path):
        """Test that download fails after max retries."""
        gcs_path = "gs://bucket/test.bgi"

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            # Mock fsspec.filesystem to always fail
            mock_fs = MagicMock()
            mock_fs.get = MagicMock(side_effect=Exception("Persistent error"))

            with patch("fsspec.filesystem", return_value=mock_fs):
                with patch("time.sleep"):  # Speed up test
                    with pytest.raises(RuntimeError, match="Failed to download BGI file"):
                        ensure_local_bgi(gcs_path)
                    assert mock_fs.get.call_count == 3  # Should try 3 times
        finally:
            os.chdir(original_cwd)

    def test_backend_not_installed(self, tmp_path, monkeypatch):
        """Friendly ImportError when the fsspec backend package is missing."""
        # Isolate the cache dir so no previously cached file short-circuits the
        # download attempt.
        monkeypatch.setenv("LAZYBGEN_BGI_CACHE_DIR", str(tmp_path))
        with patch("fsspec.filesystem", side_effect=ImportError("No module named gcsfs")):
            with pytest.raises(ImportError, match="gcsfs is required"):
                ensure_local_bgi("gs://bucket/test.bgi")
        with patch("fsspec.filesystem", side_effect=ImportError("No module named s3fs")):
            with pytest.raises(ImportError, match="s3fs is required"):
                ensure_local_bgi("s3://bucket/test.bgi")

    def test_passes_storage_options_for_s3(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LAZYBGEN_BGI_CACHE_DIR", str(tmp_path))
        captured = {}

        def fake_filesystem(scheme, **kwargs):
            captured["scheme"] = scheme
            captured["kwargs"] = kwargs
            m = MagicMock()
            m.get = MagicMock(side_effect=lambda src, dst: open(dst, "w").close())
            return m

        with patch("fsspec.filesystem", side_effect=fake_filesystem):
            ensure_local_bgi("s3://bucket/test.bgi", storage_options={"requester_pays": True})
        assert captured["scheme"] == "s3"
        assert captured["kwargs"] == {"requester_pays": True}


# ==================== Cache key / path ====================


def test_bgi_cache_path_not_in_cwd():
    local = _local_bgi_cache_path("gs://bucket/path/data.bgen.bgi")
    # Must not pollute the current working directory.
    assert os.path.isabs(local)
    assert os.path.dirname(local) != os.getcwd()
    assert os.path.basename(local).endswith("data.bgen.bgi")


def test_bgi_cache_path_no_collision_for_same_basename():
    # Same basename, different buckets/paths must map to different local files.
    a = _local_bgi_cache_path("gs://bucket-a/data.bgen.bgi")
    b = _local_bgi_cache_path("gs://bucket-b/data.bgen.bgi")
    c = _local_bgi_cache_path("s3://bucket-a/data.bgen.bgi")
    assert a != b
    assert a != c
    # Stable for the same URL.
    assert a == _local_bgi_cache_path("gs://bucket-a/data.bgen.bgi")


def test_bgi_cache_key_varies_with_storage_options():
    url = "gs://bucket/data.bgen.bgi"
    # Two different storage_options must produce different cache keys, so an index
    # fetched under one credential/billing setup is not reused under another.
    anon = _local_bgi_cache_path(url, {"anon": True})
    authed = _local_bgi_cache_path(url, {"anon": False})
    rp_a = _local_bgi_cache_path(url, {"requester_pays": "project-a"})
    rp_b = _local_bgi_cache_path(url, {"requester_pays": "project-b"})
    assert anon != authed
    assert rp_a != rp_b
    assert anon != rp_a


def test_bgi_cache_key_identical_for_identical_storage_options():
    url = "gs://bucket/data.bgen.bgi"
    # Identical options (regardless of dict ordering) must produce the same key.
    a = _local_bgi_cache_path(url, {"anon": True, "requester_pays": "p"})
    b = _local_bgi_cache_path(url, {"requester_pays": "p", "anon": True})
    assert a == b


def test_bgi_cache_key_none_matches_empty_options():
    url = "gs://bucket/data.bgen.bgi"
    # No options must be equivalent to empty options (byte-identical key).
    assert _local_bgi_cache_path(url, None) == _local_bgi_cache_path(url, {})


def test_bgi_cache_discriminator_does_not_leak_secrets():
    # A secret token value must not appear verbatim in the discriminator.
    disc = _storage_options_discriminator({"token": "super-secret-credential", "anon": True})
    assert "super-secret-credential" not in disc


def test_bgi_cache_key_varies_with_secret_value():
    url = "gs://bucket/data.bgen.bgi"
    # Different credential values must NOT share a cache entry (no under-invalidation):
    # an index fetched under one token is not reused under another.
    a = _local_bgi_cache_path(url, {"token": "credential-a"})
    b = _local_bgi_cache_path(url, {"token": "credential-b"})
    assert a != b


def test_bgi_cache_key_varies_with_unlisted_option_value():
    url = "gs://bucket/data.bgen.bgi"
    # An unlisted but byte-affecting option (e.g. endpoint_url) must still
    # discriminate so the cache is not reused across different backends.
    a = _local_bgi_cache_path(url, {"endpoint_url": "https://a.example"})
    b = _local_bgi_cache_path(url, {"endpoint_url": "https://b.example"})
    assert a != b


def test_bgi_cache_discriminator_does_not_leak_unlisted_value():
    # An unlisted option's raw value must not appear verbatim (it is hashed).
    disc = _storage_options_discriminator({"endpoint_url": "https://private.internal.example"})
    assert "private.internal.example" not in disc


def test_bgi_cache_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("LAZYBGEN_BGI_CACHE_DIR", str(tmp_path))
    local = _local_bgi_cache_path("gs://bucket/data.bgen.bgi")
    assert os.path.dirname(local) == str(tmp_path)


def test_ensure_local_bgi_failed_download_leaves_no_partial(tmp_path, monkeypatch):
    """A mid-download failure must not cache a partial file at the final path."""
    import fsspec

    from lazybgen import remote

    monkeypatch.setenv("LAZYBGEN_BGI_CACHE_DIR", str(tmp_path))
    remote.clear_bgi_cache()

    class _BoomFS:
        def get(self, src, dst):
            # Simulate a download that writes some bytes then fails partway.
            with open(dst, "wb") as fh:
                fh.write(b"partial-bytes")
            raise OSError("simulated mid-download failure")

    monkeypatch.setattr(fsspec, "filesystem", lambda scheme, **kw: _BoomFS())

    url = "gs://bucket/data.bgen.bgi"
    final = remote._local_bgi_cache_path(url)
    with pytest.raises(RuntimeError):
        remote.ensure_local_bgi(url)

    # The final cache path must not exist (no partial caching), and no leftover
    # temp files should remain in the cache dir.
    assert not os.path.exists(final)
    leftovers = [p for p in os.listdir(tmp_path) if p.startswith(".tmp-")]
    assert leftovers == []
    assert url + "\x00" not in "".join(remote.get_bgi_cache_info().keys())


def test_ensure_local_bgi_successful_download_is_atomic(tmp_path, monkeypatch):
    """A successful download lands at the final path via atomic replace."""
    import fsspec

    from lazybgen import remote

    monkeypatch.setenv("LAZYBGEN_BGI_CACHE_DIR", str(tmp_path))
    remote.clear_bgi_cache()

    seen_targets = []

    class _OkFS:
        def get(self, src, dst):
            # The download target must be a temp file, never the final path.
            seen_targets.append(dst)
            with open(dst, "wb") as fh:
                fh.write(b"index-bytes")

    monkeypatch.setattr(fsspec, "filesystem", lambda scheme, **kw: _OkFS())

    url = "gs://bucket/data.bgen.bgi"
    final = remote._local_bgi_cache_path(url)
    result = remote.ensure_local_bgi(url)

    assert result == final
    assert os.path.exists(final)
    with open(final, "rb") as fh:
        assert fh.read() == b"index-bytes"
    # fs.get wrote to a temp file, not directly to the final path.
    assert seen_targets and all(t != final for t in seen_targets)
    assert all(os.path.basename(t).startswith(".tmp-") for t in seen_targets)
