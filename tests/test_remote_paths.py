import os
from unittest.mock import patch

import pytest

from lazybgen import BgenReader
from lazybgen.remote import REMOTE_SCHEMES, is_remote_path

DATA = os.path.join(os.path.dirname(__file__), "data")


def test_s3_path_skips_local_existence_guard():
    # An s3:// bgen path must NOT be rejected by the local-existence guard.
    # Force a deterministic, network-free failure by making the s3fs import fail
    # (None in sys.modules => ImportError). The error must come from the backend
    # layer ("Failed to import s3fs ..."), never "BGEN file not found".
    # Pinned to fsspec: nulling s3fs only forces a failure on that transport, and
    # an unpinned reader would instead make real requests through obstore.
    local_bgi = os.path.join(DATA, "data.bgen.bgi")
    with patch.dict("sys.modules", {"s3fs": None}):
        with pytest.raises(Exception) as exc:
            BgenReader(
                "s3://lazybgen-no-such-bucket-xyz/data.bgen",
                bgi_path=local_bgi,
                remote_backend="fsspec",
            )
    assert "BGEN file not found" not in str(exc.value)


def test_remote_schemes_contains_gs_and_s3():
    assert "gs://" in REMOTE_SCHEMES
    assert "s3://" in REMOTE_SCHEMES


def test_is_remote_path_true_for_gs_and_s3():
    assert is_remote_path("gs://bucket/file.bgen") is True
    assert is_remote_path("s3://bucket/file.bgen") is True


def test_is_remote_path_false_for_local():
    assert is_remote_path("/data/file.bgen") is False
    assert is_remote_path("file.bgen") is False
    assert is_remote_path("./rel/file.bgen") is False


def test_storage_options_accepted_for_local_read():
    # storage_options must be accepted (and harmlessly ignored) for local files.
    from lazybgen import load_bgen

    bgen = os.path.join(DATA, "data.bgen")
    dosages, info, samples = load_bgen(bgen, storage_options={"anon": True})
    assert dosages.shape[1] > 0
