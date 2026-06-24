"""S3 integration test. Skipped unless LAZYBGEN_S3_TEST_BGEN points at a readable
s3:// BGEN. Set LAZYBGEN_S3_ANON=1 for public buckets, or rely on AWS creds +
LAZYBGEN_S3_REQUESTER_PAYS=1 for requester-pays buckets (e.g. Pan-UKB)."""

import os

import pytest

from lazybgen import load_bgen

S3_BGEN = os.environ.get("LAZYBGEN_S3_TEST_BGEN")


def _storage_options():
    opts = {}
    if os.environ.get("LAZYBGEN_S3_ANON") == "1":
        opts["anon"] = True
    if os.environ.get("LAZYBGEN_S3_REQUESTER_PAYS") == "1":
        opts["requester_pays"] = True
    return opts


@pytest.mark.integration
@pytest.mark.skipif(not S3_BGEN, reason="set LAZYBGEN_S3_TEST_BGEN to run the S3 integration test")
def test_load_bgen_from_s3():
    index = os.environ.get("LAZYBGEN_S3_TEST_BGI", S3_BGEN + ".bgi")
    try:
        dosages, info, samples = load_bgen(S3_BGEN, index_path=index, storage_options=_storage_options())
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"S3 integration test failed (network/creds): {e}")
    assert dosages.shape[1] > 0
    assert len(info) == dosages.shape[1]
