"""Batched remote range reads over the obstore transport, without a network.

Mirrors test_remote_batched_reads.py, which covers the same C++ read path over
fsspec. Here a stand-in ``obstore`` module serves one local fixture, so the real
reader, the real adapter in ``lazybgen.obstore_backend`` (option translation,
thread fan-out, buffer hand-off) and the real C++ range grouping all run; only
the network is replaced. These tests therefore run whether or not obstore is
installed.
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest
from lazybgen.reader import BgenReader

from lazybgen import obstore_backend

DATA_DIR = Path(__file__).parent / "data"
LOCAL_BGEN = DATA_DIR / "example.16bits.bgen"
LOCAL_BGI = DATA_DIR / "example.16bits.bgen.bgi"
REMOTE_URL = "gs://stand-in-bucket/example.16bits.bgen"


class _StandInStore:
    """Stands in for an obstore store object: just carries its constructor args."""

    def __init__(self, bucket, **kwargs):
        self.bucket = bucket
        self.kwargs = kwargs


def _make_stand_in_obstore(payload: bytes, calls: list):
    """Build a module object with the obstore functions the adapter calls."""

    module = types.ModuleType("obstore")
    store_module = types.ModuleType("obstore.store")
    store_module.GCSStore = _StandInStore
    store_module.S3Store = _StandInStore
    module.store = store_module

    def head(_store, path):
        return {"path": path, "size": len(payload)}

    def get_ranges(_store, _path, *, starts, ends=None, lengths=None, coalesce=0):
        if ends is None:
            ends = [s + n for s, n in zip(starts, lengths)]
        calls.append(list(zip(starts, ends)))
        # memoryview so the reader has to go through the buffer protocol, as it
        # does with the real Bytes objects.
        return [memoryview(payload[s:e]) for s, e in zip(starts, ends)]

    def get_range(_store, _path, *, start, end=None, length=None):
        if end is None:
            end = start + length
        return memoryview(payload[start:end])

    module.head = head
    module.get_ranges = get_ranges
    module.get_range = get_range
    return module, store_module


@pytest.fixture
def stand_in_obstore(monkeypatch):
    """Serve REMOTE_URL from the local fixture through a stand-in obstore."""
    payload = LOCAL_BGEN.read_bytes()
    calls: list = []
    module, store_module = _make_stand_in_obstore(payload, calls)
    monkeypatch.setitem(sys.modules, "obstore", module)
    monkeypatch.setitem(sys.modules, "obstore.store", store_module)
    yield calls


def _open_remote(**kwargs):
    return BgenReader(REMOTE_URL, bgi_path=str(LOCAL_BGI), remote_backend="obstore", **kwargs)


def test_auto_picks_obstore_when_it_is_importable(stand_in_obstore):
    """With obstore importable and plain options, "auto" resolves to obstore."""
    from lazybgen.remote import resolve_remote_backend

    assert resolve_remote_backend(REMOTE_URL, None, "auto") == "obstore"


def test_obstore_read_matches_a_local_read(stand_in_obstore):
    """The obstore path returns exactly what reading the file locally returns."""
    with BgenReader(str(LOCAL_BGEN)) as local:
        expected, _ = local.load_variants(dtype=np.float64)

    with _open_remote() as remote:
        got, _ = remote.load_variants(dtype=np.float64)

    assert stand_in_obstore, "the batched range path was never taken"
    np.testing.assert_array_equal(np.isnan(got), np.isnan(expected))
    np.testing.assert_array_equal(got[~np.isnan(got)], expected[~np.isnan(expected)])


def test_obstore_merges_adjacent_records(stand_in_obstore):
    """Adjacent records are fetched as one range, exactly as over fsspec."""
    with _open_remote() as remote:
        dosages, _ = remote.load_variants(dtype=np.float32)

    n_variants = dosages.shape[1]
    ranges = [r for call in stand_in_obstore for r in call]
    assert n_variants > 50
    assert (
        len(ranges) < 0.1 * n_variants
    ), f"{len(ranges)} range requests for {n_variants} adjacent records; adjacent records should be merged"


def test_fan_out_preserves_the_order_of_a_batch(stand_in_obstore, monkeypatch):
    """A batch split across threads comes back in the caller's order.

    Each worker gets a slice of the batch and writes its results back by index; a
    mix-up there would hand one variant's bytes to another. The ranges below are
    deliberately non-monotonic so a result reordered by file position would fail.
    """
    monkeypatch.setenv("LAZYBGEN_OBSTORE_THREADS", "4")
    monkeypatch.setattr(obstore_backend, "_MIN_RANGES_TO_SPLIT", 1)

    payload = LOCAL_BGEN.read_bytes()
    ranges = [(i * 997, i * 997 + 64) for i in range(40)]
    ranges = ranges[::2][::-1] + ranges[1::2]

    fs = obstore_backend.ObstoreFileSystem()
    got = fs.cat_ranges([REMOTE_URL] * len(ranges), [s for s, _ in ranges], [e for _, e in ranges])

    assert len(stand_in_obstore) > 1, "the batch was not split across workers"
    assert len(got) == len(ranges)
    for buffer, (start, end) in zip(got, ranges):
        assert bytes(buffer) == payload[start:end]


def test_a_failed_range_is_reported_not_decoded(stand_in_obstore, monkeypatch):
    """A store error surfaces as an error, never as silently wrong dosages."""
    import obstore

    def boom(*_args, **_kwargs):
        raise RuntimeError("range unavailable")

    reader = _open_remote()
    monkeypatch.setattr(obstore, "get_ranges", boom)
    with pytest.raises(Exception, match="range unavailable"):
        reader.load_variants(dtype=np.float64)
    reader.close()


def test_a_truncated_range_is_reported(stand_in_obstore, monkeypatch):
    """A short result is caught rather than decoded as a whole record."""
    import obstore

    original = obstore.get_ranges

    def truncating(*args, **kwargs):
        return [chunk[:-1] for chunk in original(*args, **kwargs)]

    reader = _open_remote()
    monkeypatch.setattr(obstore, "get_ranges", truncating)
    with pytest.raises(Exception):
        reader.load_variants(dtype=np.float64)
    reader.close()


def test_storage_options_reach_the_store(stand_in_obstore):
    """Translated options land on the store the adapter builds."""
    fs = obstore_backend.ObstoreFileSystem(requester_pays="billed-project")
    store, key = fs._store(REMOTE_URL)
    assert key == "example.16bits.bgen"
    assert store.bucket == "stand-in-bucket"
    assert store.kwargs["client_options"]["default_headers"]["x-goog-user-project"] == "billed-project"
