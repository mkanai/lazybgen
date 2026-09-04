"""Batched remote range reads, driven through a stand-in fsspec backend.

The block decode fetches a whole block's variant records in one call, and the
fsspec reader turns that into a single ``cat_ranges`` request per group of
nearby ranges. That code only runs for ``gs://`` / ``s3://`` paths, so these
tests swap in a filesystem class that serves a local fixture: no network, but
the real C++ read path.

What is covered here: that a remote read still returns exactly what a local read
does, that adjacent records are merged into one request (fetching them
separately made contiguous reads several times slower), and that a backend
returning something unusable is reported rather than decoded. Coalescing's other
half - a genuinely scattered selection staying at one request per variant -
needs records further apart than the bundled fixtures are, and is covered by
tests/test_gcs.py against the real bucket.
"""

from pathlib import Path

import numpy as np
import pytest
from lazybgen.reader import BgenReader

DATA_DIR = Path(__file__).parent / "data"
LOCAL_BGEN = DATA_DIR / "example.16bits.bgen"
LOCAL_BGI = DATA_DIR / "example.16bits.bgen.bgi"
REMOTE_URL = "gs://stand-in-bucket/example.16bits.bgen"


class _RecordingFileSystem:
    """Minimal fsspec-shaped filesystem serving one local file.

    Records every ``cat_ranges`` call so a test can assert how the reader
    grouped its requests, and can be told to hand back a malformed result.
    """

    payload = b""
    calls: list = []
    mode = "ok"

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def info(self, path):
        return {"size": len(type(self).payload), "name": path}

    def open(self, path, mode="rb", block_size=None, **kwargs):
        import io

        return io.BytesIO(type(self).payload)

    def cat_ranges(self, paths, starts, ends, **kwargs):
        cls = type(self)
        cls.calls.append(list(zip(starts, ends)))
        out = [cls.payload[s:e] for s, e in zip(starts, ends)]
        if cls.mode == "bad_item":
            out[0] = ValueError("range unavailable")
        elif cls.mode == "wrong_count":
            out = out[:-1]
        elif cls.mode == "short":
            out = [chunk[:-1] for chunk in out]
        return out


@pytest.fixture
def stand_in_gcs(monkeypatch):
    """Point the gs:// backend at a local fixture for the duration of a test."""
    gcsfs = pytest.importorskip("gcsfs")

    _RecordingFileSystem.payload = LOCAL_BGEN.read_bytes()
    _RecordingFileSystem.calls = []
    _RecordingFileSystem.mode = "ok"
    monkeypatch.setattr(gcsfs, "GCSFileSystem", _RecordingFileSystem)
    yield _RecordingFileSystem
    _RecordingFileSystem.payload = b""
    _RecordingFileSystem.calls = []


def _open_remote():
    return BgenReader(REMOTE_URL, bgi_path=str(LOCAL_BGI))


def test_batched_remote_read_matches_a_local_read(stand_in_gcs):
    """The batched path returns exactly what reading the file locally returns."""
    with BgenReader(str(LOCAL_BGEN)) as local:
        expected, _ = local.load_variants(dtype=np.float64)

    with _open_remote() as remote:
        got, _ = remote.load_variants(dtype=np.float64)

    assert stand_in_gcs.calls, "the batched range path was never taken"
    np.testing.assert_array_equal(np.isnan(got), np.isnan(expected))
    np.testing.assert_array_equal(got[~np.isnan(got)], expected[~np.isnan(expected)])


def test_adjacent_records_are_merged_into_few_requests(stand_in_gcs):
    """Records that sit next to each other are fetched as one range, not many.

    Every variant in this fixture is adjacent to the next, so the whole block
    should collapse to a small number of requests. Asking for them one at a time
    is what made contiguous remote reads several times slower.
    """
    with _open_remote() as remote:
        dosages, _ = remote.load_variants(dtype=np.float32)

    n_variants = dosages.shape[1]
    ranges = [r for call in stand_in_gcs.calls for r in call]
    assert n_variants > 50
    assert len(ranges) < 0.1 * n_variants, (
        f"{len(ranges)} range requests for {n_variants} adjacent records; " "adjacent records should be merged"
    )
    # The merged ranges still have to cover every byte that was asked for.
    assert sum(end - start for start, end in ranges) > 0


def test_distant_records_are_batched_but_not_merged(stand_in_gcs):
    """Records too far apart to merge still go out in one request batch.

    This is the other half of the coalescing rule: merging is what keeps a
    contiguous read from issuing one request per variant, and batching is what
    keeps a scattered read from issuing them one after another. Here the
    selection is spread far enough that nothing merges, so the ranges must
    arrive as several ranges in a single call rather than several calls.
    """
    import sqlite3

    con = sqlite3.connect(str(LOCAL_BGI))
    rows = con.execute(
        "SELECT chromosome, position, allele1, allele2 FROM Variant ORDER BY file_start_position"
    ).fetchall()
    con.close()
    # Every 40th record of this fixture is ~70 KB from its neighbour, past the
    # 64 KB merge gap, so these stay separate ranges.
    picks = rows[::40]
    assert len(picks) >= 4
    vf = {
        "chromosome": picks[0][0],
        "positions": [p[1] for p in picks],
        "allele1": [p[2] for p in picks],
        "allele2": [p[3] for p in picks],
    }

    with _open_remote() as remote:
        dosages, _ = remote.load_variants(variant_filter=vf, dtype=np.float32)

    assert dosages.shape[1] == len(picks)
    assert len(stand_in_gcs.calls) == 1, "the ranges should have been fetched together"
    assert len(stand_in_gcs.calls[0]) == len(picks), (
        f"{len(stand_in_gcs.calls[0])} ranges for {len(picks)} well-separated records; "
        "they should not have been merged"
    )


def test_a_failed_range_is_reported_not_decoded(stand_in_gcs):
    """An unusable element in the result list surfaces as an error."""
    stand_in_gcs.mode = "bad_item"
    with _open_remote() as remote:
        with pytest.raises(Exception, match="cat_ranges"):
            remote.load_variants(dtype=np.float32)


def test_a_short_result_list_is_reported(stand_in_gcs):
    """A backend returning fewer results than ranges is an error, not a decode."""
    stand_in_gcs.mode = "wrong_count"
    with _open_remote() as remote:
        with pytest.raises(Exception, match="cat_ranges returned"):
            remote.load_variants(dtype=np.float32)


def test_a_truncated_range_is_reported(stand_in_gcs):
    """A short read is caught before the record is parsed."""
    stand_in_gcs.mode = "short"
    with _open_remote() as remote:
        with pytest.raises(Exception, match="Failed to read complete variant record"):
            remote.load_variants(dtype=np.float32)
