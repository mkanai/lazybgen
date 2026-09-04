"""Integration tests for the GCS remote-read path in lazybgen.

These run against the public bucket gs://gcs-anndata-test and are the regression
gate that the remote byte-range read path stays byte-identical to a local read
and does not over-fetch. Run explicitly with ``-m integration``. The mocked,
network-free BGI cache/download unit tests live in test_bgi_cache.py.
"""

import contextlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lazybgen import BgenReader, load_bgen

# Public GCS bucket holding the same example fixtures as tests/data/.
GCS_DATA = "gs://gcs-anndata-test/lazybgen"
LOCAL_DATA = Path(__file__).parent / "data"

# A multi-block (~20 MB) fixture used to guard the remote readahead bound: a
# point read must fetch ~one small block, not a large fixed block or the whole
# file. Public; egress per run is one small block.
GCS_LARGE_BGEN = "gs://gcs-anndata-test/lazybgen/test_5000s_2000v_zlib_8bit.bgen"


@contextlib.contextmanager
def _counting_gcs_requests():
    """Count the GCS range requests a block of code makes.

    A block decode fetches its records through the filesystem's batched range
    API, while a single-record read still goes through the file handle's
    readahead, so both have to be counted for the total to mean anything.
    """
    from gcsfs.core import GCSFile, GCSFileSystem

    calls = {"n": 0}
    orig_fetch_range = GCSFile._fetch_range
    orig_cat_file = GCSFileSystem._cat_file

    def counting_fetch_range(self, start, end):
        calls["n"] += 1
        return orig_fetch_range(self, start, end)

    async def counting_cat_file(self, path, start=None, end=None, **kwargs):
        calls["n"] += 1
        return await orig_cat_file(self, path, start=start, end=end, **kwargs)

    GCSFile._fetch_range = counting_fetch_range
    GCSFileSystem._cat_file = counting_cat_file
    try:
        yield calls
    finally:
        GCSFile._fetch_range = orig_fetch_range
        GCSFileSystem._cat_file = orig_cat_file


@pytest.mark.integration
class TestGCSIntegration:
    """Integration tests requiring actual GCS access.

    These run against the public bucket gs://gcs-anndata-test/lazybgen, which
    holds copies of the example fixtures in tests/data/. They are the guard that
    the remote byte-range read path stays byte-identical to a local read. They
    must PASS (not skip) on an environment with GCS access; run explicitly with
    ``-m integration``.
    """

    @pytest.fixture(autouse=True)
    def _isolate_bgi_cache(self, tmp_path, monkeypatch):
        """Download remote .bgi indexes into a throwaway cache dir."""
        monkeypatch.setenv("LAZYBGEN_BGI_CACHE_DIR", str(tmp_path))

    def test_load_bgen_from_gcs(self):
        """Loading a BGEN from the public GCS bucket returns a well-formed result."""
        gcs_path = f"{GCS_DATA}/example.16bits.bgen"
        dosages, variant_info, sample_ids = load_bgen(gcs_path, nan_action="omit", show_progress=False)

        assert isinstance(dosages, np.ndarray)
        assert isinstance(variant_info, pd.DataFrame)
        assert isinstance(sample_ids, list)
        assert dosages.shape[0] > 0
        assert dosages.shape[1] > 0
        assert len(sample_ids) == dosages.shape[0]
        assert len(variant_info) == dosages.shape[1]
        assert {"chrom", "pos", "rsid", "ref", "alt"}.issubset(variant_info.columns)

    @pytest.mark.parametrize(
        "fname",
        ["example.8bits.bgen", "example.16bits.bgen", "example.32bits.bgen", "example.16bits.zstd.bgen"],
    )
    def test_gcs_read_matches_local(self, fname):
        """A GCS read must be byte-identical to a local read, full and filtered.

        Exercises the remote byte-range path against a local ground truth for
        every bit depth and zstd, for both the full decode and the
        sample-filtered decode (sample index 0 carries missing genotypes, so
        this also guards the filtered missing-value path over the network).
        """
        local = str(LOCAL_DATA / fname)
        remote = f"{GCS_DATA}/{fname}"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            local_full, _, local_ids = load_bgen(local, nan_action="warn")
            remote_full, _, remote_ids = load_bgen(remote, nan_action="warn")

        assert remote_ids == local_ids
        np.testing.assert_array_equal(np.isnan(remote_full), np.isnan(local_full))
        np.testing.assert_array_equal(remote_full[~np.isnan(remote_full)], local_full[~np.isnan(local_full)])

        # Sample-filtered path over GCS; index 0 has missing genotypes (-> NaN).
        requested = [local_ids[i] for i in (0, 3, 7)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            local_sub, _, _ = load_bgen(local, sample_ids=requested, nan_action="warn")
            remote_sub, _, _ = load_bgen(remote, sample_ids=requested, nan_action="warn")

        np.testing.assert_array_equal(np.isnan(remote_sub), np.isnan(local_sub))
        np.testing.assert_array_equal(remote_sub[~np.isnan(remote_sub)], local_sub[~np.isnan(local_sub)])

    def test_point_read_does_not_overfetch(self):
        """A single-variant remote read fetches ~one small block, not the whole
        file or a large fixed block.

        This is the regression guard for the access-pattern readahead bound:
        the reader sizes the remote block to the selection, so an isolated
        variant pulls roughly one variant's worth
        of bytes. The pre-fix fixed 10 MB readahead block fetched ~10 MB for the
        same point read; on this ~20 MB file that is plainly distinguishable.
        Skips if the fixture is unreachable.
        """
        import gcsfs
        from gcsfs.core import GCSFile

        fs = gcsfs.GCSFileSystem()
        try:
            file_size = fs.info(GCS_LARGE_BGEN)["size"]
        except Exception as exc:  # noqa: BLE001 - environment/fixture issue -> skip
            pytest.skip(f"large fixture unavailable: {exc}")
        assert file_size > 16 * 1024 * 1024  # must exceed the max block to be a real guard

        # Count bytes pulled from GCS. fsspec binds a file's range fetcher at
        # OPEN time, so the wrapper must be installed BEFORE the reader opens.
        fetched = {"bytes": 0}
        orig = GCSFile._fetch_range

        def counting_fetch_range(self, start, end):
            fetched["bytes"] += end - start
            return orig(self, start, end)

        GCSFile._fetch_range = counting_fetch_range
        try:
            reader = BgenReader(GCS_LARGE_BGEN, bgi_path=GCS_LARGE_BGEN + ".bgi")
            # Measure only the genotype read path, not the one-time header/sample
            # parse done at open (which fills the larger sequential buffer).
            fetched["bytes"] = 0
            # block_size=1 routes one variant through the small-block path.
            for _info, _dosage in reader.iter_variants(block_size=1):
                break
        finally:
            GCSFile._fetch_range = orig

        # One variant's record for this fixture is well under 1 MB; allow slack
        # for the readahead block + metadata window, but stay far below the
        # ~10 MB a fixed large block would pull for a single seek.
        assert 0 < fetched["bytes"] < 2 * 1024 * 1024, (
            f"point read fetched {fetched['bytes'] / 1e6:.2f} MB from a "
            f"{file_size / 1e6:.1f} MB file (expected ~one small block)"
        )

    def test_open_does_not_overfetch(self):
        """Opening a remote reader fetches a small header window, not ~10 MB.

        The reader's internal sequential buffer serves only the one-time
        header/sample parse (all genotype I/O is random read_at), so it is sized
        modestly. An oversized buffer would over-fetch at open on small files,
        which is paid on every open. Skips if the fixture is unreachable.
        """
        import gcsfs
        from gcsfs.core import GCSFile

        fs = gcsfs.GCSFileSystem()
        try:
            fs.info(GCS_LARGE_BGEN)["size"]
        except Exception as exc:  # noqa: BLE001 - environment/fixture issue -> skip
            pytest.skip(f"large fixture unavailable: {exc}")

        fetched = {"bytes": 0}
        orig = GCSFile._fetch_range

        def counting_fetch_range(self, start, end):
            fetched["bytes"] += end - start
            return orig(self, start, end)

        GCSFile._fetch_range = counting_fetch_range
        try:
            BgenReader(GCS_LARGE_BGEN, bgi_path=GCS_LARGE_BGEN + ".bgi")
        finally:
            GCSFile._fetch_range = orig

        # The header + sample block for this fixture are well under 4 MB; the
        # pre-fix 10 MB buffer fetched ~11 MB.
        assert 0 < fetched["bytes"] < 4 * 1024 * 1024, (
            f"open fetched {fetched['bytes'] / 1e6:.2f} MB (expected a small "
            "header window, not the full sequential buffer)"
        )

    def test_scattered_read_is_one_get_per_variant(self):
        """A scattered (variant-filtered) remote read costs ~1 range request per
        variant, not 2.

        The decode reads each whole record in a single range request and locates
        the genotype in-buffer, so the separate metadata-read phase is gone (a
        scattered read is 1 GET/variant, not 2). Skips if the fixture is
        unreachable.
        """
        import sqlite3

        from lazybgen.remote import ensure_local_bgi

        try:
            bgi_local = ensure_local_bgi(GCS_LARGE_BGEN + ".bgi")
            con = sqlite3.connect(bgi_local)
            rows = list(
                con.execute(
                    "SELECT chromosome, position, allele1, allele2, file_start_position "
                    "FROM Variant ORDER BY file_start_position"
                )
            )
            con.close()
        except Exception as exc:  # noqa: BLE001 - environment/fixture issue -> skip
            pytest.skip(f"large fixture unavailable: {exc}")

        # Widely-separated variants so each falls in its own readahead block (a
        # genuine scattered pattern, not coincidentally-adjacent records).
        picks = rows[:: max(1, len(rows) // 20)][:20]
        assert len(picks) >= 10
        # Span the file so the selection is unambiguously scattered.
        assert picks[-1][4] - picks[0][4] > 5 * 1024 * 1024
        vf = {
            "chromosome": picks[0][0],
            "positions": [p[1] for p in picks],
            "allele1": [p[2] for p in picks],
            "allele2": [p[3] for p in picks],
        }

        with _counting_gcs_requests() as calls:
            reader = BgenReader(GCS_LARGE_BGEN, bgi_path=GCS_LARGE_BGEN + ".bgi")
            calls["n"] = 0  # isolate the genotype reads from open-time fetches
            dosages, _info = reader.load_variants(variant_filter=vf)

        n_variants = dosages.shape[1]
        assert n_variants == len(picks)
        # One record read per variant: allow a small slack for an occasional split
        # block, but stay well under the pre-fix 2/variant. The lower bound keeps
        # the check honest if the counted call sites ever stop being the ones the
        # read actually uses.
        assert 0.9 * n_variants <= calls["n"] <= 1.3 * n_variants, (
            f"scattered read made {calls['n']} range requests for {n_variants} "
            f"variants ({calls['n'] / n_variants:.2f}/variant; expected ~1)"
        )

    def test_contiguous_region_read_is_coalesced(self):
        """Records that sit next to each other are fetched together.

        A scattered selection needs one request per variant, but a contiguous
        region would be pathological that way: the records are adjacent, so they
        are merged into a handful of large requests instead of hundreds of small
        ones. Skips if the fixture is unreachable.
        """
        import sqlite3

        from lazybgen.remote import ensure_local_bgi

        try:
            bgi_local = ensure_local_bgi(GCS_LARGE_BGEN + ".bgi")
            con = sqlite3.connect(bgi_local)
            rows = list(con.execute("SELECT chromosome, position FROM Variant ORDER BY file_start_position LIMIT 200"))
            con.close()
        except Exception as exc:  # noqa: BLE001 - environment/fixture issue -> skip
            pytest.skip(f"large fixture unavailable: {exc}")

        assert len(rows) >= 100
        chrom, start = rows[0]
        end = rows[-1][1]

        with _counting_gcs_requests() as calls:
            reader = BgenReader(GCS_LARGE_BGEN, bgi_path=GCS_LARGE_BGEN + ".bgi")
            calls["n"] = 0  # isolate the genotype reads from open-time fetches
            dosages, _info = reader.load_variants(region_chrom=chrom, region_start=start, region_end=end)

        n_variants = dosages.shape[1]
        assert n_variants >= 100
        assert calls["n"] < 0.25 * n_variants, (
            f"contiguous region made {calls['n']} range requests for {n_variants} "
            "adjacent variants; adjacent records should be fetched together"
        )
