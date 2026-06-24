"""Parallel batched load path: must be byte-identical to the serial load.

Parallel decode (the default; `num_threads != 1`) routes the unfiltered
float32/float64 load through a C++ path that pre-reads compressed bytes on the main
thread, then inflates and decodes variants across worker threads writing into
disjoint output columns. The
contract is that this produces output bit-for-bit identical to the serial
(per-variant) path; only the wall-clock time should change. These tests pin that
equivalence across bit depths, compression, dtypes, and region queries.
"""

import numpy as np
import pytest
from lazybgen.reader import BgenReader

FIXTURES = [
    "example.8bits.bgen",
    "example.16bits.bgen",
    "example.32bits.bgen",
    "example.16bits.zstd.bgen",
    "example.v11.bgen",
    "data.bgen",
    "haplotypes.bgen",  # phased biallelic diploid; unfiltered Direct path handles it
]


def _load(path, dec_type, dtype, num_threads=0, **query):
    # num_threads is the sole public knob: "sequential" maps to num_threads=1,
    # "parallel" uses the given count (0 = auto-detect cores).
    nt = 1 if dec_type == "sequential" else num_threads
    r = BgenReader(str(path), num_threads=nt)
    try:
        dosages, info = r.load_variants(dtype=dtype, **query)
        return dosages, info
    finally:
        r.close()


def _assert_identical(a, b):
    # Byte-identical, treating NaN as equal (missing genotypes decode to NaN).
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
    mask = ~np.isnan(a)
    # Exact equality on the non-NaN entries (no tolerance: same decode kernel).
    assert np.array_equal(a[mask], b[mask])


@pytest.mark.parametrize("fixture", FIXTURES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_parallel_matches_serial_full(data_dir, fixture, dtype):
    path = data_dir / fixture
    serial, _ = _load(path, "sequential", dtype)
    par, _ = _load(path, "parallel", dtype, num_threads=4)
    _assert_identical(serial, par)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_parallel_matches_serial_region(data_dir, dtype):
    # Region query feeds a metadata subset into the same batched path.
    path = data_dir / "example.8bits.bgen"
    r = BgenReader(str(path))
    info = r.load_variants(dtype=dtype)[1]
    chrom = info["chrom"].iloc[0]
    lo, hi = int(info["pos"].min()), int(info["pos"].max())
    r.close()
    q = dict(region_chrom=chrom, region_start=lo, region_end=(lo + hi) // 2)
    serial, sinfo = _load(path, "sequential", dtype, **q)
    par, pinfo = _load(path, "parallel", dtype, num_threads=8, **q)
    assert len(sinfo) > 1  # the query actually selected a subset
    _assert_identical(serial, par)
    assert list(sinfo["pos"]) == list(pinfo["pos"])


@pytest.mark.parametrize("num_threads", [1, 2, 16])
def test_parallel_thread_counts_agree(data_dir, num_threads):
    # Output must not depend on the worker-thread count.
    path = data_dir / "example.16bits.bgen"
    serial, _ = _load(path, "sequential", np.float64)
    par, _ = _load(path, "parallel", np.float64, num_threads=num_threads)
    _assert_identical(serial, par)


@pytest.mark.parametrize("fixture", FIXTURES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_parallel_sample_filtered_matches_serial(data_dir, fixture, dtype):
    # Sample-filtered (cohort) load through the parallel filtered decode path
    # must be byte-identical to the serial filtered decode, for both dtypes.
    path = data_dir / fixture
    r = BgenReader(str(path))
    n = r.nsamples
    r.close()
    idx = np.arange(0, n, 2, dtype=np.int32)  # 50% cohort
    serial, _ = _load(path, "sequential", dtype, sample_indices=idx)
    par, _ = _load(path, "parallel", dtype, num_threads=4, sample_indices=idx)
    _assert_identical(serial, par)


@pytest.mark.parametrize("num_threads", [1, 2, 16])
def test_parallel_sample_filtered_thread_counts_agree(data_dir, num_threads):
    # Filtered parallel output must not depend on the worker-thread count.
    path = data_dir / "example.16bits.bgen"
    r = BgenReader(str(path))
    n = r.nsamples
    r.close()
    idx = np.array([0, 1, 5, 7, n - 1], dtype=np.int32)  # scattered, includes last
    serial, _ = _load(path, "sequential", np.float64, sample_indices=idx)
    par, _ = _load(path, "parallel", np.float64, num_threads=num_threads, sample_indices=idx)
    _assert_identical(serial, par)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_parallel_region_plus_sample_filtered(data_dir, dtype):
    # Combined region subset + sample filter through the parallel filtered path.
    path = data_dir / "example.8bits.bgen"
    r = BgenReader(str(path))
    info = r.load_variants(dtype=dtype)[1]
    n = r.nsamples
    chrom = info["chrom"].iloc[0]
    lo, hi = int(info["pos"].min()), int(info["pos"].max())
    r.close()
    idx = np.arange(1, n, 3, dtype=np.int32)
    q = dict(region_chrom=chrom, region_start=lo, region_end=(lo + hi) // 2, sample_indices=idx)
    serial, sinfo = _load(path, "sequential", dtype, **q)
    par, _ = _load(path, "parallel", dtype, num_threads=8, **q)
    assert len(sinfo) > 1
    _assert_identical(serial, par)
