"""Validation of user-supplied sample_indices at the public reader boundary.

sample_indices flows from load_variants / iter_variants down through Cython into
a raw C pointer cast (const int* / const uint32_t*). Without coercion and bounds
checks a wrong-dtype, non-contiguous, or out-of-range array silently misreads
memory. These tests pin the contract: coerce to contiguous int32 and validate
every index is in [0, n_samples).
"""

import numpy as np
import pytest

from lazybgen import BgenReader


@pytest.fixture(scope="module")
def reader(data_dir):
    r = BgenReader(str(data_dir / "example.16bits.bgen"))
    yield r
    r.close()


def test_out_of_range_sample_index_raises(reader):
    """An index >= n_samples must raise a clear error, not read OOB memory."""
    n = reader.nsamples
    bad = np.array([0, 1, n], dtype=np.int32)  # n is one past the last valid index
    with pytest.raises((ValueError, IndexError)):
        reader.load_variants(sample_indices=bad)


def test_negative_sample_index_raises(reader):
    """A negative index must raise rather than wrap into a huge unsigned offset."""
    bad = np.array([0, -1, 2], dtype=np.int32)
    with pytest.raises((ValueError, IndexError)):
        reader.load_variants(sample_indices=bad)


def test_int64_overflow_does_not_wrap_past_bounds(reader):
    """An int64 value that would alias to a valid index after a naive int32 cast
    (e.g. 2**32 -> 0) must still be rejected as out of range."""
    bad = np.array([0, 2**32, 2**32 + 5], dtype=np.int64)
    with pytest.raises((ValueError, IndexError)):
        reader.load_variants(sample_indices=bad)


def test_int64_and_noncontiguous_match_contiguous_int32(reader):
    """int64 / non-contiguous sample_indices must yield bit-identical results to
    the equivalent contiguous int32 array (no silent memory corruption)."""
    base = [0, 5, 10, 15, 20]
    ref = np.array(base, dtype=np.int32)
    ref = np.ascontiguousarray(ref)

    dosages_ref, _ = reader.load_variants(sample_indices=ref)

    # int64 default-dtype array
    idx_int64 = np.array(base, dtype=np.int64)
    dosages_int64, _ = reader.load_variants(sample_indices=idx_int64)

    # non-contiguous int32 slice (stride 2 picks the same logical indices)
    interleaved = np.empty(len(base) * 2, dtype=np.int32)
    interleaved[0::2] = base
    interleaved[1::2] = -999  # poison the gaps; must never be read
    idx_noncontig = interleaved[0::2]
    assert not idx_noncontig.flags["C_CONTIGUOUS"]
    dosages_noncontig, _ = reader.load_variants(sample_indices=idx_noncontig)

    np.testing.assert_array_equal(dosages_int64, dosages_ref)
    np.testing.assert_array_equal(dosages_noncontig, dosages_ref)


def test_iter_variants_out_of_range_raises(reader):
    """iter_variants shares the same validation contract."""
    n = reader.nsamples
    bad = np.array([0, n + 1], dtype=np.int32)
    with pytest.raises((ValueError, IndexError)):
        list(reader.iter_variants(sample_indices=bad))


def test_iter_variants_int64_matches(reader):
    """int64 sample_indices through iter_variants match int32."""
    base = [0, 3, 9]
    ref = np.array(base, dtype=np.int32)
    rows_ref = [d.copy() for _, d in reader.iter_variants(sample_indices=ref)]
    rows_int64 = [d.copy() for _, d in reader.iter_variants(sample_indices=np.array(base, dtype=np.int64))]
    assert len(rows_ref) == len(rows_int64)
    for a, b in zip(rows_ref, rows_int64):
        np.testing.assert_array_equal(a, b)
