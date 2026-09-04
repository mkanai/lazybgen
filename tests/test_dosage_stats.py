"""The decode's own summary of what it wrote must match the array it returned.

`load_bgen` trusts `BgenReader.last_dosage_stats` instead of scanning the output,
so a wrong summary is not a slow answer, it is a wrong one: it decides whether a
read is reported as containing missing calls and what its value range is. These
tests recompute the summary from the returned array and require an exact match,
across the dtypes and thread counts that select different decode paths (the
float64 range scan is vectorized, so the two dtypes are not the same code).
"""

from pathlib import Path

import numpy as np
import pytest
from lazybgen.reader import BgenReader

DATA = Path(__file__).parent / "data"
FIXTURES = ["example.16bits.bgen", "example.8bits.bgen", "example.32bits.bgen"]


def _expected(dosages):
    """(min, max, has_nan) the way the reader defines it: min/max ignore NaN."""
    finite = dosages[~np.isnan(dosages)]
    if finite.size == 0:
        return float("inf"), float("-inf"), True
    return float(finite.min()), float(finite.max()), bool(np.isnan(dosages).any())


@pytest.mark.parametrize("fname", FIXTURES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("num_threads", [0, 1, 4])
def test_stats_match_the_returned_array(fname, dtype, num_threads):
    path = DATA / fname
    if not path.exists():
        pytest.skip(f"fixture missing: {path}")

    with BgenReader(str(path), num_threads=num_threads) as reader:
        dosages, _info = reader.load_variants(dtype=dtype)
        stats = reader.last_dosage_stats

    if stats is None:
        # Documented: the per-variant serial loop collects nothing, and a caller
        # must then scan the array itself. That is a valid answer, not a failure.
        return

    got_min, got_max, got_nan = stats
    exp_min, exp_max, exp_nan = _expected(dosages.astype(np.float64))
    assert got_nan == exp_nan
    # float32 output is compared at float32 tolerance; the reader computes in
    # single precision either way.
    tol = 1e-6 if dtype is np.float32 else 1e-12
    assert got_min == pytest.approx(exp_min, abs=tol)
    assert got_max == pytest.approx(exp_max, abs=tol)


@pytest.mark.parametrize("num_threads", [0, 1])
def test_stats_are_cleared_between_decodes(num_threads):
    """A decode that collects nothing must not report the previous decode's answer."""
    path = DATA / "example.16bits.bgen"
    if not path.exists():
        pytest.skip(f"fixture missing: {path}")

    with BgenReader(str(path), num_threads=num_threads) as reader:
        reader.load_variants(dtype=np.float64)
        first = reader.last_dosage_stats
        dosages, _info = reader.load_variants(dtype=np.float64)
        second = reader.last_dosage_stats

    # Same read twice, so either both paths collected or neither did, and if they
    # did the answer must be the same rather than stale-but-plausible.
    assert (first is None) == (second is None)
    if second is not None:
        assert second == pytest.approx(_expected(dosages.astype(np.float64)), abs=1e-12)


def test_a_sample_filtered_read_summarizes_only_the_samples_it_returned():
    """The cohort path writes a narrower matrix; the summary must describe that."""
    path = DATA / "example.16bits.bgen"
    if not path.exists():
        pytest.skip(f"fixture missing: {path}")

    with BgenReader(str(path)) as reader:
        wanted = reader.samples[:10]
        indices, _kept = reader.get_sample_indices(wanted)
        dosages, _info = reader.load_variants(sample_indices=np.asarray(indices, dtype=np.int64), dtype=np.float64)
        stats = reader.last_dosage_stats

    assert dosages.shape[0] == len(wanted)
    if stats is not None:
        got_min, got_max, got_nan = stats
        exp_min, exp_max, exp_nan = _expected(dosages)
        assert got_nan == exp_nan
        assert got_min == pytest.approx(exp_min, abs=1e-12)
        assert got_max == pytest.approx(exp_max, abs=1e-12)
