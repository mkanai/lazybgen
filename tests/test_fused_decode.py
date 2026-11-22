"""The fused 8/16-bit full-sample decode, against an independent reference.

Unphased v1.2 8- and 16-bit columns are decoded by kernels that handle many
samples per step (AVX2 on x86 when the CPU has it, NEON on arm64, a branch-free
scalar loop otherwise) and that return the column's min, max and missing-call
flag in the same pass. These tests write BGEN files byte by byte, so every
numerator, invalid probability sums and missing calls can be placed anywhere,
including in the scalar tail after the last full vector, and compare the result
bit for bit with the decode contract computed in NumPy:

    dosage = float32(2*M - 2*P(AA) - P(AB)) / float32(M)    (M = 255 or 65535)

with NaN for a missing call or for P(AA) + P(AB) > M. A float64 column must be
the exact widening of the float32 one, and the range the reader reports must
equal the range of the values it returned.
"""

import numpy as np
import pytest
from _bgen_writer import write_v12
from lazybgen.reader import BgenReader

NAN32 = np.float32(np.nan)


def _reference(columns, bits):
    m = (1 << bits) - 1
    out = []
    for aa, ab, miss in columns:
        aa = np.asarray(aa, dtype=np.int64)
        ab = np.asarray(ab, dtype=np.int64)
        k = 2 * m - 2 * aa - ab
        val = np.float32(k) / np.float32(m)
        bad = np.asarray(miss) | (aa + ab > m)
        out.append(np.where(bad, NAN32, val).astype(np.float32))
    return np.stack(out, axis=1)


def _columns(n, bits, rng):
    """Columns that between them hit every case the kernels distinguish."""
    m = (1 << bits) - 1
    cols = []
    # Uniform over all stored pairs, valid or not, with scattered missing calls.
    aa = rng.integers(0, m + 1, size=n)
    ab = rng.integers(0, m + 1, size=n)
    cols.append((aa, ab, rng.random(n) < 0.05))
    # Valid only: P(AB) drawn within what P(AA) leaves.
    aa = rng.integers(0, m + 1, size=n)
    ab = (rng.random(n) * (m - aa + 1)).astype(np.int64)
    cols.append((aa, ab, np.zeros(n, bool)))
    # The extremes of the range: dosage exactly 0 and exactly 2.
    cols.append((np.full(n, m), np.zeros(n, np.int64), np.zeros(n, bool)))
    cols.append((np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(n, bool)))
    # Nothing valid: all missing, then all invalid sums.
    cols.append((np.zeros(n, np.int64), np.zeros(n, np.int64), np.ones(n, bool)))
    cols.append((np.full(n, m), np.full(n, m), np.zeros(n, bool)))
    # A single valid sample at the very end (the tail), the rest missing.
    miss = np.ones(n, bool)
    miss[-1] = False
    cols.append((np.full(n, 3), np.full(n, 1), miss))
    # A sum exactly at the limit (valid) next to one just past it (invalid).
    aa = np.full(n, m // 2)
    ab = np.where(np.arange(n) % 2 == 0, m - m // 2, m - m // 2 + 1)
    cols.append((aa, ab, np.zeros(n, bool)))
    return cols


def _bits(a):
    a = np.ascontiguousarray(a)
    return a.view(np.uint32 if a.dtype == np.float32 else np.uint64)


def _range(values):
    finite = values[~np.isnan(values)]
    lo = float(finite.min()) if finite.size else float("inf")
    hi = float(finite.max()) if finite.size else float("-inf")
    return (lo, hi, bool(np.isnan(values).any()))


def _check(path, columns, bits, num_threads):
    expected32 = _reference(columns, bits)
    n_variants = expected32.shape[1]
    for dtype in (np.float32, np.float64):
        want = expected32.astype(dtype)
        with BgenReader(str(path), num_threads=num_threads) as reader:
            got, _ = reader.load_variants(dtype=dtype)
            stats = reader.last_dosage_stats
            assert (_bits(got) == _bits(want)).all(), f"values differ ({np.dtype(dtype).name})"
            # The block decode (two or more variants, more than one thread)
            # returns the range with the values; the serial loop returns none.
            if num_threads != 1 and n_variants > 1:
                assert stats == _range(want)
                # Every adjacent pair, so ranges narrower than [0, 2], ones
                # with no missing call, and all-missing (inf, -inf) are seen.
                for j in range(n_variants - 1):
                    reader.load_variants(dtype=dtype, region_chrom="01", region_start=1000 + j, region_end=1001 + j)
                    assert reader.last_dosage_stats == _range(want[:, j : j + 2]), f"pair {j}"
            else:
                assert stats is None


# Column lengths around the 8- and 16-lane vector widths, so the vector loop,
# its tail, and a column shorter than one vector are all exercised.
LENGTHS = [1, 2, 7, 8, 9, 15, 16, 17, 31, 33, 100, 1027]


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("n", LENGTHS)
def test_fused_decode_matches_reference(tmp_path, bits, n):
    rng = np.random.default_rng(n * 31 + bits)
    columns = _columns(n, bits, rng)
    path = tmp_path / f"f{bits}_{n}.bgen"
    write_v12(path, columns, bits)
    for num_threads in (1, 4):
        _check(path, columns, bits, num_threads)


@pytest.mark.parametrize("n_samples", [32896, 15])
def test_every_8bit_numerator(tmp_path, n_samples):
    """All 32,896 valid (P(AA), P(AB)) pairs, so all 511 numerators.

    As one long column the vector loop sees them all; as 15-sample columns
    (shorter than a vector, so decoded entirely by the scalar tail) the lookup
    table does.
    """
    aa, ab = np.meshgrid(np.arange(256), np.arange(256), indexing="ij")
    keep = (aa + ab) <= 255
    aa, ab = aa[keep], ab[keep]
    order = np.random.default_rng(1).permutation(aa.size)
    aa, ab = aa[order], ab[order]
    pad = (-aa.size) % n_samples  # fill the last column with valid zeros
    aa, ab = np.concatenate([aa, np.zeros(pad, int)]), np.concatenate([ab, np.zeros(pad, int)])
    columns = [
        (aa[k : k + n_samples], ab[k : k + n_samples], np.zeros(n_samples, bool)) for k in range(0, aa.size, n_samples)
    ]
    path = tmp_path / "all8.bgen"
    write_v12(path, columns, 8)
    want = _reference(columns, 8)
    for num_threads in (1, 2):
        with BgenReader(str(path), num_threads=num_threads) as reader:
            got, _ = reader.load_variants(dtype=np.float32)
        assert (_bits(got) == _bits(want)).all()
