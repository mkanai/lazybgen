"""Missing-genotype behavior: nan_action paths, decoder equivalence, SIMD decode.

The bundled ``example.{8,16,32}bits`` and ``example.16bits.zstd`` fixtures each
carry exactly two missing calls: sample index 0 is missing at variants 0 and 5
(no all-missing variant). These tests exercise the four nan_action paths against
that real missingness, both full and sample-filtered; pin sequential vs parallel
decoder equivalence including the NaN mask; and guard the SIMD decode paths
against returning a computed dosage for a genotype flagged missing.

This is the real coverage of the error / mean / omit / warn paths on genuinely
missing data (previous tests ran them only on clean fixtures).
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from lazybgen.reader import BgenReader

from lazybgen import load_bgen

DATA_DIR = Path(__file__).parent / "data"

# Fixtures that carry missing genotypes (sample 0 missing at variants 0 and 5).
MISSING_FIXTURES = {
    "8bit": "example.8bits.bgen",
    "16bit": "example.16bits.bgen",
    "32bit": "example.32bits.bgen",
    "zstd": "example.16bits.zstd.bgen",
}

# The known missing cells in every fixture above: (sample_index, variant_index).
MISSING_CELLS = [(0, 0), (0, 5)]
MISSING_SAMPLE = 0
MISSING_VARIANTS = (0, 5)


@pytest.fixture(params=sorted(MISSING_FIXTURES), ids=sorted(MISSING_FIXTURES))
def missing_fixture(request):
    path = DATA_DIR / MISSING_FIXTURES[request.param]
    if not path.exists():
        pytest.skip(f"{request.param} fixture not found")
    return str(path)


def _full_warn(path):
    """Full decode that preserves NaNs (warn), returning (dosages, sample_ids)."""
    dosages, _, sample_ids = load_bgen(file_path=path, nan_action="warn")
    return dosages, sample_ids


# ==================== nan_action == "error" (the default) ====================


def test_error_raises_on_real_missing(missing_fixture):
    """The default 'error' action must raise a clear ValueError when NaNs exist."""
    with pytest.raises(ValueError) as exc:
        load_bgen(file_path=missing_fixture, nan_action="error")
    msg = str(exc.value)
    assert "NaN" in msg
    # The message reports the affected sample/variant counts.
    assert "samples have NaN" in msg
    assert "variants have NaN" in msg


def test_error_is_the_default(missing_fixture):
    """Omitting nan_action behaves identically to nan_action='error'."""
    with pytest.raises(ValueError, match="NaN"):
        load_bgen(file_path=missing_fixture)


def test_error_raises_in_sample_filtered_path(missing_fixture):
    """'error' must also fire when the missing sample is in the requested subset."""
    _, sample_ids = _full_warn(missing_fixture)
    requested = [sample_ids[MISSING_SAMPLE], sample_ids[3], sample_ids[7]]
    with pytest.raises(ValueError, match="NaN"):
        load_bgen(file_path=missing_fixture, sample_ids=requested, nan_action="error")


def test_error_does_not_raise_when_missing_sample_excluded(missing_fixture):
    """If the only missing sample is filtered out, 'error' loads cleanly."""
    _, sample_ids = _full_warn(missing_fixture)
    # Pick samples that are NOT the missing one (sample 0).
    requested = [sample_ids[1], sample_ids[2], sample_ids[3]]
    dosages, _, got = load_bgen(file_path=missing_fixture, sample_ids=requested, nan_action="error")
    assert got == requested
    assert not np.any(np.isnan(dosages))


# ==================== nan_action == "warn" ====================


def test_warn_preserves_nan_at_known_cells(missing_fixture, caplog):
    """'warn' keeps NaNs at the known cells AND logs a warning about them."""
    import logging

    with caplog.at_level(logging.WARNING, logger="lazybgen.missing_data"):
        dosages, _, _ = load_bgen(file_path=missing_fixture, nan_action="warn")

    nan_mask = np.isnan(dosages)
    assert nan_mask.sum() == len(MISSING_CELLS)
    for s, v in MISSING_CELLS:
        assert nan_mask[s, v], f"expected NaN at ({s}, {v})"
    # Finite dosages stay in range.
    finite = dosages[~nan_mask]
    assert np.all((finite >= 0) & (finite <= 2))

    # 'warn' must actually warn (via the logging module, not warnings.warn).
    warn_text = "\n".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
    assert "NaN" in warn_text
    assert "preserved" in warn_text.lower()


# ==================== nan_action == "mean" ====================


def test_mean_imputes_exact_per_variant_finite_mean(missing_fixture):
    """Each imputed cell equals the per-variant mean of the finite values."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warn_dosages, _, _ = load_bgen(file_path=missing_fixture, nan_action="warn")
        mean_dosages, _, _ = load_bgen(file_path=missing_fixture, nan_action="mean")

    # No NaNs remain after imputation.
    assert not np.any(np.isnan(mean_dosages))
    # Same shape; no samples/variants dropped.
    assert mean_dosages.shape == warn_dosages.shape

    nan_mask = np.isnan(warn_dosages)
    # Non-missing cells are untouched.
    np.testing.assert_allclose(mean_dosages[~nan_mask], warn_dosages[~nan_mask], rtol=1e-6, atol=1e-8)

    # Each imputed cell equals the finite per-variant mean computed from warn output.
    for s, v in MISSING_CELLS:
        col = warn_dosages[:, v]
        expected = col[~np.isnan(col)].mean()
        np.testing.assert_allclose(mean_dosages[s, v], expected, rtol=1e-6, atol=1e-8)


def test_mean_imputes_in_sample_filtered_path(missing_fixture):
    """Mean imputation in the filtered path uses the mean over the SELECTED samples.

    handle_nan_values runs after sample filtering, so the per-variant mean is
    taken over the requested rows only (here samples 0, 3, 7), not the full file.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, sample_ids = _full_warn(missing_fixture)
        requested = [sample_ids[0], sample_ids[3], sample_ids[7]]
        warn_subset, _, ids = load_bgen(file_path=missing_fixture, sample_ids=requested, nan_action="warn")
        mean_subset, _, _ = load_bgen(file_path=missing_fixture, sample_ids=requested, nan_action="mean")

    assert ids == requested
    assert not np.any(np.isnan(mean_subset))
    nan_mask = np.isnan(warn_subset)
    # Missing only at the row corresponding to sample 0 (position 0 in the subset).
    assert nan_mask.sum() == len(MISSING_VARIANTS)

    # Non-missing selected cells must be left exactly as decoded (not imputed).
    np.testing.assert_allclose(mean_subset[~nan_mask], warn_subset[~nan_mask], rtol=1e-6, atol=1e-8)

    # Imputed cells equal the finite per-variant mean over the SELECTED rows.
    for v in MISSING_VARIANTS:
        col = warn_subset[:, v]
        expected = col[~np.isnan(col)].mean()
        np.testing.assert_allclose(mean_subset[0, v], expected, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_impute_mean_in_place_matches_reference(dtype):
    """_impute_nan_with_mean imputes in place and matches a per-variant-mean reference.

    Pins the memory-efficient in-place path: an all-NaN column imputes to 0,
    finite cells are untouched, the per-variant mean fills the missing cells, the
    dtype is preserved, and the returned array is the same object (mutated), not a
    fresh allocation.
    """
    from lazybgen.missing_data import _impute_nan_with_mean

    rng = np.random.default_rng(0)
    a = rng.random((40, 6)).astype(dtype) * 2.0  # dosages in [0, 2)
    a[3, 1] = np.nan
    a[10, 1] = np.nan
    a[7, 4] = np.nan
    a[:, 2] = np.nan  # entirely-missing variant -> imputes to 0
    reference = a.copy()
    nan_mask = np.isnan(reference)

    info = pd.DataFrame({"rsid": [f"v{i}" for i in range(a.shape[1])], "pos": list(range(a.shape[1]))})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out, _, _ = _impute_nan_with_mean(a, info, [f"s{i}" for i in range(a.shape[0])])

    assert out is a, "imputation must be in place (same array object)"
    assert out.dtype == dtype
    assert not np.any(np.isnan(out))

    # Finite cells untouched.
    np.testing.assert_array_equal(out[~nan_mask], reference[~nan_mask])

    # Each imputed cell equals the finite per-variant mean (0 for the all-NaN column).
    for v in range(a.shape[1]):
        col = reference[:, v]
        finite = col[~np.isnan(col)]
        expected = finite.mean() if finite.size else 0.0
        rows = np.where(nan_mask[:, v])[0]
        if rows.size:
            np.testing.assert_allclose(out[rows, v], expected, rtol=1e-5, atol=1e-6)


# ==================== nan_action == "omit" ====================


def test_omit_drops_exactly_the_missing_sample(missing_fixture):
    """'omit' removes exactly sample 0 (the only sample with a missing call)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warn_dosages, all_ids = _full_warn(missing_fixture)
        omit_dosages, _, omit_ids = load_bgen(file_path=missing_fixture, nan_action="omit")

    # Exactly one sample (index 0) dropped.
    assert len(omit_ids) == len(all_ids) - 1
    assert all_ids[MISSING_SAMPLE] not in omit_ids
    assert omit_ids == [sid for i, sid in enumerate(all_ids) if i != MISSING_SAMPLE]

    # Remaining rows are the original non-missing rows, unchanged, NaN-free.
    expected = np.delete(warn_dosages, MISSING_SAMPLE, axis=0)
    assert omit_dosages.shape == expected.shape
    assert not np.any(np.isnan(omit_dosages))
    np.testing.assert_array_equal(omit_dosages, expected)


def test_omit_in_sample_filtered_path(missing_fixture):
    """'omit' in the filtered path drops only the requested missing sample.

    The two retained rows must carry the exact dosages of samples 3 and 7 from
    the full decode (value check, not just shape/ids).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full, sample_ids = _full_warn(missing_fixture)
        requested = [sample_ids[0], sample_ids[3], sample_ids[7]]
        omit_subset, _, ids = load_bgen(file_path=missing_fixture, sample_ids=requested, nan_action="omit")

    # Sample 0 carries the missing calls and must be dropped; 3 and 7 remain.
    assert ids == [sample_ids[3], sample_ids[7]]
    assert omit_subset.shape[0] == 2
    assert not np.any(np.isnan(omit_subset))

    # Retained rows equal the full-decode rows for samples 3 and 7.
    expected = full[[3, 7], :]
    np.testing.assert_array_equal(omit_subset, expected)


# ==================== sequential vs parallel equivalence ====================


@pytest.mark.parametrize("num_threads", [2, 4])
def test_sequential_parallel_equivalence_full(missing_fixture, num_threads):
    """Sequential and parallel decoders agree, including the NaN mask (full decode)."""
    with BgenReader(missing_fixture, num_threads=1) as r:
        seq, _ = r.load_variants(dtype=np.float64)
    with BgenReader(missing_fixture, num_threads=num_threads) as r:
        assert r.decompressor_type == "parallel"
        par, _ = r.load_variants(dtype=np.float64)

    assert seq.shape == par.shape
    # The missing fixtures must actually contain NaNs, else this proves nothing.
    assert np.isnan(seq).sum() == len(MISSING_CELLS)
    np.testing.assert_array_equal(np.isnan(seq), np.isnan(par))
    np.testing.assert_allclose(seq[~np.isnan(seq)], par[~np.isnan(par)], rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("num_threads", [2, 4])
def test_sequential_parallel_equivalence_filtered(missing_fixture, num_threads):
    """Sequential and parallel agree on the sample-filtered path, NaN mask included."""
    # sample_indices that include the missing sample (0) and span a SIMD chunk.
    sample_indices = np.arange(12, dtype=np.int32)
    with BgenReader(missing_fixture, num_threads=1) as r:
        seq, _ = r.load_variants(sample_indices=sample_indices, dtype=np.float64)
    with BgenReader(missing_fixture, num_threads=num_threads) as r:
        par, _ = r.load_variants(sample_indices=sample_indices, dtype=np.float64)

    assert seq.shape == par.shape
    assert np.isnan(seq).sum() == len(MISSING_VARIANTS)  # sample 0 missing at 2 variants
    np.testing.assert_array_equal(np.isnan(seq), np.isnan(par))
    np.testing.assert_allclose(seq[~np.isnan(seq)], par[~np.isnan(par)], rtol=1e-6, atol=1e-8)


# ==================== sample_indices NaN-mask vs full decode ====================


@pytest.mark.parametrize("file_key", ["16bit", "32bit", "zstd"])
def test_sample_indices_nan_mask_matches_full(file_key):
    """A known-missing sample inside a full SIMD chunk decodes to NaN when filtered.

    Extends the 8-bit SIMD guard (test_filtered_8bit_missing_in_avx2_chunk_is_nan
    below) to 16/32-bit and zstd.
    The missing sample (index 0) is placed first in a length-16 sample_indices so
    it sits inside the first full SIMD lane chunk; the filtered NaN mask and the
    finite values must match the full decode's corresponding rows exactly.
    """
    path = str(DATA_DIR / MISSING_FIXTURES[file_key])
    if not Path(path).exists():
        pytest.skip(f"{file_key} fixture not found")

    with BgenReader(path) as r:
        full, _ = r.load_variants(dtype=np.float64)

    sample_indices = np.arange(16, dtype=np.int32)  # includes the missing sample 0
    with BgenReader(path) as r:
        filtered, _ = r.load_variants(sample_indices=sample_indices, dtype=np.float64)

    expected = full[sample_indices, :]
    assert np.isnan(expected).sum() == len(MISSING_VARIANTS)  # guard: chunk has a real NaN
    np.testing.assert_array_equal(np.isnan(filtered), np.isnan(expected))
    np.testing.assert_allclose(filtered[~np.isnan(filtered)], expected[~np.isnan(expected)], rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("file_key", ["8bit", "16bit", "32bit", "zstd"])
def test_iter_variants_sample_indices_nan_mask_matches_batch(file_key):
    """iter_variants(sample_indices=...) must reproduce the batch NaN mask too.

    The existing iter_variants sample_indices test compares only finite values;
    this adds the isnan-mask assertion on the missing fixtures.
    """
    path = str(DATA_DIR / MISSING_FIXTURES[file_key])
    if not Path(path).exists():
        pytest.skip(f"{file_key} fixture not found")

    sample_indices = np.arange(16, dtype=np.int32)
    with BgenReader(path) as r:
        batch, _ = r.load_variants(sample_indices=sample_indices, dtype=np.float64)
        rows = list(r.iter_variants(sample_indices=sample_indices, dtype=np.float64))

    streamed = np.column_stack([d for _, d in rows])
    assert streamed.shape == batch.shape
    assert np.isnan(batch).sum() == len(MISSING_VARIANTS)  # guard: real NaNs present
    np.testing.assert_array_equal(np.isnan(streamed), np.isnan(batch))
    np.testing.assert_allclose(streamed[~np.isnan(streamed)], batch[~np.isnan(batch)], rtol=1e-6, atol=1e-8)


# ============ SIMD 8-bit filtered missing-lane guard (targeted) ============


def _find_missing_sample_in_first_chunk(bgen_path):
    """Find a (variant, sample_index) where the genotype is missing (NaN)
    in the full decode and the sample index is small enough to sit inside
    the first full SIMD lane chunk (positions 0..7)."""
    with BgenReader(str(bgen_path)) as reader:
        dosages, _ = reader.load_variants()
        # dosages shape: (n_samples, n_variants)
        for variant in range(dosages.shape[1]):
            rows = np.where(np.isnan(dosages[:, variant]))[0]
            for row in rows:
                if row < 8:
                    return variant, int(row), dosages
    return None, None, None


def test_filtered_8bit_missing_in_avx2_chunk_is_nan():
    """A missing genotype that lands inside the first full 8-lane AVX2 chunk
    of the sample-filtered 8-bit decode must be NaN, not the computed dosage.
    """
    bgen_path = DATA_DIR / "example.8bits.bgen"
    variant, missing_row, full_dosages = _find_missing_sample_in_first_chunk(bgen_path)
    assert (
        variant is not None and full_dosages is not None
    ), "no missing genotype with sample index < 8 found in fixture"

    # Build a sample_indices vector of length >= 8 so the filtered path takes
    # the SIMD branch and the missing sample (placed first) sits inside the
    # first full 8-lane chunk (i+7 < n_indices).
    sample_indices = np.array(list(range(10)), dtype=np.int32)
    assert missing_row in sample_indices

    with BgenReader(str(bgen_path)) as reader:
        filtered, _ = reader.load_variants(sample_indices=sample_indices)

    # Row position in the filtered output equals the position in sample_indices.
    pos = int(np.where(sample_indices == missing_row)[0][0])

    # The bug: AVX2 8-bit filtered path returned 2.0 here instead of NaN.
    assert np.isnan(filtered[pos, variant]), (
        f"missing genotype decoded to {filtered[pos, variant]} instead of NaN "
        "(SIMD filtered missing-lane selection is wrong)"
    )

    # Non-missing rows in the filtered output must match the full decode.
    for i, sample_idx in enumerate(sample_indices):
        if sample_idx == missing_row:
            continue
        np.testing.assert_allclose(
            filtered[i, variant],
            full_dosages[sample_idx, variant],
            rtol=1e-6,
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# Dosage validation: range check and NaN detection share the same scan
# ---------------------------------------------------------------------------
def test_validate_dosages_reports_nan_presence():
    """_validate_dosages answers "in range?" and "any NaN?" from one min/max pair.

    np.min / np.max propagate NaN, so the range scan already reveals whether the
    matrix holds missing data; the caller must not need a separate isnan pass.
    """
    from lazybgen import _validate_dosages

    clean = np.array([[0.0, 1.0], [2.0, 0.5]])
    assert _validate_dosages(clean) is False

    with_nan = clean.copy()
    with_nan[0, 1] = np.nan
    assert _validate_dosages(with_nan) is True

    # An empty result has neither a range nor a NaN to report.
    assert _validate_dosages(np.empty((0, 0))) is False


def test_validate_dosages_rejects_out_of_range():
    """Values outside [0, 2] raise, including alongside NaNs; all-NaN does not."""
    from lazybgen import _validate_dosages

    for bad in (-0.5, 2.5):
        with pytest.raises(ValueError, match="out of valid range"):
            _validate_dosages(np.array([[0.0, bad]]))

    # A NaN must not mask an out-of-range value.
    with pytest.raises(ValueError, match="out of valid range"):
        _validate_dosages(np.array([[np.nan, 2.5]]))

    # All-NaN is missing data, not a range error.
    assert _validate_dosages(np.full((2, 2), np.nan)) is True

    # +/-inf is not NaN, so it flows through the range check and is caught.
    with pytest.raises(ValueError, match="out of valid range"):
        _validate_dosages(np.array([[0.0, np.inf]]))


@pytest.fixture(scope="module")
def test_paths_basic():
    """A fixture with no missing calls."""
    return str(DATA_DIR / "data.bgen")


# ---------------------------------------------------------------------------
# Dosage stats gathered during the parallel decode
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key,fname", sorted(MISSING_FIXTURES.items()))
def test_parallel_decode_reports_dosage_stats(key, fname):
    """The block decode reports the range and NaN presence it wrote.

    These fixtures carry real missing calls, so has_nan must be True and the
    reported min/max must ignore the NaNs, matching nanmin/nanmax exactly.
    """
    path = str(DATA_DIR / fname)
    with BgenReader(path) as reader:
        dosages, _ = reader.load_variants(dtype=np.float64)
        stats = reader.last_dosage_stats

    assert stats is not None
    lo, hi, has_nan = stats
    assert has_nan is True
    assert lo == pytest.approx(np.nanmin(dosages))
    assert hi == pytest.approx(np.nanmax(dosages))


def test_dosage_stats_match_numpy_without_missing_data(test_paths_basic):
    """On a fixture with no missing calls the stats match a plain min/max."""
    with BgenReader(test_paths_basic) as reader:
        dosages, _ = reader.load_variants(dtype=np.float32)
        stats = reader.last_dosage_stats

    assert stats is not None
    lo, hi, has_nan = stats
    assert has_nan == bool(np.isnan(dosages).any())
    assert lo == pytest.approx(np.nanmin(dosages), abs=1e-6)
    assert hi == pytest.approx(np.nanmax(dosages), abs=1e-6)


def test_dosage_stats_cover_sample_filtered_decode():
    """A sample-filtered decode reports stats for the rows it actually wrote."""
    path = str(DATA_DIR / MISSING_FIXTURES["16bit"])
    indices = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    with BgenReader(path) as reader:
        dosages, _ = reader.load_variants(sample_indices=indices, dtype=np.float64)
        stats = reader.last_dosage_stats

    assert stats is not None
    lo, hi, has_nan = stats
    assert has_nan == bool(np.isnan(dosages).any())
    assert lo == pytest.approx(np.nanmin(dosages))
    assert hi == pytest.approx(np.nanmax(dosages))


def test_dosage_stats_absent_for_the_serial_path():
    """The per-variant serial loop gathers no stats, so none are reported.

    load_bgen must fall back to scanning the matrix in that case, so reporting
    None (rather than a stale answer from an earlier load) is what makes the
    fallback safe.
    """
    path = str(DATA_DIR / MISSING_FIXTURES["16bit"])
    with BgenReader(path, num_threads=1) as reader:
        reader.load_variants(dtype=np.float64)
        assert reader.last_dosage_stats is None


def test_dosage_stats_are_reset_between_loads():
    """Stats never leak from one load into the next."""
    path = str(DATA_DIR / MISSING_FIXTURES["16bit"])
    with BgenReader(path) as reader:
        reader.load_variants(dtype=np.float64)
        assert reader.last_dosage_stats is not None
        reader.set_decompressor_type("sequential", 1)
        reader.load_variants(dtype=np.float64)
        assert reader.last_dosage_stats is None
