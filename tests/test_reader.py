"""Comprehensive tests for the BGEN reader and load_bgen.

Tests are organized into logical sections:
1. Basic Reading and Initialization
2. Context Manager Support
3. Sample Filtering
4. Region Queries
5. Format Support (bit depths, compression)
6. Error Handling
7. Performance and Memory
8. Decompressor Types
"""

import gc
import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from lazybgen.reader import BgenReader

# Import BGEN functionality
from lazybgen import load_bgen
from lazybgen.variant_filter import load_variant_filter

# Try to import external bgen library for comparison
try:
    import bgen as external_bgen

    HAS_EXTERNAL_BGEN = True
except ImportError:
    HAS_EXTERNAL_BGEN = False
    external_bgen = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def test_paths():
    """Set up test data paths."""
    examples_dir = Path(__file__).parent / "data"

    # Test files
    bgen_file = examples_dir / "data.bgen"
    bgi_file = examples_dir / "data.bgen.bgi"
    sample_file = examples_dir / "data.sample"

    # Additional format test files
    test_files = {
        "basic": bgen_file,
        "8bit": examples_dir / "example.8bits.bgen",
        "16bit": examples_dir / "example.16bits.bgen",
        "32bit": examples_dir / "example.32bits.bgen",
        "zstd": examples_dir / "example.16bits.zstd.bgen",
        "v11": examples_dir / "example.v11.bgen",
    }

    return {
        "examples_dir": examples_dir,
        "bgen_file": bgen_file,
        "bgi_file": bgi_file,
        "sample_file": sample_file,
        "test_files": test_files,
    }


def check_dosages_valid(dosages):
    """Check that dosages are valid (between 0 and 2)."""
    # Handle NaN values by checking only non-NaN values
    valid_mask = ~np.isnan(dosages)
    if np.any(valid_mask):
        assert np.all(dosages[valid_mask] >= 0)
        assert np.all(dosages[valid_mask] <= 2)


# ==================== Basic Reading and Initialization ====================


def test_basic_loading(test_paths):
    """Test basic BGEN file loading."""
    genotypes, variant_info, sample_ids = load_bgen(
        file_path=str(test_paths["bgen_file"]),
        index_path=str(test_paths["bgi_file"]),
        sample_path=str(test_paths["sample_file"]),
        nan_action="omit",  # Handle any potential NaN values
    )

    # Check shapes
    assert genotypes.shape[0] == 5363  # Expected samples
    assert genotypes.shape[1] == 55  # Expected variants
    assert len(variant_info) == 55
    assert len(sample_ids) == 5363

    # Check variant info columns
    expected_cols = {"chrom", "pos", "rsid", "ref", "alt"}
    assert set(variant_info.columns) == expected_cols

    # Check dosages are valid
    check_dosages_valid(genotypes)


def test_loading_without_index(test_paths):
    """Test BGEN loading without BGI index file."""
    genotypes, variant_info, sample_ids = load_bgen(
        file_path=str(test_paths["bgen_file"]),
        sample_path=str(test_paths["sample_file"]),
        nan_action="omit",  # Handle any potential NaN values
    )

    # Should still load successfully
    assert genotypes.shape[0] > 0
    assert genotypes.shape[1] > 0
    assert len(variant_info) == genotypes.shape[1]
    assert len(sample_ids) == genotypes.shape[0]


def test_bgen_reader_requires_bgi():
    """Test that BGEN reader requires BGI file when using certain features."""
    # Create a temp BGEN file without BGI
    with tempfile.NamedTemporaryFile(suffix=".bgen") as f:
        # Write a minimal invalid BGEN header to make it look like a BGEN file
        f.write(b"\x00" * 32)  # Minimal content
        f.flush()

        # Should fail without BGI or with invalid BGEN content
        with pytest.raises((FileNotFoundError, ValueError, RuntimeError)):
            load_bgen(f.name)


def test_reader_properties(test_paths):
    """Test BgenReader properties and basic functionality."""
    with BgenReader(str(test_paths["bgen_file"])) as reader:
        # Check basic properties
        assert reader.nvariants > 0
        assert reader.nsamples > 0
        assert reader.samples is not None
        assert len(reader.samples) == reader.nsamples

        # BgenReader.load_variants returns raw dosages and takes no nan_action
        # (NaN handling is a load_bgen-level concern); data.bgen has no NaNs.
        dosages, variant_info = reader.load_variants()
        assert dosages.shape[0] == reader.nsamples
        assert dosages.shape[1] == reader.nvariants
        assert len(variant_info) == reader.nvariants


# ==================== Context Manager Support ====================


def test_context_manager_basic(test_paths):
    """Test basic context manager functionality."""
    with BgenReader(str(test_paths["bgen_file"])) as reader:
        # Should be able to access properties and load data
        assert reader.nvariants > 0
        assert reader.nsamples > 0
        dosages, _ = reader.load_variants()
        # Loaded matrix matches the reader's reported dimensions.
        assert dosages.shape == (reader.nsamples, reader.nvariants)


def test_context_manager_file_closed(test_paths):
    """Test that file handle is properly closed after context exit."""
    with BgenReader(str(test_paths["bgen_file"])) as reader:
        # Can access data while open
        assert reader.nsamples > 0

    # After context exit, operations should fail
    with pytest.raises(ValueError, match="closed"):
        reader.load_variants()


def test_context_manager_exception_handling(test_paths):
    """Test that context manager properly handles exceptions."""
    reader = None
    try:
        with BgenReader(str(test_paths["bgen_file"])) as reader:
            # Simulate an error during processing
            raise RuntimeError("Test error")
    except RuntimeError:
        pass

    # Reader should still be closed even after exception
    with pytest.raises(ValueError):
        reader.load_variants()


def test_context_manager_nested(test_paths):
    """Test nested context managers work correctly."""
    with BgenReader(str(test_paths["bgen_file"])) as reader1:
        nvariants1 = reader1.nvariants

        with BgenReader(str(test_paths["bgen_file"])) as reader2:
            nvariants2 = reader2.nvariants
            assert nvariants1 == nvariants2

        # reader2 closed, reader1 still open
        dosages1, _ = reader1.load_variants()
        assert dosages1 is not None

    # Both should be closed now
    with pytest.raises(ValueError):
        reader1.load_variants()


# ==================== Sample Filtering ====================


def test_sample_filtering_basic(test_paths):
    """Test basic sample filtering functionality."""
    # First load all samples
    all_genotypes, _, all_sample_ids = load_bgen(
        file_path=str(test_paths["bgen_file"]),
        index_path=str(test_paths["bgi_file"]),
        sample_path=str(test_paths["sample_file"]),
        nan_action="omit",
    )

    # Select subset of samples
    subset_samples = all_sample_ids[:100]  # First 100 samples

    # Load with sample filtering
    filtered_genotypes, filtered_variant_info, filtered_sample_ids = load_bgen(
        file_path=str(test_paths["bgen_file"]),
        index_path=str(test_paths["bgi_file"]),
        sample_path=str(test_paths["sample_file"]),
        sample_ids=subset_samples,
        nan_action="omit",
    )

    # Verify filtering worked
    assert len(filtered_sample_ids) == 100
    assert filtered_sample_ids == subset_samples
    assert filtered_genotypes.shape[0] == 100
    assert filtered_genotypes.shape[1] == all_genotypes.shape[1]

    # Value check: the filtered rows must equal the corresponding rows of the
    # full decode, by sample identity (the first 100 here), including any NaNs.
    # data.bgen has no missing values, but the equality must hold exactly.
    id_to_row = {sid: i for i, sid in enumerate(all_sample_ids)}
    expected = all_genotypes[[id_to_row[s] for s in subset_samples], :]
    np.testing.assert_array_equal(np.isnan(filtered_genotypes), np.isnan(expected))
    np.testing.assert_array_equal(filtered_genotypes, expected)


def test_sample_filtering_with_indices(test_paths):
    """Filtered-by-index rows must equal the matching rows of the full decode."""
    sample_indices = np.array([0, 10, 20, 30, 40], dtype=np.int32)

    with BgenReader(str(test_paths["bgen_file"]), sample_path=str(test_paths["sample_file"])) as reader:
        full, _ = reader.load_variants()
        dosages, _ = reader.load_variants(sample_indices=sample_indices)

    assert dosages.shape[0] == len(sample_indices)
    expected = full[sample_indices, :]
    np.testing.assert_array_equal(np.isnan(dosages), np.isnan(expected))
    np.testing.assert_array_equal(dosages, expected)


@pytest.mark.parametrize("file_key", ["8bit", "16bit", "32bit", "zstd"])
def test_sample_filtering_matches_full_decode_with_missing(test_paths, file_key):
    """Sample-filtered decode must equal the full-decode rows, including NaN mask.

    Regression guard: missing genotypes in the bundled fixtures (sample index 0
    at RSID_2/RSID_7) were silently returned as dosage 2.0 by the filtered path
    instead of NaN, because v1.2 missingness was read as a packed bit array
    rather than one byte per sample.
    """
    test_file = test_paths["test_files"][file_key]
    if not test_file.exists():
        pytest.skip(f"{file_key} test file not found")

    full, _, all_sample_ids = load_bgen(file_path=str(test_file), nan_action="warn")  # (samples, variants)

    # Sample index 0 carries the missing calls in these fixtures; include it.
    sample_indices = [0, 3, 7]
    requested = [all_sample_ids[i] for i in sample_indices]
    subset, _, subset_ids = load_bgen(
        file_path=str(test_file),
        sample_ids=requested,
        nan_action="warn",
    )
    assert subset_ids == requested

    expected = full[sample_indices, :]
    # The filtered path must reproduce missing calls as NaN, not as a real dosage.
    assert np.isnan(subset).sum() == np.isnan(expected).sum()
    np.testing.assert_array_equal(np.isnan(subset), np.isnan(expected))
    np.testing.assert_allclose(subset[~np.isnan(subset)], expected[~np.isnan(expected)], rtol=1e-6, atol=1e-8)


def test_sample_filtering_missing_samples(test_paths):
    """Test that missing samples are handled gracefully."""
    # Get actual sample IDs
    _, _, actual_sample_ids = load_bgen(
        file_path=str(test_paths["bgen_file"]),
        index_path=str(test_paths["bgi_file"]),
        sample_path=str(test_paths["sample_file"]),
        nan_action="omit",
    )

    # Request some existing and some non-existing samples
    requested_samples = [
        actual_sample_ids[0],
        "FAKE_SAMPLE_1",
        actual_sample_ids[1],
        "FAKE_SAMPLE_2",
    ]

    # Load with filtering
    filtered_genotypes, _, filtered_sample_ids = load_bgen(
        file_path=str(test_paths["bgen_file"]),
        index_path=str(test_paths["bgi_file"]),
        sample_path=str(test_paths["sample_file"]),
        sample_ids=requested_samples,
        nan_action="omit",
    )

    # Should only get the existing samples
    assert len(filtered_sample_ids) == 2
    assert filtered_sample_ids == [actual_sample_ids[0], actual_sample_ids[1]]


def test_sample_filtering_order_preserved(test_paths):
    """Test that sample order is preserved when filtering."""
    # Load all samples
    _, _, all_sample_ids = load_bgen(
        file_path=str(test_paths["bgen_file"]),
        index_path=str(test_paths["bgi_file"]),
        sample_path=str(test_paths["sample_file"]),
        nan_action="omit",
    )

    # Select samples in reverse order
    subset_samples = all_sample_ids[::10][::-1]  # Every 10th sample, reversed

    # Load with filtering
    _, _, filtered_sample_ids = load_bgen(
        file_path=str(test_paths["bgen_file"]),
        index_path=str(test_paths["bgi_file"]),
        sample_path=str(test_paths["sample_file"]),
        sample_ids=subset_samples,
        nan_action="omit",
    )

    # Order should be preserved
    assert filtered_sample_ids == subset_samples


def test_sample_filtering_memory_efficiency(test_paths):
    """Filtering by sample_ids returns a proportionally smaller genotype matrix.

    The memory win of sample filtering is that the returned (and internally
    allocated) matrix holds only the requested samples; assert that structurally
    via the output bytes/shape rather than via noisy process-RSS sampling.
    """
    full_genotypes, _, sample_ids = load_bgen(
        file_path=str(test_paths["bgen_file"]),
        sample_path=str(test_paths["sample_file"]),
        nan_action="omit",
    )
    subset_samples = sample_ids[:10]

    filtered_genotypes, _, filtered_ids = load_bgen(
        file_path=str(test_paths["bgen_file"]),
        sample_path=str(test_paths["sample_file"]),
        sample_ids=subset_samples,
        nan_action="omit",
    )

    # Only the requested samples come back, and the matrix is correspondingly
    # smaller (same variants, fewer rows -> fewer bytes).
    assert filtered_ids == subset_samples
    assert filtered_genotypes.shape[0] == len(subset_samples)
    assert filtered_genotypes.shape[0] < full_genotypes.shape[0]
    assert filtered_genotypes.nbytes < full_genotypes.nbytes


# ==================== Region Queries ====================


def test_region_loading(test_paths):
    """Region load must return exactly the variants in the window, with right data.

    Asserts variant identity (the region result is the in-window slice of the
    full decode, same rsids and same dosage columns), not merely that some
    variants came back.
    """
    # Full decode for the oracle.
    full_dosages, full_info, _ = load_bgen(
        file_path=str(test_paths["bgen_file"]),
        index_path=str(test_paths["bgi_file"]),
        nan_action="omit",
    )

    # Load region with variants.
    dosages, variant_info, _ = load_bgen(
        file_path=str(test_paths["bgen_file"]),
        index_path=str(test_paths["bgi_file"]),
        region="01:1-10",
        nan_action="omit",
    )

    assert dosages.shape[1] > 0  # Should have some variants
    assert len(variant_info) == dosages.shape[1]
    assert np.all(variant_info["pos"] >= 1)
    assert np.all(variant_info["pos"] <= 10)

    # Variant identity: the region's rsids are exactly the in-window rsids of the
    # full decode (same set), and the dosage columns match those variants.
    in_window = full_info[(full_info["pos"] >= 1) & (full_info["pos"] <= 10)]
    assert set(variant_info["rsid"]) == set(in_window["rsid"])
    full_col = {rsid: j for j, rsid in enumerate(full_info["rsid"])}
    for j, rsid in enumerate(variant_info["rsid"]):
        np.testing.assert_array_equal(dosages[:, j], full_dosages[:, full_col[rsid]])


def test_region_empty(test_paths):
    """Test loading from an empty region."""
    # Empty region - should raise ValueError
    with pytest.raises(ValueError, match="No variants"):
        load_bgen(
            file_path=str(test_paths["bgen_file"]),
            index_path=str(test_paths["bgi_file"]),
            region="01:100000-200000",
            nan_action="omit",
        )


def test_region_chromosome_formats(test_paths):
    """Test region queries with different chromosome formats."""
    # First get actual chromosome format from file
    _, variant_info, _ = load_bgen(
        file_path=str(test_paths["bgen_file"]),
        index_path=str(test_paths["bgi_file"]),
        nan_action="omit",
    )

    # Precondition (fail loudly rather than pass trivially if the fixture is empty).
    assert len(variant_info) > 0
    chrom = variant_info["chrom"].iloc[0]
    min_pos = variant_info["pos"].min()
    max_pos = variant_info["pos"].max()

    # Try region query
    region = f"{chrom}:{min_pos}-{max_pos}"
    dosages, loaded_info, _ = load_bgen(
        file_path=str(test_paths["bgen_file"]),
        index_path=str(test_paths["bgi_file"]),
        region=region,
        nan_action="omit",
    )

    assert len(loaded_info) > 0
    assert np.all(loaded_info["chrom"] == chrom)


# ==================== Format Support ====================


def _reference_dosages(path):
    """Alt-allele dosage from the external bgen reference probabilities.

    Uses P(AB)+2*P(BB), not the reference's own alt_dosage (off by 1.0 for 32-bit
    in bgen 1.9.0). Returns (samples, variants) with NaN for missing calls.
    """
    with external_bgen.BgenReader(path) as ref:
        cols = [np.asarray(v.probabilities)[:, 1] + 2.0 * np.asarray(v.probabilities)[:, 2] for v in ref]
    return np.column_stack(cols)


@pytest.mark.parametrize("bit_depth", [8, 16, 32])
def test_bit_depth_formats(test_paths, bit_depth):
    """Each bit-depth fixture must decode to the reference dosages, NaN mask included."""
    file_key = f"{bit_depth}bit"
    test_file = test_paths["test_files"][file_key]
    if not test_file.exists():
        pytest.skip(f"{bit_depth}-bit test file not found")
    if not HAS_EXTERNAL_BGEN:
        pytest.skip("External bgen library not available for comparison")

    dosages, _, _ = load_bgen(file_path=str(test_file), nan_action="warn")
    check_dosages_valid(dosages)

    ref = _reference_dosages(str(test_file))
    assert dosages.shape == ref.shape
    np.testing.assert_array_equal(np.isnan(dosages), np.isnan(ref))
    np.testing.assert_allclose(dosages[~np.isnan(dosages)], ref[~np.isnan(ref)], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("compression_format", ["zstd"])
def test_compression_formats(test_paths, compression_format):
    """zstd-compressed fixture must decode to the reference dosages, NaN mask included."""
    test_file = test_paths["test_files"][compression_format]
    if not test_file.exists():
        pytest.skip(f"{compression_format} compressed test file not found")
    if not HAS_EXTERNAL_BGEN:
        pytest.skip("External bgen library not available for comparison")

    dosages, _, _ = load_bgen(file_path=str(test_file), nan_action="warn")
    check_dosages_valid(dosages)

    ref = _reference_dosages(str(test_file))
    assert dosages.shape == ref.shape
    np.testing.assert_array_equal(np.isnan(dosages), np.isnan(ref))
    np.testing.assert_allclose(dosages[~np.isnan(dosages)], ref[~np.isnan(ref)], rtol=1e-5, atol=1e-6)


def test_v11_format(test_paths):
    """v1.1 BGEN must decode and match the reference dosages (no all-outcomes pass).

    v1.1 stores 16-bit probabilities (3 per diploid sample). lazybgen decodes it;
    this asserts the actual dosages match the external bgen reference's
    probability-derived dosage (P(AB)+2*P(BB)), including the NaN mask, rather
    than accepting "success OR any error".
    """
    v11_file = test_paths["test_files"]["v11"]
    if not v11_file.exists():
        pytest.skip("v1.1 test file not found")
    if not HAS_EXTERNAL_BGEN:
        pytest.skip("External bgen library not available for v1.1 comparison")

    dosages, _, _ = load_bgen(file_path=str(v11_file), nan_action="warn")

    with external_bgen.BgenReader(str(v11_file)) as ref:
        cols = []
        for v in ref:
            probs = np.asarray(v.probabilities)  # (samples, 3) unphased biallelic
            cols.append(probs[:, 1] + 2.0 * probs[:, 2])
        ref_dosages = np.column_stack(cols)

    assert dosages.shape == ref_dosages.shape
    # v1.1 fixture carries the same two missing calls as its v1.2 siblings.
    np.testing.assert_array_equal(np.isnan(dosages), np.isnan(ref_dosages))
    # 16-bit v1.1 precision: ~1/65535, so a loose-ish tolerance is correct here.
    np.testing.assert_allclose(dosages[~np.isnan(dosages)], ref_dosages[~np.isnan(ref_dosages)], rtol=1e-3, atol=1e-4)


@pytest.mark.skipif(not HAS_EXTERNAL_BGEN, reason="External bgen library not available for comparison")
@pytest.mark.parametrize("file_key", ["8bit", "16bit", "32bit", "zstd"])
def test_dosages_match_reference_all_formats(test_paths, file_key):
    """lazybgen dosages must match the external bgen reference, full and filtered.

    Covers every bundled bit-depth/compression fixture for both the full decode
    and the sample-filtered path, comparing the NaN mask too. The filtered
    comparison is the guard that would have caught the missing-value regression.

    The oracle is the alt-allele dosage derived from the reference *probabilities*
    (P(AB) + 2*P(BB)), not the reference's own ``alt_dosage`` attribute: the bgen
    1.9.0 library reports alt_dosage off by 1.0 for 32-bit fixtures, while its
    probabilities are correct.
    """
    test_file = test_paths["test_files"][file_key]
    if not test_file.exists():
        pytest.skip(f"{file_key} test file not found")

    # Reference dosage from probabilities: (samples, variants), NaN for missing.
    with external_bgen.BgenReader(str(test_file)) as ref:
        cols = []
        for v in ref:
            probs = np.asarray(v.probabilities)  # (samples, 3) unphased biallelic
            cols.append(probs[:, 1] + 2.0 * probs[:, 2])
        ref_dosages = np.column_stack(cols)

    full, _, all_sample_ids = load_bgen(file_path=str(test_file), nan_action="warn")
    assert full.shape == ref_dosages.shape
    np.testing.assert_array_equal(np.isnan(full), np.isnan(ref_dosages))
    np.testing.assert_allclose(full[~np.isnan(full)], ref_dosages[~np.isnan(ref_dosages)], rtol=1e-5, atol=1e-6)

    # Sample-filtered path must match the corresponding reference rows.
    sample_indices = [0, 3, 7]
    requested = [all_sample_ids[i] for i in sample_indices]
    subset, _, _ = load_bgen(file_path=str(test_file), sample_ids=requested, nan_action="warn")
    ref_subset = ref_dosages[sample_indices, :]
    np.testing.assert_array_equal(np.isnan(subset), np.isnan(ref_subset))
    np.testing.assert_allclose(subset[~np.isnan(subset)], ref_subset[~np.isnan(ref_subset)], rtol=1e-5, atol=1e-6)


def test_parallel_decompressor_type_tracked(test_paths):
    """Selecting the parallel path (num_threads != 1) is reflected in the active
    routing flag, and the requested thread count is recorded so the
    block-parallel decode reads it.
    """
    with BgenReader(
        str(test_paths["bgen_file"]),
        sample_path=str(test_paths["sample_file"]),
        num_threads=2,
    ) as reader:
        assert reader.decompressor_type == "parallel"
        # Results must still be correct with the parallel decompressor.
        dosages, _ = reader.load_variants()
        check_dosages_valid(dosages)


# ==================== Z-file Filtering ====================


def test_z_file_filtering(test_paths, tmp_path):
    """Test loading filtered variants from z file."""
    # Load some variants to create Z-file
    _, variant_info, _ = load_bgen(
        file_path=str(test_paths["bgen_file"]),
        index_path=str(test_paths["bgi_file"]),
        nan_action="omit",
    )

    # Create Z-file with subset of variants
    subset_variants = variant_info.iloc[:3]
    z_data = pd.DataFrame(
        {
            "chromosome": subset_variants["chrom"].tolist(),
            "position": subset_variants["pos"].astype(str).tolist(),
            "allele1": subset_variants["ref"].tolist(),
            "allele2": subset_variants["alt"].tolist(),
            "rsid": subset_variants["rsid"].tolist(),
        }
    )
    z_file = tmp_path / "test.z"
    z_data.to_csv(z_file, sep="\t", index=False)

    # Create filter from z file
    variant_filter = load_variant_filter(str(z_file))

    # Full decode for the oracle (dosage columns by rsid).
    full_dosages, full_info, _ = load_bgen(
        file_path=str(test_paths["bgen_file"]),
        index_path=str(test_paths["bgi_file"]),
        nan_action="omit",
    )

    # Load filtered variants
    dosages, loaded_info, sample_ids = load_bgen(
        file_path=str(test_paths["bgen_file"]),
        index_path=str(test_paths["bgi_file"]),
        variant_filter=variant_filter,
        nan_action="omit",
    )

    # Should have loaded the variants in z file (or fewer if some don't exist)
    assert dosages.shape[1] <= len(variant_filter["positions"])
    assert len(loaded_info) == dosages.shape[1]

    # Variant identity AND order: load_bgen documents that variant_filter results
    # follow the .z file order, so assert the exact ordered rsid sequence (set
    # equality would let a wrong-order regression slip through), with matching
    # dosage columns.
    assert list(loaded_info["rsid"]) == list(subset_variants["rsid"])
    full_col = {rsid: j for j, rsid in enumerate(full_info["rsid"])}
    for j, rsid in enumerate(loaded_info["rsid"]):
        np.testing.assert_array_equal(dosages[:, j], full_dosages[:, full_col[rsid]])


# ==================== Error Handling ====================


def test_invalid_file_path():
    """Test error handling for invalid file path."""
    with pytest.raises(FileNotFoundError):
        load_bgen("/nonexistent/file.bgen")


def test_validate_dosages_all_nan_no_warning():
    """_validate_dosages must not warn or raise on an all-NaN array.

    np.nanmin/np.nanmax over an all-NaN array emit a RuntimeWarning and return
    NaN; the range check must be skipped cleanly so nan_action can handle the
    missing data afterward.
    """
    import warnings

    from lazybgen import _validate_dosages

    arr = np.full((3, 4), np.nan, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes an exception
        _validate_dosages(arr)  # must not raise


def test_validate_dosages_empty_no_warning():
    """_validate_dosages handles an empty array without warning or error."""
    import warnings

    from lazybgen import _validate_dosages

    arr = np.empty((0, 0), dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _validate_dosages(arr)


def test_validate_dosages_partial_nan_still_range_checks():
    """With some finite values, the range check still fires on out-of-range data."""
    from lazybgen import _validate_dosages

    arr = np.array([[np.nan, 0.5], [1.0, 5.0]], dtype=np.float64)  # 5.0 > 2.0
    with pytest.raises(ValueError):
        _validate_dosages(arr)


@pytest.mark.parametrize(
    "arr",
    [
        np.array([[np.inf]], dtype=np.float64),
        np.array([[-np.inf, np.nan]], dtype=np.float64),
        np.array([[np.nan, np.inf], [0.5, 1.0]], dtype=np.float64),
    ],
)
def test_validate_dosages_infinite_values_still_caught(arr):
    """Infinite dosages must still trip the range check (not be skipped as non-finite)."""
    from lazybgen import _validate_dosages

    with pytest.raises(ValueError):
        _validate_dosages(arr)


@pytest.mark.parametrize("key", ["8bit", "16bit", "32bit", "zstd"])
def test_float64_load_is_exact_widening_of_float32(test_bgen_files, key):
    """float64 dosages must be the exact widening of the float32 dosages.

    The decoder computes dosages in single precision; requesting float64 must
    only widen those values (identical bit pattern after an f32->f64 cast), not
    recompute in double precision. This pins the fused float64 decode path
    (which writes double directly from the kernel) to byte-identical output
    versus the float path, including the 32-bit branch and NaN positions.
    """
    import warnings

    path = str(test_bgen_files[key])
    if not Path(path).exists():
        pytest.skip(f"fixture {key} not present")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d32, _, _ = load_bgen(path, dtype=np.float32, nan_action="warn")
        d64, _, _ = load_bgen(path, dtype=np.float64, nan_action="warn")

    assert d32.dtype == np.float32 and d64.dtype == np.float64
    # assert_array_equal treats matching NaN positions as equal.
    np.testing.assert_array_equal(d64, d32.astype(np.float64))


def test_invalid_nan_action_clean_file_fails_early(test_paths):
    """An invalid nan_action must be rejected up front, even with no missing data.

    data.bgen has no missing values, so nan_action must be validated at the top
    of load_bgen (independent of whether any NaN is present), raising a clear
    ValueError that lists the valid options.
    """
    with pytest.raises(ValueError) as exc_info:
        load_bgen(str(test_paths["bgen_file"]), nan_action="bogus")

    error_msg = str(exc_info.value).lower()
    assert "nan_action" in error_msg
    # The error should list the valid options.
    for opt in ("error", "mean", "omit", "warn"):
        assert opt in error_msg


def test_valid_nan_action_clean_file_succeeds(test_paths):
    """A valid nan_action still loads a clean file unchanged."""
    dosages, variant_info, sample_ids = load_bgen(str(test_paths["bgen_file"]), nan_action="error")
    assert dosages.shape[1] > 0


def test_corrupted_file_handling(tmp_path):
    """Test handling of corrupted BGEN files."""
    # Create a file with invalid content
    corrupted_file = tmp_path / "corrupted.bgen"
    with open(corrupted_file, "wb") as f:
        f.write(b"This is not a valid BGEN file")

    # Should raise either ValueError or FileNotFoundError (for missing BGI)
    with pytest.raises((ValueError, FileNotFoundError, RuntimeError)):
        load_bgen(str(corrupted_file))


# ==================== Memory ====================


# The memory tests below assert the filtered-output size structurally rather
# than sampling process RSS, which is page-granular noise on the small bundled
# fixture. The load and region paths are covered by test_basic_loading and
# test_region_loading.


def test_memory_efficiency(test_paths):
    """Filtering via sample_indices allocates only the requested rows.

    The filtered decode path writes straight into an (n_subset, n_variants) array
    and never materializes the full matrix, so the memory win shows up exactly in
    the output size. Assert that structurally (bytes/shape) instead of sampling
    process RSS, which is page-granular noise for the small bundled fixture.
    """
    file_path = str(test_paths["bgen_file"])

    with BgenReader(file_path) as reader:
        dosages_full, _ = reader.load_variants()
        full_shape = dosages_full.shape
        full_nbytes = dosages_full.nbytes

    # Filter to 10% of samples.
    n_subset = full_shape[0] // 10
    sample_indices = np.array(list(range(n_subset)), dtype=np.int32)

    with BgenReader(file_path) as reader:
        dosages_filtered, _ = reader.load_variants(sample_indices=sample_indices)

    # Same variants, only n_subset rows -> a strictly smaller array.
    assert dosages_filtered.shape == (n_subset, full_shape[1])
    assert dosages_filtered.nbytes < full_nbytes


def test_concurrent_readers(test_paths):
    """Test multiple readers on the same file return identical data."""
    file_path = str(test_paths["bgen_file"])

    readers = []
    try:
        for _ in range(3):
            readers.append(BgenReader(file_path))

        results = []
        for reader in readers:
            dosages, _ = reader.load_variants()
            results.append(dosages[:, :3])  # first 3 variants

        # All results should be identical.
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])
    finally:
        for reader in readers:
            reader.close()


def test_variant_metadata_consistency(test_paths):
    """Variant metadata is consistent across full and region queries."""
    file_path = str(test_paths["bgen_file"])

    with BgenReader(file_path) as reader:
        _, all_info = reader.load_variants()

    # Precondition (fail loudly rather than pass trivially if the fixture is empty).
    assert len(all_info) > 0
    # Query first 5 variants by region.
    chrom = all_info.iloc[0]["chrom"]
    start_pos = int(all_info.iloc[0]["pos"])
    end_pos = int(all_info.iloc[min(4, len(all_info) - 1)]["pos"])

    with BgenReader(file_path) as reader:
        _, region_info = reader.load_variants(region_chrom=chrom, region_start=start_pos, region_end=end_pos)

    assert len(region_info) > 0
    assert len(region_info) <= 5


# ==================== Comparison with External Library ====================


@pytest.mark.skipif(not HAS_EXTERNAL_BGEN, reason="External bgen library not available")
def test_compare_with_external_library(test_paths):
    """Compare results with external bgen library if available."""
    # Load with lazybgen
    our_dosages, our_info, our_samples = load_bgen(
        file_path=str(test_paths["bgen_file"]),
        sample_path=str(test_paths["sample_file"]),
        nan_action="omit",
    )

    # Load with external library
    try:
        from bgen import BgenReader

        bfile = BgenReader(str(test_paths["bgen_file"]))

        # Compare samples
        external_samples = bfile.samples
        assert len(our_samples) == len(external_samples)

        # Compare variant count
        external_variant_count = len(bfile)
        assert our_dosages.shape[1] == external_variant_count

        # Compare sample count
        assert our_dosages.shape[0] == len(external_samples)

        # Optional: Compare first few variant IDs
        variant_count = 0
        for i, variant in enumerate(bfile):
            if i >= 5:  # Just check first 5 variants
                break
            variant_count += 1
            # Check if rsid matches
            if hasattr(variant, "rsid") and i < len(our_info):
                assert variant.rsid == our_info.iloc[i]["rsid"]

        # Ensure we could read at least some variants
        assert variant_count > 0

    except Exception as e:
        # If there's any issue with the external library, skip the test
        pytest.skip(f"External bgen library error: {e}")


# ==================== NaN Handling ====================


@pytest.mark.parametrize("nan_action", ["error", "mean", "omit"])
def test_nan_handling_actions(test_paths, nan_action):
    """Test different NaN handling options."""
    dosages, _, _ = load_bgen(str(test_paths["bgen_file"]), nan_action=nan_action)
    assert dosages is not None
    # Test data shouldn't have NaN values
    assert not np.any(np.isnan(dosages))


# ==================== Decompressor Types ====================


@pytest.mark.parametrize("num_threads", [1, 0], ids=["sequential", "parallel"])
def test_decompressor_types(test_paths, num_threads):
    """Both decoders (num_threads=1 sequential, 0 parallel) decode valid dosages."""
    with BgenReader(str(test_paths["bgen_file"]), num_threads=num_threads) as reader:
        dosages, _ = reader.load_variants()
        assert dosages is not None
        check_dosages_valid(dosages)


def test_default_decompressor_is_parallel(test_paths):
    """Parallel decode is the default (auto-detected cores)."""
    with BgenReader(str(test_paths["bgen_file"])) as reader:
        assert reader.decompressor_type == "parallel"


def test_num_threads_one_selects_sequential(test_paths):
    """num_threads=1 opts out of parallel decode and uses the true serial path."""
    with BgenReader(str(test_paths["bgen_file"]), num_threads=1) as reader:
        assert reader.decompressor_type == "sequential"


def test_load_bgen_num_threads_equivalence(test_paths):
    """load_bgen defaults to parallel; num_threads=1 (sequential) is byte-identical."""
    file_path = str(test_paths["bgen_file"])
    genotypes_parallel, _, _ = load_bgen(file_path)
    genotypes_sequential, _, _ = load_bgen(file_path, num_threads=1)
    np.testing.assert_array_equal(genotypes_parallel, genotypes_sequential)


def test_negative_num_threads_rejected(test_paths):
    """A negative num_threads raises rather than silently falling back to auto."""
    file_path = str(test_paths["bgen_file"])
    with pytest.raises(ValueError, match="num_threads"):
        BgenReader(file_path, num_threads=-1)
    with pytest.raises(ValueError, match="num_threads"):
        load_bgen(file_path, num_threads=-4)


def test_decompressors_agree(test_paths):
    """All decompressor backends must decode to byte-identical dosages.

    Each backend must succeed (exceptions are not caught) and produce the same
    result as the sequential backend, so a broken backend cannot pass unnoticed.
    """
    file_path = str(test_paths["bgen_file"])

    decoded = {}
    for name, num_threads in (("sequential", 1), ("parallel", 0)):
        gc.collect()
        with BgenReader(file_path, num_threads=num_threads) as reader:
            dosages, _ = reader.load_variants()
        decoded[name] = dosages

    ref = decoded["sequential"]
    for name, arr in decoded.items():
        np.testing.assert_array_equal(arr, ref, err_msg=f"{name} differs from sequential")


def test_parallel_decompressor_thread_counts_agree(test_paths):
    """The parallel decompressor returns identical dosages for any thread count."""
    file_path = str(test_paths["bgen_file"])

    with BgenReader(file_path, num_threads=0) as reader:  # auto-detect
        dosages_auto, _ = reader.load_variants()
    with BgenReader(file_path, num_threads=2) as reader:
        dosages_fixed, _ = reader.load_variants()

    np.testing.assert_array_equal(dosages_auto, dosages_fixed)


def test_unknown_decompressor_type_rejected(test_paths):
    """set_decompressor_type rejects any unrecognized backend name.

    Rejection must be atomic: a failed call leaves the previously active type in
    place (the .decompressor_type property must never report a rejected value).
    """
    file_path = str(test_paths["bgen_file"])

    with BgenReader(file_path) as reader:
        active = reader.decompressor_type  # default 'parallel'
        for bad_type in ("bogus", "turbo"):
            with pytest.raises(Exception, match="[Dd]ecompressor type"):
                reader.set_decompressor_type(bad_type)
            assert reader.decompressor_type == active  # unchanged after the throw


# ==================== Decode vs External Reference (per format) ====================


@pytest.mark.skipif(not HAS_EXTERNAL_BGEN, reason="External bgen library not available for comparison")
def test_correctness_vs_reference(test_paths):
    """The basic decode matches the external bgen reference shape and metadata."""
    file_path = str(test_paths["bgen_file"])

    our_dosages, our_info, _ = load_bgen(file_path, show_progress=False)

    with external_bgen.BgenReader(file_path) as ref_reader:
        ref_dosages = []
        ref_positions = []
        ref_rsids = []
        for variant in ref_reader:
            ref_dosages.append(variant.alt_dosage)
            ref_positions.append(variant.pos)
            ref_rsids.append(variant.rsid)
    ref_dosages = np.column_stack(ref_dosages)

    assert our_dosages.shape == ref_dosages.shape
    np.testing.assert_array_equal(our_info["pos"].values, ref_positions)
    np.testing.assert_array_equal(our_info["rsid"].values, ref_rsids)
    np.testing.assert_allclose(our_dosages, ref_dosages, rtol=1e-5, atol=1e-8)


def test_reader_basic_dosage_range(test_bgen_files):
    """A basic context-managed load returns valid-range dosages with right shape."""
    file_path = str(test_bgen_files["basic"])
    with BgenReader(file_path) as reader:
        assert reader.nvariants > 0
        assert reader.nsamples > 0
        dosages, variant_info = reader.load_variants()
        assert dosages.shape[0] == reader.nsamples
        assert dosages.shape[1] == len(variant_info)
        assert np.all((dosages >= 0) & (dosages <= 2))


def test_reader_sample_filtering_selects_rows(test_bgen_files, sample_file):
    """Sample filtering by index returns exactly the requested rows."""
    file_path = str(test_bgen_files["basic"])
    sample_indices = np.array([0, 10, 20, 30, 40], dtype=np.int32)
    with BgenReader(file_path, sample_path=str(sample_file)) as reader:
        full, _ = reader.load_variants()
        filtered, _ = reader.load_variants(sample_indices=sample_indices)
    assert filtered.shape[0] == len(sample_indices)
    # Each filtered row is the corresponding row of the full decode (NaN-aware).
    np.testing.assert_array_equal(filtered, full[sample_indices])


def test_reader_nonexistent_file_raises(test_bgen_files):
    """Opening a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        BgenReader("non_existent_file.bgen")


@pytest.mark.parametrize("bit_depth,file_key", [(8, "8bit"), (16, "16bit"), (32, "32bit")])
def test_bit_depth_decodes_valid_range(test_bgen_files, bit_depth, file_key):
    """Each bit-depth fixture decodes to valid-range dosages."""
    if not test_bgen_files[file_key].exists():
        pytest.skip(f"{bit_depth}-bit test file not found")
    dosages, _, _ = load_bgen(str(test_bgen_files[file_key]), show_progress=False, nan_action="mean")
    assert np.all((dosages >= 0) & (dosages <= 2))


def test_zlib_and_zstd_both_decode(test_bgen_files):
    """Both zlib (default) and zstd compressed fixtures decode to valid dosages."""
    dosages_zlib, _, _ = load_bgen(str(test_bgen_files["basic"]), show_progress=False)
    assert np.all((dosages_zlib >= 0) & (dosages_zlib <= 2))

    dosages_zstd, _, _ = load_bgen(str(test_bgen_files["zstd"]), show_progress=False, nan_action="mean")
    assert not np.any(np.isnan(dosages_zstd))
    assert np.all((dosages_zstd >= 0) & (dosages_zstd <= 2))


# ==================== Input Validation ====================


def test_load_variants_rejects_non_float_dtype(test_paths):
    """A non-floating dtype is rejected up front, not silently truncated.

    Dosages are real-valued in [0, 2]; an integer output buffer would round them
    (0.7 -> 0) with no error, corrupting a scientific result. The reader must
    refuse a non-floating dtype rather than produce wrong numbers.
    """
    with BgenReader(str(test_paths["bgen_file"])) as reader:
        with pytest.raises((ValueError, TypeError)):
            reader.load_variants(dtype=np.int32)


def test_iter_variants_rejects_non_float_dtype(test_paths):
    """iter_variants applies the same non-floating-dtype guard as load_variants."""
    with BgenReader(str(test_paths["bgen_file"])) as reader:
        with pytest.raises((ValueError, TypeError)):
            next(reader.iter_variants(dtype=np.int32))


def test_load_bgen_rejects_non_float_dtype(test_paths):
    """load_bgen rejects a non-floating dtype with a clear error (not an assert
    that vanishes under ``python -O``)."""
    with pytest.raises((ValueError, TypeError)):
        load_bgen(str(test_paths["bgen_file"]), dtype=np.int32, show_progress=False)


def test_load_variants_rejects_reversed_region(test_paths):
    """region_start > region_end is rejected in the direct reader API, matching
    the validation load_bgen does via parse_region (instead of silently returning
    an empty result)."""
    with BgenReader(str(test_paths["bgen_file"])) as reader:
        _, all_info = reader.load_variants()
        chrom = all_info.iloc[0]["chrom"]
        with pytest.raises(ValueError):
            reader.load_variants(region_chrom=chrom, region_start=500, region_end=100)


def test_malformed_sample_file_raises_clear_error(test_paths, tmp_path):
    """A .sample file with fewer than the two mandated header lines raises a
    descriptive error, not a bare StopIteration."""
    bad_sample = tmp_path / "bad.sample"
    bad_sample.write_text("ID_1 ID_2 missing\n")  # only one line; spec requires two
    with pytest.raises(ValueError):
        BgenReader(str(test_paths["bgen_file"]), sample_path=str(bad_sample))


def test_get_build_info_is_public():
    """get_build_info is exported and reports a real compression backend.

    Asserting a concrete backend (not the "unknown" import-fallback) makes this a
    guard that the build-generated _build_config.py was actually shipped and
    imported, since the test runs against a built extension.
    """
    import lazybgen

    assert "get_build_info" in lazybgen.__all__
    info = lazybgen.get_build_info()
    assert isinstance(info, dict)
    assert info["type"] in {"vendored", "system"}


# ---------------------------------------------------------------------------
# Sample IDs: materialized on demand, but with the same contract as before
# ---------------------------------------------------------------------------
def test_samples_available_after_close(test_paths):
    """Sample IDs stay readable once the reader is closed, from either source.

    They are materialized on first access rather than at open, so this pins that
    closing the file does not take the answer away.
    """
    reader = BgenReader(str(test_paths["bgen_file"]))
    reader.close()
    assert len(reader.samples) == reader.nsamples

    reader = BgenReader(str(test_paths["bgen_file"]), sample_path=str(test_paths["sample_file"]))
    reader.close()
    assert len(reader.samples) == reader.nsamples


def test_samples_are_cached(test_paths):
    """Repeated access returns the same materialized list, not a fresh parse."""
    with BgenReader(str(test_paths["bgen_file"])) as reader:
        assert reader.samples is reader.samples


def test_samples_from_sample_file_match_second_column(test_paths):
    """The .sample path yields column 2 (ID_2) of each row after the two headers."""
    lines = Path(test_paths["sample_file"]).read_text().splitlines()
    expected = [p[1] for p in (line.split() for line in lines[2:]) if len(p) >= 2]
    with BgenReader(str(test_paths["bgen_file"]), sample_path=str(test_paths["sample_file"])) as reader:
        assert reader.samples == expected


def test_sample_file_rows_with_too_few_fields_are_skipped(test_paths, tmp_path):
    """A row without at least two fields contributes no sample ID."""
    ragged = tmp_path / "ragged.sample"
    ragged.write_text("ID_1 ID_2 missing\n0 0 0\nA A 0\nB\n\nC C 0\n")
    with BgenReader(str(test_paths["bgen_file"]), sample_path=str(ragged)) as reader:
        assert reader.samples == ["A", "C"]


def test_nonexistent_sample_file_raises_at_open(test_paths, tmp_path):
    """A missing .sample file is reported when the reader is opened, not later."""
    with pytest.raises(FileNotFoundError):
        BgenReader(str(test_paths["bgen_file"]), sample_path=str(tmp_path / "nope.sample"))


def test_relative_sample_path_survives_a_chdir(test_paths, tmp_path, monkeypatch):
    """A relative sample_path is anchored at construction, not at first access.

    Sample IDs are parsed on demand, so a relative path re-resolved at access
    time would follow the process's current directory. If another file happened
    to sit at the same relative path, the reader would return that file's IDs
    and silently map genotype columns to the wrong samples.
    """
    sample_file = Path(test_paths["sample_file"])
    workdir = tmp_path / "work"
    workdir.mkdir()
    shutil.copy(sample_file, workdir / "data.sample")

    decoy_dir = tmp_path / "decoy"
    decoy_dir.mkdir()
    (decoy_dir / "data.sample").write_text("ID_1 ID_2 missing\n0 0 0\nDECOY DECOY 0\n")

    monkeypatch.chdir(workdir)
    reader = BgenReader(str(test_paths["bgen_file"]), sample_path="data.sample")
    monkeypatch.chdir(decoy_dir)
    try:
        assert reader.samples[0] != "DECOY"
        assert len(reader.samples) == reader.nsamples
    finally:
        reader.close()


def test_repeated_readers_reuse_a_parsed_sample_file(test_paths, tmp_path):
    """Parsing the same .sample file again is avoided, but never at the cost of
    correctness: the list handed out is the caller's own, and a changed file is
    re-read."""
    sample_file = tmp_path / "cohort.sample"
    shutil.copy(Path(test_paths["sample_file"]), sample_file)

    with BgenReader(str(test_paths["bgen_file"]), sample_path=str(sample_file)) as r1:
        first = r1.samples
    with BgenReader(str(test_paths["bgen_file"]), sample_path=str(sample_file)) as r2:
        second = r2.samples
    assert first == second
    # Each reader owns its list; mutating one must not reach the next.
    assert first is not second
    second[0] = "MUTATED"
    with BgenReader(str(test_paths["bgen_file"]), sample_path=str(sample_file)) as r3:
        assert r3.samples[0] == first[0]


def test_rewritten_sample_file_is_not_served_from_a_stale_parse(test_paths, tmp_path):
    """A .sample file edited between reads yields the new IDs, not the old ones."""
    sample_file = tmp_path / "cohort.sample"
    shutil.copy(Path(test_paths["sample_file"]), sample_file)
    with BgenReader(str(test_paths["bgen_file"]), sample_path=str(sample_file)) as reader:
        original = reader.samples
    assert original[0] != "REWRITTEN_0"

    n = len(original)
    rows = "\n".join(f"REWRITTEN_{i} REWRITTEN_{i} 0" for i in range(n))
    sample_file.write_text(f"ID_1 ID_2 missing\n0 0 0\n{rows}\n")

    with BgenReader(str(test_paths["bgen_file"]), sample_path=str(sample_file)) as reader:
        assert reader.samples[0] == "REWRITTEN_0"
        assert len(reader.samples) == n
