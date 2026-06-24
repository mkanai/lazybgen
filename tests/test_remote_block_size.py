"""Unit tests for the remote readahead block-size policy.

These pin the access-pattern heuristic that dampens scattered-read over-fetch on
remote (gs://, s3://) BGEN files. The policy is pure Python so it is tested
directly, without network access; the C++ reader applies and clamps the value.
"""

from lazybgen.remote import choose_remote_block_size

NSAMPLES = 500_000
# One variant's estimated on-disk record, mirroring the policy's own estimate.
VSIZE = NSAMPLES * 2 + 128 * 1024


def test_empty_selection_returns_zero():
    # 0 means "leave the block unchanged" - nothing to read.
    assert choose_remote_block_size(NSAMPLES, []) == 0


def test_single_variant_uses_one_variant_block():
    assert choose_remote_block_size(NSAMPLES, [1_000_000]) == VSIZE


def test_contiguous_selection_is_dense_and_uses_large_block():
    # Variants packed back-to-back (gap ~ one record) -> coalesce several.
    offsets = [i * VSIZE for i in range(100)]
    assert choose_remote_block_size(NSAMPLES, offsets) == VSIZE * 8


def test_scattered_selection_uses_one_variant_block():
    # Variants spread far apart (gap >> one record) -> no coalescing to gain.
    offsets = [i * 50 * VSIZE for i in range(100)]
    assert choose_remote_block_size(NSAMPLES, offsets) == VSIZE


def test_unsorted_offsets_are_handled():
    # Density is decided on sorted gaps, so input order must not matter.
    contiguous = [i * VSIZE for i in range(50)]
    shuffled = contiguous[::-1]
    assert choose_remote_block_size(NSAMPLES, shuffled) == VSIZE * 8


def test_large_dense_selection_classified_dense():
    # A large dense whole-file selection (every variant adjacent) is classified
    # dense and returns the multi-variant block size.
    offsets = [i * VSIZE for i in range(200_000)]
    assert choose_remote_block_size(NSAMPLES, offsets) == VSIZE * 8


def test_block_scales_with_sample_count():
    small = choose_remote_block_size(10_000, [0])
    big = choose_remote_block_size(1_000_000, [0])
    assert big > small
