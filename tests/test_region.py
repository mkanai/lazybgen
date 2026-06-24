"""
Tests for region parsing utilities in the lazybgen package.

This module tests the region.parse_region function for:
- Different chromosome formats
- Error handling for invalid formats
- Edge cases
"""

import pytest

from lazybgen.region import parse_region


@pytest.mark.parametrize(
    "region_str,expected_chrom,expected_start,expected_end",
    [
        ("1:1000000-2000000", "1", 1000000, 2000000),
        ("chr1:1000000-2000000", "chr1", 1000000, 2000000),
        ("01:1000000-2000000", "01", 1000000, 2000000),
        ("X:1000000-2000000", "X", 1000000, 2000000),
        ("Y:500000-1000000", "Y", 500000, 1000000),
    ],
)
def test_parse_region_formats(region_str, expected_chrom, expected_start, expected_end):
    """Test region parsing with different chromosome formats."""
    chrom, (start, end) = parse_region(region_str)
    assert chrom == expected_chrom
    assert start == expected_start
    assert end == expected_end


@pytest.mark.parametrize(
    "invalid_region",
    [
        "1-1000000-2000000",  # Missing colon
        "1:10000002000000",  # Missing hyphen
        "1:abc-def",  # Invalid positions
    ],
)
def test_parse_region_invalid_format(invalid_region):
    """Test error handling for invalid region formats."""
    with pytest.raises(ValueError):
        parse_region(invalid_region)


def test_parse_region_edge_cases():
    """Test edge cases in region parsing."""
    # Single position (start = end)
    chrom, (start, end) = parse_region("1:1000000-1000000")
    assert start == end

    # Large positions
    chrom, (start, end) = parse_region("1:200000000-300000000")
    assert start == 200000000
    assert end == 300000000


def test_parse_region_start_greater_than_end():
    """A reversed range (start > end) is an error, not a silent empty region."""
    with pytest.raises(ValueError):
        parse_region("1:200-100")


@pytest.mark.parametrize(
    "bad_region",
    [
        "1:0-100",  # zero start
        "1:-5-100",  # negative start
        "1:100-0",  # zero end (also start > end)
    ],
)
def test_parse_region_non_positive_coords(bad_region):
    """Non-positive coordinates are rejected (BGEN positions are 1-based, >= 1)."""
    with pytest.raises(ValueError):
        parse_region(bad_region)


def test_parse_region_valid_unchanged():
    """Valid regions still parse exactly as before."""
    chrom, (start, end) = parse_region("1:100-200")
    assert chrom == "1"
    assert start == 100
    assert end == 200
