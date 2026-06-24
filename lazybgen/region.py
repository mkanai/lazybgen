"""
Utility functions for working with genomic regions.
"""

from typing import Tuple


def parse_region(region_str: str) -> Tuple[str, Tuple[int, int]]:
    """
    Parse a genomic region string.

    Parameters:
    -----------
    region_str : str
        Genomic region in format "chr:start-end"

    Returns:
    --------
    tuple
        (chrom, (start_pos, end_pos))

    Raises:
    -------
    ValueError
        If the region string is not in the expected format
    """
    if region_str is None:
        raise ValueError("Region string cannot be None")

    try:
        chrom, pos_range = region_str.split(":")
        start_pos, end_pos = map(int, pos_range.split("-"))
    except ValueError:
        raise ValueError(f"Invalid region format: {region_str}. Expected format: 'chr:start-end'")

    # Reject non-positive coordinates and reversed ranges so an invalid region
    # fails loudly instead of silently yielding an empty/confusing query.
    # BGEN positions are 1-based.
    if start_pos < 1 or end_pos < 1:
        raise ValueError(f"Invalid region coordinates in {region_str!r}: positions must be >= 1.")
    if start_pos > end_pos:
        raise ValueError(f"Invalid region {region_str!r}: start ({start_pos}) is greater than end ({end_pos}).")

    return chrom, (start_pos, end_pos)


def validate_region_bounds(chrom, start, end) -> None:
    """Validate integer region bounds for the direct reader API.

    Mirrors ``parse_region``'s coordinate checks for callers that pass
    ``region_start``/``region_end`` as integers rather than a ``"chr:start-end"``
    string, so the two entry points reject the same bad input. No-op when
    ``chrom`` is None (no region) or a bound is None (open-ended on that side).
    BGEN positions are 1-based.
    """
    if chrom is None:
        return
    if start is not None and start < 1:
        raise ValueError(f"Invalid region start ({start}): positions must be >= 1.")
    if end is not None and end < 1:
        raise ValueError(f"Invalid region end ({end}): positions must be >= 1.")
    if start is not None and end is not None and start > end:
        raise ValueError(f"Invalid region: start ({start}) is greater than end ({end}).")
