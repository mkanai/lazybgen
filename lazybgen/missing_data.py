"""
NaN handling strategies for BGEN genotype data.

This module provides different strategies for handling NaN values in genotype matrices:
- error: Raise an error with detailed information about NaN locations
- mean: Impute NaN values with variant-wise mean. WARNING: a variant whose values
  are ALL missing has no mean, so it is imputed with dosage 0.0 for every sample,
  i.e. confident hom-ref. This is a silent assumption; prefer 'omit' or filtering
  such variants if that is not what you want.
- omit: Remove samples containing any NaN values
- warn: Issue a warning but preserve NaN values in the data
"""

import logging
from typing import List, NoReturn, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Single source of truth for the accepted nan_action values.
VALID_NAN_ACTIONS = ("error", "mean", "omit", "warn")


def validate_nan_action(action: str) -> None:
    """Validate a nan_action against the allowed set, raising ValueError if unknown.

    Kept separate from handle_nan_values so callers can validate up front (before
    a full read) rather than only when NaNs are actually encountered.
    """
    if action not in VALID_NAN_ACTIONS:
        raise ValueError(f"Invalid nan_action: {action!r}. Valid options are: {', '.join(VALID_NAN_ACTIONS)}")


def handle_nan_values(
    dosages: np.ndarray, variant_info: pd.DataFrame, sample_ids: List[str], action: str = "error"
) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """
    Handle NaN values in genotype matrix based on specified action.

    Parameters
    ----------
    dosages : np.ndarray
        Genotype dosage matrix (samples x variants)
    variant_info : pd.DataFrame
        Variant information
    sample_ids : List[str]
        Sample IDs
    action : str
        Action to take: "error", "mean", "omit", or "warn"

    Returns
    -------
    Tuple[np.ndarray, pd.DataFrame, List[str]]
        (processed_dosages, variant_info, sample_ids)
    """
    # Validate up front so a bogus action is rejected even when no NaNs are present.
    validate_nan_action(action)

    if not np.any(np.isnan(dosages)):
        return dosages, variant_info, sample_ids

    if action == "error":
        _report_nan_error(dosages, variant_info, sample_ids)
    elif action == "mean":
        return _impute_nan_with_mean(dosages, variant_info, sample_ids)
    elif action == "omit":
        return _omit_nan_samples(dosages, variant_info, sample_ids)
    else:  # "warn" (validate_nan_action guarantees one of the four actions)
        return _warn_about_nan(dosages, variant_info, sample_ids)


def _impute_nan_with_mean(
    dosages: np.ndarray, variant_info: pd.DataFrame, sample_ids: List[str]
) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """
    Impute NaN values with variant-wise mean, in place.

    The missing cells of ``dosages`` are overwritten with their per-variant mean
    (and the array is returned), so the only large temporary is the boolean NaN
    mask. Going through ``np.nanmean`` (which copies the whole array) and
    ``np.where`` (which allocates a whole new array) would cost ~2x the
    matrix in temporaries; on a biobank-scale cohort with missing genotypes that
    is enough to exhaust memory. Computing the column sums/counts directly and
    scattering with ``np.copyto`` keeps peak memory at one extra mask.

    WARNING: a variant whose values are ALL missing has no mean and is imputed
    with dosage 0.0 for every sample (confident hom-ref). That is a strong,
    silent assumption; consider 'omit' or pre-filtering all-missing variants if
    that is not the desired behavior.

    Parameters:
    -----------
    dosages : np.ndarray
        Genotype dosage matrix (samples x variants); imputed in place.
    variant_info : pd.DataFrame
        Variant information
    sample_ids : List[str]
        Sample IDs

    Returns:
    --------
    tuple
        (imputed_dosages, variant_info, sample_ids)
    """
    nan_mask = np.isnan(dosages)
    n_nan_total = np.sum(nan_mask)
    valid_counts = dosages.shape[0] - np.sum(nan_mask, axis=0)
    n_variants_with_nan = np.count_nonzero(valid_counts < dosages.shape[0])

    logger.warning(
        f"Found {n_nan_total} NaN values across {n_variants_with_nan} variants. " f"Imputing with variant-wise mean."
    )

    # Per-variant (column) mean of the finite values, without copying the matrix:
    # zero the NaN cells in place, sum each column, divide by the finite count.
    # Accumulate in float64 so the mean is dtype-independent before the cast back.
    np.copyto(dosages, 0, where=nan_mask)
    col_sums = dosages.sum(axis=0, dtype=np.float64)
    all_nan = valid_counts == 0
    col_means = col_sums / np.where(all_nan, 1, valid_counts)

    # Variants that are entirely NaN have no mean: warn (in column order) and impute with
    # 0.0, i.e. confident hom-ref for every sample. This is a silent assumption; see the
    # module/function docstring.
    col_means[all_nan] = 0.0
    for j in np.nonzero(all_nan)[0]:
        logger.warning(
            f"Variant {variant_info.iloc[int(j)]['rsid']} at position "
            f"{variant_info.iloc[int(j)]['pos']} has all NaN values. Imputing with 0."
        )

    # Scatter each column mean into that column's missing cells (the NaNs are
    # currently zeroed); finite cells are untouched. Casting to the dosage dtype
    # preserves the original array dtype (e.g. float32).
    np.copyto(dosages, col_means.astype(dosages.dtype, copy=False), where=nan_mask)

    return dosages, variant_info, sample_ids


def _omit_nan_samples(
    dosages: np.ndarray, variant_info: pd.DataFrame, sample_ids: List[str]
) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """
    Omit samples with any NaN values.

    Parameters:
    -----------
    dosages : np.ndarray
        Genotype dosage matrix (samples x variants)
    variant_info : pd.DataFrame
        Variant information
    sample_ids : list of str
        Sample IDs

    Returns:
    --------
    tuple
        (filtered_dosages, variant_info, filtered_sample_ids)
    """
    # Find samples with any NaN values
    nan_mask = np.isnan(dosages)
    samples_with_nan = np.any(nan_mask, axis=1)
    n_samples_with_nan = np.sum(samples_with_nan)

    if n_samples_with_nan == 0:
        return dosages, variant_info, sample_ids

    # Log warning about samples being removed
    logger.warning(f"Removing {n_samples_with_nan} samples with NaN values out of {len(sample_ids)} total samples.")

    # Show first few sample IDs being removed
    removed_sample_ids = [sample_ids[i] for i in np.where(samples_with_nan)[0][:5]]
    if n_samples_with_nan <= 5:
        logger.warning(f"Removed samples: {', '.join(removed_sample_ids)}")
    else:
        logger.warning(
            f"First 5 removed samples: {', '.join(removed_sample_ids)} " f"(and {n_samples_with_nan - 5} more)"
        )

    # Keep only samples without NaN
    keep_mask = ~samples_with_nan
    filtered_dosages = dosages[keep_mask, :]
    filtered_sample_ids = [sid for i, sid in enumerate(sample_ids) if keep_mask[i]]

    # Also check if any variants now have all missing values
    all_nan_variants = np.all(np.isnan(filtered_dosages), axis=0)
    if np.any(all_nan_variants):
        n_all_nan = np.sum(all_nan_variants)
        logger.warning(
            f"After removing samples, {n_all_nan} variants have no valid data. "
            f"Consider using 'mean' imputation instead."
        )

    return filtered_dosages, variant_info, filtered_sample_ids


def _report_nan_error(dosages: np.ndarray, variant_info: pd.DataFrame, sample_ids: List[str]) -> NoReturn:
    """
    Report detailed error message for NaN values in genotype matrix.

    Parameters:
    -----------
    dosages : np.ndarray
        Genotype dosage matrix
    variant_info : pd.DataFrame
        Variant information
    sample_ids : list of str
        Sample IDs

    Raises:
    -------
    ValueError
        Always raises with detailed NaN information
    """
    nan_mask = np.isnan(dosages)

    # Count samples and variants with NaN
    samples_with_nan = np.any(nan_mask, axis=1)
    variants_with_nan = np.any(nan_mask, axis=0)
    n_samples_with_nan = np.sum(samples_with_nan)
    n_variants_with_nan = np.sum(variants_with_nan)

    # Find first 5 sample/variant pairs with NaN
    nan_locations = np.argwhere(nan_mask)[:5]

    # Build detailed error message
    error_msg = (
        f"Genotype matrix contains NaN values:\n"
        f"  - {n_samples_with_nan} out of {dosages.shape[0]} samples have NaN values\n"
        f"  - {n_variants_with_nan} out of {dosages.shape[1]} variants have NaN values\n"
    )

    if len(nan_locations) > 0:
        error_msg += "\nFirst (up to 5) sample/variant pairs with NaN:\n"
        for i, (sample_idx, variant_idx) in enumerate(nan_locations):
            sample_id = sample_ids[sample_idx]
            variant_rsid = variant_info.iloc[variant_idx]["rsid"]
            variant_pos = variant_info.iloc[variant_idx]["pos"]
            error_msg += (
                f"  {i+1}. Sample '{sample_id}' (index {sample_idx}), "
                f"Variant '{variant_rsid}' at position {variant_pos} (index {variant_idx})\n"
            )

    error_msg += "\nThis may indicate issues with the input BGEN file or variant filtering."

    raise ValueError(error_msg)


def _warn_about_nan(
    dosages: np.ndarray, variant_info: pd.DataFrame, sample_ids: List[str]
) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """
    Warn about NaN values but return data unchanged.

    Parameters
    ----------
    dosages : np.ndarray
        Genotype dosage matrix
    variant_info : pd.DataFrame
        Variant information
    sample_ids : List[str]
        Sample IDs

    Returns
    -------
    Tuple[np.ndarray, pd.DataFrame, List[str]]
        Original data unchanged
    """
    nan_mask = np.isnan(dosages)

    # Count samples and variants with NaN
    samples_with_nan = np.any(nan_mask, axis=1)
    variants_with_nan = np.any(nan_mask, axis=0)
    n_samples_with_nan = np.sum(samples_with_nan)
    n_variants_with_nan = np.sum(variants_with_nan)

    # Find first 5 sample/variant pairs with NaN
    nan_locations = np.argwhere(nan_mask)[:5]

    # Build warning message
    warning_msg = (
        f"Genotype matrix contains NaN values:\n"
        f"  - {n_samples_with_nan} out of {dosages.shape[0]} samples have NaN values\n"
        f"  - {n_variants_with_nan} out of {dosages.shape[1]} variants have NaN values"
    )

    if len(nan_locations) > 0:
        warning_msg += "\nFirst (up to 5) sample/variant pairs with NaN:\n"
        for i, (sample_idx, variant_idx) in enumerate(nan_locations):
            sample_id = sample_ids[sample_idx]
            variant_rsid = variant_info.iloc[variant_idx]["rsid"]
            variant_pos = variant_info.iloc[variant_idx]["pos"]
            warning_msg += (
                f"  {i+1}. Sample '{sample_id}' (index {sample_idx}), "
                f"Variant '{variant_rsid}' at position {variant_pos} (index {variant_idx})\n"
            )

    warning_msg += "\nNaN values preserved for analysis. Use 'mean' or 'omit' to handle missing data."

    logger.warning(warning_msg)

    # Return data unchanged
    return dosages, variant_info, sample_ids
