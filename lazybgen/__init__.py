"""
lazybgen: a high-performance BGEN reader with cloud (GCS/S3) partial-read support.

Provides a Cython implementation of BGEN file reading with random-access,
region/variant-filtered loading directly from local files or Google Cloud Storage or Amazon S3.
"""

import logging
import os
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, List, Optional

from .missing_data import handle_nan_values, validate_nan_action

# Import the high-performance BGEN reader
from .reader import BgenReader
from .region import parse_region
from .remote import is_remote_path
from .variant_filter import load_variant_filter

try:
    # Generated at build time; records the compression backend (vendored vs system).
    from ._build_config import get_build_info
except ImportError:  # pragma: no cover - only when imported without a completed build

    def get_build_info() -> Dict[str, str]:
        """Build-time compression-backend info (unavailable: package not built)."""
        return {"type": "unknown", "note": "build configuration not available"}


try:
    __version__ = version("lazybgen")
except PackageNotFoundError:  # package not installed (e.g. running from a source checkout)
    __version__ = "0.0.0+unknown"

logger = logging.getLogger(__name__)

__all__ = ["BgenReader", "load_bgen", "load_variant_filter", "get_build_info"]


def _create_progress_callback(show_progress: bool, total: int, desc: str = "Loading variants"):
    """Create a progress callback function using tqdm if requested."""
    if not show_progress:
        return None

    # Lazy import tqdm only when needed
    from tqdm import tqdm

    pbar = tqdm(total=total, desc=desc, unit="variants")

    # Update every 100 variants or at least every 1%
    update_freq = min(100, max(1, total // 100))

    def callback(current):
        # Only update progress bar at specified frequency
        if current % update_freq == 0 or current >= total:
            pbar.n = current
            pbar.refresh()
        if current >= total:
            pbar.close()

    return callback


def _validate_dosages(dosages) -> bool:
    """Validate dosage values are within expected range; report missing data.

    Returns True if the matrix contains any NaN (missing genotypes).

    NaN propagates through np.min / np.max, so this single min/max pair answers
    both questions at once: a NaN result means missing data is present, and any
    real value outside [0, 2] still shows up in the min or the max. That keeps
    the common (no missing data) case to two scans of the matrix, instead of a
    third isnan scan plus a full-size boolean temporary, which at biobank scale
    is gigabytes of avoidable traffic.
    """
    # Lazy import numpy
    import numpy as np

    # Empty array: nothing to check (np.min/np.max would raise on it).
    if dosages.size == 0:
        return False

    min_dosage = dosages.min()
    max_dosage = dosages.max()
    has_nan = bool(min_dosage != min_dosage or max_dosage != max_dosage)

    if has_nan:
        # Re-scan ignoring NaN so the range check still sees the real values.
        # For an all-NaN array nanmin and nanmax return NaN (with an "All-NaN
        # slice" RuntimeWarning we suppress), and both NaN < 0.0 and NaN > 2.0
        # are False, so the range check is correctly skipped. +/-inf is not NaN,
        # so it never reaches here and is caught by the plain min/max above.
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            min_dosage = np.nanmin(dosages)
            max_dosage = np.nanmax(dosages)

    _check_dosage_range(min_dosage, max_dosage)

    return has_nan


def _check_dosage_range(min_dosage, max_dosage) -> None:
    """Raise if the observed dosage bounds fall outside [0, 2].

    Both bounds ignore missing calls, so an all-missing result has no real value
    to be out of range and correctly passes: it arrives as inf / -inf from the
    reader's own summary, or as NaN from nanmin / nanmax, and neither compares
    true against the bounds.
    """
    if min_dosage < 0.0 or max_dosage > 2.0:
        raise ValueError(
            f"Dosage values out of valid range [0, 2] detected "
            f"(min: {min_dosage:.6f}, max: {max_dosage:.6f}). "
            f"This may indicate: "
            f"1) Corrupted BGEN file data, "
            f"2) Memory initialization issue in the reader, "
            f"3) Invalid genotype probabilities that don't sum to 1.0"
        )


def load_bgen(
    file_path: str,
    index_path: Optional[str] = None,
    sample_path: Optional[str] = None,
    region: Optional[str] = None,
    variant_filter: Optional[Dict[str, Any]] = None,
    sample_ids: Optional[List[str]] = None,
    dtype=None,
    show_progress: bool = False,
    nan_action: str = "error",
    num_threads: int = 0,
    storage_options: Optional[Dict[str, Any]] = None,
):
    """
    Load genotype data from BGEN file.

    Parameters
    ----------
    file_path : str
        Path to BGEN file
    index_path : str, optional
        Path to BGI index file. If None, will look for file_path + '.bgi'
    sample_path : str, optional
        Path to sample file
    region : str, optional
        Genomic region in format "chr:start-end"
    variant_filter : dict, optional
        Variant filter from .z file (from load_variant_filter)
    sample_ids : list of str, optional
        Sample IDs to keep. If None, all samples are loaded.
    dtype : numpy.dtype, optional
        Data type for the dosage array (default: numpy.float64)
    show_progress : bool, optional
        Whether to show progress bars during loading (default: False)
    nan_action : str, optional
        Action for handling missing (NaN) dosages: 'error' (default, raise),
        'mean' (impute with the per-variant mean), 'omit' (drop affected
        samples), or 'warn' (keep NaNs and log a warning).
    num_threads : int, optional
        Worker threads for decoding. 0 (default) auto-detects the CPU core count
        and decodes blocks in parallel (several times faster on multi-core
        machines, byte-identical to serial); 1 forces single-threaded decoding;
        N > 1 uses N threads.
    storage_options : dict, optional
        Backend kwargs forwarded to the fsspec filesystem (e.g. {"anon": True}
        for public S3, or for GCS requester-pays buckets {"requester_pays": True}
        to bill the environment's default project or {"requester_pays":
        "billing-project-id"} to bill a specific one). Ignored for local files.

    Returns
    -------
    tuple
        (genotypes, variant_info, sample_ids)
        genotypes: numpy.ndarray
        variant_info: pandas.DataFrame
        sample_ids: list of str
        Note: genotypes are returned as floating point values of the specified dtype
        If variant_filter is provided, variants are ordered according to the .z file order
        If sample_ids is provided, only those samples are returned
    """
    # Validate nan_action up front so a bogus value fails immediately, before a
    # full read, rather than only when missing data happens to be encountered.
    validate_nan_action(nan_action)

    # Lazy import heavy modules
    import numpy as np

    # Set default dtype if not provided
    if dtype is None:
        dtype = np.float64

    # Check BGEN file exists (skip for remote paths)
    if not is_remote_path(file_path) and not os.path.exists(file_path):
        raise FileNotFoundError(f"BGEN file not found: {file_path}")

    # Determine BGI path
    if index_path is not None:
        bgi_path = index_path
    else:
        bgi_path = file_path + ".bgi"

    # BGI is mandatory (skip check for remote paths as they'll be handled by reader)
    if not is_remote_path(bgi_path) and not os.path.exists(bgi_path):
        raise FileNotFoundError(
            f"BGI index required but not found: {bgi_path}\n" f"Please create index using: bgenix -g {file_path}"
        )

    logger.info(f"Opening BGEN file: {file_path}")

    # Create reader with explicit BGI path
    reader = BgenReader(
        file_path,
        sample_path=sample_path if sample_path else None,
        bgi_path=bgi_path,
        num_threads=num_threads,
        storage_options=storage_options,
    )

    try:
        # Process sample filtering if requested
        sample_indices = None
        filtered_sample_ids = reader.samples

        if sample_ids is not None:
            logger.info(f"Filtering BGEN to {len(sample_ids)} requested samples")
            sample_indices, filtered_sample_ids = reader.get_sample_indices(sample_ids)

            if not sample_indices:
                raise ValueError(
                    "No requested samples found in BGEN file. " "Please check that sample IDs match between files."
                )

            # Only log if there's a difference between requested and found
            if len(sample_indices) < len(sample_ids):
                missing = len(sample_ids) - len(sample_indices)
                logger.warning(
                    f"Found {len(sample_indices)} out of {len(sample_ids)} requested samples. "
                    f"Missing {missing} samples."
                )

        # Parse region if provided
        region_chrom = None
        region_start = None
        region_end = None
        if region:
            region_chrom, (region_start, region_end) = parse_region(region)

        # Get total variant count for progress bar
        if variant_filter is not None:
            total_variants = len(variant_filter["positions"])
        elif region:
            # We don't know exact count for region, so estimate
            total_variants = None
        else:
            # Get variant count from reader
            total_variants = reader.nvariants

        # Create progress callback
        progress_callback = None
        if show_progress and total_variants:
            progress_callback = _create_progress_callback(show_progress, total_variants)

        # Convert sample indices to numpy array if needed
        if sample_indices is not None:
            sample_indices = np.array(sample_indices, dtype=np.int32)

        # Load variants using unified method
        dosages, variant_info = reader.load_variants(
            region_chrom=region_chrom,
            region_start=region_start,
            region_end=region_end,
            variant_filter=variant_filter,
            sample_indices=sample_indices,
            dtype=dtype,
            progress_callback=progress_callback,
        )

        # Check if we loaded any variants
        if dosages.size == 0 or dosages.shape[1] == 0:
            raise ValueError(
                "No variants were loaded from the BGEN file. "
                "This may be due to: "
                "1) An empty genomic region, "
                "2) No variants passing the filter criteria, "
                "3) Issues with the BGEN file format"
            )

        # Validate genotypes. The output dtype is validated as floating-point at
        # the reader boundary (BgenReader.load_variants), so by here the dosages
        # are guaranteed real-valued; only the value range needs checking.
        # The block decode summarizes the values as it writes them, so prefer
        # its answer; scanning the matrix here would re-read every byte of it
        # from memory, which at biobank scale costs more than the decode did.
        # The per-variant serial path reports nothing, and then we scan.
        stats = reader.last_dosage_stats
        if stats is None:
            has_nan = _validate_dosages(dosages)
        else:
            min_dosage, max_dosage, has_nan = stats
            _check_dosage_range(min_dosage, max_dosage)

        # Handle NaN values if present, reusing the answer from above rather
        # than scanning again.
        if has_nan:
            dosages, variant_info, filtered_sample_ids = handle_nan_values(
                dosages, variant_info, filtered_sample_ids, nan_action
            )

        return dosages, variant_info, filtered_sample_ids

    except Exception as e:
        logger.error(f"Error loading BGEN file: {e}")
        raise
    finally:
        reader.close()
