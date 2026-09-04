# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
# distutils: language=c++
# Compile flags and NumPy macros are set on the Extension in setup.py
# (single source: -std=c++14, NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION).

import os
import struct
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.memory cimport unique_ptr, make_unique
from libc.stdint cimport uint32_t, uint64_t, uint8_t, uint16_t
from libc.string cimport memcpy
import pandas as pd
from typing import Optional, List, Tuple, Dict, Any, Callable, Union
import logging

from .remote import is_remote_path, choose_remote_block_size
from .region import validate_region_bounds


def _validate_dtype(dtype):
    """Reject a non-floating output dtype before any decode.

    Dosages are real-valued in [0, 2]; an integer buffer would silently round
    them (0.7 -> 0), corrupting a scientific result, so refuse rather than
    truncate.
    """
    if not np.issubdtype(np.dtype(dtype), np.floating):
        raise ValueError(
            f"dtype must be a floating-point type (dosages are real-valued in "
            f"[0, 2]); got {np.dtype(dtype)!r}"
        )

# Note: Since this file is reader.pyx and has a corresponding reader.pxd,
# we don't need to explicitly import - the declarations are automatically available

logger = logging.getLogger(__name__)

np.import_array()


# Most recently parsed .sample file: {(path, mtime_ns, size): tuple(ids)}.
# Keyed on the file's identity and a content stamp, so an edited file is re-read
# rather than served stale. Holds one entry; see _load_samples_from_file.
_SAMPLE_ID_CACHE = {}


def _sample_file_key(sample_path):
    """Identity + content stamp for a .sample file, or None if it cannot be stat'd."""
    try:
        st = os.stat(sample_path)
    except OSError:
        return None
    return (sample_path, st.st_mtime_ns, st.st_size)


cdef class BgenReader:
    """
    High-performance BGEN file reader with C++ integration.
    
    This reader uses an optimized decompressor architecture for better performance
    and automatic optimization based on access patterns.
    """
    
    def __init__(self, file_path: str, bgi_path: Optional[str] = None,
                 sample_path: Optional[str] = None,
                 num_threads: int = 0,
                 storage_options: Optional[dict] = None):
        """
        Initialize BGEN reader.

        Parameters
        ----------
        file_path : str
            Path to BGEN file
        bgi_path : str, optional
            Path to BGI index file. If None, will look for file_path + '.bgi'
        sample_path : str, optional
            Path to sample file
        num_threads : int, optional
            Worker threads for decoding. Must be >= 0. 0 (default) auto-detects the
            CPU core count and decodes blocks in parallel (several times faster on
            multi-core machines, byte-identical to single-threaded decoding); 1
            decodes on a single thread; N > 1 uses N threads. This is the only knob
            for selecting parallel vs sequential decoding.
        storage_options : dict, optional
            Backend kwargs forwarded to the fsspec filesystem (e.g. {"anon": True}
            for public S3, or for GCS requester-pays buckets {"requester_pays": True}
            to bill the environment's default project or {"requester_pays":
            "billing-project-id"} to bill a specific one). Ignored for local files.
        """
        self.file_path = file_path
        self.bgi_path = bgi_path or (file_path + '.bgi')
        self.storage_options = storage_options
        self.is_open = False
        # Anchored at construction: the rows are parsed on first access to the
        # `samples` property, so a relative path left unresolved would follow the
        # process's current directory to whatever file sits there by then.
        self.sample_path = os.path.abspath(sample_path) if sample_path else None
        # Sample IDs are materialized on first access (see the `samples`
        # property). At biobank scale building them is hundreds of thousands of
        # Python strings, which a read that never asks for sample IDs should not
        # pay for. None means "not built yet"; an empty list is a real answer.
        self.sample_ids = None
        self.dosage_stats = None

        # Check files exist (skip for remote paths as they'll be handled by C++)
        if not is_remote_path(self.file_path):
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"BGEN file not found: {self.file_path}")
        if not is_remote_path(self.bgi_path):
            if not os.path.exists(self.bgi_path):
                raise FileNotFoundError(f"BGI index not found: {self.bgi_path}")
        
        # Initialize C++ components
        self._init_reader()
        
        # Mark as open before configuring decompressor
        self.is_open = True

        # Select the decoder from num_threads: 1 uses the true sequential path (no
        # worker-thread overhead); anything else uses parallel decode (0 auto-detects
        # the core count at decode time). set_decompressor_type remains available for
        # advanced callers that want to force a backend after construction.
        if num_threads < 0:
            raise ValueError(
                f"num_threads must be >= 0 (0 auto-detects the core count); got {num_threads}"
            )
        decompressor_type = 'sequential' if num_threads == 1 else 'parallel'
        self.set_decompressor_type(decompressor_type, num_threads)
        
        # Check the .sample file up front so a missing or truncated one is
        # reported when the reader is opened, not on some later attribute access.
        # Only the two mandated header lines are read here; the per-sample rows
        # are parsed on demand.
        if sample_path:
            self._check_sample_file(self.sample_path)
    
    cdef void _init_reader(self) except *:
        """Initialize C++ reader components."""
        cdef string cpp_file_path = self.file_path.encode('utf-8')
        cdef string cpp_bgi_path
        
        # Handle BGI cache for remote paths
        if is_remote_path(self.bgi_path):
            # Download the remote BGI index into the local cache directory, then
            # pass the local path to C++.
            from .remote import ensure_local_bgi
            local_bgi_path = ensure_local_bgi(self.bgi_path, self.storage_options)
            cpp_bgi_path = local_bgi_path.encode('utf-8')
            # Update self.bgi_path to the local path for consistency
            self.bgi_path = local_bgi_path
        else:
            cpp_bgi_path = self.bgi_path.encode('utf-8')
        
        try:
            # Create main reader
            self.impl.reset(new BgenReaderImpl(cpp_file_path, cpp_bgi_path, self.storage_options))
            
            # Store header info
            self.header_info = self.impl.get().header()
            
            # Create BGI reader separately
            self.bgi_reader.reset(new BgiReader(cpp_bgi_path))
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize BGEN reader: {e}")
    
    def _check_sample_file(self, sample_path: str):
        """Validate a .sample file's header without parsing its rows.

        The .sample format mandates two header lines (column names, then column
        types) before the per-sample rows. Reading just those two catches a
        missing or truncated file at open, while leaving the per-sample rows
        (which dominate the cost at biobank scale) to _load_samples_from_file.
        Only the file's state at open is checked; a file removed or truncated
        afterwards surfaces when the rows are read.
        """
        with open(sample_path, 'r') as f:
            if not f.readline() or not f.readline():
                raise ValueError(
                    f"Malformed .sample file (expected at least 2 header lines): {sample_path}"
                )

    def _load_samples_from_file(self, sample_path: str):
        """Load sample IDs from .sample file."""
        # Reuse the last file parsed, if this is that same file unchanged.
        # Parsing is ~500K Python strings at biobank scale and dominates a
        # load_bgen call that reads only a handful of variants, so a loop of
        # loads over one cohort would otherwise repeat it every time.
        cache_key = _sample_file_key(sample_path)
        if cache_key is not None:
            cached = _SAMPLE_ID_CACHE.get(cache_key)
            if cached is not None:
                # A fresh list each time: the caller owns what it is handed and
                # may mutate it, which must not reach the next reader.
                self.sample_ids = list(cached)
                return

        with open(sample_path, 'r') as f:
            # Skip the two header lines validated at open.
            if not f.readline() or not f.readline():
                raise ValueError(
                    f"Malformed .sample file (expected at least 2 header lines): {sample_path}"
                )
            # Column 2 (ID_2) is the primary ID. split(None, 2) stops after the
            # field we want instead of splitting every remaining column, and it
            # already ignores surrounding whitespace, so no separate strip() is
            # needed. A row without at least two fields carries no ID.
            self.sample_ids = [
                parts[1]
                for parts in (line.split(None, 2) for line in f)
                if len(parts) >= 2
            ]

        if cache_key is not None:
            # One file only: the pattern this serves is repeated loads over a
            # single cohort, and the IDs are tens of megabytes to hold.
            _SAMPLE_ID_CACHE.clear()
            _SAMPLE_ID_CACHE[cache_key] = tuple(self.sample_ids)

    cdef void _load_samples(self) except *:
        """Materialize the sample IDs from whichever source was configured."""
        if self.sample_path:
            self._load_samples_from_file(self.sample_path)
        else:
            self._load_samples_from_bgen()

    cdef void _load_samples_from_bgen(self) except *:
        """Load sample IDs from BGEN file."""
        # Bound by reference: copying the C++ vector would duplicate every string
        # before any of them reach Python.
        cdef const vector[string]* cpp_samples = &self.impl.get().sample_ids()
        cdef size_t i
        cdef list out = []
        for i in range(cpp_samples.size()):
            out.append(cpp_samples.at(i).decode('utf-8'))
        self.sample_ids = out
    
    def set_decompressor_type(self, decompressor_type: str, num_threads: int = 0):
        """
        Set the decompressor type.
        
        Parameters
        ----------
        decompressor_type : str
            Decode routing: 'sequential' (per-variant loop) or 'parallel' (block
            decode across worker threads). A default-constructed reader uses
            'parallel' unless num_threads == 1.
        num_threads : int
            Number of worker threads for the parallel path (0 = auto-detect)
        """
        self._ensure_open()

        cdef string cpp_type = decompressor_type.encode('utf-8')
        # Record the worker-thread count so the block-parallel decode path (which
        # reads num_threads() directly) uses it. set_decompressor_type only sets
        # the routing flag; the stored decompressor is always sequential.
        if decompressor_type == 'parallel' and num_threads > 0:
            self.impl.get().set_num_threads(num_threads)
        self.impl.get().set_decompressor_type(cpp_type)

    @property
    def decompressor_type(self) -> str:
        """The active decompressor type ('sequential' or 'parallel')."""
        self._ensure_open()
        return self.impl.get().decompressor_type().decode('utf-8')

    cdef np.ndarray _validate_sample_indices(self, object sample_indices):
        """Coerce and validate a user-supplied sample_indices array.

        The array is forwarded down to a raw ``const int*`` / ``const uint32_t*``
        cast in C++, so it MUST be contiguous, int32, and in range. Returns a
        contiguous int32 ndarray (a copy when coercion is needed), or None if the
        input was None.
        """
        if sample_indices is None:
            return None
        # Validate in a WIDE signed dtype first. Casting to int32 up front would let
        # out-of-range values wrap into the valid range (e.g. 2**32 -> 0) and slip
        # past the bounds check, so we range-check in int64 before narrowing.
        cdef np.ndarray wide = np.asarray(sample_indices)
        if wide.ndim != 1:
            raise ValueError("sample_indices must be a 1-D array")
        wide = np.ascontiguousarray(wide, dtype=np.int64)
        cdef Py_ssize_t n = wide.shape[0]
        if n == 0:
            return np.empty(0, dtype=np.int32)
        cdef long long n_samples = <long long>self.header_info.n_samples
        cdef long long lo = <long long>wide.min()
        cdef long long hi = <long long>wide.max()
        if lo < 0:
            raise IndexError(
                f"sample_indices contains a negative index ({lo}); "
                "indices must be in [0, n_samples)"
            )
        if hi >= n_samples:
            raise IndexError(
                f"sample_indices contains an out-of-range index ({hi}); "
                f"valid range is [0, {n_samples})"
            )
        # Now safe to narrow to the contiguous int32 the C++ parser expects: every
        # value is in [0, n_samples) and n_samples fits in int32 (BGEN nsamples is
        # a uint32 file field but the parser indexes with int).
        return np.ascontiguousarray(wide, dtype=np.int32)

    def load_variants(
        self,
        region_chrom: Optional[str] = None,
        region_start: Optional[int] = None,
        region_end: Optional[int] = None,
        variant_filter: Optional[Dict] = None,
        sample_indices: Optional[np.ndarray] = None,
        dtype = np.float64,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Load variants with various filtering options.
        
        Parameters
        ----------
        region_chrom : str, optional
            Chromosome for region query
        region_start : int, optional
            Start position for region query (inclusive)
        region_end : int, optional
            End position for region query (inclusive)
        variant_filter : dict, optional
            Variant filter from .z file
        sample_indices : np.ndarray, optional
            Sample indices to keep
        dtype : np.dtype
            Data type for dosages
        progress_callback : callable, optional
            Function to call with progress updates
        
        Returns
        -------
        Tuple[np.ndarray, pd.DataFrame]
            (dosages, variant_info)
        """
        self._ensure_open()

        _validate_dtype(dtype)
        validate_region_bounds(region_chrom, region_start, region_end)
        # Validate / coerce sample_indices at the public boundary before it is
        # cast to a raw C pointer downstream.
        sample_indices = self._validate_sample_indices(sample_indices)

        # Get variant metadata based on query type
        cdef vector[VariantMetadata] variant_metadata
        cdef vector[VariantInfo] variant_infos
        cdef string cpp_chrom

        if variant_filter is not None:
            # Filtered variants
            variant_metadata = self._filtered_variant_metadata(variant_filter)
        else:
            if region_chrom is not None:
                # Region query
                cpp_chrom = region_chrom.encode('utf-8')
                variant_infos = self.bgi_reader.get().query_region(
                    cpp_chrom, region_start or 0, region_end or 0xFFFFFFFF
                )
            else:
                # All variants - use efficient batch query
                variant_infos = self.bgi_reader.get().get_all_variants()
            # Size the remote readahead block to this selection before reading.
            self._tune_remote_block_size([info.file_offset for info in variant_infos])
            # Build decode-ready metadata straight from the BGI (no per-variant
            # metadata read): identity from the index, genotype located in-buffer
            # by the one-GET record read at decode time.
            variant_metadata = self._metadata_from_variant_infos(variant_infos)

        # Load variant data
        return self._load_variants_from_metadata(
            variant_metadata, sample_indices, dtype, progress_callback
        )

    def iter_variants(
        self,
        region_chrom: Optional[str] = None,
        region_start: Optional[int] = None,
        region_end: Optional[int] = None,
        variant_filter: Optional[Dict] = None,
        sample_indices: Optional[np.ndarray] = None,
        dtype = np.float64,
        block_size: Optional[int] = None,
    ):
        """
        Stream variants one at a time without materializing the full matrix.

        Memory-bounded counterpart to ``load_variants``: variant dosages are
        read in blocks of ``block_size`` and yielded per variant, so peak memory
        is O(n_samples x block_size) rather than O(n_samples x n_variants).
        Accepts the same region/variant_filter/sample_indices selection as
        ``load_variants``. When the reader uses parallel decode (the default; see
        ``num_threads``), each block is decoded across worker threads (the
        per-block path is the same as ``load_variants``).

        Parameters
        ----------
        region_chrom, region_start, region_end : optional
            Restrict to a genomic region (uses the BGI index to seek).
        variant_filter : dict, optional
            Variant filter from a .z file (see load_variant_filter).
        sample_indices : np.ndarray, optional
            Sample indices to keep.
        dtype : np.dtype
            Data type for the per-variant dosage array.
        block_size : int, optional
            Number of variants whose dosages are decoded (and held) at once. The
            block matrix is O(n_samples x block_size), so a fixed default is a
            footgun at large sample counts (1000 x 500K x 8B = 4 GB). When None
            (the default), block_size auto-scales to keep each block near a 256 MB
            budget (so it shrinks as n_samples grows); pass an explicit value to
            override. block_size affects only memory/throughput, never the yielded
            values.

        Yields
        ------
        Tuple[dict, np.ndarray]
            (info, dosage) per variant, where ``info`` is a dict with keys
            chrom, pos, rsid, ref, alt (access as ``info["chrom"]``) and
            ``dosage`` is a 1-D array of per-sample dosages (NaN for missing),
            length n_samples (after any sample filtering).
        """
        if block_size is not None and block_size < 1:
            raise ValueError("block_size must be a positive integer")
        _validate_dtype(dtype)
        validate_region_bounds(region_chrom, region_start, region_end)
        # Validate / coerce sample_indices at the public boundary before it is
        # cast to a raw C pointer downstream.
        sample_indices = self._validate_sample_indices(sample_indices)

        # Auto-scale block_size to a memory budget when not given explicitly, so
        # the per-block matrix (n_samples_out x block_size) stays bounded as the
        # sample count grows. Output is identical for any block_size.
        if block_size is None:
            n_out = len(sample_indices) if sample_indices is not None else self.header_info.n_samples
            itemsize = np.dtype(dtype).itemsize
            budget = 256 * 1024 * 1024
            block_size = max(1, min(4096, budget // max(1, n_out * itemsize)))

        cdef vector[VariantInfo] variant_infos = self._resolve_variant_infos(
            region_chrom, region_start, region_end, variant_filter
        )
        cdef Py_ssize_t total = variant_infos.size()
        cdef Py_ssize_t start, end, j

        for start in range(0, total, block_size):
            end = min(start + block_size, total)
            dosages, info = self._load_block_from_variant_infos(variant_infos, start, end, sample_indices, dtype)
            # Convert the block's variant_info to plain dicts once, instead of
            # building a pandas Series per variant via info.iloc[j] (which
            # dominated this loop). Dict subscript preserves info["chrom"] access.
            records = info.to_dict("records")
            for j in range(len(records)):
                yield records[j], np.ascontiguousarray(dosages[:, j])

    cdef vector[VariantInfo] _resolve_variant_infos(
        self, region_chrom, region_start, region_end, variant_filter
    ):
        """Resolve a variant selection to BGI index entries (no genotype decode).

        Uses only the BGI index, so it is cheap even for whole-file streaming.
        """
        self._ensure_open()
        cdef string cpp_chrom
        if variant_filter is not None:
            return self._filtered_variant_infos(variant_filter)
        elif region_chrom is not None:
            cpp_chrom = region_chrom.encode('utf-8')
            return self.bgi_reader.get().query_region(
                cpp_chrom, region_start or 0, region_end or 0xFFFFFFFF
            )
        else:
            return self.bgi_reader.get().get_all_variants()

    cdef void _tune_remote_block_size(self, list offsets) except *:
        """Size the remote readahead block to the selection's access pattern.

        Random-access remote reads over-fetch by one readahead block per seek.
        A large block is what we want for a dense/contiguous selection (it
        coalesces neighboring variants into shared range GETs), but it is pure
        waste for a scattered selection (each isolated seek drags a full block).
        Using the BGI offsets (known before any genotype read), pick a large
        block when variants are packed and a one-variant block when they are
        scattered. No-op for local readers; the C++ reader clamps the value.

        The policy lives in ``choose_remote_block_size`` (pure Python, unit
        tested); this wires it to the selection's offsets and the reader.
        """
        # Local readers ignore the block size, so skip the offset analysis
        # entirely (it would otherwise sort every selection for nothing).
        if not is_remote_path(self.file_path):
            return
        cdef unsigned long long block = choose_remote_block_size(
            self.header_info.n_samples, offsets
        )
        if block > 0:
            self.impl.get().set_read_block_size(<size_t>block)

    cdef vector[VariantMetadata] _metadata_from_variant_infos(self, vector[VariantInfo]& variant_infos):
        """Build decode-ready metadata from BGI entries (no file read).

        Identity (chrom/pos/rsid/alleles) comes from the BGI; the genotype is
        located in-buffer by the one-GET record read at decode time.
        """
        cdef const VariantInfo* ptr = NULL
        if variant_infos.size() > 0:
            ptr = &variant_infos[0]
        return self.impl.get().build_metadata_from_index(ptr, variant_infos.size())

    cdef tuple _load_block_from_variant_infos(self, vector[VariantInfo]& variant_infos,
                                              Py_ssize_t start, Py_ssize_t end,
                                              np.ndarray sample_indices, dtype):
        """Decode one block of variants [start, end) from BGI entries into dosages."""
        cdef size_t n = <size_t>(end - start)
        cdef const VariantInfo* ptr = NULL
        if n > 0:
            ptr = &variant_infos[start]
        cdef vector[VariantMetadata] block = self.impl.get().build_metadata_from_index(ptr, n)
        self._tune_remote_block_size(
            [variant_infos[k].file_offset for k in range(start, end)]
        )
        return self._load_variants_from_metadata(block, sample_indices, dtype, None)

    cdef vector[VariantInfo] _filtered_variant_infos(self, variant_filter) except *:
        """BGI lookup of variants matching a .z-style filter (in filter order)."""
        cdef vector[VariantInfo] variant_infos
        cdef string cpp_chrom = variant_filter["chromosome"].encode('utf-8')
        cdef vector[uint32_t] positions
        cdef vector[string] alleles1
        cdef vector[string] alleles2

        # Convert Python lists to C++ vectors
        for pos in variant_filter["positions"]:
            positions.push_back(pos)

        for a1 in variant_filter["allele1"]:
            alleles1.push_back(a1.encode('utf-8'))

        for a2 in variant_filter["allele2"]:
            alleles2.push_back(a2.encode('utf-8'))

        # Use the optimized find_variants_by_filter method
        variant_infos = self.bgi_reader.get().find_variants_by_filter(
            cpp_chrom, positions, alleles1, alleles2, 1000
        )
        return variant_infos

    cdef vector[VariantMetadata] _filtered_variant_metadata(self, variant_filter) except *:
        """Decode-ready metadata for a .z-style filter, plus remote block tuning.

        The metadata counterpart to ``_filtered_variant_infos``: it resolves the
        filter to BGI entries, sizes the remote readahead block to that (often
        scattered) selection, and returns decode-ready ``VariantMetadata``.
        """
        cdef vector[VariantInfo] variant_infos = self._filtered_variant_infos(variant_filter)

        # Size the remote readahead block to this (often scattered) selection
        # before any file read.
        self._tune_remote_block_size([info.file_offset for info in variant_infos])

        # Build decode-ready metadata straight from the BGI (no per-variant read).
        return self._metadata_from_variant_infos(variant_infos)

    cdef tuple _load_variants_from_metadata(
        self,
        vector[VariantMetadata]& variant_metadata,
        np.ndarray sample_indices,
        dtype,
        progress_callback
    ):
        """Load variant dosages from metadata."""
        # Cleared up front so a decode path that gathers no stats reports None
        # rather than an answer left over from a previous load.
        self.dosage_stats = None
        cdef int n_variants = variant_metadata.size()
        if n_variants == 0:
            n_samples = len(sample_indices) if sample_indices is not None else self.header_info.n_samples
            return np.empty((n_samples, 0), dtype=dtype), pd.DataFrame()
        
        # Determine output dimensions
        cdef int n_samples_out
        if sample_indices is not None:
            n_samples_out = len(sample_indices)
        else:
            n_samples_out = self.header_info.n_samples
        
        # OPTIMIZATION: Pre-allocate the entire dosage array at once
        # This avoids reallocation and improves memory locality
        dosages = np.empty((n_samples_out, n_variants), dtype=dtype, order='F')  # Fortran order for column-wise access

        # The genotype kernel decodes in single precision. Three output paths:
        #  - float32: parse directly into column i of the F-order buffer (its
        #    elements are contiguous), avoiding a per-variant allocation.
        #  - float64 unfiltered: parse directly into the float64 column via the
        #    double decode overload, which widens on store - this fuses away the
        #    separate float32->float64 cast that otherwise costs ~20% of the load.
        #  - everything else (e.g. float64 with a sample filter, or other dtypes):
        #    parse into a reused float32 scratch and cast once per column.
        cdef bint out_is_float32 = (np.dtype(dtype) == np.float32)
        cdef bint out_is_float64 = (np.dtype(dtype) == np.float64)
        # Parallel batched path: float32/float64 load with decompressor_type=
        # 'parallel'. Reads compressed bytes on this thread, then inflates +
        # decodes variants across worker threads writing into the F-order output
        # columns. Byte-identical to the serial loop; only the wall-clock changes.
        # Covers both the unfiltered (all-samples) and sample-filtered (cohort)
        # decode; other dtypes keep the serial path.
        cdef bint parallel_active = (
            (out_is_float32 or out_is_float64)
            and n_variants > 1
            and self.impl.get().decompressor_type() == b'parallel'
        )
        # Serial-loop discriminator: float64 all-samples decodes straight into the
        # float64 column via the double kernel (fuses away the f32->f64 cast).
        cdef bint fuse_f64 = out_is_float64 and (sample_indices is None)
        cdef np.ndarray[np.float32_t, ndim=2] f32_out = None
        cdef np.ndarray[np.float64_t, ndim=2] f64_out = None
        cdef np.ndarray[np.float32_t, ndim=1] scratch = None
        cdef float* f32_col_ptr
        cdef double* f64_col_ptr
        if out_is_float32:
            f32_out = dosages
        elif out_is_float64 and (fuse_f64 or parallel_active):
            # Direct float64 column: serial unfiltered fuse, or any parallel path
            # (the parallel filtered kernel widens its float result into out).
            f64_out = dosages
        else:
            # Serial float64 with a sample filter, or any non-f32/f64 dtype.
            scratch = np.empty(n_samples_out, dtype=np.float32)

        cdef size_t num_threads, out_stride, chunk_bytes
        cdef int chunk_start, chunk_end, chunk_n_variants, n_indices
        cdef const int* si_ptr = NULL
        # Range / missing-call summary, folded across chunks. The block decode
        # gathers it while each column is still in cache, which is what lets
        # load_bgen validate the result without scanning the whole matrix again.
        cdef const DosageStats* chunk_stats
        cdef double stats_min = float('inf')
        cdef double stats_max = float('-inf')
        cdef bint stats_has_nan = False
        # Cap on compressed bytes held in memory per parallel chunk. The C++ path
        # reads a whole chunk's compressed bytes up front (on this thread) before
        # the parallel inflate+decode, so chunking keeps peak memory bounded
        # regardless of how many variants are selected (e.g. 500K x 5000), and
        # lets progress fire per chunk. The output matrix itself is still O(all).
        cdef size_t chunk_byte_budget = <size_t>256 * 1024 * 1024
        if parallel_active:
            num_threads = self.impl.get().num_threads()
            out_stride = <size_t>n_samples_out
            if sample_indices is not None:
                si_ptr = <const int*>np.PyArray_DATA(sample_indices)
                n_indices = <int>len(sample_indices)
            chunk_start = 0
            while chunk_start < n_variants:
                # Grow the chunk until its compressed bytes reach the budget
                # (always at least one variant, even if a single block exceeds it).
                chunk_bytes = 0
                chunk_end = chunk_start
                while chunk_end < n_variants:
                    # variant_size (full record) is the BGI-sourced size; for
                    # BGI metadata genotype_length is unset (0) until decode, so
                    # use variant_size to bound the chunk's compressed bytes.
                    chunk_bytes += variant_metadata[chunk_end].variant_size
                    chunk_end += 1
                    if chunk_bytes >= chunk_byte_budget:
                        break
                chunk_n_variants = chunk_end - chunk_start
                if sample_indices is None:
                    if out_is_float32:
                        self.impl.get().read_decode_block_parallel(
                            &variant_metadata[chunk_start], <size_t>chunk_n_variants,
                            &f32_out[0, chunk_start], out_stride, num_threads)
                    else:  # fuse_f64
                        self.impl.get().read_decode_block_parallel(
                            &variant_metadata[chunk_start], <size_t>chunk_n_variants,
                            &f64_out[0, chunk_start], out_stride, num_threads)
                else:
                    if out_is_float32:
                        self.impl.get().read_decode_block_filtered_parallel(
                            &variant_metadata[chunk_start], <size_t>chunk_n_variants, si_ptr, n_indices,
                            &f32_out[0, chunk_start], out_stride, num_threads)
                    else:
                        self.impl.get().read_decode_block_filtered_parallel(
                            &variant_metadata[chunk_start], <size_t>chunk_n_variants, si_ptr, n_indices,
                            &f64_out[0, chunk_start], out_stride, num_threads)
                chunk_stats = &self.impl.get().last_block_stats()
                if chunk_stats.min_value < stats_min:
                    stats_min = chunk_stats.min_value
                if chunk_stats.max_value > stats_max:
                    stats_max = chunk_stats.max_value
                if chunk_stats.has_nan:
                    stats_has_nan = True
                if progress_callback is not None:
                    progress_callback(chunk_end)
                chunk_start = chunk_end
            # `bool` here is the C++ type from the .pxd, so build the Python
            # bool explicitly.
            self.dosage_stats = (stats_min, stats_max, True if stats_has_nan else False)
            variant_info = self._build_variant_info_frame(variant_metadata)
            return dosages, variant_info

        cdef int i, batch_start, batch_end
        # Process variants in batches for progress reporting cadence. (The previous
        # dynamic batch size only gated the progress callback; decoding is per-variant.)
        cdef int batch_size
        if n_variants < 1000:
            batch_size = 100
        elif n_variants < 10000:
            batch_size = 1000
        else:
            batch_size = 5000

        for batch_start in range(0, n_variants, batch_size):
            batch_end = min(batch_start + batch_size, n_variants)

            for i in range(batch_start, batch_end):
                if out_is_float32:
                    # Column i of an F-order array is contiguous; parse straight in.
                    f32_col_ptr = &f32_out[0, i]
                    self._parse_variant_into_f32(variant_metadata[i], sample_indices, f32_col_ptr)
                elif fuse_f64:
                    # Contiguous float64 column; decode straight in as double.
                    f64_col_ptr = &f64_out[0, i]
                    self._parse_variant_into_f64(variant_metadata[i], f64_col_ptr)
                else:
                    self._parse_variant_into_f32(
                        variant_metadata[i], sample_indices,
                        <float*>np.PyArray_DATA(scratch)
                    )
                    dosages[:, i] = scratch

                # Progress callback
                if progress_callback is not None:
                    progress_callback(i + 1)

        # Create variant info DataFrame
        variant_info = self._build_variant_info_frame(variant_metadata)

        return dosages, variant_info
    
    cdef void _parse_variant_into_f32(self, const VariantMetadata& metadata,
                                      np.ndarray sample_indices, float* out) except *:
        """Read, decompress, and parse one variant directly into ``out``.

        ``out`` must point to space for n_samples_out float32 values (the full
        sample count, or len(sample_indices) when filtering). ``sample_indices``,
        if not None, must already be a contiguous int32 array validated against
        n_samples (see _validate_sample_indices).
        """
        cdef uint32_t n_samples = self.header_info.n_samples
        cdef const uint32_t* sample_indices_ptr = NULL
        cdef uint32_t n_indices = 0

        if sample_indices is not None:
            sample_indices_ptr = <const uint32_t*>np.PyArray_DATA(sample_indices)
            n_indices = <uint32_t>len(sample_indices)

        cdef unique_ptr[DecompressedData] data_ptr
        cdef LayoutType layout_type = LayoutType_V11 if self.header_info.layout == 1 else LayoutType_V12

        # Get the decompressed data
        data_ptr = move(self.impl.get().read_variant_genotypes(metadata))

        if not data_ptr or not data_ptr.get().is_valid():
            error_msg = "Unknown error" if not data_ptr else data_ptr.get().error_message.decode('utf-8')
            raise RuntimeError(f"Failed to decompress variant at offset {metadata.file_offset}: {error_msg}")

        # Get parser buffer info
        cdef const uint8_t* parser_buffer = data_ptr.get().data()
        cdef size_t parser_size = data_ptr.get().size

        # Parse genotypes - data is already decompressed, so pass CompressionType.None
        if sample_indices_ptr != NULL:
            GenotypeParser.compute_dosages_filtered(
                parser_buffer,
                parser_size,
                layout_type,
                CompressionType_None,  # Data is already decompressed
                n_samples,
                metadata.n_alleles,
                <const int*>sample_indices_ptr,
                n_indices,
                out
            )
        else:
            GenotypeParser.compute_dosages_direct(
                parser_buffer,
                parser_size,
                layout_type,
                CompressionType_None,  # Data is already decompressed
                n_samples,
                metadata.n_alleles,
                out
            )

    cdef void _parse_variant_into_f64(self, const VariantMetadata& metadata,
                                      double* out) except *:
        """Decode one variant directly into a float64 buffer (unfiltered, all samples).

        Like _parse_variant_into but for the full (no sample filter) float64 case:
        the double compute_dosages_direct overload writes widened dosages straight
        into ``out`` (n_samples values), so there is no float32 scratch + cast.
        Output is byte-identical to the float32 decode widened to float64.
        """
        cdef uint32_t n_samples = self.header_info.n_samples
        cdef unique_ptr[DecompressedData] data_ptr
        cdef LayoutType layout_type = LayoutType_V11 if self.header_info.layout == 1 else LayoutType_V12

        data_ptr = move(self.impl.get().read_variant_genotypes(metadata))

        if not data_ptr or not data_ptr.get().is_valid():
            error_msg = "Unknown error" if not data_ptr else data_ptr.get().error_message.decode('utf-8')
            raise RuntimeError(f"Failed to decompress variant at offset {metadata.file_offset}: {error_msg}")

        cdef const uint8_t* parser_buffer = data_ptr.get().data()
        cdef size_t parser_size = data_ptr.get().size

        GenotypeParser.compute_dosages_direct(
            parser_buffer,
            parser_size,
            layout_type,
            CompressionType_None,  # Data is already decompressed
            n_samples,
            metadata.n_alleles,
            out
        )

    def _build_variant_info_frame(self, vector[VariantMetadata]& metadata) -> pd.DataFrame:
        """Build the variant-info DataFrame from metadata.

        Builds one Python list per column and constructs the DataFrame from a
        dict of columns, instead of a per-variant dict fed to
        ``pd.DataFrame(list_of_dicts)`` (which pandas reassembles into columns
        with per-row type inference). Same columns, order, and dtypes; cheaper
        for many variants.
        """
        cdef Py_ssize_t n = metadata.size()
        if n == 0:
            return pd.DataFrame()

        chroms = [None] * n
        positions = [0] * n
        rsids = [None] * n
        refs = [None] * n
        alts = [None] * n

        cdef VariantMetadata var
        cdef Py_ssize_t i
        for i in range(n):
            var = metadata[i]
            chroms[i] = var.chromosome.decode('utf-8')
            positions[i] = var.position
            rsids[i] = var.rsid.decode('utf-8')
            refs[i] = var.alleles[0].decode('utf-8') if var.alleles.size() > 0 else ''
            alts[i] = var.alleles[1].decode('utf-8') if var.alleles.size() > 1 else ''

        return pd.DataFrame({
            'chrom': chroms,
            'pos': positions,
            'rsid': rsids,
            'ref': refs,
            'alt': alts,
        })
    
    cdef void _ensure_open(self) except *:
        """Ensure reader is open."""
        if not self.is_open:
            raise ValueError("BGEN reader is closed")
    
    def close(self):
        """Close the BGEN reader."""
        if self.is_open:
            if self.impl:
                self.impl.get().close()
            self.is_open = False
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __dealloc__(self):
        """Cleanup."""
        self.close()
    
    # Properties
    @property
    def nsamples(self) -> int:
        """Number of samples."""
        return self.header_info.n_samples
    
    @property
    def nvariants(self) -> int:
        """Number of variants."""
        if self.bgi_reader:
            return self.bgi_reader.get().get_variant_count()
        return self.header_info.n_variants
    
    @property
    def last_dosage_stats(self):
        """Summary of the values the most recent decode wrote, or None.

        Returns ``(min, max, has_nan)``, where min and max ignore missing calls
        (so an all-missing result reports ``inf`` and ``-inf``) and has_nan says
        whether any missing call was written. The block decode collects this as
        it goes, at no meaningful cost, which saves callers a second pass over
        the result. It is None when the decode ran through the per-variant
        serial loop, which does not collect it; callers that need the answer
        must then compute it from the returned array.
        """
        return self.dosage_stats

    @property
    def samples(self) -> List[str]:
        """List of sample IDs (materialized on first access, then cached)."""
        if self.sample_ids is None:
            self._load_samples()
        return self.sample_ids
    
    @property
    def compression(self) -> str:
        """Compression type."""
        if self.header_info.compression == 0:
            return "none"
        elif self.header_info.compression == 1:
            return "zlib"
        elif self.header_info.compression == 2:
            return "zstd"
        else:
            return "unknown"
    
    @property
    def layout(self) -> int:
        """Layout version."""
        return self.header_info.layout
    
    def get_sample_indices(self, sample_ids: List[str]) -> Tuple[List[int], List[str]]:
        """
        Get indices of requested samples.
        
        Parameters
        ----------
        sample_ids : List[str]
            Sample IDs to find
        
        Returns
        -------
        Tuple[List[int], List[str]]
            (indices, found_sample_ids)
        """
        sample_map = {sid: i for i, sid in enumerate(self.samples)}
        indices = []
        found_ids = []
        
        for sid in sample_ids:
            if sid in sample_map:
                indices.append(sample_map[sid])
                found_ids.append(sid)
        
        return indices, found_ids
