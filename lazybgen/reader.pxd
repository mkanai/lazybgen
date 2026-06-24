# cython: language_level=3
# Cython declarations for new BGEN reader implementation

from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, int32_t
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr, shared_ptr
from libcpp.utility cimport move
from libcpp cimport bool
from libcpp.unordered_map cimport unordered_map

# Import numpy array API
cimport numpy as np

# Decompressed variant payload returned by BgenReaderImpl::read_variant_genotypes
cdef extern from "decompress/decompressor.h" namespace "lazybgen::bgen::decompress":
    cdef cppclass DecompressedData:
        uint8_t* data() const
        size_t size
        uint64_t offset
        bool success
        bool is_valid() const
        string error_message

# BGEN structures from C++
cdef extern from "bgen_reader_impl.h" namespace "lazybgen::io::bgen":
    cdef struct BgenHeader:
        uint32_t offset
        uint32_t n_variants
        uint32_t n_samples
        uint32_t flags
        uint8_t compression
        uint8_t layout
        bool has_sample_ids

cdef extern from "format/variant_parser.h" namespace "lazybgen::bgen":
    cdef struct VariantMetadata:
        uint64_t file_offset
        string varid
        string rsid
        string chromosome
        uint32_t position
        uint16_t n_alleles
        vector[string] alleles
        uint64_t genotype_offset
        uint32_t genotype_length
        uint32_t variant_size

# BGI variant info structure
cdef extern from "bgi_reader.h" namespace "lazybgen::io::bgen::index":
    cdef struct VariantInfo:
        uint64_t file_offset
        uint32_t variant_size
        string chromosome
        uint32_t position
        string rsid
        string varid
        uint16_t n_alleles
        string allele1
        string allele2

# BGI index reader
cdef extern from "bgi_reader.h" namespace "lazybgen::io::bgen::index":
    cdef cppclass BgiReader:
        BgiReader(const string& filename) except +

        # Query methods
        vector[VariantInfo] query_region(
            const string& chromosome, uint32_t start_pos, uint32_t end_pos) except +
        vector[VariantInfo] find_variants_by_filter(
            const string& chromosome,
            const vector[uint32_t]& positions,
            const vector[string]& alleles1,
            const vector[string]& alleles2,
            size_t batch_size) except +
        vector[VariantInfo] get_all_variants() except +

        # Index info
        size_t get_variant_count() except +

# Genotype parser
cdef extern from "format/genotype_parser.h" namespace "lazybgen::bgen":
    cdef enum LayoutType:
        LayoutType_V11 "lazybgen::bgen::LayoutType::V11"
        LayoutType_V12 "lazybgen::bgen::LayoutType::V12"
    
    cdef enum CompressionType:
        CompressionType_None "lazybgen::bgen::CompressionType::None"
        CompressionType_Zlib "lazybgen::bgen::CompressionType::Zlib"
        CompressionType_Zstd "lazybgen::bgen::CompressionType::Zstd"
        CompressionType_Unknown "lazybgen::bgen::CompressionType::Unknown"
    
    cdef cppclass GenotypeParser:
        @staticmethod
        void compute_dosages_filtered(
            const uint8_t* buffer,
            size_t size,
            LayoutType layout,
            CompressionType compression,
            uint32_t n_samples,
            uint16_t n_alleles,
            const int* sample_indices,
            int n_indices,
            float* output
        ) except +
        
        @staticmethod
        void compute_dosages_direct(
            const uint8_t* buffer,
            size_t size,
            LayoutType layout,
            CompressionType compression,
            uint32_t n_samples,
            uint16_t n_alleles,
            float* output
        ) except +

        @staticmethod
        void compute_dosages_direct(
            const uint8_t* buffer,
            size_t size,
            LayoutType layout,
            CompressionType compression,
            uint32_t n_samples,
            uint16_t n_alleles,
            double* output
        ) except +

# Main BGEN reader class
cdef extern from "bgen_reader_impl.h" namespace "lazybgen::io::bgen":
    cdef cppclass BgenReaderImpl:
        BgenReaderImpl(const string& filename, const string& bgi_filename, object storage_options) except +
        
        # Header access
        const BgenHeader& header() except +

        # Sample access
        vector[string] sample_ids() except +

        # Variant access
        vector[VariantMetadata] build_metadata_from_index(const VariantInfo* infos, size_t n) except +
        unique_ptr[DecompressedData] read_variant_genotypes(const VariantMetadata& metadata) except +

        # Batched parallel read + decode (unfiltered, all samples). Overloaded on
        # output precision; writes column-major into out (out_stride per variant).
        void read_decode_block_parallel(const VariantMetadata* block, size_t n_variants,
                                        float* out, size_t out_stride, size_t num_threads) except +
        void read_decode_block_parallel(const VariantMetadata* block, size_t n_variants,
                                        double* out, size_t out_stride, size_t num_threads) except +
        # Sample-filtered (cohort) parallel decode. Overloaded on output precision.
        void read_decode_block_filtered_parallel(const VariantMetadata* block, size_t n_variants,
                                                 const int* sample_indices, int n_indices,
                                                 float* out, size_t out_stride,
                                                 size_t num_threads) except +
        void read_decode_block_filtered_parallel(const VariantMetadata* block, size_t n_variants,
                                                 const int* sample_indices, int n_indices,
                                                 double* out, size_t out_stride,
                                                 size_t num_threads) except +

        # Decompressor configuration
        void set_decompressor_type(const string& type) except +
        string decompressor_type() except +
        size_t num_threads() except +
        void set_num_threads(size_t n) except +
        void set_read_block_size(size_t block_size) except +
        
        # File info
        bool is_open() except +
        void close() except +

# Cython wrapper classes
cdef class BgenReader:
    cdef unique_ptr[BgenReaderImpl] impl
    cdef unique_ptr[BgiReader] bgi_reader
    cdef BgenHeader header_info
    cdef list sample_ids
    cdef bool is_open
    cdef str file_path
    cdef str bgi_path
    cdef object storage_options

    # Private methods
    cdef void _init_reader(self) except *
    cdef void _load_samples_from_bgen(self) except *
    cdef void _ensure_open(self) except *
    cdef np.ndarray _validate_sample_indices(self, object sample_indices)
    cdef vector[VariantMetadata] _filtered_variant_metadata(self, variant_filter) except *
    cdef vector[VariantInfo] _filtered_variant_infos(self, variant_filter) except *
    cdef vector[VariantMetadata] _metadata_from_variant_infos(self, vector[VariantInfo]& variant_infos)
    cdef vector[VariantInfo] _resolve_variant_infos(self, region_chrom, region_start, region_end, variant_filter)
    cdef void _tune_remote_block_size(self, list offsets) except *
    cdef tuple _load_block_from_variant_infos(self, vector[VariantInfo]& variant_infos, Py_ssize_t start, Py_ssize_t end, np.ndarray sample_indices, dtype)
    cdef tuple _load_variants_from_metadata(
        self, vector[VariantMetadata]& metadata, np.ndarray sample_indices,
        dtype, progress_callback)
    cdef void _parse_variant_into_f32(self, const VariantMetadata& metadata,
                                      np.ndarray sample_indices, float* out) except *
    cdef void _parse_variant_into_f64(self, const VariantMetadata& metadata,
                                      double* out) except *