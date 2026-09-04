#ifndef LAZYBGEN_BGEN_READER_IMPL_H
#define LAZYBGEN_BGEN_READER_IMPL_H

#include <Python.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "decompress/decompressor.h"
#include "format/variant_parser.h"
#include "index/bgi_reader.h"
#include "io/reader_interface.h"

namespace lazybgen {
namespace io {
namespace bgen {

// Import VariantMetadata from the format namespace
using ::lazybgen::bgen::VariantMetadata;

// BGEN header structure
struct BgenHeader {
    uint32_t offset;
    uint32_t n_variants;
    uint32_t n_samples;
    uint32_t flags;
    uint8_t compression;
    uint8_t layout;
    bool has_sample_ids;
};

/**
 * BgenReaderImpl - Main implementation of BGEN file reader
 *
 * This class handles reading BGEN files, managing decompression,
 * and providing efficient access to genetic variants.
 */
/**
 * Summary of the dosage values a block decode wrote.
 *
 * min_value / max_value ignore NaN, so an all-NaN (or empty) block leaves them
 * at +inf / -inf and any range test on them is correctly a no-op. has_nan
 * reports whether any missing call was written.
 *
 * The decode touches every value while it is still hot in cache, so gathering
 * this there costs almost nothing, whereas scanning the finished matrix for the
 * same answers is gigabytes of DRAM traffic per read at biobank scale.
 */
struct DosageStats {
    // Defaulted so a value-initialized instance (an unwritten slot in the
    // per-variant array a block decode fills) is the identity for the fold
    // rather than a zero that would clamp the range to include 0.
    double min_value = std::numeric_limits<double>::infinity();
    double max_value = -std::numeric_limits<double>::infinity();
    bool has_nan = false;
};

class BgenReaderImpl {
   public:
    /**
     * Constructor
     *
     * @param filename Path to BGEN file
     * @param bgi_filename Path to BGI index file
     * @throws std::runtime_error if files cannot be opened
     */
    BgenReaderImpl(const std::string& filename, const std::string& bgi_filename,
                   PyObject* storage_options = nullptr);

    /**
     * Destructor
     */
    ~BgenReaderImpl();

    /**
     * Get BGEN header information
     *
     * @return Reference to header structure
     */
    const BgenHeader& header() const;

    /**
     * Get sample IDs from BGEN file
     *
     * Returned by reference: at biobank scale this vector holds hundreds of
     * thousands of strings, so callers must not pay for a copy they do not
     * want. A file with no sample block has its placeholder IDs built on the
     * first call rather than at open.
     *
     * @return Reference to the vector of sample IDs
     */
    const std::vector<std::string>& sample_ids();

    /**
     * Stats for the values written by the most recent block decode
     *
     * Only the read_decode_block_* entry points set this; it is meaningless
     * before the first such call.
     *
     * @return Reference to the last block's dosage stats
     */
    const DosageStats& last_block_stats() const;

    /**
     * Build decode-ready variant metadata from BGI index entries, without
     * reading the BGEN file. The genotype location is left unset; variant_size
     * carries the BGI size_in_bytes so the decode path reads the whole record in
     * a single range request (1 GET/variant) and locates the genotype in-buffer.
     * Identity fields (chrom/pos/rsid/alleles) are sourced from the BGI.
     *
     * @param infos Pointer to a contiguous array of BGI index entries
     * @param n Number of entries
     * @return Decode-ready metadata, in input order
     */
    std::vector<VariantMetadata> build_metadata_from_index(const index::VariantInfo* infos,
                                                           size_t n);

    /**
     * Read and decompress variant genotype data
     *
     * @param metadata Variant metadata
     * @return Unique pointer to decompressed genotype data
     */
    std::unique_ptr<::lazybgen::bgen::decompress::DecompressedData> read_variant_genotypes(
        const VariantMetadata& metadata);

    /**
     * Batched parallel read + decode of an unfiltered (all-samples) block of
     * variants into a column-major (Fortran-order) output buffer. Variant i is
     * written to out[i * out_stride .. i * out_stride + n_samples). Compressed
     * bytes are read on the calling thread; inflate + decode run across
     * num_threads worker threads (0 = auto-detect). Overloaded on output
     * precision (float / double) to match GenotypeParser::compute_dosages_direct.
     *
     * @param block Pointer to a contiguous array of variant metadata
     * @param n_variants Number of variants in the block
     * @param out Output buffer (column-major; out_stride elements per variant)
     * @param out_stride Elements between consecutive variant columns (n_samples)
     * @param num_threads Worker threads (0 = hardware_concurrency)
     */
    void read_decode_block_parallel(const VariantMetadata* block, size_t n_variants, float* out,
                                    size_t out_stride, size_t num_threads);
    void read_decode_block_parallel(const VariantMetadata* block, size_t n_variants, double* out,
                                    size_t out_stride, size_t num_threads);

    /**
     * Sample-filtered (cohort) counterpart of read_decode_block_parallel. Each
     * variant's full block is still inflated (decompression is independent of
     * cohort size), but only the n_indices requested samples are decoded, via
     * the filtered SIMD kernel. Variant i is written to
     * out[i * out_stride .. i * out_stride + n_indices). out_stride = n_indices.
     * Overloaded on output precision (float / double).
     */
    void read_decode_block_filtered_parallel(const VariantMetadata* block, size_t n_variants,
                                             const int* sample_indices, int n_indices, float* out,
                                             size_t out_stride, size_t num_threads);
    void read_decode_block_filtered_parallel(const VariantMetadata* block, size_t n_variants,
                                             const int* sample_indices, int n_indices, double* out,
                                             size_t out_stride, size_t num_threads);

    /**
     * Number of worker threads used by the parallel block decode path
     * (0 = auto-detect).
     */
    size_t num_threads() const;

    /**
     * Set the decode routing flag.
     *
     * @param type "sequential" (per-variant loop) or "parallel" (block decode
     *             across worker threads). The stored decompressor is always
     *             sequential; this flag selects the multi-variant decode path.
     */
    void set_decompressor_type(const std::string& type);

    /**
     * Get the active decode routing flag
     *
     * @return "sequential" or "parallel"
     */
    std::string decompressor_type() const;

    /**
     * Set the worker-thread count used by the parallel block decode path
     *
     * @param n Number of threads (0 = auto-detect)
     */
    void set_num_threads(size_t n);

    /**
     * Size the remote readahead block (in bytes) to the access pattern.
     *
     * For remote (fsspec) readers this controls how much each random-access
     * read fetches: a large block coalesces neighboring variants on dense
     * selections, a one-variant block avoids over-fetching on scattered ones.
     * No-op for local readers. The reader clamps to a sane range.
     *
     * @param block_size Suggested readahead block size in bytes
     */
    void set_read_block_size(size_t block_size);

    /**
     * Check if file is open
     *
     * @return true if file is open
     */
    bool is_open() const;

    /**
     * Close the file
     */
    void close();

   private:
    // Implementation details hidden with pimpl
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

}  // namespace bgen
}  // namespace io
}  // namespace lazybgen

#endif  // LAZYBGEN_BGEN_READER_IMPL_H