#include "bgen_reader_impl.h"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include "format/bgen_header.h"
#include "format/genotype_parser.h"
#include "format/variant_parser.h"
#include "index/bgi_reader.h"
#include "io/fsspec_file_reader.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <thread>

#include "decompress/compression_utils.h"
#include "decompress/decompressor_factory.h"

namespace lazybgen {
namespace io {
namespace bgen {

namespace {
bool is_remote_scheme(const std::string& f) {
    return f.rfind("gs://", 0) == 0 || f.rfind("s3://", 0) == 0;
}
}  // namespace

// Using directives for nested namespaces
using index::BgiReader;
using ::lazybgen::bgen::BgenHeaderParser;
using ::lazybgen::bgen::CompressionType;
using ::lazybgen::bgen::LayoutType;
using ::lazybgen::bgen::SampleBlockParser;
using ::lazybgen::bgen::VariantMetadata;
using ::lazybgen::bgen::VariantParser;

// Decompress namespace
namespace decompress = ::lazybgen::bgen::decompress;

namespace {

// Filtered (sample-subset) decode of one already-decompressed variant into an
// output column. Overloaded on output precision so the parallel filtered path
// works for both dtypes: GenotypeParser::compute_dosages_filtered is float-only,
// so the float overload writes straight into the column, while the double
// overload decodes into a thread-local float scratch and widens (matching the
// serial float32-scratch+cast path; output is its exact widening).
inline void filtered_decode_into(float* out_col, const uint8_t* buf, size_t size,
                                 ::lazybgen::bgen::LayoutType layout, uint32_t n_samples,
                                 uint16_t n_alleles, const int* sample_indices, int n_indices) {
    ::lazybgen::bgen::GenotypeParser::compute_dosages_filtered(
        buf, size, layout, ::lazybgen::bgen::CompressionType::None, n_samples, n_alleles,
        sample_indices, n_indices, out_col);
}

inline void filtered_decode_into(double* out_col, const uint8_t* buf, size_t size,
                                 ::lazybgen::bgen::LayoutType layout, uint32_t n_samples,
                                 uint16_t n_alleles, const int* sample_indices, int n_indices) {
    thread_local std::vector<float> scratch;
    if (static_cast<int>(scratch.size()) < n_indices) {
        scratch.resize(n_indices);
    }
    ::lazybgen::bgen::GenotypeParser::compute_dosages_filtered(
        buf, size, layout, ::lazybgen::bgen::CompressionType::None, n_samples, n_alleles,
        sample_indices, n_indices, scratch.data());
    for (int k = 0; k < n_indices; ++k) {
        out_col[k] = static_cast<double>(scratch[k]);
    }
}

}  // namespace

// Regular file reader implementation
class RegularFileReader : public FileReader {
   public:
    explicit RegularFileReader(const std::string& filename) : filename_(filename), file_size_(0) {
        file_.open(filename, std::ios::binary);
        if (!file_.is_open()) {
            throw std::runtime_error("Failed to open file: " + filename);
        }

        // Get file size
        file_.seekg(0, std::ios::end);
        file_size_ = file_.tellg();
        file_.seekg(0, std::ios::beg);
    }

    ~RegularFileReader() override {
        close();
    }

    size_t read(uint8_t* buffer, size_t size) override {
        if (!file_.is_open()) {
            return 0;
        }
        file_.read(reinterpret_cast<char*>(buffer), size);
        return file_.gcount();
    }

    size_t read_at(uint64_t offset, uint8_t* buffer, size_t size) override {
        if (!file_.is_open()) {
            return 0;
        }
        auto current_pos = file_.tellg();
        file_.seekg(offset);
        size_t bytes_read = read(buffer, size);
        file_.seekg(current_pos);
        return bytes_read;
    }

    void seek(uint64_t offset) override {
        if (file_.is_open()) {
            file_.seekg(offset);
        }
    }

    uint64_t tell() const override {
        if (!file_.is_open()) {
            return 0;
        }
        return file_.tellg();
    }

    uint64_t size() const override {
        return file_size_;
    }

    bool is_open() const override {
        return file_.is_open();
    }

    void close() override {
        if (file_.is_open()) {
            file_.close();
        }
    }

    const std::string& filename() const override {
        return filename_;
    }

   private:
    std::string filename_;
    mutable std::ifstream file_;
    uint64_t file_size_;
};

// Memory-mapped file reader implementation
class MMapFileReader : public FileReader {
   public:
    explicit MMapFileReader(const std::string& filename)
        : filename_(filename), data_(nullptr), file_size_(0), current_pos_(0), fd_(-1) {
        // Open file
        fd_ = ::open(filename.c_str(), O_RDONLY);
        if (fd_ < 0) {
            throw std::runtime_error("Failed to open file: " + filename);
        }

        // Get file size
        struct stat st;
        if (fstat(fd_, &st) < 0) {
            ::close(fd_);
            throw std::runtime_error("Failed to stat file: " + filename);
        }
        file_size_ = st.st_size;

        // Memory map the file
        data_ = static_cast<uint8_t*>(mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0));
        if (data_ == MAP_FAILED) {
            ::close(fd_);
            throw std::runtime_error("Failed to mmap file: " + filename);
        }

        // Let the kernel adapt its readahead to the actual access pattern
        // (MADV_NORMAL): it ramps up for sequential scans (full-file / large
        // contiguous region decodes) and stays modest for indexed lookups. The
        // previous MADV_RANDOM disabled readahead entirely, which is fine for
        // scattered point reads but makes a cold full/region scan fault one page
        // at a time off disk (an order of magnitude slower than warm). Local
        // readahead over-fetch is nearly free (page cache, fast disk), unlike the
        // remote byte-range path, which sizes its own block (choose_remote_block_size).
        madvise(data_, file_size_, MADV_NORMAL);
    }

    ~MMapFileReader() override {
        close();
    }

    size_t read(uint8_t* buffer, size_t size) override {
        if (!data_ || current_pos_ >= file_size_) {
            return 0;
        }

        size_t bytes_to_read = std::min(size, static_cast<size_t>(file_size_ - current_pos_));
        memcpy(buffer, data_ + current_pos_, bytes_to_read);
        current_pos_ += bytes_to_read;
        return bytes_to_read;
    }

    size_t read_at(uint64_t offset, uint8_t* buffer, size_t size) override {
        if (!data_ || offset >= file_size_) {
            return 0;
        }

        size_t bytes_to_read = std::min(size, static_cast<size_t>(file_size_ - offset));
        memcpy(buffer, data_ + offset, bytes_to_read);
        return bytes_to_read;
    }

    // The whole file is already mapped, so any in-bounds range can be handed
    // out in place. Out-of-bounds ranges return nullptr and the caller falls
    // back to read_at, which clamps to EOF and reports the short read.
    const uint8_t* view_at(uint64_t offset, size_t size) const override {
        if (!data_ || offset > file_size_ || size > file_size_ - offset) {
            return nullptr;
        }
        return data_ + offset;
    }

    void seek(uint64_t offset) override {
        current_pos_ = std::min(offset, file_size_);
    }

    uint64_t tell() const override {
        return current_pos_;
    }

    uint64_t size() const override {
        return file_size_;
    }

    bool is_open() const override {
        return data_ != nullptr;
    }

    void close() override {
        if (data_ && data_ != MAP_FAILED) {
            munmap(data_, file_size_);
            data_ = nullptr;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

    const std::string& filename() const override {
        return filename_;
    }

   private:
    std::string filename_;
    uint8_t* data_;
    uint64_t file_size_;
    uint64_t current_pos_;
    int fd_;
};

// Implementation class (pimpl idiom)
class BgenReaderImpl::Impl {
   public:
    // Constructor
    Impl(const std::string& filename, const std::string& bgi_filename,
         PyObject* storage_options = nullptr)
        : filename_(filename),
          bgi_filename_(bgi_filename),
          file_reader_(),
          bgi_reader_(),
          decompressor_(),
          header_(),
          sample_ids_(),
          is_open_(false),
          num_threads_(0),
          decompressor_type_("sequential") {
        try {
            // Open the BGEN file
            open_file(storage_options);

            // Read and parse header
            parse_header();

            // Open BGI index
            open_index();

            // Read sample IDs if present
            read_sample_ids();

            // Create default decompressor (sequential)
            create_default_decompressor();

            is_open_ = true;
        } catch (const std::exception& e) {
            // Clean up on error
            close();
            throw;
        }
    }

    // Destructor
    ~Impl() {
        close();
    }

    // Get header
    const BgenHeader& header() const {
        return header_;
    }

    // Get sample IDs. A file with no sample block carries placeholder IDs, which
    // are built here on demand: generating them at open costs one string per
    // sample (hundreds of thousands at biobank scale) for a list most reads
    // never look at, and callers that supply their own .sample file discard
    // them outright.
    const std::vector<std::string>& sample_ids() {
        if (sample_ids_.empty() && !header_.has_sample_ids && header_.n_samples > 0) {
            sample_ids_.reserve(header_.n_samples);
            for (uint32_t i = 0; i < header_.n_samples; ++i) {
                sample_ids_.push_back("sample_" + std::to_string(i));
            }
        }
        return sample_ids_;
    }

    // Build decode-ready metadata from BGI index entries without touching the
    // file. The genotype location (genotype_offset / genotype_length) is left
    // unset (0); variant_size carries the BGI size_in_bytes so the decode path
    // reads the whole record in a single range request (1 GET/variant) and parses
    // the ID block in-buffer to locate the genotype. Identity fields
    // (chrom/pos/rsid/alleles) are sourced from the BGI, not the bgen record.
    std::vector<VariantMetadata> build_metadata_from_index(const index::VariantInfo* infos,
                                                           size_t n) {
        std::vector<VariantMetadata> out;
        out.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            const index::VariantInfo& info = infos[i];
            // The reader trusts the BGI; only a zero record size is unusable
            // (the one-GET read would fetch nothing), so guard against just that.
            if (info.variant_size == 0) {
                throw std::runtime_error("BGI entry has zero size_in_bytes at file offset " +
                                         std::to_string(info.file_offset));
            }
            VariantMetadata md;
            md.file_offset = info.file_offset;
            md.variant_size = info.variant_size;
            md.varid = info.varid;
            md.rsid = info.rsid;
            md.chromosome = info.chromosome;
            md.position = info.position;
            md.n_alleles = info.n_alleles;
            md.alleles.reserve(2);
            md.alleles.push_back(info.allele1);
            md.alleles.push_back(info.allele2);
            md.genotype_offset = 0;
            md.genotype_length = 0;
            out.push_back(std::move(md));
        }
        return out;
    }

    // Locate the genotype payload to decompress within [gdata, gdata + glen) and
    // compute its declared uncompressed size. For v1.2 this strips the 4-byte D
    // field (the uncompressed length) and validates it against a per-sample
    // bound; for v1.1 the uncompressed size is fixed at n_samples * 6.
    // Uncompressed BGEN is rejected. All size arithmetic is done in size_t so the
    // byte reconstruction and the bound cannot overflow on a crafted header. The
    // single source of truth for the v1.2 prologue, shared by the per-variant and
    // batched block paths.
    void locate_genotype_payload(const uint8_t* gdata, size_t glen,
                                 decompress::CompressionType comp_type, const uint8_t*& out_ptr,
                                 size_t& out_compressed, size_t& out_uncompressed) const {
        if (comp_type == decompress::CompressionType::None) {
            throw std::runtime_error(
                "Uncompressed BGEN files are not supported. "
                "Please use compressed BGEN files (zlib or zstd). "
                "You can compress your BGEN file using bgenix or qctool2.");
        }
        const uint8_t* dp = gdata;
        size_t compressed_size = glen;
        size_t uncompressed_size = 0;
        if (header_.layout == 2) {  // v1.2: block starts with the 4-byte D field.
            if (compressed_size < 4) {
                throw std::runtime_error("Invalid compressed genotype data size for v1.2 variant");
            }
            uncompressed_size = static_cast<size_t>(dp[0]) | (static_cast<size_t>(dp[1]) << 8) |
                                (static_cast<size_t>(dp[2]) << 16) |
                                (static_cast<size_t>(dp[3]) << 24);
            size_t max_expected_size = static_cast<size_t>(header_.n_samples) * 10 + 1000;
            if (uncompressed_size > max_expected_size || uncompressed_size == 0) {
                throw std::runtime_error(
                    "Invalid uncompressed size in BGEN file: " + std::to_string(uncompressed_size) +
                    " bytes. Expected at most " + std::to_string(max_expected_size) +
                    " bytes for " + std::to_string(header_.n_samples) + " samples.");
            }
            dp += 4;
            compressed_size -= 4;
        } else {  // v1.1
            uncompressed_size = static_cast<size_t>(header_.n_samples) * 6;  // 3 * 2 bytes/sample
        }
        out_ptr = dp;
        out_compressed = compressed_size;
        out_uncompressed = uncompressed_size;
    }

    // Locate + decompress a single variant's genotype block out of a buffer that
    // holds at least [gdata, gdata + glen). Backs the per-variant read path.
    std::unique_ptr<decompress::DecompressedData> decompress_genotype_block(
        const uint8_t* gdata, size_t glen, uint64_t file_offset, const std::string& varid) {
        auto comp_type = static_cast<decompress::CompressionType>(header_.compression);
        const uint8_t* data_ptr;
        size_t compressed_size;
        size_t uncompressed_size;
        locate_genotype_payload(gdata, glen, comp_type, data_ptr, compressed_size,
                                uncompressed_size);

        decompress::CompressedVariant comp_variant(file_offset, data_ptr, compressed_size,
                                                   uncompressed_size, comp_type);
        comp_variant.variant_id = varid;

        auto result = decompressor_->decompress(comp_variant);
        return std::unique_ptr<decompress::DecompressedData>(
            new decompress::DecompressedData(std::move(result)));
    }

    // Read variant genotypes. Metadata is BGI-sourced (variant_size known), so
    // read the whole record in one range request and parse the ID block in-buffer
    // to locate the genotype - a scattered remote read costs 1 GET/variant.
    std::unique_ptr<decompress::DecompressedData> read_variant_genotypes(
        const VariantMetadata& metadata) {
        if (!is_open_) {
            throw std::runtime_error("BGEN file is not open");
        }
        // A memory-mapped reader hands the record back in place; every other
        // reader fills a buffer we own.
        std::vector<uint8_t> buffer;
        const size_t got = metadata.variant_size;
        const uint8_t* record = file_reader_->view_at(metadata.file_offset, metadata.variant_size);
        if (!record) {
            buffer.resize(metadata.variant_size);
            if (file_reader_->read_at(metadata.file_offset, buffer.data(), metadata.variant_size) !=
                got) {
                // BGI-sourced metadata carries no varid, so identify the record
                // by where it starts as well.
                throw std::runtime_error("Failed to read complete variant record for variant " +
                                         metadata.varid + " at offset " +
                                         std::to_string(metadata.file_offset));
            }
            record = buffer.data();
        }

        auto layout_type = static_cast<LayoutType>(header_.layout);
        auto compression_type = static_cast<CompressionType>(header_.compression);
        auto parse_result =
            VariantParser::parse(record, got, layout_type, compression_type, header_.n_samples);
        size_t goff = parse_result.first.genotype_offset;   // relative to record start
        size_t glen = parse_result.first.genotype_length;
        // Guard against a BGI size_in_bytes that understates the record (would
        // otherwise read past the buffer).
        if (goff + glen > got) {
            throw std::runtime_error("Variant record smaller than BGI size_in_bytes for variant " +
                                     metadata.varid);
        }
        return decompress_genotype_block(record + goff, glen, metadata.file_offset,
                                         metadata.varid);
    }

    // Range and missing-call summary for one decoded column, gathered while the
    // values are still in cache. NaN compares false against everything, so it
    // never moves the bounds; it is picked up by the self-comparison instead.
    template <typename T>
    static DosageStats scan_dosages_scalar(const T* values, size_t n) {
        double lo = std::numeric_limits<double>::infinity();
        double hi = -std::numeric_limits<double>::infinity();
        bool nan_seen = false;
        for (size_t k = 0; k < n; ++k) {
            const double v = static_cast<double>(values[k]);
            nan_seen |= (v != v);
            if (v < lo) {
                lo = v;
            }
            if (v > hi) {
                hi = v;
            }
        }
        return DosageStats{lo, hi, nan_seen};
    }

    template <typename T>
    static DosageStats scan_dosages(const T* values, size_t n) {
        return scan_dosages_scalar(values, n);
    }

#if defined(__AVX2__)
    // The scalar loop above will not auto-vectorize, and cannot be made to:
    // vminpd returns its second operand when either is NaN, so a NaN would
    // poison the accumulator, and the compiler will not reorder around that.
    // std::fmin / std::fmax have the semantics we want but compile to calls and
    // measure three times slower still. Replacing NaN lanes with the identity
    // before the vector min/max makes the vector form exact, and it runs 2.6x
    // faster than the scalar loop on this scan, which is 5-12% of a decode.
    //
    // Only the double column is vectorized. It is the default output type and
    // the one the measurements above are for; a float column is half the bytes,
    // so it keeps the scalar loop rather than a second copy of this.
    // A non-template overload, which wins over the template above for an exact
    // double match. An explicit specialization is not allowed at class scope.
    static DosageStats scan_dosages(const double* values, size_t n) {
        const __m256d pos_inf = _mm256_set1_pd(std::numeric_limits<double>::infinity());
        const __m256d neg_inf = _mm256_set1_pd(-std::numeric_limits<double>::infinity());
        const __m256d all_ones = _mm256_castsi256_pd(_mm256_set1_epi32(-1));
        __m256d lo_v = pos_inf;
        __m256d hi_v = neg_inf;
        __m256d nan_v = _mm256_setzero_pd();

        size_t k = 0;
        for (; k + 4 <= n; k += 4) {
            const __m256d v = _mm256_loadu_pd(values + k);
            // Lanes that are not NaN compare ordered against themselves.
            const __m256d ordered = _mm256_cmp_pd(v, v, _CMP_ORD_Q);
            nan_v = _mm256_or_pd(nan_v, _mm256_andnot_pd(ordered, all_ones));
            lo_v = _mm256_min_pd(lo_v, _mm256_blendv_pd(pos_inf, v, ordered));
            hi_v = _mm256_max_pd(hi_v, _mm256_blendv_pd(neg_inf, v, ordered));
        }

        double lo_lanes[4];
        double hi_lanes[4];
        _mm256_storeu_pd(lo_lanes, lo_v);
        _mm256_storeu_pd(hi_lanes, hi_v);
        double lo = lo_lanes[0];
        double hi = hi_lanes[0];
        for (int lane = 1; lane < 4; ++lane) {
            if (lo_lanes[lane] < lo) {
                lo = lo_lanes[lane];
            }
            if (hi_lanes[lane] > hi) {
                hi = hi_lanes[lane];
            }
        }
        bool nan_seen = _mm256_movemask_pd(nan_v) != 0;

        // Tail, and the whole column when it is shorter than one vector.
        for (; k < n; ++k) {
            const double v = values[k];
            nan_seen |= (v != v);
            if (v < lo) {
                lo = v;
            }
            if (v > hi) {
                hi = v;
            }
        }
        return DosageStats{lo, hi, nan_seen};
    }
#endif  // __AVX2__

    // Fold per-variant summaries into one. DosageStats defaults to +inf/-inf,
    // so an entry no worker wrote leaves the combined bounds unchanged instead
    // of dragging them toward zero.
    static DosageStats combine_dosage_stats(const std::vector<DosageStats>& parts) {
        DosageStats out{std::numeric_limits<double>::infinity(),
                        -std::numeric_limits<double>::infinity(), false};
        for (const DosageStats& p : parts) {
            if (p.min_value < out.min_value) {
                out.min_value = p.min_value;
            }
            if (p.max_value > out.max_value) {
                out.max_value = p.max_value;
            }
            out.has_nan = out.has_nan || p.has_nan;
        }
        return out;
    }

    // One variant's compressed payload, located on the main thread in Phase 1.
    struct CompressedJob {
        // Owned copy of the record, used only when the reader cannot hand out a
        // view; left empty when compressed_ptr points into a mapping instead.
        std::vector<uint8_t> raw;
        const uint8_t* compressed_ptr;      // compressed payload (D field stripped)
        size_t compressed_size;
        size_t uncompressed_size;  // decompressed size
        uint64_t offset;
        uint16_t n_alleles;
    };

    // Phase 1 (main thread): locate each variant's compressed payload and its
    // decoded size. All file I/O happens here, so the GIL (for the Python-backed
    // fsspec reader) and the non-thread-safe RegularFileReader stream are only
    // touched on the calling thread; the parallel phase below never does I/O.
    // A memory-mapped reader can point at the record in place, so for a local
    // file this phase copies nothing and the pages are first touched by the
    // worker threads instead.
    std::vector<CompressedJob> read_compressed_block(const VariantMetadata* block,
                                                     size_t n_variants,
                                                     decompress::CompressionType comp_type) {
        const uint32_t n_samples = header_.n_samples;
        const auto layout = static_cast<LayoutType>(header_.layout);
        // VariantParser::parse expects the format-layer CompressionType (distinct
        // from the decompress:: one this method is parameterized on).
        const auto fmt_comp = static_cast<CompressionType>(header_.compression);
        std::vector<CompressedJob> jobs(n_variants);

        // One-GET path: take each whole record (variant_size bytes at
        // file_offset) and parse the ID block in-buffer to locate the genotype
        // slice, so a scattered remote read costs 1 GET/variant.
        //
        // Records the reader can show us in place cost nothing; the rest are
        // fetched in a single read_many so a remote store has them all in flight
        // at once instead of paying one round trip after another.
        std::vector<const uint8_t*> records(n_variants, nullptr);
        std::vector<uint64_t> fetch_offsets;
        std::vector<size_t> fetch_sizes;
        std::vector<uint8_t*> fetch_buffers;
        std::vector<size_t> fetch_indices;
        for (size_t i = 0; i < n_variants; ++i) {
            const VariantMetadata& md = block[i];
            records[i] = file_reader_->view_at(md.file_offset, md.variant_size);
            if (!records[i]) {
                jobs[i].raw.resize(md.variant_size);
                fetch_offsets.push_back(md.file_offset);
                fetch_sizes.push_back(md.variant_size);
                fetch_buffers.push_back(jobs[i].raw.data());
                fetch_indices.push_back(i);
            }
        }
        if (!fetch_indices.empty()) {
            std::vector<size_t> bytes_read(fetch_indices.size(), 0);
            file_reader_->read_many(fetch_offsets.data(), fetch_sizes.data(), fetch_buffers.data(),
                                    bytes_read.data(), fetch_indices.size());
            for (size_t k = 0; k < fetch_indices.size(); ++k) {
                const size_t i = fetch_indices[k];
                if (bytes_read[k] != fetch_sizes[k]) {
                    // BGI-sourced metadata carries no varid, so identify the
                    // record by where it starts as well.
                    throw std::runtime_error("Failed to read complete variant record for variant " +
                                             block[i].varid + " at offset " +
                                             std::to_string(block[i].file_offset));
                }
                records[i] = jobs[i].raw.data();
            }
        }

        for (size_t i = 0; i < n_variants; ++i) {
            const VariantMetadata& md = block[i];
            CompressedJob& j = jobs[i];
            const size_t got = md.variant_size;
            const uint8_t* record = records[i];
            auto pr = VariantParser::parse(record, got, layout, fmt_comp, n_samples);
            size_t goff = pr.first.genotype_offset;
            size_t glen = pr.first.genotype_length;
            if (goff + glen > got) {
                throw std::runtime_error(
                    "Variant record smaller than BGI size_in_bytes for variant " + md.varid);
            }
            const uint8_t* dp;
            size_t compressed_size;
            size_t uncompressed_size;
            locate_genotype_payload(record + goff, glen, comp_type, dp, compressed_size,
                                    uncompressed_size);
            j.compressed_ptr = dp;
            j.compressed_size = compressed_size;
            j.uncompressed_size = uncompressed_size;
            j.offset = md.file_offset;
            j.n_alleles = md.n_alleles;
        }
        return jobs;
    }

    // Phase 2 (worker threads): pure-CPU inflate of each job into a thread-local
    // buffer, then call `decode(i, decompressed_ptr, decompressed_size, job)` to
    // write variant i's dosages. No file I/O, no Python, no shared mutable state,
    // so it is safe to run with the GIL held; worker threads never call into the
    // interpreter. The first decode/inflate error is captured and rethrown here.
    template <typename DecodeFn>
    void parallel_inflate_decode(std::vector<CompressedJob>& jobs,
                                 decompress::CompressionType comp_type, size_t num_threads,
                                 DecodeFn decode, size_t column_bytes = 0) {
        const size_t n_variants = jobs.size();
        if (n_variants == 0) {
            return;
        }
        size_t hw = num_threads ? num_threads : std::thread::hardware_concurrency();
        if (hw == 0) {
            hw = 1;
        }
        if (hw > n_variants) {
            hw = n_variants;
        }

        // Workers claim runs of adjacent variants rather than one at a time.
        // The output is a fresh column-major matrix whose pages the kernel maps
        // on first touch. Claiming one variant at a time puts every worker on
        // adjacent columns at the same moment, so they all first-touch the same
        // region and serialize in the kernel: page-table locking, made worse by
        // the transparent huge pages NumPy asks for on large arrays. The decode
        // then runs no faster than a single-threaded write of its own output.
        // Giving each worker a run of columns spreads those first touches out.
        //
        // The run targets a few megabytes of output. That is tuned, not derived:
        // throughput is flat between roughly 2 and 8 MB and falls off past about
        // 16 MB as the tail of the work stops dividing evenly. Runs are capped
        // so every worker still gets several claims, which bounds how unevenly
        // the work can land. At biobank sample counts a single column already
        // exceeds the target, so the run collapses to one variant and this is
        // exactly the old path. column_bytes == 0 (a caller that does not know
        // its output geometry) also keeps one-at-a-time claiming.
        size_t chunk = 1;
        if (column_bytes > 0) {
            const size_t run_target_bytes = static_cast<size_t>(4) << 20;
            const size_t per_run = (run_target_bytes + column_bytes - 1) / column_bytes;
            const size_t claims_per_worker = n_variants / (hw * 4);
            chunk = std::max<size_t>(1, std::min(per_run, claims_per_worker));
        }

        std::atomic<size_t> next{0};
        std::atomic<bool> failed{false};
        std::mutex err_mu;
        std::string err_msg;

        auto worker = [&]() {
            std::vector<uint8_t> decomp;
            size_t run_start;
            while ((run_start = next.fetch_add(chunk)) < n_variants) {
              const size_t run_end = std::min(n_variants, run_start + chunk);
              for (size_t i = run_start; i < run_end; ++i) {
                if (failed.load(std::memory_order_relaxed)) {
                    return;
                }
                const CompressedJob& j = jobs[i];
                try {
                    decomp.resize(j.uncompressed_size);
                    decompress::CompressionResult r;
                    if (comp_type == decompress::CompressionType::Zlib) {
                        r = decompress::decompress_zlib(j.compressed_ptr, j.compressed_size, decomp.data(), j.uncompressed_size);
                    } else if (comp_type == decompress::CompressionType::Zstd) {
                        r = decompress::decompress_zstd(j.compressed_ptr, j.compressed_size, decomp.data(), j.uncompressed_size);
                    } else {
                        throw std::runtime_error("Unsupported compression type");
                    }
                    if (!r.success) {
                        throw std::runtime_error("Decompression failed for variant at offset " +
                                                 std::to_string(j.offset) + ": " + r.error_message);
                    }
                    // Match the serial decompressor's validate_size behavior: the
                    // BGEN D field is the exact uncompressed size, so a mismatch
                    // means a malformed/truncated block. Reject rather than decode
                    // a short or partial buffer.
                    if (r.bytes_processed != j.uncompressed_size) {
                        throw std::runtime_error(
                            "Decompressed size mismatch for variant at offset " +
                            std::to_string(j.offset) + ": got " +
                            std::to_string(r.bytes_processed) + ", expected " +
                            std::to_string(j.uncompressed_size));
                    }
                    decode(i, decomp.data(), r.bytes_processed, j);
                } catch (const std::exception& e) {
                    bool expected = false;
                    if (failed.compare_exchange_strong(expected, true)) {
                        std::lock_guard<std::mutex> lk(err_mu);
                        err_msg = e.what();
                    }
                    return;
                }
              }
            }
        };

        // Spawn workers. If thread creation itself throws (e.g. resource
        // exhaustion), signal the already-started workers to stop and join them
        // before rethrowing - otherwise a joinable std::thread destructs during
        // unwinding and calls std::terminate.
        std::vector<std::thread> pool;
        pool.reserve(hw - 1);
        try {
            for (size_t t = 1; t < hw; ++t) {
                pool.emplace_back(worker);
            }
        } catch (...) {
            failed.store(true);
            for (auto& th : pool) {
                if (th.joinable()) {
                    th.join();
                }
            }
            throw;
        }
        worker();  // main thread participates
        for (auto& th : pool) {
            th.join();
        }

        if (failed.load()) {
            throw std::runtime_error(err_msg);
        }
    }

    // Batched parallel read + decode of an unfiltered (all-samples) block of
    // variants into a column-major (Fortran-order) output. Variant i is written
    // to out[i * out_stride .. i * out_stride + n_samples). This is the parallel
    // counterpart to the per-variant Cython loop: at UKBB scale (~500K samples)
    // both the inflate (~50%) and the scalar decode kernel (~41%) are per-variant
    // independent work, so spreading them across worker threads is the dominant
    // speedup.
    // Shared read + parallel-inflate prologue for the block decode paths. Reads
    // the compressed block (main-thread I/O), validates the file is open,
    // compressed, and non-empty, derives the layout, then drives
    // parallel_inflate_decode with the caller's per-variant decode step. The two
    // public block paths (all-samples and sample-filtered, each float/double)
    // differ ONLY in that decode step; everything else is identical here. The
    // decode callback is invoked as decode(i, buf, size, job, layout, n_samples).
    template <typename DecodeStep>
    void read_decode_block_impl(const VariantMetadata* block, size_t n_variants, size_t num_threads,
                                DecodeStep decode_step, size_t column_bytes = 0) {
        if (!is_open_) {
            throw std::runtime_error("BGEN file is not open");
        }
        if (n_variants == 0) {
            return;
        }
        const uint32_t n_samples = header_.n_samples;
        const auto comp_type = static_cast<decompress::CompressionType>(header_.compression);
        const auto layout = (header_.layout == 1) ? ::lazybgen::bgen::LayoutType::V11
                                                  : ::lazybgen::bgen::LayoutType::V12;
        if (comp_type == decompress::CompressionType::None) {
            throw std::runtime_error(
                "Uncompressed BGEN files are not supported. "
                "Please use compressed BGEN files (zlib or zstd).");
        }
        auto jobs = read_compressed_block(block, n_variants, comp_type);
        parallel_inflate_decode(
            jobs, comp_type, num_threads,
            [&](size_t i, const uint8_t* buf, size_t size, const CompressedJob& j) {
                decode_step(i, buf, size, j, layout, n_samples);
            },
            column_bytes);
    }

    template <typename T>
    void read_decode_block_parallel(const VariantMetadata* block, size_t n_variants, T* out,
                                    size_t out_stride, size_t num_threads) {
        // One slot per variant, written only by the worker that decoded it, so
        // the summary costs no synchronization.
        std::vector<DosageStats> per_variant(n_variants);
        read_decode_block_impl(
            block, n_variants, num_threads,
            [&](size_t i, const uint8_t* buf, size_t size, const CompressedJob& j,
                ::lazybgen::bgen::LayoutType layout, uint32_t n_samples) {
                // Data is already decompressed; pass the format-layer
                // CompressionType::None (distinct from decompress::).
                T* column = out + i * out_stride;
                ::lazybgen::bgen::GenotypeParser::compute_dosages_direct(
                    buf, size, layout, ::lazybgen::bgen::CompressionType::None, n_samples,
                    j.n_alleles, column);
                per_variant[i] = scan_dosages(column, n_samples);
            },
            out_stride * sizeof(T));
        last_block_stats_ = combine_dosage_stats(per_variant);
    }

    // Batched parallel read + decode of a sample-FILTERED (cohort) block. Variant
    // i is written to out[i * out_stride .. i * out_stride + n_indices). This is
    // the cohort-extraction path: it still inflates each full variant block (the
    // probability stream is contiguous, so decompression is independent of cohort
    // size) but decodes only the requested samples via the filtered SIMD kernel.
    // Decompression is the floor here, so spreading it across workers is the win.
    template <typename T>
    void read_decode_block_filtered_parallel(const VariantMetadata* block, size_t n_variants,
                                             const int* sample_indices, int n_indices, T* out,
                                             size_t out_stride, size_t num_threads) {
        std::vector<DosageStats> per_variant(n_variants);
        read_decode_block_impl(
            block, n_variants, num_threads,
            [&](size_t i, const uint8_t* buf, size_t size, const CompressedJob& j,
                ::lazybgen::bgen::LayoutType layout, uint32_t n_samples) {
                T* column = out + i * out_stride;
                filtered_decode_into(column, buf, size, layout, n_samples, j.n_alleles,
                                     sample_indices, n_indices);
                per_variant[i] = scan_dosages(column, static_cast<size_t>(n_indices));
            },
            out_stride * sizeof(T));
        last_block_stats_ = combine_dosage_stats(per_variant);
    }

    const DosageStats& last_block_stats() const {
        return last_block_stats_;
    }

    size_t num_threads() const {
        return num_threads_;
    }

    // Select the decode backend. The multi-variant block path
    // (read_decode_block_*) reads num_threads() directly and runs its own
    // parallel inflate, so the stored decompressor_ only backs the
    // single-variant fallback, where a sequential decode is what we want
    // regardless of type. The recorded type is purely the routing flag the
    // Cython layer reads to send multi-variant loads through the parallel block
    // path ('parallel') or the per-variant loop ('sequential').
    void set_decompressor_type(const std::string& type) {
        // Reject before mutating any state so decompressor_type() never reports
        // a rejected value (atomic rejection).
        if (type != "sequential" && type != "parallel") {
            throw std::runtime_error("Unknown decompressor type: " + type);
        }
        decompress::VariantDecompressor::Config config;
        config.validate_size = true;
        decompressor_ = decompress::create_sequential_decompressor(file_reader_.get(), config);
        decompressor_type_ = type;
    }

    // Get the active decompressor type
    const std::string& decompressor_type() const {
        return decompressor_type_;
    }

    // Set the worker-thread count used by the parallel block decode path.
    void set_num_threads(size_t n) {
        num_threads_ = n;
    }

    // Size the remote readahead block (bytes) to the access pattern. No-op for
    // local readers; the remote reader clamps to a sane range. Called by the
    // Cython layer per load once the variant offsets (hence pattern) are known.
    void set_read_block_size(size_t block_size) {
        if (file_reader_) {
            file_reader_->set_read_block_size(block_size);
        }
    }

    // Check if open
    bool is_open() const {
        return is_open_;
    }

    // Close file
    void close() {
        if (file_reader_) {
            file_reader_->close();
            file_reader_.reset();
        }
        if (bgi_reader_) {
            bgi_reader_.reset();
        }
        decompressor_.reset();
        is_open_ = false;
    }

   private:
    // Open the BGEN file
    void open_file(PyObject* storage_options) {
        // Create appropriate reader based on filename
        if (is_remote_scheme(filename_)) {
            file_reader_ = std::unique_ptr<FsspecFileReader>(
                new FsspecFileReader(filename_, storage_options));
        } else {
            // Try memory-mapped file first for local files
            try {
                file_reader_ = std::unique_ptr<MMapFileReader>(new MMapFileReader(filename_));
                if (file_reader_->is_open()) {
                    return;
                }
            } catch (...) {
                // Fall back to regular file reader
            }

            // Use regular file reader
            file_reader_ = std::unique_ptr<RegularFileReader>(new RegularFileReader(filename_));
        }

        if (!file_reader_ || !file_reader_->is_open()) {
            throw std::runtime_error("Failed to open BGEN file: " + filename_);
        }
    }

    // Parse BGEN header
    void parse_header() {
        // Read header size first (need at least 20 bytes for initial fields)
        std::vector<uint8_t> initial_buffer(20);
        size_t bytes_read = file_reader_->read(initial_buffer.data(), 20);
        if (bytes_read < 20) {
            throw std::runtime_error("BGEN file too small to contain valid header");
        }

        // Get full header size
        size_t header_size = BgenHeaderParser::getHeaderSize(initial_buffer.data(), bytes_read);

        // Read full header
        std::vector<uint8_t> header_buffer(header_size);
        file_reader_->seek(0);
        bytes_read = file_reader_->read(header_buffer.data(), header_size);
        if (bytes_read < header_size) {
            throw std::runtime_error("Failed to read complete BGEN header");
        }

        // Parse header
        auto parsed_header = BgenHeaderParser::parse(header_buffer.data(), bytes_read);

        // Convert to our header structure
        header_.offset = parsed_header.offset;
        header_.n_variants = parsed_header.n_variants;
        header_.n_samples = parsed_header.n_samples;
        header_.flags = parsed_header.flags;
        header_.compression = static_cast<uint8_t>(parsed_header.compression);
        header_.layout = static_cast<uint8_t>(parsed_header.layout);
        header_.has_sample_ids = parsed_header.has_sample_ids;
    }

    // Open BGI index
    void open_index() {
        try {
            // BGI path should already be local (Python handles GCS download)
            bgi_reader_ = std::unique_ptr<BgiReader>(new BgiReader(bgi_filename_));
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to open BGI index: " + std::string(e.what()));
        }
    }

    // Read sample IDs
    void read_sample_ids() {
        if (header_.has_sample_ids) {
            // Current position should be right after header
            uint64_t sample_block_pos = file_reader_->tell();

            // Read sample block size
            uint8_t size_buffer[4];
            size_t bytes_read = file_reader_->read(size_buffer, 4);
            if (bytes_read < 4) {
                throw std::runtime_error("Failed to read sample block size");
            }

            // Convert from little-endian
            uint32_t block_size_with_n = static_cast<uint32_t>(size_buffer[0]) |
                                         (static_cast<uint32_t>(size_buffer[1]) << 8) |
                                         (static_cast<uint32_t>(size_buffer[2]) << 16) |
                                         (static_cast<uint32_t>(size_buffer[3]) << 24);

            // Sanity check the block size (should include sample count)
            if (block_size_with_n < 4 || block_size_with_n > 100000000) {  // 100MB max
                throw std::runtime_error("Invalid sample block size: " +
                                         std::to_string(block_size_with_n));
            }

            // Read full sample block (including the 4-byte size prefix)
            std::vector<uint8_t> sample_buffer(block_size_with_n + 4);
            file_reader_->seek(sample_block_pos);
            bytes_read = file_reader_->read(sample_buffer.data(), block_size_with_n + 4);
            if (bytes_read < block_size_with_n + 4) {
                throw std::runtime_error("Failed to read sample block");
            }

            // Parse sample block
            auto sample_block =
                SampleBlockParser::parse(sample_buffer.data(), bytes_read, header_.n_samples);
            sample_ids_ = std::move(sample_block.sample_ids);

            // Update offset to skip sample block
            header_.offset = sample_block_pos + block_size_with_n + 4;
        }
        // With no sample block there is nothing to read; sample_ids() builds the
        // placeholder IDs if anyone asks for them.
    }

    // Create default decompressor
    void create_default_decompressor() {
        decompress::VariantDecompressor::Config config;
        config.validate_size = true;

        // Use the sequential decompressor by default
        decompressor_ = decompress::create_sequential_decompressor(file_reader_.get(), config);
        decompressor_type_ = "sequential";
    }

   private:
    std::string filename_;
    std::string bgi_filename_;
    std::unique_ptr<FileReader> file_reader_;
    std::unique_ptr<BgiReader> bgi_reader_;
    std::unique_ptr<decompress::VariantDecompressor> decompressor_;
    BgenHeader header_;
    std::vector<std::string> sample_ids_;
    DosageStats last_block_stats_{std::numeric_limits<double>::infinity(),
                                  -std::numeric_limits<double>::infinity(), false};
    bool is_open_;
    size_t num_threads_ = 0;
    std::string decompressor_type_;
};

// BgenReaderImpl public methods (forwarding to pimpl)

BgenReaderImpl::BgenReaderImpl(const std::string& filename, const std::string& bgi_filename,
                               PyObject* storage_options)
    : pimpl_(std::unique_ptr<Impl>(new Impl(filename, bgi_filename, storage_options))) {}

BgenReaderImpl::~BgenReaderImpl() = default;

const BgenHeader& BgenReaderImpl::header() const {
    return pimpl_->header();
}

const std::vector<std::string>& BgenReaderImpl::sample_ids() {
    return pimpl_->sample_ids();
}

const DosageStats& BgenReaderImpl::last_block_stats() const {
    return pimpl_->last_block_stats();
}

std::vector<VariantMetadata> BgenReaderImpl::build_metadata_from_index(
    const index::VariantInfo* infos, size_t n) {
    return pimpl_->build_metadata_from_index(infos, n);
}

std::unique_ptr<decompress::DecompressedData> BgenReaderImpl::read_variant_genotypes(
    const VariantMetadata& metadata) {
    return pimpl_->read_variant_genotypes(metadata);
}

void BgenReaderImpl::read_decode_block_parallel(const VariantMetadata* block, size_t n_variants,
                                                float* out, size_t out_stride,
                                                size_t num_threads) {
    pimpl_->read_decode_block_parallel<float>(block, n_variants, out, out_stride, num_threads);
}

void BgenReaderImpl::read_decode_block_parallel(const VariantMetadata* block, size_t n_variants,
                                                double* out, size_t out_stride,
                                                size_t num_threads) {
    pimpl_->read_decode_block_parallel<double>(block, n_variants, out, out_stride, num_threads);
}

void BgenReaderImpl::read_decode_block_filtered_parallel(const VariantMetadata* block,
                                                         size_t n_variants,
                                                         const int* sample_indices, int n_indices,
                                                         float* out, size_t out_stride,
                                                         size_t num_threads) {
    pimpl_->read_decode_block_filtered_parallel<float>(block, n_variants, sample_indices, n_indices,
                                                       out, out_stride, num_threads);
}

void BgenReaderImpl::read_decode_block_filtered_parallel(const VariantMetadata* block,
                                                         size_t n_variants,
                                                         const int* sample_indices, int n_indices,
                                                         double* out, size_t out_stride,
                                                         size_t num_threads) {
    pimpl_->read_decode_block_filtered_parallel<double>(block, n_variants, sample_indices,
                                                        n_indices, out, out_stride, num_threads);
}

size_t BgenReaderImpl::num_threads() const {
    return pimpl_->num_threads();
}

void BgenReaderImpl::set_decompressor_type(const std::string& type) {
    pimpl_->set_decompressor_type(type);
}

void BgenReaderImpl::set_read_block_size(size_t block_size) {
    pimpl_->set_read_block_size(block_size);
}

std::string BgenReaderImpl::decompressor_type() const {
    return pimpl_->decompressor_type();
}

void BgenReaderImpl::set_num_threads(size_t n) {
    pimpl_->set_num_threads(n);
}

bool BgenReaderImpl::is_open() const {
    return pimpl_->is_open();
}

void BgenReaderImpl::close() {
    pimpl_->close();
}

}  // namespace bgen
}  // namespace io
}  // namespace lazybgen