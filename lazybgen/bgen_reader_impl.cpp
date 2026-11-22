#include "bgen_reader_impl.h"

#include "format/simd_target.h"
#if defined(LAZYBGEN_X86_AVX2)
#include <immintrin.h>
#endif

#include "format/bgen_header.h"
#include "format/genotype_parser.h"
#include "format/genotype_parser_simd.h"  // cpu_has_avx2
#include "format/variant_parser.h"
#include "index/bgi_reader.h"
#include "io/fsspec_file_reader.h"
#include <fcntl.h>
#if defined(__linux__)
#include <sched.h>
#endif
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <sstream>
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

// CPUs a cgroup quota allows ("max" or unset -> 0, meaning no limit). v2 keeps
// "quota period" in cpu.max; v1 keeps the two in cpu.cfs_quota_us and
// cpu.cfs_period_us, with quota -1 for unlimited. A fractional quota rounds up.
size_t quota_cpus(long long quota, long long period) {
    if (quota <= 0 || period <= 0) {
        return 0;
    }
    return static_cast<size_t>(std::max<long long>(1, (quota + period - 1) / period));
}

size_t cgroup_v2_cpus(const std::string& dir) {
    std::ifstream f(dir + "/cpu.max");
    std::string quota;
    long long period = 0;
    if (!(f >> quota >> period) || quota == "max") {
        return 0;
    }
    try {
        return quota_cpus(std::stoll(quota), period);
    } catch (const std::exception&) {
        return 0;
    }
}

size_t cgroup_v1_cpus(const std::string& dir) {
    long long quota = -1;
    long long period = 0;
    std::ifstream fq(dir + "/cpu.cfs_quota_us");
    std::ifstream fp(dir + "/cpu.cfs_period_us");
    if (!(fq >> quota) || !(fp >> period)) {
        return 0;
    }
    return quota_cpus(quota, period);
}

// mountinfo escapes space, tab, newline and backslash in paths as \ooo.
std::string unescape_mount_path(const std::string& s) {
    std::string out;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 3 < s.size() &&
            std::isdigit(static_cast<unsigned char>(s[i + 1])) &&
            std::isdigit(static_cast<unsigned char>(s[i + 2])) &&
            std::isdigit(static_cast<unsigned char>(s[i + 3]))) {
            out.push_back(static_cast<char>((s[i + 1] - '0') * 64 + (s[i + 2] - '0') * 8 +
                                            (s[i + 3] - '0')));
            i += 3;
        } else {
            out.push_back(s[i]);
        }
    }
    return out;
}

bool has_token(const std::string& list, const std::string& token) {
    return ("," + list + ",").find("," + token + ",") != std::string::npos;
}

}  // namespace

// Tightest CPU quota on this process's cgroup or any ancestor visible in its
// mount, or 0. Mount points and mount roots come from mountinfo rather than
// being assumed: a v1 cpu controller may be mounted at cpu,cpuacct while
// /proc/self/cgroup lists it as cpuacct,cpu, and a container's cgroup mount is
// often rooted below the hierarchy's root, so the process's cgroup path must
// be taken relative to that root. A path outside the mount's root (a container
// without a cgroup namespace) falls back to the mount point's own files.
// `prefix` is prepended to every mount point, so tests can use a fake tree.
size_t cgroup_cpu_limit_from(const std::string& mountinfo_path, const std::string& cgroup_path,
                             const std::string& prefix) {
    struct Mount {
        bool v2;
        std::string root;
        std::string point;
    };
    std::vector<Mount> mounts;
    {
        std::ifstream f(mountinfo_path);
        std::string line;
        while (std::getline(f, line)) {
            // id parent major:minor root mount-point options [optional...] - fstype source super-options
            std::istringstream in(line);
            std::vector<std::string> fields;
            std::string w;
            while (in >> w) {
                fields.push_back(w);
            }
            size_t dash = 0;
            while (dash < fields.size() && fields[dash] != "-") {
                ++dash;
            }
            if (fields.size() < 5 || dash + 3 > fields.size()) {
                continue;
            }
            const std::string& fstype = fields[dash + 1];
            const std::string& superopts = dash + 3 < fields.size() ? fields[dash + 3] : std::string();
            if (fstype == "cgroup2") {
                mounts.push_back({true, unescape_mount_path(fields[3]), unescape_mount_path(fields[4])});
            } else if (fstype == "cgroup" && has_token(superopts, "cpu")) {
                mounts.push_back({false, unescape_mount_path(fields[3]), unescape_mount_path(fields[4])});
            }
        }
    }

    size_t limit = 0;
    auto take = [&limit](size_t n) {
        if (n > 0 && (limit == 0 || n < limit)) {
            limit = n;
        }
    };
    std::ifstream f(cgroup_path);
    std::string line;
    while (std::getline(f, line)) {
        // "hierarchy-id:controllers:path"; v2 is "0::path".
        const size_t a = line.find(':');
        const size_t b = a == std::string::npos ? a : line.find(':', a + 1);
        if (b == std::string::npos) {
            continue;
        }
        const std::string controllers = line.substr(a + 1, b - a - 1);
        const std::string path = line.substr(b + 1);
        const bool v2 = controllers.empty();
        if (!v2 && !has_token(controllers, "cpu")) {
            continue;
        }
        // The process's cgroup relative to each mount's root, for the mounts
        // whose root contains it. A mount of some other subtree (a runtime or
        // monitoring agent bind-mounting a container's cgroup, say) says
        // nothing about this process, so its quota is not taken. Only when no
        // mount resolves the path (a container whose cgroup namespace hides
        // the real path) is the process taken to sit at the mount point.
        std::vector<std::pair<const Mount*, std::string>> resolved;
        for (const Mount& m : mounts) {
            if (m.v2 != v2) {
                continue;
            }
            if (m.root == "/") {
                resolved.emplace_back(&m, path.empty() ? "/" : path);
            } else if (path == m.root) {
                resolved.emplace_back(&m, "/");
            } else if (path.compare(0, m.root.size() + 1, m.root + "/") == 0) {
                resolved.emplace_back(&m, path.substr(m.root.size()));
            }
        }
        if (resolved.empty()) {
            for (const Mount& m : mounts) {
                if (m.v2 == v2) {
                    resolved.emplace_back(&m, "/");
                }
            }
        }
        for (auto& entry : resolved) {
            const Mount& m = *entry.first;
            std::string rel = entry.second;
            // Walk from the process's cgroup up to the mount point; a parent's
            // quota caps its children.
            while (true) {
                const std::string dir = prefix + m.point + (rel == "/" ? std::string() : rel);
                take(v2 ? cgroup_v2_cpus(dir) : cgroup_v1_cpus(dir));
                if (rel == "/") {
                    break;
                }
                const size_t slash = rel.find_last_of('/');
                rel = slash == 0 || slash == std::string::npos ? "/" : rel.substr(0, slash);
            }
        }
    }
    return limit;
}

namespace {
#if defined(__linux__)
// CPUs in this thread's affinity mask, or 0 if unknown. The mask is read with a
// buffer that grows until the kernel accepts it, so a machine with more CPUs
// than a cpu_set_t holds still reports them.
size_t affinity_cpus() {
    for (int ncpu = CPU_SETSIZE; ncpu <= (1 << 20); ncpu *= 2) {
        cpu_set_t* set = CPU_ALLOC(ncpu);
        if (!set) {
            return 0;
        }
        const size_t bytes = CPU_ALLOC_SIZE(ncpu);
        CPU_ZERO_S(bytes, set);
        const int rc = sched_getaffinity(0, bytes, set);
        const int count = rc == 0 ? CPU_COUNT_S(bytes, set) : 0;
        const bool too_small = rc != 0 && errno == EINVAL;
        CPU_FREE(set);
        if (rc == 0) {
            return count > 0 ? static_cast<size_t>(count) : 0;
        }
        if (!too_small) {
            return 0;
        }
    }
    return 0;
}

// The cgroup quota, read once per process: parsing mountinfo costs tens of
// microseconds, and this is consulted on every block decode. A forked child
// re-reads it, since it may have been moved to another cgroup. No lock: a
// child forked while another thread held one would deadlock on its first
// decode. Threads racing on the first call each compute the same value, and
// the release/acquire pair publishes the value before the pid that marks it.
size_t cached_cgroup_cpu_limit() {
    static std::atomic<pid_t> cached_pid{-1};
    static std::atomic<size_t> cached{0};
    const pid_t pid = getpid();
    if (cached_pid.load(std::memory_order_acquire) != pid) {
        cached.store(cgroup_cpu_limit_from("/proc/self/mountinfo", "/proc/self/cgroup", ""),
                     std::memory_order_relaxed);
        cached_pid.store(pid, std::memory_order_release);
    }
    return cached.load(std::memory_order_relaxed);
}
#endif

}  // namespace

size_t default_num_threads() {
    size_t cpus = std::thread::hardware_concurrency();
#if defined(__linux__)
    // A SLURM allocation, taskset or a container cpuset narrows the CPUs this
    // process may run on without changing the machine's core count, and a
    // cgroup quota limits how much of them it may use. Starting one worker per
    // core of the machine then oversubscribes the job. The affinity mask is read
    // on every call (one system call), so a later taskset or a forked child
    // pinned to fewer CPUs is seen.
    const size_t allowed = affinity_cpus();
    if (allowed > 0 && (cpus == 0 || allowed < cpus)) {
        cpus = allowed;
    }
    const size_t quota = cached_cgroup_cpu_limit();
    if (quota > 0 && (cpus == 0 || quota < cpus)) {
        cpus = quota;
    }
#endif
    return cpus == 0 ? static_cast<size_t>(1) : cpus;
}

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
        // A short read sets eof/fail, and a stream in that state ignores
        // seekg, so clear it before restoring the position.
        file_.clear();
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
        // The mapping stays valid without the descriptor, so do not hold one
        // per open reader (a process keeping many readers would otherwise run
        // into its open-file limit).
        ::close(fd_);
        fd_ = -1;

        // Let the kernel adapt its readahead to the actual access pattern
        // (MADV_NORMAL): it ramps up for sequential scans (full-file / large
        // contiguous region decodes) and stays modest for indexed lookups. The
        // alternative, MADV_RANDOM, disables readahead entirely, which suits
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
         PyObject* storage_options = nullptr, const std::string& remote_transport = "fsspec")
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
            open_file(storage_options, remote_transport);

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
        release_id_reader();
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
        if (header_.has_sample_ids && !embedded_ids_loaded_) {
            load_embedded_sample_ids();
        }
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

#if defined(LAZYBGEN_X86_AVX2)
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
    // It takes the AVX2 path only on a CPU that has it.
    static DosageStats scan_dosages(const double* values, size_t n) {
        return ::lazybgen::bgen::cpu_has_avx2() ? scan_dosages_avx2(values, n)
                                                : scan_dosages_scalar(values, n);
    }

    LAZYBGEN_TARGET_AVX2 static DosageStats scan_dosages_avx2(const double* values, size_t n) {
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
#endif  // LAZYBGEN_X86_AVX2

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
        size_t hw = num_threads ? num_threads : default_num_threads();
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
        // exceeds the target, so the run collapses to one variant, i.e. to
        // claiming one at a time. column_bytes == 0 (a caller that does not know
        // its output geometry) claims one at a time too.
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
                ::lazybgen::bgen::ColumnStats fused;
                ::lazybgen::bgen::GenotypeParser::compute_dosages_direct(
                    buf, size, layout, ::lazybgen::bgen::CompressionType::None, n_samples,
                    j.n_alleles, column, &fused);
                // The 8/16-bit kernels return the range with the decode; only
                // the other layouts need a second pass over the column.
                per_variant[i] = fused.valid
                                     ? DosageStats{fused.min_value, fused.max_value, fused.has_nan}
                                     : scan_dosages(column, n_samples);
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
            if (header_.has_sample_ids && !embedded_ids_loaded_ && !embedded_ids_discarded_ &&
                sample_block_bytes_.empty()) {
                // A local reader whose embedded IDs nobody has read yet keeps
                // its mapping for them, so samples still work after close()
                // without open() copying the block for every reader; the
                // mapping goes once the IDs are parsed or the reader is freed.
                id_reader_ = std::move(file_reader_);
            } else {
                file_reader_->close();
                file_reader_.reset();
            }
        }
        if (bgi_reader_) {
            bgi_reader_.reset();
        }
        decompressor_.reset();
        is_open_ = false;
    }

   private:
    // Open the BGEN file
    void open_file(PyObject* storage_options, const std::string& remote_transport) {
        // Create appropriate reader based on filename
        if (is_remote_scheme(filename_)) {
            file_reader_ = std::unique_ptr<FsspecFileReader>(
                new FsspecFileReader(filename_, storage_options, remote_transport));
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

    // Check the embedded sample-ID block's shape without building the IDs.
    // Building one string per sample is the bulk of opening a biobank-scale
    // file, and a reader given a .sample file never uses them, so sample_ids()
    // parses the block the first time anyone asks. A local file is read from
    // its mapping then (copying the block at open cost every reader a few
    // milliseconds at 500k samples); a remote reader keeps the bytes now, so it
    // pays no second fetch and holds no connection open after close().
    void read_sample_ids() {
        if (header_.has_sample_ids) {
            // The block starts right after the header, at 4 + L_H.
            const uint64_t sample_block_pos = file_reader_->tell();

            // L_SI (the block's length, counting these 4 bytes) and N.
            uint8_t prefix[8];
            if (file_reader_->read(prefix, 8) < 8) {
                throw std::runtime_error("Failed to read sample block size");
            }
            const uint64_t block_size = read_u32_le(prefix);
            const uint64_t n_in_block = read_u32_le(prefix + 4);

            if (n_in_block != header_.n_samples) {
                throw std::runtime_error("Sample count mismatch: expected " +
                                         std::to_string(header_.n_samples) + ", got " +
                                         std::to_string(n_in_block));
            }
            // The block must hold L_SI, N and a 2-byte length per ID, can hold
            // no more than N IDs of the longest length a 2-byte field allows,
            // and must end where the variant data begins (offset + 4) or
            // before it. The upper bound keeps a corrupt L_SI from sizing an
            // allocation (or a remote download) by the file instead of by N.
            const uint64_t variants_pos = static_cast<uint64_t>(header_.offset) + 4;
            if (block_size < 8 + 2 * n_in_block || block_size > 8 + 65537 * n_in_block ||
                sample_block_pos + block_size > variants_pos ||
                variants_pos > file_reader_->size()) {
                throw std::runtime_error("Invalid sample block size: " +
                                         std::to_string(block_size));
            }
            sample_block_pos_ = sample_block_pos;
            sample_block_size_ = block_size;
            if (is_remote_scheme(filename_)) {
                sample_block_bytes_.resize(block_size);
                std::memcpy(sample_block_bytes_.data(), prefix, 8);
                const size_t rest = static_cast<size_t>(block_size - 8);
                if (rest > 0 && file_reader_->read(sample_block_bytes_.data() + 8, rest) < rest) {
                    throw std::runtime_error("Failed to read sample block");
                }
            }
        }
        // With no sample block there is nothing to read; sample_ids() builds the
        // placeholder IDs if anyone asks for them.
    }

    static uint32_t read_u32_le(const uint8_t* b) {
        return static_cast<uint32_t>(b[0]) | (static_cast<uint32_t>(b[1]) << 8) |
               (static_cast<uint32_t>(b[2]) << 16) | (static_cast<uint32_t>(b[3]) << 24);
    }

    // Parse the embedded sample-ID block: the bytes a remote reader kept at
    // open, or the block read from the local file (or the mapping close() kept
    // for it). Then drop whatever held them.
    void load_embedded_sample_ids() {
        const uint8_t* block = sample_block_bytes_.data();
        const size_t size = static_cast<size_t>(sample_block_size_);
        if (sample_block_bytes_.empty()) {
            FileReader* src = file_reader_ ? file_reader_.get() : id_reader_.get();
            if (!src) {
                throw std::runtime_error("Sample IDs are no longer available");
            }
            // Positional reads only, so a decode running on the same reader
            // is undisturbed.
            block = src->view_at(sample_block_pos_, size);
            if (!block) {
                // Publish the bytes only once the read is complete, so a short
                // read raises on every access instead of leaving a partial
                // buffer that the next access would parse as the block.
                std::vector<uint8_t> bytes(size);
                if (src->read_at(sample_block_pos_, bytes.data(), size) < size) {
                    throw std::runtime_error("Failed to read sample block");
                }
                sample_block_bytes_.swap(bytes);
                block = sample_block_bytes_.data();
            }
        }
        auto sample_block = SampleBlockParser::parse(block, size, header_.n_samples);
        sample_ids_ = std::move(sample_block.sample_ids);
        std::vector<uint8_t>().swap(sample_block_bytes_);
        embedded_ids_loaded_ = true;
        release_id_reader();
    }

   public:
    // The caller takes its sample IDs from elsewhere (a .sample file), so the
    // embedded block will not be read: drop a remote reader's copy of it now,
    // and let close() release a local mapping instead of keeping it for IDs
    // nobody will ask for. sample_ids() still works while the reader is open.
    void discard_embedded_sample_ids() {
        embedded_ids_discarded_ = true;
        std::vector<uint8_t>().swap(sample_block_bytes_);
        release_id_reader();
    }

   private:
    void release_id_reader() {
        if (id_reader_) {
            id_reader_->close();
            id_reader_.reset();
        }
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
    std::unique_ptr<FileReader> id_reader_;    // kept by close() for unread IDs
    uint64_t sample_block_pos_ = 0;
    uint64_t sample_block_size_ = 0;
    std::vector<uint8_t> sample_block_bytes_;  // remote: embedded ID block, until parsed
    bool embedded_ids_loaded_ = false;
    bool embedded_ids_discarded_ = false;
    DosageStats last_block_stats_{std::numeric_limits<double>::infinity(),
                                  -std::numeric_limits<double>::infinity(), false};
    bool is_open_;
    size_t num_threads_ = 0;
    std::string decompressor_type_;
};

// BgenReaderImpl public methods (forwarding to pimpl)

BgenReaderImpl::BgenReaderImpl(const std::string& filename, const std::string& bgi_filename,
                               PyObject* storage_options, const std::string& remote_transport)
    : pimpl_(std::unique_ptr<Impl>(
          new Impl(filename, bgi_filename, storage_options, remote_transport))) {}

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

void BgenReaderImpl::discard_embedded_sample_ids() {
    pimpl_->discard_embedded_sample_ids();
}

}  // namespace bgen
}  // namespace io
}  // namespace lazybgen