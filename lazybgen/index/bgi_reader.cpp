#include "bgi_reader.h"

#include <sqlite3.h>

#include <algorithm>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

// C++17 feature detection for shared_mutex
#if __cplusplus >= 201703L
#include <shared_mutex>
#endif

namespace lazybgen {
namespace io {
namespace bgen {
namespace index {

namespace {
// RAII finalizer for ad-hoc (non-cached) prepared statements. Guarantees
// sqlite3_finalize on every exit path, including a throw from execute_query /
// stepping, so the statement is never leaked. The cached statements (stmt_*_)
// are owned by Impl and finalized in its destructor, so they are NOT wrapped.
class StmtFinalizer {
   public:
    explicit StmtFinalizer(sqlite3_stmt* stmt) : stmt_(stmt) {}
    ~StmtFinalizer() {
        if (stmt_) {
            sqlite3_finalize(stmt_);
        }
    }
    StmtFinalizer(const StmtFinalizer&) = delete;
    StmtFinalizer& operator=(const StmtFinalizer&) = delete;

   private:
    sqlite3_stmt* stmt_;
};
}  // namespace

// Implementation class
class BgiReader::Impl {
   private:
// C++17 feature detection for lock types
#if __cplusplus >= 201703L
    using mutex_type = std::shared_mutex;
    using read_lock = std::shared_lock<std::shared_mutex>;
    using write_lock = std::unique_lock<std::shared_mutex>;
#else
    using mutex_type = std::mutex;
    using read_lock = std::lock_guard<std::mutex>;
    using write_lock = std::lock_guard<std::mutex>;
#endif

   public:
    Impl(const std::string& bgi_path) : db_(nullptr), variant_count_(0) {
        // Open the BGI as a read-only SQLite database (default threading mode).
        int rc = sqlite3_open_v2(bgi_path.c_str(), &db_, SQLITE_OPEN_READONLY, nullptr);
        if (rc != SQLITE_OK) {
            throw std::runtime_error("Failed to open BGI file: " + bgi_path + " - " +
                                     sqlite3_errmsg(db_));
        }

        // Configure SQLite for optimal performance
        sqlite3_exec(db_, "PRAGMA cache_size = 10000", nullptr, nullptr, nullptr);
        sqlite3_exec(db_, "PRAGMA mmap_size = 268435456", nullptr, nullptr, nullptr);  // 256MB
        sqlite3_exec(db_, "PRAGMA temp_store = MEMORY", nullptr, nullptr, nullptr);

        // Verify database structure
        verify_database();

        // Prepare frequently used statements
        prepare_statements();
    }

    ~Impl() {
        // Finalize prepared statements
        if (stmt_by_region_)
            sqlite3_finalize(stmt_by_region_);

        // Close database
        if (db_)
            sqlite3_close(db_);
    }

    std::vector<VariantInfo> query_region(const std::string& chromosome, uint32_t start_pos,
                                          uint32_t end_pos) {
        read_lock lock(mutex_);

        // Query database directly without cache
        sqlite3_reset(stmt_by_region_);
        sqlite3_bind_text(stmt_by_region_, 1, chromosome.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_int(stmt_by_region_, 2, start_pos);
        sqlite3_bind_int(stmt_by_region_, 3, end_pos);

        return execute_query(stmt_by_region_);
    }

    size_t get_variant_count() const {
        return variant_count_;
    }

    std::vector<VariantInfo> get_all_variants() {
        read_lock lock(mutex_);

        std::vector<VariantInfo> results;
        results.reserve(variant_count_);

        // Query all variants in one go
        const char* query =
            "SELECT file_start_position, size_in_bytes, chromosome, position, "
            "rsid, number_of_alleles, allele1, allele2 "
            "FROM Variant "
            "ORDER BY file_start_position";

        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_, query, -1, &stmt, nullptr) != SQLITE_OK) {
            throw std::runtime_error("Failed to prepare all variants query");
        }
        StmtFinalizer finalizer(stmt);  // finalize on every path (incl. throw)

        results = execute_query(stmt);

        return results;
    }

    std::vector<VariantInfo> find_variants_by_filter(const std::string& chromosome,
                                                     const std::vector<uint32_t>& positions,
                                                     const std::vector<std::string>& alleles1,
                                                     const std::vector<std::string>& alleles2,
                                                     size_t batch_size) {
        read_lock lock(mutex_);

        if (positions.size() != alleles1.size() || positions.size() != alleles2.size()) {
            throw std::invalid_argument("Input vectors must have same size");
        }

        if (positions.empty()) {
            return std::vector<VariantInfo>();
        }

        // Build lookup table for allele matching with original indices
        struct AlleleMatch {
            std::string allele1;
            std::string allele2;
            size_t original_index;
        };
        std::unordered_map<uint32_t, std::vector<AlleleMatch>> allele_map;
        for (size_t i = 0; i < positions.size(); ++i) {
            allele_map[positions[i]].push_back({alleles1[i], alleles2[i], i});
        }

        // Result map to maintain original order
        std::unordered_map<size_t, VariantInfo> result_map;

        // Get unique positions for efficient querying
        std::set<uint32_t> unique_positions(positions.begin(), positions.end());
        std::vector<uint32_t> unique_pos_vec(unique_positions.begin(), unique_positions.end());

        // Process in batches
        for (size_t i = 0; i < unique_pos_vec.size(); i += batch_size) {
            size_t batch_end = std::min(i + batch_size, unique_pos_vec.size());
            size_t batch_size_actual = batch_end - i;

            // Build query with IN clause
            std::stringstream query;
            query << "SELECT file_start_position, size_in_bytes, chromosome, position, "
                  << "rsid, number_of_alleles, allele1, allele2 "
                  << "FROM Variant WHERE chromosome = ? AND position IN (";

            for (size_t j = 0; j < batch_size_actual; ++j) {
                if (j > 0)
                    query << ",";
                query << "?";
            }
            query << ") ORDER BY file_start_position";

            // Prepare statement
            sqlite3_stmt* stmt;
            if (sqlite3_prepare_v2(db_, query.str().c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
                throw std::runtime_error("Failed to prepare batch query: " +
                                         std::string(sqlite3_errmsg(db_)));
            }

            // Bind parameters
            sqlite3_bind_text(stmt, 1, chromosome.c_str(), -1, SQLITE_STATIC);
            for (size_t j = 0; j < batch_size_actual; ++j) {
                sqlite3_bind_int(stmt, 2 + j, unique_pos_vec[i + j]);
            }

            // Execute and collect results
            int rc;
            while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
                VariantInfo info;

                // Extract data from columns
                info.file_offset = static_cast<uint64_t>(sqlite3_column_int64(stmt, 0));
                info.variant_size = static_cast<uint32_t>(sqlite3_column_int(stmt, 1));

                const char* chr = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
                if (chr)
                    info.chromosome = chr;

                info.position = static_cast<uint32_t>(sqlite3_column_int(stmt, 3));

                const char* rsid = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
                if (rsid)
                    info.rsid = rsid;

                info.n_alleles = static_cast<uint16_t>(sqlite3_column_int(stmt, 5));

                const char* a1 = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 6));
                if (a1)
                    info.allele1 = a1;

                const char* a2 = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 7));
                if (a2)
                    info.allele2 = a2;
                else
                    info.allele2 = "";  // Handle NULL as empty string

                // Check if this variant matches any of our allele combinations
                auto it = allele_map.find(info.position);
                if (it != allele_map.end()) {
                    for (const auto& match : it->second) {
                        if (info.allele1 == match.allele1 && info.allele2 == match.allele2) {
                            result_map[match.original_index] = info;
                        }
                    }
                }
            }

            if (rc != SQLITE_DONE) {
                sqlite3_finalize(stmt);
                throw std::runtime_error("Batch query execution failed: " +
                                         std::string(sqlite3_errmsg(db_)));
            }

            sqlite3_finalize(stmt);
        }

        // Build results in original order
        std::vector<VariantInfo> results;
        results.reserve(result_map.size());

        for (size_t i = 0; i < positions.size(); ++i) {
            auto it = result_map.find(i);
            if (it != result_map.end()) {
                results.push_back(it->second);
            }
        }

        return results;
    }

   private:
    void verify_database() {
        // Check if required tables exist
        const char* check_tables =
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE type='table' AND name IN ('Variant', 'Metadata')";

        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_, check_tables, -1, &stmt, nullptr) != SQLITE_OK) {
            throw std::runtime_error("Failed to prepare table check query");
        }

        int table_count = 0;
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            table_count = sqlite3_column_int(stmt, 0);
        }
        sqlite3_finalize(stmt);

        if (table_count < 1) {  // At least Variant table must exist
            throw std::runtime_error("Invalid BGI file: missing required tables");
        }

        // Get variant count
        const char* count_query = "SELECT COUNT(*) FROM Variant";
        if (sqlite3_prepare_v2(db_, count_query, -1, &stmt, nullptr) != SQLITE_OK) {
            throw std::runtime_error("Failed to prepare count query");
        }

        if (sqlite3_step(stmt) == SQLITE_ROW) {
            variant_count_ = static_cast<size_t>(sqlite3_column_int64(stmt, 0));
        }
        sqlite3_finalize(stmt);
    }

    void prepare_statements() {
        // Query by region
        const char* region_query =
            "SELECT file_start_position, size_in_bytes, chromosome, position, "
            "rsid, number_of_alleles, allele1, allele2 "
            "FROM Variant "
            "WHERE chromosome = ? AND position >= ? AND position <= ? "
            "ORDER BY position";

        if (sqlite3_prepare_v2(db_, region_query, -1, &stmt_by_region_, nullptr) != SQLITE_OK) {
            throw std::runtime_error("Failed to prepare region query");
        }
    }

    std::vector<VariantInfo> execute_query(sqlite3_stmt* stmt) {
        std::vector<VariantInfo> results;

        int rc;
        while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
            VariantInfo info;

            // Extract data from columns
            info.file_offset = static_cast<uint64_t>(sqlite3_column_int64(stmt, 0));
            info.variant_size = static_cast<uint32_t>(sqlite3_column_int(stmt, 1));

            const char* chr = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
            if (chr)
                info.chromosome = chr;

            info.position = static_cast<uint32_t>(sqlite3_column_int(stmt, 3));

            const char* rsid = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
            if (rsid)
                info.rsid = rsid;

            info.n_alleles = static_cast<uint16_t>(sqlite3_column_int(stmt, 5));

            const char* allele1 = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 6));
            if (allele1)
                info.allele1 = allele1;

            const char* allele2 = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 7));
            if (allele2)
                info.allele2 = allele2;

            results.push_back(info);
        }

        if (rc != SQLITE_DONE) {
            throw std::runtime_error("Query execution failed: " + std::string(sqlite3_errmsg(db_)));
        }

        return results;
    }

   private:
    sqlite3* db_;
    size_t variant_count_;

    // Prepared statements
    sqlite3_stmt* stmt_by_region_ = nullptr;

    // Thread safety
    mutable mutex_type mutex_;
};

// BgiReader implementation

BgiReader::BgiReader(const std::string& bgi_path)
    : pimpl_(std::unique_ptr<Impl>(new Impl(bgi_path))) {}

BgiReader::~BgiReader() = default;

BgiReader::BgiReader(BgiReader&&) noexcept = default;

BgiReader& BgiReader::operator=(BgiReader&&) noexcept = default;

std::vector<VariantInfo> BgiReader::query_region(const std::string& chromosome, uint32_t start_pos,
                                                 uint32_t end_pos) {
    return pimpl_->query_region(chromosome, start_pos, end_pos);
}

std::vector<VariantInfo> BgiReader::get_all_variants() {
    return pimpl_->get_all_variants();
}

size_t BgiReader::get_variant_count() const {
    return pimpl_->get_variant_count();
}

std::vector<VariantInfo> BgiReader::find_variants_by_filter(
    const std::string& chromosome, const std::vector<uint32_t>& positions,
    const std::vector<std::string>& alleles1, const std::vector<std::string>& alleles2,
    size_t batch_size) {
    return pimpl_->find_variants_by_filter(chromosome, positions, alleles1, alleles2, batch_size);
}

}  // namespace index
}  // namespace bgen
}  // namespace io
}  // namespace lazybgen