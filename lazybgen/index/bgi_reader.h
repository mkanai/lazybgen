#ifndef LAZYBGEN_BGEN_INDEX_BGI_READER_H
#define LAZYBGEN_BGEN_INDEX_BGI_READER_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lazybgen {
namespace io {
namespace bgen {
namespace index {

/**
 * VariantInfo - Information about a variant stored in the BGI index
 */
struct VariantInfo {
    uint64_t file_offset;    // Offset in BGEN file
    uint32_t variant_size;   // Size of variant data in BGEN file
    std::string chromosome;  // Chromosome name
    uint32_t position;       // 1-based position
    std::string rsid;        // RS ID
    std::string varid;       // Variant ID
    uint16_t n_alleles;      // Number of alleles
    std::string allele1;     // First allele (reference)
    std::string allele2;     // Second allele (alternate)

    // Constructor
    VariantInfo() : file_offset(0), variant_size(0), position(0), n_alleles(0) {}
};

/**
 * BgiReader - Reader for BGEN index files (.bgi)
 *
 * The BGI format is a SQLite database with specific tables:
 * - Variant: contains variant metadata and file offsets
 * - Metadata: contains index metadata
 *
 * This implementation provides thread-safe access to the index.
 */
class BgiReader {
   public:
    /**
     * Constructor
     *
     * @param bgi_path Path to BGI file
     * @throws std::runtime_error if file cannot be opened or is invalid
     */
    explicit BgiReader(const std::string& bgi_path);

    /**
     * Destructor
     */
    ~BgiReader();

    // Disable copy constructor and assignment
    BgiReader(const BgiReader&) = delete;
    BgiReader& operator=(const BgiReader&) = delete;

    // Enable move constructor and assignment
    BgiReader(BgiReader&&) noexcept;
    BgiReader& operator=(BgiReader&&) noexcept;

    /**
     * Query variants by genomic region
     *
     * @param chromosome Chromosome name
     * @param start_pos Start position (1-based, inclusive)
     * @param end_pos End position (1-based, inclusive)
     * @return Vector of variant information
     */
    std::vector<VariantInfo> query_region(const std::string& chromosome, uint32_t start_pos,
                                          uint32_t end_pos);

    /**
     * Get all variant info efficiently
     *
     * @return Vector of all variant information
     */
    std::vector<VariantInfo> get_all_variants();

    /**
     * Get total number of variants in the index
     *
     * @return Number of variants
     */
    size_t get_variant_count() const;

    /**
     * Find variants matching chromosome, position, and allele combinations
     *
     * This method performs exact matching on chromosome, position, allele1, and allele2.
     * It uses batch queries for efficiency when searching for many variants.
     *
     * @param chromosome Chromosome to filter on
     * @param positions Positions to match
     * @param alleles1 First alleles (must match exactly)
     * @param alleles2 Second alleles (must match exactly)
     * @param batch_size Number of positions to query at once (default: 1000)
     * @return Vector of matched variants in order found
     * @throws std::invalid_argument if input vectors have different sizes
     */
    std::vector<VariantInfo> find_variants_by_filter(const std::string& chromosome,
                                                     const std::vector<uint32_t>& positions,
                                                     const std::vector<std::string>& alleles1,
                                                     const std::vector<std::string>& alleles2,
                                                     size_t batch_size = 1000);

   private:
    // Forward declaration of implementation
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

}  // namespace index
}  // namespace bgen
}  // namespace io
}  // namespace lazybgen

#endif  // LAZYBGEN_BGEN_INDEX_BGI_READER_H
