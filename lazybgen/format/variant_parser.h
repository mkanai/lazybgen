#ifndef LAZYBGEN_BGEN_FORMAT_VARIANT_PARSER_H
#define LAZYBGEN_BGEN_FORMAT_VARIANT_PARSER_H

#include <cstdint>
#include <string>
#include <vector>

#include "bgen_header.h"

namespace lazybgen {
namespace bgen {

// Structure to hold variant metadata
struct VariantMetadata {
    uint64_t file_offset;              // Offset in file where variant starts
    std::string varid;                 // Variant ID
    std::string rsid;                  // RS ID
    std::string chromosome;            // Chromosome
    uint32_t position;                 // Position
    uint16_t n_alleles;                // Number of alleles
    std::vector<std::string> alleles;  // Allele strings
    uint64_t genotype_offset;          // Offset to genotype data
    uint32_t genotype_length;          // Length of genotype data block
    uint32_t variant_size;             // Total record size (BGI size_in_bytes); 0 = unknown

    VariantMetadata()
        : file_offset(0),
          position(0),
          n_alleles(0),
          genotype_offset(0),
          genotype_length(0),
          variant_size(0) {}
};

// Variant parser class
class VariantParser {
   public:
    /**
     * Parse variant metadata from buffer
     * @param buffer Pointer to variant data
     * @param size Size of buffer
     * @param layout BGEN layout version
     * @param compression Compression type (needed for v1.1 uncompressed size calculation)
     * @param expected_samples Expected number of samples (for v1.1 validation)
     * @return Parsed variant metadata and bytes consumed
     */
    static std::pair<VariantMetadata, size_t> parse(const uint8_t* buffer, size_t size,
                                                    LayoutType layout, CompressionType compression,
                                                    uint32_t expected_samples);

   private:
    // Parse v1.1 format variant
    static std::pair<VariantMetadata, size_t> parseV11(const uint8_t* buffer, size_t size,
                                                       CompressionType compression,
                                                       uint32_t expected_samples);

    // Parse v1.2 format variant
    static std::pair<VariantMetadata, size_t> parseV12(const uint8_t* buffer, size_t size);

    // Helper to read little-endian integers
    template <typename T>
    static T read_le(const uint8_t* ptr) {
        T value = 0;
        for (size_t i = 0; i < sizeof(T); ++i) {
            value |= static_cast<T>(ptr[i]) << (8 * i);
        }
        return value;
    }

    // Helper to read a length-prefixed string
    static std::string readLengthPrefixedString(const uint8_t* buffer, size_t& pos, size_t max_size,
                                                bool use_32bit_length = false);
};

}  // namespace bgen
}  // namespace lazybgen

#endif  // LAZYBGEN_BGEN_FORMAT_VARIANT_PARSER_H