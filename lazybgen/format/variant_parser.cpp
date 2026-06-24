#include "variant_parser.h"

#include <cstring>

namespace lazybgen {
namespace bgen {

std::string VariantParser::readLengthPrefixedString(const uint8_t* buffer, size_t& pos,
                                                    size_t max_size, bool use_32bit_length) {
    size_t length_size = use_32bit_length ? 4 : 2;

    if (pos + length_size > max_size) {
        throw std::runtime_error("Buffer too small for string length");
    }

    uint32_t length;
    if (use_32bit_length) {
        length = read_le<uint32_t>(buffer + pos);
    } else {
        length = read_le<uint16_t>(buffer + pos);
    }
    pos += length_size;

    if (pos + length > max_size) {
        throw std::runtime_error("Buffer too small for string data");
    }

    std::string result;
    if (length > 0) {
        result.assign(reinterpret_cast<const char*>(buffer + pos), length);
    }
    pos += length;

    return result;
}

std::pair<VariantMetadata, size_t> VariantParser::parseV11(const uint8_t* buffer, size_t size,
                                                           CompressionType compression,
                                                           uint32_t expected_samples) {
    VariantMetadata variant;
    size_t pos = 0;

    // Read number of samples (4 bytes)
    if (pos + 4 > size) {
        throw std::runtime_error("Buffer too small for v1.1 sample count");
    }
    uint32_t n_samples = read_le<uint32_t>(buffer + pos);
    pos += 4;

    if (n_samples != expected_samples) {
        throw std::runtime_error("Sample count mismatch in variant");
    }

    // Read variant ID
    variant.varid = readLengthPrefixedString(buffer, pos, size, false);

    // Read rsID
    variant.rsid = readLengthPrefixedString(buffer, pos, size, false);

    // Read chromosome
    variant.chromosome = readLengthPrefixedString(buffer, pos, size, false);

    // Read position (4 bytes)
    if (pos + 4 > size) {
        throw std::runtime_error("Buffer too small for position");
    }
    variant.position = read_le<uint32_t>(buffer + pos);
    pos += 4;

    // v1.1 is always biallelic
    variant.n_alleles = 2;
    variant.alleles.reserve(2);

    // Read alleles (each with 32-bit length prefix)
    for (int i = 0; i < 2; ++i) {
        variant.alleles.push_back(readLengthPrefixedString(buffer, pos, size, true));
    }

    // Calculate genotype data length
    if (compression == CompressionType::None) {
        // Uncompressed: 6 bytes per sample (v1.1 format only)
        // For v1.1, there's no length prefix
        variant.genotype_length = n_samples * 6;
    } else {
        // Compressed: read the length
        if (pos + 4 > size) {
            throw std::runtime_error("Buffer too small for compressed genotype length");
        }
        variant.genotype_length = read_le<uint32_t>(buffer + pos);
        pos += 4;
    }

    // Store genotype offset (relative to start of variant)
    variant.genotype_offset = pos;

    // Total size includes genotype data
    size_t total_size = pos + variant.genotype_length;

    return {std::move(variant), total_size};
}

std::pair<VariantMetadata, size_t> VariantParser::parseV12(const uint8_t* buffer, size_t size) {
    VariantMetadata variant;
    size_t pos = 0;

    // v1.2 format goes straight to variant ID (no variant data block length)

    // Read variant ID
    variant.varid = readLengthPrefixedString(buffer, pos, size, false);

    // Read rsID
    variant.rsid = readLengthPrefixedString(buffer, pos, size, false);

    // Read chromosome
    variant.chromosome = readLengthPrefixedString(buffer, pos, size, false);

    // Read position (4 bytes)
    if (pos + 4 > size) {
        throw std::runtime_error("Buffer too small for position");
    }
    variant.position = read_le<uint32_t>(buffer + pos);
    pos += 4;

    // Read number of alleles (2 bytes)
    if (pos + 2 > size) {
        throw std::runtime_error("Buffer too small for allele count");
    }
    variant.n_alleles = read_le<uint16_t>(buffer + pos);
    pos += 2;

    // Read each allele
    variant.alleles.reserve(variant.n_alleles);
    for (uint16_t i = 0; i < variant.n_alleles; ++i) {
        variant.alleles.push_back(readLengthPrefixedString(buffer, pos, size, true));
    }

    // Read genotype data block length (4 bytes)
    if (pos + 4 > size) {
        throw std::runtime_error("Buffer too small for genotype block length");
    }
    variant.genotype_length = read_le<uint32_t>(buffer + pos);
    pos += 4;

    // Store genotype offset (relative to start of variant)
    variant.genotype_offset = pos;

    // Total size includes genotype data
    size_t total_size = pos + variant.genotype_length;

    return {std::move(variant), total_size};
}

std::pair<VariantMetadata, size_t> VariantParser::parse(const uint8_t* buffer, size_t size,
                                                        LayoutType layout,
                                                        CompressionType compression,
                                                        uint32_t expected_samples) {
    if (layout == LayoutType::V11) {
        return parseV11(buffer, size, compression, expected_samples);
    } else if (layout == LayoutType::V12) {
        return parseV12(buffer, size);
    } else {
        throw std::runtime_error("Unsupported BGEN layout version");
    }
}

// Explicit template instantiation
template uint16_t VariantParser::read_le<uint16_t>(const uint8_t*);
template uint32_t VariantParser::read_le<uint32_t>(const uint8_t*);

}  // namespace bgen
}  // namespace lazybgen