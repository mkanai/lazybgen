#include "genotype_parser.h"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "genotype_parser_simd.h"
#include "simd_target.h"

namespace lazybgen {
namespace bgen {

namespace {
// Read nbits (1..32) from a bit-packed little-endian stream starting at the given
// absolute bit offset. Bits are consumed least-significant-first within each byte
// and accumulate from the least-significant end of the value, matching the BGEN
// layout-2 probability encoding.
inline uint32_t read_bits_le(const uint8_t* data, uint64_t bit_offset, uint8_t nbits) {
    uint32_t value = 0;
    for (uint8_t i = 0; i < nbits; ++i) {
        uint64_t b = bit_offset + i;
        value |= static_cast<uint32_t>((data[b >> 3] >> (b & 7u)) & 1u) << i;
    }
    return value;
}
}  // namespace

// GenotypeData member implementations
void GenotypeData::extract_dosages_filtered(const int* sample_indices, int n_indices,
                                            float* output) const {
    if (n_alleles != 2) {
        throw std::runtime_error("Dosage computation only supported for biallelic variants");
    }

    for (int i = 0; i < n_indices; ++i) {
        int idx = sample_indices[i];
        if (idx < 0 || idx >= static_cast<int>(n_samples)) {
            throw std::runtime_error("Sample index out of bounds");
        }

        if (missing[idx]) {
            output[i] = std::nanf("");
        } else {
            size_t offset = static_cast<size_t>(idx) * 3;
            output[i] = probabilities[offset + 1] + 2.0f * probabilities[offset + 2];
        }
    }
}

// GenotypeParser implementations
std::unique_ptr<GenotypeData> GenotypeParser::parse_v11(const uint8_t* buffer, size_t size,
                                                       uint32_t n_samples) {
    auto data = std::unique_ptr<GenotypeData>(new GenotypeData());
    data->n_samples = n_samples;
    data->n_alleles = 2;  // v1.1 is always biallelic
    data->phased = false;
    data->constant_ploidy = true;
    data->min_ploidy = 2;
    data->max_ploidy = 2;

    // v1.1 format: 6 bytes per sample
    if (size < static_cast<size_t>(n_samples) * 6) {
        throw std::runtime_error("Buffer too small for v1.1 genotype data");
    }

    data->ploidy.resize(n_samples, 2);
    data->probabilities.resize(n_samples * 3);
    data->missing.resize(n_samples);

    const uint8_t* ptr = buffer;

    for (uint32_t i = 0; i < n_samples; ++i) {
        // Read 3 probabilities (2 bytes each)
        uint16_t prob_aa = read_le<uint16_t>(ptr);
        ptr += 2;
        uint16_t prob_ab = read_le<uint16_t>(ptr);
        ptr += 2;
        uint16_t prob_bb = read_le<uint16_t>(ptr);
        ptr += 2;

        // Check for missing data (all probs = 0)
        if (prob_aa == 0 && prob_ab == 0 && prob_bb == 0) {
            data->missing[i] = true;
            data->probabilities[i * 3] = 0.0f;
            data->probabilities[i * 3 + 1] = 0.0f;
            data->probabilities[i * 3 + 2] = 0.0f;
        } else {
            data->missing[i] = false;
            // The spec scales each stored integer by 1/32768 (exact: a power of
            // two); the three need not sum to exactly 1.
            data->probabilities[i * 3] = prob_aa * kV11Scale;
            data->probabilities[i * 3 + 1] = prob_ab * kV11Scale;
            data->probabilities[i * 3 + 2] = prob_bb * kV11Scale;
        }
    }

    return data;
}

std::unique_ptr<GenotypeData> GenotypeParser::parse(const uint8_t* buffer, size_t size,
                                                    LayoutType layout, CompressionType compression,
                                                    uint32_t n_samples, uint16_t n_alleles) {
    // Handle decompression if needed
    std::vector<uint8_t> decompressed;
    const uint8_t* data_ptr = buffer;
    size_t data_size = size;

    if (compression != CompressionType::None) {
        // For now, throw an error - decompression should be handled externally
        throw std::runtime_error("Compressed genotype data should be decompressed before parsing");
    }

    return parse_decompressed(data_ptr, data_size, layout, n_samples, n_alleles);
}

std::unique_ptr<GenotypeData> GenotypeParser::parse_decompressed(const uint8_t* buffer, size_t size,
                                                                LayoutType layout,
                                                                uint32_t n_samples,
                                                                uint16_t n_alleles) {
    (void)n_alleles;
    if (layout == LayoutType::V11) {
        return parse_v11(buffer, size, n_samples);
    }
    // v1.2/v1.3 dosages are computed via the dedicated compute_dosages_v12_direct /
    // compute_dosages_v12_filtered fast paths; the generic parse path only handles
    // v1.1, so anything else here is unexpected.
    throw std::runtime_error("Generic genotype parsing is only supported for BGEN v1.1");
}

void GenotypeParser::compute_dosages_v11_direct(const uint8_t* buffer, size_t size, uint32_t n_samples,
                                             float* output) {
    if (size < static_cast<size_t>(n_samples) * 6) {
        throw std::runtime_error("Buffer too small for v1.1 genotype data");
    }

    // Check if we can use SIMD optimization
    if (can_use_simd_dosage()) {
        // Use SIMD-optimized implementation
        simd::compute_dosages_v11_simd(buffer, n_samples, output);
        return;
    }

    // Fallback to scalar implementation
    const uint8_t* ptr = buffer;

    for (uint32_t i = 0; i < n_samples; ++i) {
        // Read 3 probabilities (2 bytes each)
        uint16_t prob_aa = read_le<uint16_t>(ptr);
        ptr += 2;
        uint16_t prob_ab = read_le<uint16_t>(ptr);
        ptr += 2;
        uint16_t prob_bb = read_le<uint16_t>(ptr);
        ptr += 2;

        output[i] = decode_dosage_v11(prob_aa, prob_ab, prob_bb);
    }
}

template <typename T>
void GenotypeParser::compute_dosages_v12_direct(const uint8_t* buffer, size_t size, uint32_t n_samples,
                                             uint16_t n_alleles, T* output, ColumnStats* stats) {
#ifdef LAZYBGEN_X86_AVX2
    if (cpu_has_avx2()) {
        compute_dosages_v12_direct_avx2<T>(buffer, size, n_samples, n_alleles, output, stats);
        return;
    }
#endif
    compute_dosages_v12_direct_body<T>(buffer, size, n_samples, n_alleles, output, stats);
}

#ifdef LAZYBGEN_X86_AVX2
template <typename T>
LAZYBGEN_TARGET_AVX2 void GenotypeParser::compute_dosages_v12_direct_avx2(
    const uint8_t* buffer, size_t size, uint32_t n_samples, uint16_t n_alleles, T* output,
    ColumnStats* stats) {
    compute_dosages_v12_direct_body<T>(buffer, size, n_samples, n_alleles, output, stats);
}
#endif

template <typename T>
LAZYBGEN_ALWAYS_INLINE void GenotypeParser::compute_dosages_v12_direct_body(
    const uint8_t* buffer, size_t size, uint32_t n_samples, uint16_t n_alleles, T* output,
    ColumnStats* stats) {
    if (n_alleles != 2) {
        throw std::runtime_error("Direct dosage computation only supported for biallelic variants");
    }

    // Uncompressed data should not reach this point as it's blocked at the reader level
    // If we somehow get here with what looks like uncompressed data, reject it

    if (size < 10) {  // Minimum size for header
        throw std::runtime_error("Buffer too small for v1.2 genotype data");
    }

    // Parse header to get to probability data
    const uint8_t* ptr = buffer;

    // Verify n_samples
    uint32_t n_samples_check = read_le<uint32_t>(ptr);
    ptr += 4;
    if (n_samples_check != n_samples) {
        throw std::runtime_error("Sample count mismatch in genotype data");
    }

    // Verify n_alleles
    uint16_t n_alleles_check = read_le<uint16_t>(ptr);
    ptr += 2;
    if (n_alleles_check != n_alleles) {
        throw std::runtime_error("Allele count mismatch in genotype data");
    }

    uint8_t min_ploidy = *ptr++;
    uint8_t max_ploidy = *ptr++;

    min_ploidy &= 0x3F;
    max_ploidy &= 0x3F;
    // The dosage layout assumes diploid (two stored probabilities per sample).
    if (min_ploidy != 2 || max_ploidy != 2) {
        throw std::runtime_error(
            "Only diploid (ploidy 2) BGEN data is supported. lazybgen computes "
            "dosages for biallelic diploid variants (phased or unphased) only.");
    }

    // Save pointer to missing data for single-pass processing
    const uint8_t* missing_data_ptr = ptr;

    // Validate the buffer holds the missing/ploidy array plus the two following
    // header bytes (phased flag, bits per probability) BEFORE skipping over the
    // missing section and reading those bytes. Without this, a buffer with the
    // fixed 10-byte prefix and a matching n_samples could drive an OOB read.
    // ptr currently sits just past the 8-byte prefix consumed so far. Use
    // subtraction against the remaining byte count so the bound check cannot
    // overflow (relevant on 32-bit size_t).
    size_t consumed = static_cast<size_t>(ptr - buffer);
    size_t remaining = size - consumed;  // consumed <= size: size >= 10 checked above
    if (remaining < static_cast<size_t>(n_samples) || remaining - n_samples < 2) {
        throw std::runtime_error("Buffer too small for v1.2 genotype data");
    }

    // Skip missing data section
    ptr += n_samples;

    uint8_t phased_flag = *ptr++;
    bool phased = (phased_flag & 1) != 0;

    // Read bits per probability
    uint8_t bits_per_prob = *ptr++;

    if (phased) {
        // Phased biallelic diploid: decode the per-haplotype probabilities into
        // the alt-allele dosage. The phased path uses a general bit reader and
        // accepts any bits-per-probability in [1,32].
        size_t header_bytes = static_cast<size_t>(ptr - buffer);
        compute_dosages_v12_phased(missing_data_ptr, ptr, size - header_bytes, n_samples,
                                bits_per_prob, nullptr, 0, output);
        return;
    }

    if (bits_per_prob != 8 && bits_per_prob != 16 && bits_per_prob != 32) {
        throw std::runtime_error("Unsupported bits per probability: " +
                                 std::to_string(bits_per_prob));
    }

    // Validate the buffer holds all probability data BEFORE reading it. Two
    // stored probabilities per sample (diploid biallelic), bits_per_prob bits
    // each. Checking after the loop instead would mean the read had already
    // walked past the end of a truncated buffer. Use division against
    // the remaining byte count so the bound check cannot overflow (relevant on
    // 32-bit size_t); bytes_per_sample is 2, 4, or 8 (never 0).
    size_t bytes_per_sample = (bits_per_prob / 8) * 2;
    size_t header_bytes = static_cast<size_t>(ptr - buffer);
    size_t prob_remaining = size - header_bytes;  // header_bytes <= size: validated above
    if (prob_remaining / bytes_per_sample < n_samples) {
        throw std::runtime_error("Buffer too small for v1.2 genotype data");
    }

    // Single-pass processing: read missing status and probabilities together
    const uint8_t* prob_ptr = ptr;

    // 8- and 16-bit go through the fused kernels, which decode branch-free
    // (vectorized where the CPU allows) and return the column's range in the
    // same pass. The earlier finding that vectorizing only the arithmetic of
    // this loop bought ~3% still holds; the cost was in the per-sample branches
    // and one-at-a-time loads and stores, which these kernels remove, and in the
    // caller's second pass over the column, which the range replaces.
    if (bits_per_prob == 8 || bits_per_prob == 16) {
        const simd::NumeratorRange r =
            bits_per_prob == 8
                ? simd::decode_v12_unphased_8bit<T>(missing_data_ptr, prob_ptr, n_samples, output)
                : simd::decode_v12_unphased_16bit<T>(missing_data_ptr, prob_ptr, n_samples, output);
        if (stats) {
            const float m = bits_per_prob == 8 ? 255.0f : 65535.0f;
            stats->has_nan = r.any_nan;
            if (r.k_min <= r.k_max) {
                stats->min_value = static_cast<double>(static_cast<float>(r.k_min) / m);
                stats->max_value = static_cast<double>(static_cast<float>(r.k_max) / m);
            } else {
                stats->min_value = std::numeric_limits<double>::infinity();
                stats->max_value = -std::numeric_limits<double>::infinity();
            }
            stats->valid = true;
        }
        return;
    }

    // 32-bit: scalar, one sample at a time, through the shared kernel (the
    // single source of truth for the arithmetic and the invalid-sum -> NaN rule).
    for (uint32_t i = 0; i < n_samples; ++i, prob_ptr += 8) {
        if (missing_data_ptr[i] & 0x80) {
            output[i] = static_cast<T>(std::nanf(""));
        } else {
            output[i] = decode_dosage_32bit<T>(read_le<uint32_t>(prob_ptr),
                                               read_le<uint32_t>(prob_ptr + 4));
        }
    }

    // Verify we consumed the expected amount of data
    size_t expected_ptr_offset = prob_ptr - buffer;
    if (expected_ptr_offset > size) {
        throw std::runtime_error("Buffer overrun while parsing genotype data");
    }
}

void GenotypeParser::compute_dosages_v12_filtered(const uint8_t* buffer, size_t size,
                                               uint32_t n_samples, uint16_t n_alleles,
                                               const int* sample_indices, int n_indices,
                                               float* output) {
    if (n_alleles != 2) {
        throw std::runtime_error(
            "Filtered dosage computation only supported for biallelic variants");
    }

    if (size < 10) {  // Minimum size for header
        throw std::runtime_error("Buffer too small for v1.2 genotype data");
    }

    // Parse header
    const uint8_t* ptr = buffer;

    // Verify n_samples
    uint32_t n_samples_check = read_le<uint32_t>(ptr);
    ptr += 4;
    if (n_samples_check != n_samples) {
        throw std::runtime_error("Sample count mismatch in genotype data");
    }

    // Verify n_alleles
    uint16_t n_alleles_check = read_le<uint16_t>(ptr);
    ptr += 2;
    if (n_alleles_check != n_alleles) {
        throw std::runtime_error("Allele count mismatch in genotype data");
    }

    uint8_t min_ploidy = *ptr++;
    uint8_t max_ploidy = *ptr++;

    min_ploidy &= 0x3F;
    max_ploidy &= 0x3F;
    // The dosage layout assumes diploid (two stored probabilities per sample).
    // This also guarantees constant ploidy, so missingness is one byte per
    // sample for every sample below.
    if (min_ploidy != 2 || max_ploidy != 2) {
        throw std::runtime_error(
            "Only diploid (ploidy 2) BGEN data is supported. lazybgen computes "
            "dosages for biallelic diploid variants (phased or unphased) only.");
    }

    // Save pointer to missing data start
    const uint8_t* missing_data_ptr = ptr;

    // Validate the buffer holds the missing/ploidy array plus the two following
    // header bytes (phased flag, bits per probability) BEFORE skipping over the
    // missing section and reading those bytes. ptr currently sits just past the
    // 8-byte prefix consumed so far. Use subtraction against the remaining byte
    // count so the bound check cannot overflow (relevant on 32-bit size_t).
    size_t consumed = static_cast<size_t>(ptr - buffer);
    size_t remaining = size - consumed;  // consumed <= size: size >= 10 checked above
    if (remaining < static_cast<size_t>(n_samples) || remaining - n_samples < 2) {
        throw std::runtime_error("Buffer too small for v1.2 genotype data");
    }

    // Skip the per-sample ploidy/missingness section: BGEN v1.2 uses n_samples
    // bytes here for both constant and variable ploidy.
    ptr += n_samples;

    uint8_t phased_flag = *ptr++;
    bool phased = (phased_flag & 1) != 0;

    // Read bits per probability
    uint8_t bits_per_prob = *ptr++;

    if (phased) {
        // Phased biallelic diploid: decode the requested samples' per-haplotype
        // probabilities into alt-allele dosages. Accepts any bits-per-probability
        // in [1,32] via a general bit reader.
        size_t header_bytes = static_cast<size_t>(ptr - buffer);
        compute_dosages_v12_phased(missing_data_ptr, ptr, size - header_bytes, n_samples,
                                bits_per_prob, sample_indices, n_indices, output);
        return;
    }

    if (bits_per_prob != 8 && bits_per_prob != 16 && bits_per_prob != 32) {
        throw std::runtime_error("Unsupported bits per probability: " +
                                 std::to_string(bits_per_prob));
    }

    // Save pointer to probability data start
    const uint8_t* prob_data_ptr = ptr;

    // Validate the buffer holds all probability data before indexing into it.
    // Two stored probabilities per sample (diploid biallelic), bits_per_prob
    // bits each. Use division against the remaining byte count so the bound
    // check cannot overflow (relevant on 32-bit size_t); bytes_per_sample is
    // 2, 4, or 8 (never 0).
    size_t bytes_per_sample = (bits_per_prob / 8) * 2;
    size_t header_bytes = static_cast<size_t>(ptr - buffer);
    size_t prob_remaining = size - header_bytes;  // header_bytes <= size: validated above
    if (prob_remaining / bytes_per_sample < n_samples) {
        throw std::runtime_error("Buffer too small for v1.2 genotype data");
    }

    // The SIMD helpers index missing_mask[idx] and prob_data + idx * stride
    // directly with no bounds check, so an out-of-range sample index would be an
    // out-of-bounds read. The scalar contract emits NaN for such indices instead.
    // Validate up front: only take the SIMD fast path when every requested index
    // is in range; otherwise fall through to the scalar loop, which handles
    // out-of-range indices by emitting NaN.
    bool all_indices_in_range = true;
    for (int idx = 0; idx < n_indices; ++idx) {
        int sample_idx = sample_indices[idx];
        if (sample_idx < 0 || sample_idx >= static_cast<int>(n_samples)) {
            all_indices_in_range = false;
            break;
        }
    }

    // Check if we can use SIMD for filtered computation
    if (all_indices_in_range && can_use_simd_dosage()) {
        // Use SIMD-optimized filtered computation
        simd::compute_dosages_filtered_simd(prob_data_ptr, output, sample_indices, n_indices,
                                            bits_per_prob, missing_data_ptr);
        return;
    }

    // Fallback to scalar processing
    // Process only requested samples
    for (int idx = 0; idx < n_indices; ++idx) {
        int sample_idx = sample_indices[idx];

        // Validate sample index
        if (sample_idx < 0 || sample_idx >= static_cast<int>(n_samples)) {
            output[idx] = std::nanf("");
            continue;
        }

        // Check if sample is missing. BGEN v1.2/v1.3 stores one byte per sample
        // (high bit 0x80 = missing, low 6 bits = ploidy) for both constant and
        // variable ploidy, so the read is the same in both cases.
        bool is_missing = (missing_data_ptr[sample_idx] & 0x80) != 0;

        if (is_missing) {
            output[idx] = std::nanf("");
            continue;
        }

        // Calculate offset to this sample's probability data
        const uint8_t* sample_prob_ptr = prob_data_ptr;

        if (bits_per_prob == 8) {
            // 2 bytes per sample
            sample_prob_ptr += static_cast<size_t>(sample_idx) * 2;

            uint8_t prob_aa = sample_prob_ptr[0];
            uint8_t prob_ab = sample_prob_ptr[1];
            output[idx] = decode_dosage_8bit<float>(prob_aa, prob_ab);
        } else if (bits_per_prob == 16) {
            // 4 bytes per sample
            sample_prob_ptr += static_cast<size_t>(sample_idx) * 4;

            uint16_t prob_aa = read_le<uint16_t>(sample_prob_ptr);
            uint16_t prob_ab = read_le<uint16_t>(sample_prob_ptr + 2);
            output[idx] = decode_dosage_16bit<float>(prob_aa, prob_ab);
        } else {  // 32 bits
            // 8 bytes per sample
            sample_prob_ptr += static_cast<size_t>(sample_idx) * 8;

            uint32_t prob_aa = read_le<uint32_t>(sample_prob_ptr);
            uint32_t prob_ab = read_le<uint32_t>(sample_prob_ptr + 4);
            output[idx] = decode_dosage_32bit<float>(prob_aa, prob_ab);
        }
    }
}

template <typename T>
LAZYBGEN_ALWAYS_INLINE void GenotypeParser::compute_dosages_v12_phased_body(
    const uint8_t* missing_ptr, const uint8_t* prob_ptr, size_t prob_bytes_available,
    uint32_t n_samples, uint8_t bits_per_prob, const int* sample_indices, int n_indices,
    T* output) {
    if (bits_per_prob < 1 || bits_per_prob > 32) {
        throw std::runtime_error("Unsupported bits per probability: " +
                                 std::to_string(bits_per_prob));
    }

    // Phased biallelic diploid stores two probabilities per sample (one per
    // haplotype), bits_per_prob bits each, bit-packed consecutively. Validate the
    // buffer holds all of them before decoding. uint64 math avoids overflow on
    // large sample counts (relevant on 32-bit size_t).
    uint64_t total_bits = static_cast<uint64_t>(n_samples) * 2u * bits_per_prob;
    uint64_t need_bytes = (total_bits + 7u) / 8u;
    if (static_cast<uint64_t>(prob_bytes_available) < need_bytes) {
        throw std::runtime_error("Buffer too small for v1.2 genotype data");
    }

    // For B == 32, (1u << 32) is undefined, so spell out the maximum directly.
    const double max_val =
        (bits_per_prob == 32) ? 4294967295.0 : static_cast<double>((1u << bits_per_prob) - 1u);

    auto dosage_for = [&](uint32_t sample) -> float {
        // Constant diploid ploidy guarantees one missingness byte per sample.
        if ((missing_ptr[sample] & 0x80) != 0) {
            return std::nanf("");
        }
        uint64_t bit0 = static_cast<uint64_t>(sample) * 2u * bits_per_prob;
        uint32_t h1 = read_bits_le(prob_ptr, bit0, bits_per_prob);
        uint32_t h2 = read_bits_le(prob_ptr, bit0 + bits_per_prob, bits_per_prob);
        // The stored value is P(haplotype == first allele), so the alt-allele
        // probability is (max - stored)/max. Alt dosage sums over both haplotypes.
        double dosage = (static_cast<double>(max_val - h1) + static_cast<double>(max_val - h2)) /
                        max_val;
        return static_cast<float>(dosage);
    };

    if (sample_indices == nullptr) {
        for (uint32_t s = 0; s < n_samples; ++s) {
            output[s] = static_cast<T>(dosage_for(s));
        }
    } else {
        for (int i = 0; i < n_indices; ++i) {
            int sidx = sample_indices[i];
            if (sidx < 0 || sidx >= static_cast<int>(n_samples)) {
                output[i] = static_cast<T>(std::nanf(""));
            } else {
                output[i] = static_cast<T>(dosage_for(static_cast<uint32_t>(sidx)));
            }
        }
    }
}

void GenotypeParser::compute_dosages_v11_typed(const uint8_t* buffer, size_t size, uint32_t n_samples,
                                     float* output) {
    compute_dosages_v11_direct(buffer, size, n_samples, output);
}

void GenotypeParser::compute_dosages_v11_typed(const uint8_t* buffer, size_t size, uint32_t n_samples,
                                     double* output) {
    // v1.1 uses a float-only SIMD kernel; for double output decode into a float
    // scratch and widen. v1.1 is a legacy path, so the extra copy is acceptable.
    std::vector<float> tmp(n_samples);
    compute_dosages_v11_direct(buffer, size, n_samples, tmp.data());
    for (uint32_t i = 0; i < n_samples; ++i) {
        output[i] = static_cast<double>(tmp[i]);
    }
}

template <typename T>
void GenotypeParser::compute_dosages_direct_impl(const uint8_t* buffer, size_t size, LayoutType layout,
                                              CompressionType compression, uint32_t n_samples,
                                              uint16_t n_alleles, T* output,
                                              ColumnStats* stats) {
    if (stats) {
        stats->valid = false;
    }
    if (compression != CompressionType::None) {
        throw std::runtime_error(
            "Compressed genotype data should be decompressed before computing dosages");
    }

    if (layout == LayoutType::V11) {
        compute_dosages_v11_typed(buffer, size, n_samples, output);
    } else if (layout == LayoutType::V12) {
        try {
            compute_dosages_v12_direct<T>(buffer, size, n_samples, n_alleles, output, stats);
        } catch (const std::runtime_error& e) {
            // Some BGEN files have V1.2 headers but V1.1 genotype blocks
            // If V1.2 parsing fails, try V1.1 format
            std::string error_msg = e.what();
            if (error_msg.find("Sample count mismatch") != std::string::npos ||
                error_msg.find("Allele count mismatch") != std::string::npos) {
                // Some BGEN files have malformed genotype data

                // Check if this could be V1.1 format (6 bytes per sample)
                if (size == static_cast<size_t>(n_samples) * 6) {
                    compute_dosages_v11_typed(buffer, size, n_samples, output);
                    return;
                }
            }
            // Re-throw the original error if fallback doesn't apply
            throw;
        }
    } else {
        throw std::runtime_error("Unsupported BGEN layout version");
    }
}

void GenotypeParser::compute_dosages_direct(const uint8_t* buffer, size_t size, LayoutType layout,
                                          CompressionType compression, uint32_t n_samples,
                                          uint16_t n_alleles, float* output, ColumnStats* stats) {
    compute_dosages_direct_impl<float>(buffer, size, layout, compression, n_samples, n_alleles, output,
                                       stats);
}

void GenotypeParser::compute_dosages_direct(const uint8_t* buffer, size_t size, LayoutType layout,
                                          CompressionType compression, uint32_t n_samples,
                                          uint16_t n_alleles, double* output, ColumnStats* stats) {
    compute_dosages_direct_impl<double>(buffer, size, layout, compression, n_samples, n_alleles,
                                        output, stats);
}

void GenotypeParser::compute_dosages_filtered(const uint8_t* buffer, size_t size, LayoutType layout,
                                            CompressionType compression, uint32_t n_samples,
                                            uint16_t n_alleles, const int* sample_indices,
                                            int n_indices, float* output) {
    // Use optimized implementation for v1.2 that only processes requested samples
    if (layout == LayoutType::V12 && compression == CompressionType::None) {
        compute_dosages_v12_filtered(buffer, size, n_samples, n_alleles, sample_indices, n_indices,
                                  output);
        return;
    }

    // v1.1: decode only the requested samples, with the same per-sample kernel
    // as the full path, so a filtered read is bitwise the full read.
    if (layout == LayoutType::V11 && compression == CompressionType::None) {
        if (size < static_cast<size_t>(n_samples) * 6) {
            throw std::runtime_error("Buffer too small for v1.1 genotype data");
        }
        for (int i = 0; i < n_indices; ++i) {
            const int idx = sample_indices[i];
            if (idx < 0 || static_cast<uint32_t>(idx) >= n_samples) {
                throw std::runtime_error("Sample index out of bounds");
            }
            const uint8_t* p = buffer + static_cast<size_t>(idx) * 6;
            output[i] = decode_dosage_v11(read_le<uint16_t>(p), read_le<uint16_t>(p + 2),
                                          read_le<uint16_t>(p + 4));
        }
        return;
    }

    // Anything else falls back to full parsing
    // (compressed data should already be decompressed before reaching here)
    auto data = parse(buffer, size, layout, compression, n_samples, n_alleles);
    data->extract_dosages_filtered(sample_indices, n_indices, output);
}

#ifdef LAZYBGEN_X86_AVX2
template <typename T>
LAZYBGEN_TARGET_AVX2 void GenotypeParser::compute_dosages_v12_phased_avx2(
    const uint8_t* missing_ptr, const uint8_t* prob_ptr, size_t prob_bytes_available,
    uint32_t n_samples, uint8_t bits_per_prob, const int* sample_indices, int n_indices,
    T* output) {
    compute_dosages_v12_phased_body<T>(missing_ptr, prob_ptr, prob_bytes_available, n_samples,
                                       bits_per_prob, sample_indices, n_indices, output);
}
#endif

// Phased decode, dispatched like compute_dosages_v12_direct: the AVX2/FMA build
// of the same body when the CPU has it, as 0.2.x compiled it everywhere.
template <typename T>
void GenotypeParser::compute_dosages_v12_phased(const uint8_t* missing_ptr, const uint8_t* prob_ptr,
                                             size_t prob_bytes_available, uint32_t n_samples,
                                             uint8_t bits_per_prob, const int* sample_indices,
                                             int n_indices, T* output) {
#ifdef LAZYBGEN_X86_AVX2
    if (cpu_has_avx2()) {
        compute_dosages_v12_phased_avx2<T>(missing_ptr, prob_ptr, prob_bytes_available, n_samples,
                                           bits_per_prob, sample_indices, n_indices, output);
        return;
    }
#endif
    compute_dosages_v12_phased_body<T>(missing_ptr, prob_ptr, prob_bytes_available, n_samples,
                                       bits_per_prob, sample_indices, n_indices, output);
}

template void GenotypeParser::compute_dosages_v12_phased<float>(const uint8_t*, const uint8_t*,
                                                             size_t, uint32_t, uint8_t, const int*,
                                                             int, float*);
template void GenotypeParser::compute_dosages_v12_phased<double>(const uint8_t*, const uint8_t*,
                                                              size_t, uint32_t, uint8_t,
                                                              const int*, int, double*);

// Explicit template instantiation
template uint16_t GenotypeParser::read_le<uint16_t>(const uint8_t*);
template uint32_t GenotypeParser::read_le<uint32_t>(const uint8_t*);

// Instantiate the direct-dosage decode for both output precisions. Instantiating
// the impl pulls in compute_dosages_v12_direct<T> and compute_dosages_v12_phased<T>.
template void GenotypeParser::compute_dosages_direct_impl<float>(const uint8_t*, size_t, LayoutType,
                                                             CompressionType, uint32_t, uint16_t,
                                                             float*, ColumnStats*);
template void GenotypeParser::compute_dosages_direct_impl<double>(const uint8_t*, size_t, LayoutType,
                                                              CompressionType, uint32_t, uint16_t,
                                                              double*, ColumnStats*);

}  // namespace bgen
}  // namespace lazybgen