#ifndef LAZYBGEN_BGEN_FORMAT_GENOTYPE_PARSER_H
#define LAZYBGEN_BGEN_FORMAT_GENOTYPE_PARSER_H

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "bgen_header.h"

namespace lazybgen {
namespace bgen {

// Shared scalar per-sample dosage kernels for unphased biallelic diploid
// BGEN v1.2/v1.3 data. These are the single source of truth for the decode
// contract, shared by the scalar direct path, the scalar filtered path, and
// the scalar tail loops of the SIMD paths:
//
//   * two stored probabilities per sample (prob_aa, prob_ab), each with
//     bits_per_prob bits; prob_bb is implicit = max - prob_aa - prob_ab;
//   * dosage = (prob_ab + 2 * prob_bb) / max;
//   * an invalid probability sum (prob_aa + prob_ab > max) decodes to NaN.
//
// Missingness is handled by the callers (it is read from the separate
// missing/ploidy byte array), so these helpers only see already-non-missing
// samples and decide NaN solely on the probability-sum validity check.
//
// They are templated on the output type T so the float and double decode paths
// stay byte-identical: the double result is the exact widening of the float
// result. The NaN value is produced as static_cast<T>(std::nanf("")) so the
// float and double paths emit byte-identical NaNs.
template <typename T>
inline T decode_dosage_8bit(uint8_t prob_aa, uint8_t prob_ab) {
    if (prob_aa + prob_ab > 255) {
        return static_cast<T>(std::nanf(""));
    }
    uint8_t prob_bb = 255 - prob_aa - prob_ab;
    return static_cast<T>((prob_ab + 2.0f * prob_bb) / 255.0f);
}

template <typename T>
inline T decode_dosage_16bit(uint16_t prob_aa, uint16_t prob_ab) {
    if (prob_aa + prob_ab > 65535) {
        return static_cast<T>(std::nanf(""));
    }
    uint16_t prob_bb = 65535 - prob_aa - prob_ab;
    return static_cast<T>((prob_ab + 2.0f * prob_bb) / 65535.0f);
}

template <typename T>
inline T decode_dosage_32bit(uint32_t prob_aa, uint32_t prob_ab) {
    if (static_cast<uint64_t>(prob_aa) + prob_ab > 4294967295UL) {
        return static_cast<T>(std::nanf(""));
    }
    uint32_t prob_bb = 4294967295UL - prob_aa - prob_ab;
    // Use double precision to avoid overflow, then round to float first and
    // widen, so the double output is the exact widening of the float result.
    double dosage = (static_cast<double>(prob_ab) + 2.0 * prob_bb) / 4294967295.0;
    return static_cast<T>(static_cast<float>(dosage));
}

// BGEN v1.1 (layout 1) stores P(AA), P(AB) and P(BB) as 16-bit integers that
// the spec converts to probabilities by dividing by 32768; all three zero
// marks a missing call. The dosage is therefore (P(AB) + 2*P(BB)) / 32768.
// The numerator is an integer below 2^18 and 32768 is a power of two, so the
// float result is exact, and every v1.1 path (full, filtered, SIMD lanes and
// scalar tails, float and double) produces the same bits.
constexpr float kV11Scale = 1.0f / 32768.0f;

inline float decode_dosage_v11(uint16_t prob_aa, uint16_t prob_ab, uint16_t prob_bb) {
    if ((prob_aa | prob_ab | prob_bb) == 0) {
        return std::nanf("");
    }
    return static_cast<float>(static_cast<uint32_t>(prob_ab) + 2u * prob_bb) * kV11Scale;
}

// Structure to hold genotype data
struct GenotypeData {
    uint32_t n_samples;
    uint16_t n_alleles;
    bool phased;
    std::vector<uint8_t> ploidy;       // Ploidy for each sample
    std::vector<float> probabilities;  // Genotype probabilities
    std::vector<bool> missing;         // Missing data flags
    uint8_t min_ploidy;
    uint8_t max_ploidy;
    bool constant_ploidy;

    GenotypeData()
        : n_samples(0),
          n_alleles(0),
          phased(false),
          min_ploidy(0),
          max_ploidy(0),
          constant_ploidy(true) {}

    // Calculate dosages for specific samples only
    void extract_dosages_filtered(const int* sample_indices, int n_indices, float* output) const;
};

// Genotype parser class
// Range and missing-call summary of one decoded column, filled by the decode
// kernels that can produce it in the same pass (valid = true). When valid is
// false the caller computes it from the column instead.
struct ColumnStats {
    double min_value = 0.0;
    double max_value = 0.0;
    bool has_nan = false;
    bool valid = false;
};

class GenotypeParser {
   public:
    /**
     * Parse genotype data from buffer
     * @param buffer Pointer to genotype data (may be compressed)
     * @param size Size of buffer
     * @param layout BGEN layout version
     * @param compression Compression type
     * @param n_samples Number of samples
     * @param n_alleles Number of alleles
     * @return Parsed genotype data
     */
    static std::unique_ptr<GenotypeData> parse(const uint8_t* buffer, size_t size,
                                               LayoutType layout, CompressionType compression,
                                               uint32_t n_samples, uint16_t n_alleles);

    /**
     * Parse genotype data from already decompressed buffer
     * @param buffer Pointer to decompressed genotype data
     * @param size Size of buffer
     * @param layout BGEN layout version
     * @param n_samples Number of samples
     * @param n_alleles Number of alleles
     * @return Parsed genotype data
     */
    static std::unique_ptr<GenotypeData> parse_decompressed(const uint8_t* buffer, size_t size,
                                                           LayoutType layout, uint32_t n_samples,
                                                           uint16_t n_alleles);

    /**
     * Compute dosages directly without full parsing (for efficiency)
     * @param buffer Pointer to genotype data
     * @param size Size of buffer
     * @param layout BGEN layout version
     * @param compression Compression type
     * @param n_samples Number of samples
     * @param n_alleles Number of alleles
     * @param output Pre-allocated array for dosages (size: n_samples)
     */
    // Overloaded on output precision. The float overload writes float32 dosages
    // directly; the double overload writes widened dosages directly from the
    // decode kernel, avoiding a separate float32->float64 cast pass in the caller
    // (~20% of a default float64 full load). Both produce byte-identical values
    // (the double output is the exact widening of the float result).
    // If stats is non-null, it is set valid when the decode also produced the
    // column's range (unphased 8/16-bit v1.2), and left invalid otherwise.
    static void compute_dosages_direct(const uint8_t* buffer, size_t size, LayoutType layout,
                                     CompressionType compression, uint32_t n_samples,
                                     uint16_t n_alleles, float* output,
                                     ColumnStats* stats = nullptr);
    static void compute_dosages_direct(const uint8_t* buffer, size_t size, LayoutType layout,
                                     CompressionType compression, uint32_t n_samples,
                                     uint16_t n_alleles, double* output,
                                     ColumnStats* stats = nullptr);

    /**
     * Compute dosages for specific samples only
     * @param buffer Pointer to genotype data
     * @param size Size of buffer
     * @param layout BGEN layout version
     * @param compression Compression type
     * @param n_samples Number of samples in data
     * @param n_alleles Number of alleles
     * @param sample_indices Array of sample indices to extract
     * @param n_indices Number of indices
     * @param output Pre-allocated array for dosages (size: n_indices)
     */
    static void compute_dosages_filtered(const uint8_t* buffer, size_t size, LayoutType layout,
                                       CompressionType compression, uint32_t n_samples,
                                       uint16_t n_alleles, const int* sample_indices, int n_indices,
                                       float* output);

   private:
    // Parse v1.1 format genotypes (generic path; v1.2/v1.3 use the dedicated
    // compute_dosages_v12_direct/filtered fast paths)
    static std::unique_ptr<GenotypeData> parse_v11(const uint8_t* buffer, size_t size,
                                                  uint32_t n_samples);

    // Shared implementation behind the two compute_dosages_direct overloads.
    template <typename T>
    static void compute_dosages_direct_impl(const uint8_t* buffer, size_t size, LayoutType layout,
                                         CompressionType compression, uint32_t n_samples,
                                         uint16_t n_alleles, T* output, ColumnStats* stats);

    // Direct dosage computation for v1.1 (float-only: uses a float SIMD kernel).
    static void compute_dosages_v11_direct(const uint8_t* buffer, size_t size, uint32_t n_samples,
                                        float* output);

    // Type-dispatched v1.1 helper: float writes directly; double decodes into a
    // float scratch and widens (v1.1 is a legacy path, so this stays simple).
    static void compute_dosages_v11_typed(const uint8_t* buffer, size_t size, uint32_t n_samples,
                                float* output);
    static void compute_dosages_v11_typed(const uint8_t* buffer, size_t size, uint32_t n_samples,
                                double* output);

    // Direct dosage computation for v1.2 (scalar; templated on output precision).
    // Dispatches at runtime to the same body compiled with or without AVX2/FMA
    // (see simd_target.h): the scalar loop runs measurably faster with VEX
    // encoding, and the AVX2 build keeps results identical to earlier releases.
    template <typename T>
    static void compute_dosages_v12_direct(const uint8_t* buffer, size_t size, uint32_t n_samples,
                                        uint16_t n_alleles, T* output,
                                        ColumnStats* stats);
    template <typename T>
    static void compute_dosages_v12_direct_body(const uint8_t* buffer, size_t size,
                                             uint32_t n_samples, uint16_t n_alleles, T* output,
                                        ColumnStats* stats);
    template <typename T>
    static void compute_dosages_v12_direct_avx2(const uint8_t* buffer, size_t size,
                                             uint32_t n_samples, uint16_t n_alleles, T* output,
                                        ColumnStats* stats);

    // Optimized filtered dosage computation for v1.2
    static void compute_dosages_v12_filtered(const uint8_t* buffer, size_t size, uint32_t n_samples,
                                          uint16_t n_alleles, const int* sample_indices,
                                          int n_indices, float* output);

    // Dosage computation for phased biallelic diploid v1.2/v1.3 data. Phased data
    // stores one probability per haplotype (P of the first allele), so the
    // alt-allele dosage is the sum over the two haplotypes of (max - stored)/max.
    // Handles any bits-per-probability in [1,32] via a general bit reader.
    // prob_bytes_available is the number of bytes left in the buffer at prob_ptr.
    // If sample_indices is null, all n_samples dosages are written to output;
    // otherwise the n_indices requested samples are written (out-of-range -> NaN).
    template <typename T>
    static void compute_dosages_v12_phased(const uint8_t* missing_ptr, const uint8_t* prob_ptr,
                                        size_t prob_bytes_available, uint32_t n_samples,
                                        uint8_t bits_per_prob, const int* sample_indices,
                                        int n_indices, T* output);
    // The phased body, compiled with and without AVX2/FMA like the direct one.
    template <typename T>
    static void compute_dosages_v12_phased_body(const uint8_t* missing_ptr, const uint8_t* prob_ptr,
                                        size_t prob_bytes_available, uint32_t n_samples,
                                        uint8_t bits_per_prob, const int* sample_indices,
                                        int n_indices, T* output);
    template <typename T>
    static void compute_dosages_v12_phased_avx2(const uint8_t* missing_ptr, const uint8_t* prob_ptr,
                                        size_t prob_bytes_available, uint32_t n_samples,
                                        uint8_t bits_per_prob, const int* sample_indices,
                                        int n_indices, T* output);

    // Helper to read little-endian integers
    template <typename T>
    static T read_le(const uint8_t* ptr) {
        T value = 0;
        for (size_t i = 0; i < sizeof(T); ++i) {
            value |= static_cast<T>(ptr[i]) << (8 * i);
        }
        return value;
    }
};

}  // namespace bgen
}  // namespace lazybgen

#endif  // LAZYBGEN_BGEN_FORMAT_GENOTYPE_PARSER_H