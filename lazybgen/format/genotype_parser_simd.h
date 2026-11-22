#ifndef LAZYBGEN_BGEN_FORMAT_GENOTYPE_PARSER_SIMD_H
#define LAZYBGEN_BGEN_FORMAT_GENOTYPE_PARSER_SIMD_H

#include <cstddef>
#include <cstdint>

namespace lazybgen {
namespace bgen {

/**
 * SIMD-optimized genotype parsing functions.
 *
 * These functions provide vectorized implementations for computing dosages
 * from genotype probabilities, achieving 2-3x speedup over scalar code.
 *
 * Features:
 * - AVX2 support for x86-64 (processes 8-16 samples at once)
 * - NEON support for ARM (processes 4-8 samples at once)
 * - Runtime CPU feature detection
 * - Fallback to scalar code when SIMD not available
 */

// Check if this sample's dosage computation can use SIMD
bool can_use_simd_dosage();

// True when the CPU supports AVX2 and FMA (x86-64 only; false elsewhere). The
// extension is built for baseline x86-64, so AVX2 code paths must check this.
bool cpu_has_avx2();

// SIMD implementations for different bit depths
namespace simd {

/**
 * Compute filtered dosages using SIMD (processes only selected samples).
 *
 * @param prob_data Pointer to probability data
 * @param output Output dosages array
 * @param sample_indices Array of sample indices to process
 * @param n_indices Number of indices
 * @param bits_per_prob Bits per probability (8, 16, or 32)
 * @param missing_mask Bit mask for missing samples (can be nullptr)
 */
void compute_dosages_filtered_simd(const uint8_t* prob_data, float* output,
                                   const int* sample_indices, size_t n_indices,
                                   uint8_t bits_per_prob, const uint8_t* missing_mask = nullptr);

/**
 * Compute dosages for BGEN v1.1 format using SIMD.
 *
 * @param buffer Pointer to v1.1 genotype data (6 bytes per sample)
 * @param n_samples Number of samples to process
 * @param output Output dosages array
 */
void compute_dosages_v11_simd(const uint8_t* buffer, size_t n_samples, float* output);

/**
 * Range of the integer dosage numerators a fused full-sample kernel saw.
 *
 * For unphased v1.2 data with b bits per probability and M = 2^b - 1, a valid
 * sample's dosage is float(k) / M with k = 2*M - 2*P(AA) - P(AB), an integer.
 * Correctly rounded division is monotone in k, so the column's min and max
 * dosage are float(k_min) / M and float(k_max) / M exactly, and the kernels
 * track k instead of rereading the decoded column. k_min > k_max means no
 * sample was valid; any_nan reports a missing call or an invalid sum.
 */
struct NumeratorRange {
    uint32_t k_min;
    uint32_t k_max;
    bool any_nan;
};

/**
 * Decode all n samples of an unphased 8-bit (or 16-bit) v1.2 probability block
 * into dosages, and return the numerator range in the same pass.
 *
 * Output is bitwise what decode_dosage_8bit / decode_dosage_16bit produce: a
 * missing call (bit 7 of its ploidy byte) or an invalid sum decodes to
 * static_cast<T>(std::nanf("")), and a double column is the exact widening of
 * the float result. Uses AVX2 when the CPU has it, NEON on AArch64, and a
 * branch-free scalar loop otherwise. The caller has already checked that
 * missing holds n bytes and prob holds n pairs.
 */
template <typename T>
NumeratorRange decode_v12_unphased_8bit(const uint8_t* missing, const uint8_t* prob, size_t n,
                                        T* output);
template <typename T>
NumeratorRange decode_v12_unphased_16bit(const uint8_t* missing, const uint8_t* prob, size_t n,
                                         T* output);

}  // namespace simd

}  // namespace bgen
}  // namespace lazybgen

#endif  // LAZYBGEN_BGEN_FORMAT_GENOTYPE_PARSER_SIMD_H