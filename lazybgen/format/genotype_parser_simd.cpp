#include "genotype_parser_simd.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>

#include "genotype_parser.h"  // shared scalar dosage kernels (decode_dosage_*bit)

#ifdef __x86_64__
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace lazybgen {
namespace bgen {

// CPU feature detection. Initialized exactly once via std::call_once: the decode
// kernels (filtered v1.2, v1.1) call can_use_simd_dosage() from the parallel
// decode workers, so a lazy unsynchronized init would be a data race.
static std::once_flag g_simd_once;
static bool g_has_avx2 = false;
static bool g_has_neon = false;

static void detect_cpu_features() {
    std::call_once(g_simd_once, []() {
#ifdef __x86_64__
        // Check for AVX2 support
        __builtin_cpu_init();
        g_has_avx2 = __builtin_cpu_supports("avx2");
#elif defined(__aarch64__)
        // ARM NEON is always available on AArch64
        g_has_neon = true;
#endif
    });
}

bool can_use_simd_dosage() {
    detect_cpu_features();
    return g_has_avx2 || g_has_neon;
}

namespace simd {

// Helper to read little-endian 16-bit value
static inline uint16_t read_le16(const uint8_t* ptr) {
    return ptr[0] | (static_cast<uint16_t>(ptr[1]) << 8);
}

// Helper to read little-endian 32-bit value
static inline uint32_t read_le32(const uint8_t* ptr) {
    return ptr[0] | (static_cast<uint32_t>(ptr[1]) << 8) | (static_cast<uint32_t>(ptr[2]) << 16) |
           (static_cast<uint32_t>(ptr[3]) << 24);
}

// Forward declarations of helper functions
static void compute_dosages_filtered_8bit_simd(const uint8_t* prob_data, float* output,
                                               const int* sample_indices, size_t n_indices,
                                               const uint8_t* missing_mask);

static void compute_dosages_filtered_16bit_simd(const uint8_t* prob_data, float* output,
                                                const int* sample_indices, size_t n_indices,
                                                const uint8_t* missing_mask);

static void compute_dosages_filtered_32bit_simd(const uint8_t* prob_data, float* output,
                                                const int* sample_indices, size_t n_indices,
                                                const uint8_t* missing_mask);

// Optimized filtered dosage computation
void compute_dosages_filtered_simd(const uint8_t* prob_data, float* output,
                                   const int* sample_indices, size_t n_indices,
                                   uint8_t bits_per_prob, const uint8_t* missing_mask) {
    if (bits_per_prob == 8) {
        compute_dosages_filtered_8bit_simd(prob_data, output, sample_indices, n_indices,
                                           missing_mask);
    } else if (bits_per_prob == 16) {
        compute_dosages_filtered_16bit_simd(prob_data, output, sample_indices, n_indices,
                                            missing_mask);
    } else if (bits_per_prob == 32) {
        compute_dosages_filtered_32bit_simd(prob_data, output, sample_indices, n_indices,
                                            missing_mask);
    }
}

// Helper functions for filtered computation
static void compute_dosages_filtered_8bit_simd(const uint8_t* prob_data, float* output,
                                               const int* sample_indices, size_t n_indices,
                                               const uint8_t* missing_mask) {
    size_t i = 0;

#ifdef __x86_64__
    if (g_has_avx2) {
        // Process 8 samples at a time with AVX2
        const __m256 scale = _mm256_set1_ps(1.0f / 255.0f);
        const __m256 two = _mm256_set1_ps(2.0f);

        for (; i + 7 < n_indices; i += 8) {
            // Gather data for 8 selected samples
            uint8_t prob_aa[8], prob_ab[8];
            bool is_missing[8] = {false};

            for (int j = 0; j < 8; ++j) {
                int idx = sample_indices[i + j];
                prob_aa[j] = prob_data[idx * 2];
                prob_ab[j] = prob_data[idx * 2 + 1];

                if (missing_mask) {
                    // BGEN v1.2/v1.3: one byte per sample, high bit 0x80 = missing.
                    is_missing[j] = (missing_mask[idx] & 0x80) != 0;
                }
            }

            // Pack into vectors
            __m256i prob_aa_vec = _mm256_set_epi32(prob_aa[7], prob_aa[6], prob_aa[5], prob_aa[4],
                                                   prob_aa[3], prob_aa[2], prob_aa[1], prob_aa[0]);

            __m256i prob_ab_vec = _mm256_set_epi32(prob_ab[7], prob_ab[6], prob_ab[5], prob_ab[4],
                                                   prob_ab[3], prob_ab[2], prob_ab[1], prob_ab[0]);

            // Calculate P(BB) = 255 - P(AA) - P(AB)
            __m256i sum = _mm256_add_epi32(prob_aa_vec, prob_ab_vec);
            __m256i max_val = _mm256_set1_epi32(255);
            __m256i prob_bb = _mm256_sub_epi32(max_val, sum);

            // Convert to float and calculate dosage
            __m256 prob_ab_f = _mm256_cvtepi32_ps(prob_ab_vec);
            __m256 prob_bb_f = _mm256_cvtepi32_ps(prob_bb);
            __m256 dosage = _mm256_mul_ps(_mm256_fmadd_ps(two, prob_bb_f, prob_ab_f), scale);

            float result[8];
            _mm256_storeu_ps(result, dosage);

            // Per-lane fixup, matching the scalar / 16-bit / 32-bit / NEON
            // contract: missing samples decode to NaN, and an invalid
            // probability sum (prob_aa + prob_ab > 255) also decodes to NaN
            // (the vector chunk above computed prob_bb = 255 - sum, which goes
            // negative for such corrupt input). Using a scalar store here also
            // avoids the _mm256_blendv_ps sign-bit pitfall: nanf("") has its
            // sign bit clear, so blendv would keep the computed dosage.
            for (int j = 0; j < 8; ++j) {
                if (is_missing[j] || (static_cast<int>(prob_aa[j]) + prob_ab[j] > 255)) {
                    output[i + j] = std::nanf("");
                } else {
                    output[i + j] = result[j];
                }
            }
        }
    }
#elif defined(__aarch64__)
    if (g_has_neon) {
        // Process 4 samples at a time with NEON
        const float32x4_t scale = vdupq_n_f32(1.0f / 255.0f);
        const float32x4_t two = vdupq_n_f32(2.0f);

        for (; i + 3 < n_indices; i += 4) {
            // Gather data for 4 selected samples
            uint8_t prob_aa[4], prob_ab[4];
            bool is_missing[4] = {false};

            for (int j = 0; j < 4; ++j) {
                int idx = sample_indices[i + j];
                prob_aa[j] = prob_data[idx * 2];
                prob_ab[j] = prob_data[idx * 2 + 1];

                if (missing_mask) {
                    // BGEN v1.2/v1.3: one byte per sample, high bit 0x80 = missing.
                    is_missing[j] = (missing_mask[idx] & 0x80) != 0;
                }
            }

            // Convert to vectors
            uint32x4_t prob_aa_vec = {prob_aa[0], prob_aa[1], prob_aa[2], prob_aa[3]};
            uint32x4_t prob_ab_vec = {prob_ab[0], prob_ab[1], prob_ab[2], prob_ab[3]};

            // Calculate P(BB) = 255 - P(AA) - P(AB)
            uint32x4_t sum = vaddq_u32(prob_aa_vec, prob_ab_vec);
            uint32x4_t prob_bb = vsubq_u32(vdupq_n_u32(255), sum);

            // Convert to float and calculate dosage
            float32x4_t prob_ab_f = vcvtq_f32_u32(prob_ab_vec);
            float32x4_t prob_bb_f = vcvtq_f32_u32(prob_bb);
            float32x4_t dosage = vmulq_f32(vmlaq_f32(prob_ab_f, two, prob_bb_f), scale);

            // Store results with missing handling
            float result[4];
            vst1q_f32(result, dosage);

            for (int j = 0; j < 4; ++j) {
                // Match the scalar contract: missing or an invalid probability
                // sum (prob_aa + prob_ab > 255, which makes the vector chunk's
                // prob_bb wrap as unsigned) decodes to NaN.
                if (is_missing[j] || (static_cast<int>(prob_aa[j]) + prob_ab[j] > 255)) {
                    output[i + j] = std::nanf("");
                } else {
                    output[i + j] = result[j];
                }
            }
        }
    }
#endif

    // Scalar fallback for remaining samples
    for (; i < n_indices; ++i) {
        int idx = sample_indices[i];

        // Check if missing
        bool is_missing = false;
        if (missing_mask) {
            // BGEN v1.2/v1.3: one byte per sample, high bit 0x80 = missing.
            is_missing = (missing_mask[idx] & 0x80) != 0;
        }

        if (is_missing) {
            output[i] = std::nanf("");
            continue;
        }

        uint8_t prob_aa = prob_data[idx * 2];
        uint8_t prob_ab = prob_data[idx * 2 + 1];
        output[i] = decode_dosage_8bit<float>(prob_aa, prob_ab);
    }
}

static void compute_dosages_filtered_16bit_simd(const uint8_t* prob_data, float* output,
                                                const int* sample_indices, size_t n_indices,
                                                const uint8_t* missing_mask) {
    size_t i = 0;

#ifdef __x86_64__
    if (g_has_avx2) {
        // Process 8 samples at a time with AVX2
        const __m256 scale = _mm256_set1_ps(1.0f / 65535.0f);
        const __m256 two = _mm256_set1_ps(2.0f);

        for (; i + 7 < n_indices; i += 8) {
            // Gather data for 8 selected samples
            uint32_t prob_aa_arr[8], prob_ab_arr[8];
            for (int j = 0; j < 8; ++j) {
                int idx = sample_indices[i + j];
                prob_aa_arr[j] = read_le16(prob_data + idx * 4);
                prob_ab_arr[j] = read_le16(prob_data + idx * 4 + 2);
            }

            __m256i prob_aa =
                _mm256_set_epi32(prob_aa_arr[7], prob_aa_arr[6], prob_aa_arr[5], prob_aa_arr[4],
                                 prob_aa_arr[3], prob_aa_arr[2], prob_aa_arr[1], prob_aa_arr[0]);

            __m256i prob_ab =
                _mm256_set_epi32(prob_ab_arr[7], prob_ab_arr[6], prob_ab_arr[5], prob_ab_arr[4],
                                 prob_ab_arr[3], prob_ab_arr[2], prob_ab_arr[1], prob_ab_arr[0]);

            // Calculate P(BB) = 65535 - P(AA) - P(AB)
            __m256i sum = _mm256_add_epi32(prob_aa, prob_ab);
            __m256i max_val = _mm256_set1_epi32(65535);
            __m256i prob_bb = _mm256_sub_epi32(max_val, sum);

            // Convert to float and calculate dosage
            __m256 prob_ab_f = _mm256_cvtepi32_ps(prob_ab);
            __m256 prob_bb_f = _mm256_cvtepi32_ps(prob_bb);
            __m256 dosage = _mm256_mul_ps(_mm256_fmadd_ps(two, prob_bb_f, prob_ab_f), scale);

            float result[8];
            _mm256_storeu_ps(result, dosage);

            // Per-lane fixup matching the scalar contract: missing samples and
            // an invalid probability sum (prob_aa + prob_ab > 65535) decode to
            // NaN.
            for (int j = 0; j < 8; ++j) {
                int idx = sample_indices[i + j];
                bool is_missing = missing_mask && (missing_mask[idx] & 0x80) != 0;
                if (is_missing || (prob_aa_arr[j] + prob_ab_arr[j] > 65535)) {
                    output[i + j] = std::nanf("");
                } else {
                    output[i + j] = result[j];
                }
            }
        }
    }
#elif defined(__aarch64__)
    if (g_has_neon) {
        // Process 4 samples at a time with NEON
        const float32x4_t scale = vdupq_n_f32(1.0f / 65535.0f);
        const float32x4_t two = vdupq_n_f32(2.0f);

        for (; i + 3 < n_indices; i += 4) {
            // Gather data for 4 selected samples
            uint32_t prob_aa_arr[4], prob_ab_arr[4];
            for (int j = 0; j < 4; ++j) {
                int idx = sample_indices[i + j];
                prob_aa_arr[j] = read_le16(prob_data + idx * 4);
                prob_ab_arr[j] = read_le16(prob_data + idx * 4 + 2);
            }

            uint16x4_t prob_aa = {static_cast<uint16_t>(prob_aa_arr[0]),
                                  static_cast<uint16_t>(prob_aa_arr[1]),
                                  static_cast<uint16_t>(prob_aa_arr[2]),
                                  static_cast<uint16_t>(prob_aa_arr[3])};

            uint16x4_t prob_ab = {static_cast<uint16_t>(prob_ab_arr[0]),
                                  static_cast<uint16_t>(prob_ab_arr[1]),
                                  static_cast<uint16_t>(prob_ab_arr[2]),
                                  static_cast<uint16_t>(prob_ab_arr[3])};

            // Calculate P(BB) = 65535 - P(AA) - P(AB)
            uint32x4_t prob_aa_32 = vmovl_u16(prob_aa);
            uint32x4_t prob_ab_32 = vmovl_u16(prob_ab);
            uint32x4_t sum = vaddq_u32(prob_aa_32, prob_ab_32);
            uint32x4_t prob_bb_32 = vsubq_u32(vdupq_n_u32(65535), sum);

            // Convert to float and calculate dosage
            float32x4_t prob_ab_f = vcvtq_f32_u32(prob_ab_32);
            float32x4_t prob_bb_f = vcvtq_f32_u32(prob_bb_32);
            float32x4_t dosage = vmulq_f32(vmlaq_f32(prob_ab_f, two, prob_bb_f), scale);

            float result[4];
            vst1q_f32(result, dosage);

            // Per-lane fixup matching the scalar contract: missing samples and
            // an invalid probability sum (prob_aa + prob_ab > 65535, which makes
            // the vector chunk's prob_bb wrap as unsigned) decode to NaN.
            for (int j = 0; j < 4; ++j) {
                int idx = sample_indices[i + j];
                bool is_missing = missing_mask && (missing_mask[idx] & 0x80) != 0;
                if (is_missing || (prob_aa_arr[j] + prob_ab_arr[j] > 65535)) {
                    output[i + j] = std::nanf("");
                } else {
                    output[i + j] = result[j];
                }
            }
        }
    }
#endif

    // Scalar fallback for remaining samples
    for (; i < n_indices; ++i) {
        int idx = sample_indices[i];

        // Check if missing
        bool is_missing = false;
        if (missing_mask) {
            // BGEN v1.2/v1.3: one byte per sample, high bit 0x80 = missing.
            is_missing = (missing_mask[idx] & 0x80) != 0;
        }

        if (is_missing) {
            output[i] = std::nanf("");
            continue;
        }

        uint16_t prob_aa = read_le16(prob_data + idx * 4);
        uint16_t prob_ab = read_le16(prob_data + idx * 4 + 2);
        output[i] = decode_dosage_16bit<float>(prob_aa, prob_ab);
    }
}

static void compute_dosages_filtered_32bit_simd(const uint8_t* prob_data, float* output,
                                                const int* sample_indices, size_t n_indices,
                                                const uint8_t* missing_mask) {
    size_t i = 0;

#ifdef __x86_64__
    if (g_has_avx2) {
        // Process 4 samples at a time with AVX2
        for (; i + 3 < n_indices; i += 4) {
            // Gather data for 4 selected samples
            uint64_t prob_aa[4], prob_ab[4];
            for (int j = 0; j < 4; ++j) {
                int idx = sample_indices[i + j];
                prob_aa[j] = read_le32(prob_data + idx * 8);
                prob_ab[j] = read_le32(prob_data + idx * 8 + 4);
            }

            // Calculate P(BB) and dosage
            uint64_t prob_bb[4];
            double dosages[4];
            for (int j = 0; j < 4; ++j) {
                if (prob_aa[j] + prob_ab[j] > 4294967295UL) {
                    dosages[j] = std::nan("");
                    prob_bb[j] = 0;
                } else {
                    prob_bb[j] = 4294967295UL - prob_aa[j] - prob_ab[j];
                    dosages[j] =
                        (static_cast<double>(prob_ab[j]) + 2.0 * prob_bb[j]) / 4294967295.0;
                }
            }

            // Store results
            for (int j = 0; j < 4; ++j) {
                output[i + j] = static_cast<float>(dosages[j]);
            }

            // Handle missing values
            if (missing_mask) {
                for (int j = 0; j < 4; ++j) {
                    int idx = sample_indices[i + j];
                    // BGEN v1.2/v1.3: one byte per sample, high bit 0x80 = missing.
                    if (missing_mask[idx] & 0x80) {
                        output[i + j] = std::nanf("");
                    }
                }
            }
        }
    }
#endif

    // Scalar fallback (also used for ARM)
    for (; i < n_indices; ++i) {
        int idx = sample_indices[i];

        // Check if missing
        bool is_missing = false;
        if (missing_mask) {
            // BGEN v1.2/v1.3: one byte per sample, high bit 0x80 = missing.
            is_missing = (missing_mask[idx] & 0x80) != 0;
        }

        if (is_missing) {
            output[i] = std::nanf("");
            continue;
        }

        uint32_t prob_aa = read_le32(prob_data + idx * 8);
        uint32_t prob_ab = read_le32(prob_data + idx * 8 + 4);
        output[i] = decode_dosage_32bit<float>(prob_aa, prob_ab);
    }
}

// BGEN v1.1 SIMD implementation
void compute_dosages_v11_simd(const uint8_t* buffer, size_t n_samples, float* output) {
    size_t i = 0;

#ifdef __x86_64__
    if (g_has_avx2) {
        // Process 8 samples at a time with AVX2
        const __m256 two = _mm256_set1_ps(2.0f);

        for (; i + 7 < n_samples; i += 8) {
            // Load 48 bytes (8 samples × 6 bytes)
            // Each sample has 3 uint16_t values: prob_aa, prob_ab, prob_bb

            // Gather P(AA) and P(AB) values for 8 samples
            __m256i prob_aa =
                _mm256_set_epi32(read_le16(buffer + (i + 7) * 6), read_le16(buffer + (i + 6) * 6),
                                 read_le16(buffer + (i + 5) * 6), read_le16(buffer + (i + 4) * 6),
                                 read_le16(buffer + (i + 3) * 6), read_le16(buffer + (i + 2) * 6),
                                 read_le16(buffer + (i + 1) * 6), read_le16(buffer + (i + 0) * 6));

            __m256i prob_ab = _mm256_set_epi32(
                read_le16(buffer + (i + 7) * 6 + 2), read_le16(buffer + (i + 6) * 6 + 2),
                read_le16(buffer + (i + 5) * 6 + 2), read_le16(buffer + (i + 4) * 6 + 2),
                read_le16(buffer + (i + 3) * 6 + 2), read_le16(buffer + (i + 2) * 6 + 2),
                read_le16(buffer + (i + 1) * 6 + 2), read_le16(buffer + (i + 0) * 6 + 2));

            __m256i prob_bb = _mm256_set_epi32(
                read_le16(buffer + (i + 7) * 6 + 4), read_le16(buffer + (i + 6) * 6 + 4),
                read_le16(buffer + (i + 5) * 6 + 4), read_le16(buffer + (i + 4) * 6 + 4),
                read_le16(buffer + (i + 3) * 6 + 4), read_le16(buffer + (i + 2) * 6 + 4),
                read_le16(buffer + (i + 1) * 6 + 4), read_le16(buffer + (i + 0) * 6 + 4));

            // Check for missing data (all probabilities == 0)
            __m256i sum_check = _mm256_or_si256(_mm256_or_si256(prob_aa, prob_ab), prob_bb);
            __m256i is_missing_mask = _mm256_cmpeq_epi32(sum_check, _mm256_setzero_si256());

            // Calculate sum for normalization
            __m256i sum = _mm256_add_epi32(_mm256_add_epi32(prob_aa, prob_ab), prob_bb);

            // Convert to float for division
            __m256 sum_f = _mm256_cvtepi32_ps(sum);
            __m256 prob_ab_f = _mm256_cvtepi32_ps(prob_ab);
            __m256 prob_bb_f = _mm256_cvtepi32_ps(prob_bb);

            // Calculate dosage = (prob_ab + 2 * prob_bb) / sum
            __m256 dosage = _mm256_div_ps(_mm256_fmadd_ps(two, prob_bb_f, prob_ab_f), sum_f);

            // Create NaN mask for missing values
            __m256 nan_val = _mm256_set1_ps(std::nanf(""));
            dosage = _mm256_blendv_ps(dosage, nan_val, _mm256_castsi256_ps(is_missing_mask));

            _mm256_storeu_ps(output + i, dosage);
        }
    }
#elif defined(__aarch64__)
    if (g_has_neon) {
        // Process 4 samples at a time with NEON
        const float32x4_t two = vdupq_n_f32(2.0f);

        for (; i + 3 < n_samples; i += 4) {
            // Load P(AA), P(AB), and P(BB) for 4 samples
            uint16x4_t prob_aa = {read_le16(buffer + (i + 0) * 6), read_le16(buffer + (i + 1) * 6),
                                  read_le16(buffer + (i + 2) * 6), read_le16(buffer + (i + 3) * 6)};

            uint16x4_t prob_ab = {
                read_le16(buffer + (i + 0) * 6 + 2), read_le16(buffer + (i + 1) * 6 + 2),
                read_le16(buffer + (i + 2) * 6 + 2), read_le16(buffer + (i + 3) * 6 + 2)};

            uint16x4_t prob_bb = {
                read_le16(buffer + (i + 0) * 6 + 4), read_le16(buffer + (i + 1) * 6 + 4),
                read_le16(buffer + (i + 2) * 6 + 4), read_le16(buffer + (i + 3) * 6 + 4)};

            // Check for missing data
            uint16x4_t sum_check = vorr_u16(vorr_u16(prob_aa, prob_ab), prob_bb);
            uint16x4_t is_missing = vceq_u16(sum_check, vdup_n_u16(0));

            // Convert to 32-bit for calculation
            uint32x4_t prob_aa_32 = vmovl_u16(prob_aa);
            uint32x4_t prob_ab_32 = vmovl_u16(prob_ab);
            uint32x4_t prob_bb_32 = vmovl_u16(prob_bb);

            // Calculate sum
            uint32x4_t sum = vaddq_u32(vaddq_u32(prob_aa_32, prob_ab_32), prob_bb_32);

            // Convert to float
            float32x4_t sum_f = vcvtq_f32_u32(sum);
            float32x4_t prob_ab_f = vcvtq_f32_u32(prob_ab_32);
            float32x4_t prob_bb_f = vcvtq_f32_u32(prob_bb_32);

            // Calculate dosage = (prob_ab + 2 * prob_bb) / sum
            float32x4_t dosage = vdivq_f32(vmlaq_f32(prob_ab_f, two, prob_bb_f), sum_f);

            // Store results with missing handling
            float result[4];
            vst1q_f32(result, dosage);

            // Apply missing mask
            uint16_t missing_mask[4];
            vst1_u16(missing_mask, is_missing);

            for (int j = 0; j < 4; ++j) {
                output[i + j] = missing_mask[j] ? std::nanf("") : result[j];
            }
        }
    }
#endif

    // Scalar fallback for remaining samples
    for (; i < n_samples; ++i) {
        // Read 3 probabilities (2 bytes each)
        uint16_t prob_aa = read_le16(buffer + i * 6);
        uint16_t prob_ab = read_le16(buffer + i * 6 + 2);
        uint16_t prob_bb = read_le16(buffer + i * 6 + 4);

        // Check for missing data
        if (prob_aa == 0 && prob_ab == 0 && prob_bb == 0) {
            output[i] = std::nanf("");
        } else {
            // Compute dosage directly
            float sum = static_cast<float>(prob_aa + prob_ab + prob_bb);
            output[i] = (prob_ab + 2.0f * prob_bb) / sum;
        }
    }
}

}  // namespace simd
}  // namespace bgen
}  // namespace lazybgen