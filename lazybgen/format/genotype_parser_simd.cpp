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

#include "simd_target.h"

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
        // The AVX2 kernels also use FMA, so require both. Every AVX2 CPU has
        // FMA, but a hypervisor can mask one without the other.
        __builtin_cpu_init();
        g_has_avx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
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

bool cpu_has_avx2() {
    detect_cpu_features();
    return g_has_avx2;
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
#ifdef LAZYBGEN_X86_AVX2
// AVX2 part of the kernel below; returns the index the scalar tail resumes from.
LAZYBGEN_TARGET_AVX2 static size_t filtered_8bit_avx2(const uint8_t* prob_data, float* output,
                                               const int* sample_indices, size_t n_indices,
                                               const uint8_t* missing_mask, size_t i) {
    // Process 8 samples at a time with AVX2
    // Divide (not multiply by 1/255) so every lane rounds exactly like the
    // scalar tail and the full-sample decode.
    const __m256 denom = _mm256_set1_ps(255.0f);
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
        __m256 dosage = _mm256_div_ps(_mm256_fmadd_ps(two, prob_bb_f, prob_ab_f), denom);

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
    return i;
}
#endif

static void compute_dosages_filtered_8bit_simd(const uint8_t* prob_data, float* output,
                                               const int* sample_indices, size_t n_indices,
                                               const uint8_t* missing_mask) {
    size_t i = 0;

#ifdef LAZYBGEN_X86_AVX2
    if (g_has_avx2) {
        i = filtered_8bit_avx2(prob_data, output, sample_indices, n_indices, missing_mask, i);
    }
#elif defined(__aarch64__)
    if (g_has_neon) {
        // Process 4 samples at a time with NEON
        // Divide (not multiply by 1/255) to match the scalar decode exactly.
        const float32x4_t denom = vdupq_n_f32(255.0f);
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
            float32x4_t dosage = vdivq_f32(vmlaq_f32(prob_ab_f, two, prob_bb_f), denom);

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

#ifdef LAZYBGEN_X86_AVX2
// AVX2 part of the kernel below; returns the index the scalar tail resumes from.
LAZYBGEN_TARGET_AVX2 static size_t filtered_16bit_avx2(const uint8_t* prob_data, float* output,
                                               const int* sample_indices, size_t n_indices,
                                               const uint8_t* missing_mask, size_t i) {
    // Process 8 samples at a time with AVX2
    // Divide, as in the 8-bit kernel, to match the scalar decode exactly.
    const __m256 denom = _mm256_set1_ps(65535.0f);
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
        __m256 dosage = _mm256_div_ps(_mm256_fmadd_ps(two, prob_bb_f, prob_ab_f), denom);

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
    return i;
}
#endif

static void compute_dosages_filtered_16bit_simd(const uint8_t* prob_data, float* output,
                                                const int* sample_indices, size_t n_indices,
                                                const uint8_t* missing_mask) {
    size_t i = 0;

#ifdef LAZYBGEN_X86_AVX2
    if (g_has_avx2) {
        i = filtered_16bit_avx2(prob_data, output, sample_indices, n_indices, missing_mask, i);
    }
#elif defined(__aarch64__)
    if (g_has_neon) {
        // Process 4 samples at a time with NEON
        // Divide, as in the 8-bit kernel, to match the scalar decode exactly.
        const float32x4_t denom = vdupq_n_f32(65535.0f);
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
            float32x4_t dosage = vdivq_f32(vmlaq_f32(prob_ab_f, two, prob_bb_f), denom);

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
#ifdef LAZYBGEN_X86_AVX2
// AVX2 part of the kernel below; returns the index the scalar tail resumes from.
LAZYBGEN_TARGET_AVX2 static size_t v11_avx2(const uint8_t* buffer, size_t n_samples, float* output,
                                            size_t i) {
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
    return i;
}
#endif

void compute_dosages_v11_simd(const uint8_t* buffer, size_t n_samples, float* output) {
    size_t i = 0;

#ifdef LAZYBGEN_X86_AVX2
    if (g_has_avx2) {
        i = v11_avx2(buffer, n_samples, output, i);
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

// ---------------------------------------------------------------------------
// Fused full-sample decode for unphased v1.2 data (8- and 16-bit).
//
// Each sample's dosage is float(k) / M, where k = 2*M - 2*P(AA) - P(AB) is an
// exact integer (below 2^24, so float(k) is exact) and M = 255 or 65535. That is
// the same float expression decode_dosage_8bit/16bit evaluate, so dividing k in
// a vector lane gives the same bits. A lane is NaN when its call is missing or
// its probabilities sum past M. The NaN is selected with an all-ones integer
// mask, never with the NaN itself as the mask, so its sign bit cannot steer a
// blend. The kernels also track the min and max valid k, which replaces a
// second pass over the decoded column to find its range.
// ---------------------------------------------------------------------------

namespace {

constexpr uint32_t kNoNumerator = 0xFFFFFFFFu;

// Scalar loop for [i, n): the whole column on CPUs without the vector kernels,
// and the tail after them. Branch-free: the missing/invalid test selects
// rather than jumps, since missing calls scattered through a column would
// otherwise mispredict.
template <typename T, uint32_t M>
inline void fused_tail(const uint8_t* missing, const uint8_t* prob, size_t i, size_t n, T* output,
                       const float* lut, NumeratorRange& r) {
    const T nan_value = static_cast<T>(std::nanf(""));
    uint32_t k_min = r.k_min;
    uint32_t k_max = r.k_max;
    uint32_t any_bad = r.any_nan ? 1u : 0u;
    for (; i < n; ++i) {
        uint32_t aa;
        uint32_t ab;
        if (M == 255u) {
            aa = prob[2 * i];
            ab = prob[2 * i + 1];
        } else {
            aa = read_le16(prob + 4 * i);
            ab = read_le16(prob + 4 * i + 2);
        }
        const uint32_t sum = aa + ab;
        const uint32_t bad = static_cast<uint32_t>(sum > M) | (missing[i] >> 7);
        const uint32_t k = bad ? 0u : 2u * M - aa - sum;
        const float value = lut ? lut[k] : static_cast<float>(k) / static_cast<float>(M);
        output[i] = bad ? nan_value : static_cast<T>(value);
        any_bad |= bad;
        k_min = (!bad && k < k_min) ? k : k_min;
        k_max = (!bad && k > k_max) ? k : k_max;
    }
    r.k_min = k_min;
    r.k_max = k_max;
    r.any_nan = any_bad != 0;
}

// float(k) / 255 for every 8-bit numerator, so the scalar path does a load
// instead of a divide. Built from the same expression, so the values match.
const float* dosage_lut_8bit() {
    static const struct Table {
        float v[511];
        Table() {
            for (uint32_t k = 0; k <= 510; ++k) {
                v[k] = static_cast<float>(k) / 255.0f;
            }
        }
    } table;
    return table.v;
}

#ifdef LAZYBGEN_X86_AVX2
LAZYBGEN_TARGET_AVX2 inline void store8(float* out, __m256 v) {
    _mm256_storeu_ps(out, v);
}
LAZYBGEN_TARGET_AVX2 inline void store8(double* out, __m256 v) {
    _mm256_storeu_pd(out, _mm256_cvtps_pd(_mm256_castps256_ps128(v)));
    _mm256_storeu_pd(out + 4, _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1)));
}

LAZYBGEN_TARGET_AVX2 inline uint32_t hmin_epu16(__m256i v) {
    __m128i m = _mm_min_epu16(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    return static_cast<uint32_t>(_mm_extract_epi16(_mm_minpos_epu16(m), 0));
}
LAZYBGEN_TARGET_AVX2 inline uint32_t hmax_epu16(__m256i v) {
    // max(x) = 0xFFFF - min(0xFFFF - x)
    const __m256i ones = _mm256_set1_epi16(-1);
    return 0xFFFFu - hmin_epu16(_mm256_xor_si256(v, ones));
}
LAZYBGEN_TARGET_AVX2 inline uint32_t hmin_epu32(__m256i v) {
    alignas(32) uint32_t lanes[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), v);
    uint32_t m = lanes[0];
    for (int j = 1; j < 8; ++j) m = lanes[j] < m ? lanes[j] : m;
    return m;
}
LAZYBGEN_TARGET_AVX2 inline uint32_t hmax_epu32(__m256i v) {
    alignas(32) uint32_t lanes[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), v);
    uint32_t m = lanes[0];
    for (int j = 1; j < 8; ++j) m = lanes[j] > m ? lanes[j] : m;
    return m;
}

// 16 samples per iteration. Returns the index the scalar tail resumes from.
template <typename T>
LAZYBGEN_TARGET_AVX2 size_t fused_8bit_avx2(const uint8_t* missing, const uint8_t* prob, size_t n,
                                            T* output, NumeratorRange& r) {
    const __m256 denom = _mm256_set1_ps(255.0f);
    const __m256 nan_v = _mm256_set1_ps(std::nanf(""));
    const __m256i w_k = _mm256_set1_epi16(0x0102);    // bytes (2, 1): 2*aa + ab
    const __m256i w_sum = _mm256_set1_epi16(0x0101);  // aa + ab
    const __m256i c510 = _mm256_set1_epi16(510);
    const __m256i c255 = _mm256_set1_epi16(255);
    __m256i v_min = _mm256_set1_epi16(-1);
    __m256i v_max = _mm256_setzero_si256();
    __m256i v_bad = _mm256_setzero_si256();
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(prob + 2 * i));
        const __m256i k = _mm256_sub_epi16(c510, _mm256_maddubs_epi16(x, w_k));
        const __m256i sum = _mm256_maddubs_epi16(x, w_sum);
        // Sign-extend the ploidy bytes: a missing call (bit 7) becomes negative.
        const __m256i miss = _mm256_cvtepi8_epi16(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(missing + i)));
        const __m256i bad =
            _mm256_or_si256(_mm256_cmpgt_epi16(sum, c255), _mm256_srai_epi16(miss, 15));
        v_bad = _mm256_or_si256(v_bad, bad);
        v_min = _mm256_min_epu16(v_min, _mm256_or_si256(k, bad));      // bad lanes -> 0xFFFF
        v_max = _mm256_max_epu16(v_max, _mm256_andnot_si256(bad, k));  // bad lanes -> 0
        for (int half = 0; half < 2; ++half) {
            const __m128i k8 = half ? _mm256_extracti128_si256(k, 1) : _mm256_castsi256_si128(k);
            const __m128i b8 =
                half ? _mm256_extracti128_si256(bad, 1) : _mm256_castsi256_si128(bad);
            __m256 f = _mm256_div_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(k8)), denom);
            f = _mm256_blendv_ps(f, nan_v, _mm256_castsi256_ps(_mm256_cvtepi16_epi32(b8)));
            store8(output + i + 8 * half, f);
        }
    }
    if (i > 0) {
        const uint32_t lo = hmin_epu16(v_min);
        r.k_min = lo == 0xFFFFu ? kNoNumerator : lo;
        r.k_max = hmax_epu16(v_max);
        if (r.k_min == kNoNumerator) r.k_max = 0;
        r.any_nan = !_mm256_testz_si256(v_bad, v_bad);
    }
    return i;
}

// 8 samples per iteration; each 32-bit lane holds one (aa, ab) pair.
template <typename T>
LAZYBGEN_TARGET_AVX2 size_t fused_16bit_avx2(const uint8_t* missing, const uint8_t* prob, size_t n,
                                             T* output, NumeratorRange& r) {
    const __m256 denom = _mm256_set1_ps(65535.0f);
    const __m256 nan_v = _mm256_set1_ps(std::nanf(""));
    const __m256i low16 = _mm256_set1_epi32(0xFFFF);
    const __m256i c131070 = _mm256_set1_epi32(131070);
    __m256i v_min = _mm256_set1_epi32(-1);
    __m256i v_max = _mm256_setzero_si256();
    __m256i v_bad = _mm256_setzero_si256();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(prob + 4 * i));
        const __m256i aa = _mm256_and_si256(x, low16);
        const __m256i ab = _mm256_srli_epi32(x, 16);
        const __m256i sum = _mm256_add_epi32(aa, ab);
        const __m256i k = _mm256_sub_epi32(c131070, _mm256_add_epi32(aa, sum));
        const __m256i miss = _mm256_cvtepi8_epi32(
            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(missing + i)));
        const __m256i bad =
            _mm256_or_si256(_mm256_cmpgt_epi32(sum, low16), _mm256_srai_epi32(miss, 31));
        v_bad = _mm256_or_si256(v_bad, bad);
        v_min = _mm256_min_epu32(v_min, _mm256_or_si256(k, bad));
        v_max = _mm256_max_epu32(v_max, _mm256_andnot_si256(bad, k));
        __m256 f = _mm256_div_ps(_mm256_cvtepi32_ps(k), denom);
        f = _mm256_blendv_ps(f, nan_v, _mm256_castsi256_ps(bad));
        store8(output + i, f);
    }
    if (i > 0) {
        r.k_min = hmin_epu32(v_min);
        r.k_max = r.k_min == kNoNumerator ? 0 : hmax_epu32(v_max);
        r.any_nan = !_mm256_testz_si256(v_bad, v_bad);
    }
    return i;
}
#endif  // LAZYBGEN_X86_AVX2

#if defined(__aarch64__)
inline void store4(float* out, float32x4_t v) {
    vst1q_f32(out, v);
}
inline void store4(double* out, float32x4_t v) {
    vst1q_f64(out, vcvt_f64_f32(vget_low_f32(v)));
    vst1q_f64(out + 2, vcvt_high_f64_f32(v));
}

// 16 samples per iteration.
template <typename T>
size_t fused_8bit_neon(const uint8_t* missing, const uint8_t* prob, size_t n, T* output,
                       NumeratorRange& r) {
    const float32x4_t denom = vdupq_n_f32(255.0f);
    const uint32x4_t nan_bits = vreinterpretq_u32_f32(vdupq_n_f32(std::nanf("")));
    const uint16x8_t c510 = vdupq_n_u16(510);
    const uint16x8_t c255 = vdupq_n_u16(255);
    uint16x8_t v_min = vdupq_n_u16(0xFFFF);
    uint16x8_t v_max = vdupq_n_u16(0);
    uint16x8_t v_bad = vdupq_n_u16(0);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const uint8x16x2_t pairs = vld2q_u8(prob + 2 * i);  // val[0] = aa, val[1] = ab
        const uint8x16_t miss = vld1q_u8(missing + i);
        for (int half = 0; half < 2; ++half) {
            const uint8x8_t aa = half ? vget_high_u8(pairs.val[0]) : vget_low_u8(pairs.val[0]);
            const uint8x8_t ab = half ? vget_high_u8(pairs.val[1]) : vget_low_u8(pairs.val[1]);
            const uint8x8_t mm = half ? vget_high_u8(miss) : vget_low_u8(miss);
            const uint16x8_t sum = vaddl_u8(aa, ab);
            const uint16x8_t k = vsubq_u16(c510, vaddw_u8(sum, aa));
            // Missing bit 7 -> all-ones lane: move it to the sign bit, shift across.
            const uint16x8_t miss_mask = vreinterpretq_u16_s16(
                vshrq_n_s16(vreinterpretq_s16_u16(vshll_n_u8(mm, 8)), 15));
            const uint16x8_t bad = vorrq_u16(vcgtq_u16(sum, c255), miss_mask);
            v_bad = vorrq_u16(v_bad, bad);
            v_min = vminq_u16(v_min, vorrq_u16(k, bad));
            v_max = vmaxq_u16(v_max, vbicq_u16(k, bad));
            for (int q = 0; q < 2; ++q) {
                const uint16x4_t k4 = q ? vget_high_u16(k) : vget_low_u16(k);
                const uint16x4_t b4 = q ? vget_high_u16(bad) : vget_low_u16(bad);
                float32x4_t f = vdivq_f32(vcvtq_f32_u32(vmovl_u16(k4)), denom);
                const uint32x4_t lane_bad =
                    vreinterpretq_u32_s32(vmovl_s16(vreinterpret_s16_u16(b4)));
                f = vreinterpretq_f32_u32(vbslq_u32(lane_bad, nan_bits, vreinterpretq_u32_f32(f)));
                store4(output + i + 8 * half + 4 * q, f);
            }
        }
    }
    if (i > 0) {
        const uint32_t lo = vminvq_u16(v_min);
        r.k_min = lo == 0xFFFFu ? kNoNumerator : lo;
        r.k_max = r.k_min == kNoNumerator ? 0 : vmaxvq_u16(v_max);
        r.any_nan = vmaxvq_u16(v_bad) != 0;
    }
    return i;
}

// 8 samples per iteration.
template <typename T>
size_t fused_16bit_neon(const uint8_t* missing, const uint8_t* prob, size_t n, T* output,
                        NumeratorRange& r) {
    const float32x4_t denom = vdupq_n_f32(65535.0f);
    const uint32x4_t nan_bits = vreinterpretq_u32_f32(vdupq_n_f32(std::nanf("")));
    const uint32x4_t c65535 = vdupq_n_u32(65535);
    const uint32x4_t c131070 = vdupq_n_u32(131070);
    uint32x4_t v_min = vdupq_n_u32(0xFFFFFFFFu);
    uint32x4_t v_max = vdupq_n_u32(0);
    uint32x4_t v_bad = vdupq_n_u32(0);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const uint16x8x2_t pairs = vld2q_u16(reinterpret_cast<const uint16_t*>(prob + 4 * i));
        const uint16x8_t miss16 = vreinterpretq_u16_s16(
            vshrq_n_s16(vreinterpretq_s16_u16(vshll_n_u8(vld1_u8(missing + i), 8)), 15));
        for (int q = 0; q < 2; ++q) {
            const uint32x4_t aa =
                vmovl_u16(q ? vget_high_u16(pairs.val[0]) : vget_low_u16(pairs.val[0]));
            const uint32x4_t ab =
                vmovl_u16(q ? vget_high_u16(pairs.val[1]) : vget_low_u16(pairs.val[1]));
            const uint32x4_t miss = vreinterpretq_u32_s32(
                vmovl_s16(vreinterpret_s16_u16(q ? vget_high_u16(miss16) : vget_low_u16(miss16))));
            const uint32x4_t sum = vaddq_u32(aa, ab);
            const uint32x4_t k = vsubq_u32(c131070, vaddq_u32(sum, aa));
            const uint32x4_t bad = vorrq_u32(vcgtq_u32(sum, c65535), miss);
            v_bad = vorrq_u32(v_bad, bad);
            v_min = vminq_u32(v_min, vorrq_u32(k, bad));
            v_max = vmaxq_u32(v_max, vbicq_u32(k, bad));
            float32x4_t f = vdivq_f32(vcvtq_f32_u32(k), denom);
            f = vreinterpretq_f32_u32(vbslq_u32(bad, nan_bits, vreinterpretq_u32_f32(f)));
            store4(output + i + 4 * q, f);
        }
    }
    if (i > 0) {
        r.k_min = vminvq_u32(v_min);
        r.k_max = r.k_min == kNoNumerator ? 0 : vmaxvq_u32(v_max);
        r.any_nan = vmaxvq_u32(v_bad) != 0;
    }
    return i;
}
#endif  // __aarch64__

}  // namespace

template <typename T>
NumeratorRange decode_v12_unphased_8bit(const uint8_t* missing, const uint8_t* prob, size_t n,
                                        T* output) {
    NumeratorRange r{kNoNumerator, 0, false};
    size_t i = 0;
#ifdef LAZYBGEN_X86_AVX2
    if (cpu_has_avx2()) {
        i = fused_8bit_avx2<T>(missing, prob, n, output, r);
    }
#elif defined(__aarch64__)
    i = fused_8bit_neon<T>(missing, prob, n, output, r);
#endif
    fused_tail<T, 255u>(missing, prob, i, n, output, dosage_lut_8bit(), r);
    return r;
}

template <typename T>
NumeratorRange decode_v12_unphased_16bit(const uint8_t* missing, const uint8_t* prob, size_t n,
                                         T* output) {
    NumeratorRange r{kNoNumerator, 0, false};
    size_t i = 0;
#ifdef LAZYBGEN_X86_AVX2
    if (cpu_has_avx2()) {
        i = fused_16bit_avx2<T>(missing, prob, n, output, r);
    }
#elif defined(__aarch64__)
    i = fused_16bit_neon<T>(missing, prob, n, output, r);
#endif
    fused_tail<T, 65535u>(missing, prob, i, n, output, nullptr, r);
    return r;
}

template NumeratorRange decode_v12_unphased_8bit<float>(const uint8_t*, const uint8_t*, size_t,
                                                        float*);
template NumeratorRange decode_v12_unphased_8bit<double>(const uint8_t*, const uint8_t*, size_t,
                                                         double*);
template NumeratorRange decode_v12_unphased_16bit<float>(const uint8_t*, const uint8_t*, size_t,
                                                         float*);
template NumeratorRange decode_v12_unphased_16bit<double>(const uint8_t*, const uint8_t*, size_t,
                                                          double*);

}  // namespace simd
}  // namespace bgen
}  // namespace lazybgen