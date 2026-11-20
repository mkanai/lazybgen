#ifndef LAZYBGEN_FORMAT_SIMD_TARGET_H
#define LAZYBGEN_FORMAT_SIMD_TARGET_H

// The extension is compiled for baseline x86-64, so it imports and runs on CPUs
// without AVX2 (issue #8). Code that should use AVX2/FMA is marked
// LAZYBGEN_TARGET_AVX2, which enables those instructions for that one function,
// and runs only after cpu_has_avx2() (genotype_parser_simd.h) confirms both.
// Compiling the whole extension with -mavx2 instead let the compiler emit AVX2
// anywhere, which killed the interpreter at import on older CPUs.
//
// LAZYBGEN_ALWAYS_INLINE lets one function body be compiled twice: inlined into
// an AVX2-targeted wrapper, and into a baseline caller.
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define LAZYBGEN_X86_AVX2 1
#define LAZYBGEN_TARGET_AVX2 __attribute__((target("avx2,fma")))
#define LAZYBGEN_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define LAZYBGEN_ALWAYS_INLINE inline
#endif

#endif  // LAZYBGEN_FORMAT_SIMD_TARGET_H
