#include "compression_utils.h"

#include <cstring>
#include <sstream>

// Include vendored compression libraries
#include "libdeflate.h"  // Will be provided by the build's include paths
#include "zstd.h"        // Will be provided by the build's include paths

namespace lazybgen {
namespace bgen {
namespace decompress {

namespace {

/**
 * Per-thread libdeflate decompressor.
 *
 * A libdeflate_decompressor holds mutable decode state, so it must not be
 * shared between threads. Allocating one is cheap, but the block decode path
 * calls decompress_zlib once per variant across many worker threads, so each
 * thread keeps its own for its lifetime and the allocation never lands on the
 * per-variant hot path.
 */
struct DecompressorHandle {
    libdeflate_decompressor* d;

    DecompressorHandle() : d(libdeflate_alloc_decompressor()) {}

    ~DecompressorHandle() {
        if (d) {
            libdeflate_free_decompressor(d);
        }
    }

    DecompressorHandle(const DecompressorHandle&) = delete;
    DecompressorHandle& operator=(const DecompressorHandle&) = delete;
};

libdeflate_decompressor* thread_decompressor() {
    static thread_local DecompressorHandle handle;
    return handle.d;
}

const char* result_name(enum libdeflate_result result) {
    switch (result) {
        case LIBDEFLATE_SUCCESS:
            return "success";
        case LIBDEFLATE_BAD_DATA:
            return "invalid or corrupt compressed data";
        case LIBDEFLATE_SHORT_OUTPUT:
            return "stream decompressed to fewer bytes than expected";
        case LIBDEFLATE_INSUFFICIENT_SPACE:
            return "output buffer too small";
        default:
            return "unknown error";
    }
}

}  // namespace

CompressionResult decompress_zlib(const uint8_t* compressed, size_t compressed_size,
                                  uint8_t* output, size_t output_size) {
    if (!compressed || !output || compressed_size == 0 || output_size == 0) {
        return CompressionResult(false, "Invalid input parameters");
    }

    libdeflate_decompressor* decompressor = thread_decompressor();
    if (!decompressor) {
        return CompressionResult(false, "Failed to allocate DEFLATE decompressor");
    }

    // The BGEN spec stores zlib-compressed blocks in the zlib format (RFC 1950:
    // a 2-byte header, the deflate stream, an Adler-32 trailer) for every
    // layout. Recognize any valid header, not only the 32 KiB-window 0x78 ones a
    // default zlib writes: CM = 8 (deflate), CINFO <= 7 (window up to 32 KiB),
    // no preset dictionary (BGEN defines none), and the FCHECK bits making the
    // big-endian 16-bit header a multiple of 31. Streams without such a header
    // are read as raw deflate, which some non-conforming writers produce.
    const bool zlib_wrapped = compressed_size >= 2 && (compressed[0] & 0x0F) == 8 &&
                              (compressed[0] >> 4) <= 7 && (compressed[1] & 0x20) == 0 &&
                              ((compressed[0] << 8) | compressed[1]) % 31 == 0;

    // The caller supplies the exact size the block declares, so pass a non-NULL
    // actual_out_nbytes_ret: a stream that yields fewer bytes returns the short
    // count here rather than an error, leaving the size check to the caller
    // (which knows whether a short block is fatal).
    size_t bytes_written = 0;
    enum libdeflate_result result =
        zlib_wrapped ? libdeflate_zlib_decompress(decompressor, compressed, compressed_size, output,
                                                  output_size, &bytes_written)
                     : libdeflate_deflate_decompress(decompressor, compressed, compressed_size,
                                                     output, output_size, &bytes_written);
    if (zlib_wrapped && result == LIBDEFLATE_BAD_DATA) {
        // A raw deflate stream can begin with bytes that happen to form a valid
        // zlib header; read it the way it was read before the header check.
        result = libdeflate_deflate_decompress(decompressor, compressed, compressed_size, output,
                                               output_size, &bytes_written);
    }

    if (result == LIBDEFLATE_SUCCESS) {
        return CompressionResult(true, "", bytes_written);
    }

    if (result == LIBDEFLATE_INSUFFICIENT_SPACE) {
        return CompressionResult(false, "Output buffer too small", 0);
    }

    std::stringstream ss;
    ss << "Zlib decompression failed: " << static_cast<int>(result) << " ("
       << result_name(result) << ")";
    return CompressionResult(false, ss.str(), 0);
}

CompressionResult decompress_zstd(const uint8_t* compressed, size_t compressed_size,
                                  uint8_t* output, size_t output_size) {
    if (!compressed || !output || compressed_size == 0 || output_size == 0) {
        return CompressionResult(false, "Invalid input parameters");
    }

    // Perform decompression
    size_t result = ZSTD_decompress(output, output_size, compressed, compressed_size);

    if (ZSTD_isError(result)) {
        std::stringstream ss;
        ss << "Zstd decompression failed: " << ZSTD_getErrorName(result);
        return CompressionResult(false, ss.str());
    }

    // Check if output buffer was large enough
    if (result > output_size) {
        return CompressionResult(false, "Output buffer too small", 0);
    }

    return CompressionResult(true, "", result);
}

bool initialize_compression_libraries() {
    // Both libdeflate and zstd are statically linked and don't require
    // explicit initialization in our case
    return true;
}

}  // namespace decompress
}  // namespace bgen
}  // namespace lazybgen
