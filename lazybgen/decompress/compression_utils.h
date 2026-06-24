#ifndef LAZYBGEN_BGEN_DECOMPRESS_COMPRESSION_UTILS_H
#define LAZYBGEN_BGEN_DECOMPRESS_COMPRESSION_UTILS_H

#include <cstddef>
#include <cstdint>
#include <string>

namespace lazybgen {
namespace bgen {
namespace decompress {

/**
 * CompressionResult - Result of compression/decompression operations
 */
struct CompressionResult {
    bool success;
    std::string error_message;
    size_t bytes_processed;  // Actual bytes written/read

    CompressionResult(bool s = false, const std::string& msg = "", size_t bytes = 0)
        : success(s), error_message(msg), bytes_processed(bytes) {}
};

/**
 * Decompress zlib-compressed data
 *
 * @param compressed Pointer to compressed data
 * @param compressed_size Size of compressed data
 * @param output Output buffer for decompressed data
 * @param output_size Size of output buffer (must be large enough)
 * @return CompressionResult with success status and actual decompressed size
 */
CompressionResult decompress_zlib(const uint8_t* compressed, size_t compressed_size,
                                  uint8_t* output, size_t output_size);

/**
 * Decompress zstd-compressed data
 *
 * @param compressed Pointer to compressed data
 * @param compressed_size Size of compressed data
 * @param output Output buffer for decompressed data
 * @param output_size Size of output buffer (must be large enough)
 * @return CompressionResult with success status and actual decompressed size
 */
CompressionResult decompress_zstd(const uint8_t* compressed, size_t compressed_size,
                                  uint8_t* output, size_t output_size);

/**
 * Initialize compression libraries
 *
 * This function initializes any required compression libraries.
 * It's called automatically but can be called explicitly for eager initialization.
 *
 * @return True if all libraries initialized successfully
 */
bool initialize_compression_libraries();

}  // namespace decompress
}  // namespace bgen
}  // namespace lazybgen

#endif  // LAZYBGEN_BGEN_DECOMPRESS_COMPRESSION_UTILS_H
