#include "compression_utils.h"

#include <cstring>
#include <sstream>

// Include vendored compression libraries
extern "C" {
#include "zlib.h"  // Will be provided by CMake include paths
}
#include "zstd.h"  // Will be provided by CMake include paths

namespace lazybgen {
namespace bgen {
namespace decompress {

CompressionResult decompress_zlib(const uint8_t* compressed, size_t compressed_size,
                                  uint8_t* output, size_t output_size) {
    if (!compressed || !output || compressed_size == 0 || output_size == 0) {
        return CompressionResult(false, "Invalid input parameters");
    }

    // Validate input parameters

    // Initialize zlib stream
    z_stream stream;
    std::memset(&stream, 0, sizeof(stream));

    stream.next_in = const_cast<Bytef*>(compressed);
    stream.avail_in = static_cast<uInt>(compressed_size);
    stream.next_out = output;
    stream.avail_out = static_cast<uInt>(output_size);

    // Initialize inflation based on format
    // Check for zlib header to determine format
    // BGEN v1.1 uses standard zlib (with header)
    // BGEN v1.2 uses raw deflate (no header)
    int ret;
    if (compressed_size >= 2 && compressed[0] == 0x78 &&
        (compressed[1] == 0x01 || compressed[1] == 0x5E || compressed[1] == 0x9C ||
         compressed[1] == 0xDA)) {
        // Standard zlib format detected (v1.1)
        ret = inflateInit(&stream);
    } else {
        // Raw deflate format (v1.2)
        ret = inflateInit2(&stream, -15);
    }

    if (ret != Z_OK) {
        std::stringstream ss;
        ss << "Failed to initialize zlib decompression: " << ret;
        if (stream.msg)
            ss << " (" << stream.msg << ")";
        return CompressionResult(false, ss.str());
    }

    // Perform decompression
    ret = inflate(&stream, Z_FINISH);
    size_t bytes_written = output_size - stream.avail_out;

    // Clean up
    inflateEnd(&stream);

    if (ret == Z_STREAM_END) {
        return CompressionResult(true, "", bytes_written);
    } else if (ret == Z_OK) {
        // Partial decompression (output buffer too small)
        return CompressionResult(false, "Output buffer too small", bytes_written);
    } else {
        // Error
        std::stringstream ss;
        ss << "Zlib decompression failed: " << ret;
        if (stream.msg)
            ss << " (" << stream.msg << ")";
        return CompressionResult(false, ss.str(), bytes_written);
    }
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
    // Both zlib-ng and zstd are statically linked and don't require
    // explicit initialization in our case
    return true;
}

}  // namespace decompress
}  // namespace bgen
}  // namespace lazybgen
