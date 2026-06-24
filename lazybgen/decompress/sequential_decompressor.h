#ifndef LAZYBGEN_BGEN_DECOMPRESS_SEQUENTIAL_DECOMPRESSOR_H
#define LAZYBGEN_BGEN_DECOMPRESS_SEQUENTIAL_DECOMPRESSOR_H

#include <memory>

#include "../io/reader_interface.h"
#include "compression_utils.h"
#include "decompressor.h"

namespace lazybgen {
namespace bgen {
namespace decompress {

// Import FileReader into this namespace
using FileReader = ::lazybgen::io::bgen::FileReader;

/**
 * SequentialDecompressor - Single-block decompressor
 *
 * Decompresses one already-read compressed variant block at a time. The caller
 * supplies the compressed data pointer (read elsewhere), so this class only
 * runs the decompression kernel into a pooled output buffer.
 */
class SequentialDecompressor : public VariantDecompressor {
   public:
    // Extended configuration for sequential decompressor
    struct SequentialConfig : public VariantDecompressor::Config {
        // File reader to use (required)
        FileReader* file_reader = nullptr;
    };

    /**
     * Constructor
     * @param config Configuration for the decompressor
     */
    explicit SequentialDecompressor(const SequentialConfig& config);

    /**
     * Destructor
     */
    ~SequentialDecompressor() override;

    // Delete copy operations
    SequentialDecompressor(const SequentialDecompressor&) = delete;
    SequentialDecompressor& operator=(const SequentialDecompressor&) = delete;

    // Delete move operations for simplicity
    SequentialDecompressor(SequentialDecompressor&&) = delete;
    SequentialDecompressor& operator=(SequentialDecompressor&&) = delete;

    /**
     * Decompress a single variant
     *
     * @param variant Compressed variant data (data pointer must be set)
     * @return Decompressed data
     */
    DecompressedData decompress(const CompressedVariant& variant) override;

   private:
    // Configuration
    SequentialConfig config_;

    // File reader (not owned)
    FileReader* file_reader_;

    // Long-lived decode buffer recycled across decompress() calls. Each call
    // decompresses into this buffer, hands its storage to the result, then
    // re-allocates a fresh buffer, so a steady-state read reuses one buffer-sized
    // allocation per variant rather than going through a shared pool each time.
    std::unique_ptr<uint8_t[]> decode_buffer_;
    size_t decode_buffer_size_;

    /**
     * Decompress data using the appropriate algorithm
     *
     * @param compressed Compressed data
     * @param compressed_size Size of compressed data
     * @param expected_size Expected uncompressed size
     * @param compression_type Type of compression
     * @return Decompressed data
     */
    DecompressedData decompress_data(const uint8_t* compressed, size_t compressed_size,
                                     size_t expected_size, CompressionType compression_type,
                                     uint64_t offset);

    // Hand decode_buffer_'s storage to a result of the given size, then refill
    // decode_buffer_ from the pool.
    DecompressedData finish(size_t size, uint64_t offset);
};

}  // namespace decompress
}  // namespace bgen
}  // namespace lazybgen

#endif  // LAZYBGEN_BGEN_DECOMPRESS_SEQUENTIAL_DECOMPRESSOR_H
