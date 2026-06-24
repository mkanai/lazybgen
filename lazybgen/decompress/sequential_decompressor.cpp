#include "sequential_decompressor.h"

#include <cstring>
#include <sstream>
#include <utility>

#include "../io/reader_interface.h"

namespace lazybgen {
namespace bgen {
namespace decompress {

SequentialDecompressor::SequentialDecompressor(const SequentialConfig& config)
    : VariantDecompressor(config), config_(config), file_reader_(config.file_reader) {
    // Validate configuration
    if (!file_reader_) {
        throw std::invalid_argument("SequentialDecompressor: file_reader cannot be null");
    }

    if (!file_reader_->is_open()) {
        throw std::invalid_argument("SequentialDecompressor: file_reader must have an open file");
    }

    // Prime the recycled decode buffer. A 4MB initial size matches the original
    // sequential default so the common single-block decompress reuses it without
    // reallocating per variant.
    decode_buffer_size_ = 4 * 1024 * 1024;  // 4MB
    decode_buffer_.reset(new uint8_t[decode_buffer_size_]);
}

SequentialDecompressor::~SequentialDecompressor() = default;

DecompressedData SequentialDecompressor::decompress(const CompressedVariant& variant) {
    try {
        // The caller always supplies already-read compressed data via variant.data.
        return decompress_data(variant.data, variant.compressed_size, variant.uncompressed_size,
                               variant.compression_type, variant.offset);
    } catch (const std::exception& e) {
        return DecompressedData(variant.offset, DecompressedData::COMPRESSION_ERROR,
                                std::string("Exception during decompression: ") + e.what());
    }
}

DecompressedData SequentialDecompressor::decompress_data(const uint8_t* compressed,
                                                         size_t compressed_size,
                                                         size_t expected_size,
                                                         CompressionType compression_type,
                                                         uint64_t offset) {
    // Ensure the recycled decode buffer is large enough for this variant.
    if (decode_buffer_size_ < expected_size) {
        decode_buffer_.reset(new uint8_t[expected_size]);
        decode_buffer_size_ = expected_size;
    }

    // Handle uncompressed data
    if (compression_type == CompressionType::None) {
        if (compressed_size != expected_size) {
            std::ostringstream oss;
            oss << "Uncompressed size mismatch: expected " << expected_size << ", got "
                << compressed_size;
            return DecompressedData(offset, DecompressedData::SIZE_MISMATCH, oss.str());
        }

        std::memcpy(decode_buffer_.get(), compressed, compressed_size);
        return finish(expected_size, offset);
    }

    // Perform decompression
    CompressionResult result;

    switch (compression_type) {
        case CompressionType::Zlib:
            result =
                decompress_zlib(compressed, compressed_size, decode_buffer_.get(), expected_size);
            break;

        case CompressionType::Zstd:
            result =
                decompress_zstd(compressed, compressed_size, decode_buffer_.get(), expected_size);
            break;

        default:
            return DecompressedData(offset, DecompressedData::UNSUPPORTED_COMPRESSION,
                                    "Unsupported compression type");
    }

    if (!result.success) {
        return DecompressedData(offset, DecompressedData::COMPRESSION_ERROR, result.error_message);
    }

    // Validate decompressed size if configured
    if (config_.validate_size && result.bytes_processed != expected_size) {
        std::ostringstream oss;
        oss << "Decompressed size mismatch: expected " << expected_size << ", got "
            << result.bytes_processed;
        return DecompressedData(offset, DecompressedData::SIZE_MISMATCH, oss.str());
    }

    return finish(result.bytes_processed, offset);
}

DecompressedData SequentialDecompressor::finish(size_t size, uint64_t offset) {
    // Hand the decode buffer's storage to the result, then re-allocate a fresh
    // recycled buffer of the same capacity for the next call.
    DecompressedData out(std::move(decode_buffer_), size, offset);
    decode_buffer_.reset(new uint8_t[decode_buffer_size_]);
    return out;
}

}  // namespace decompress
}  // namespace bgen
}  // namespace lazybgen
