#include <stdexcept>

#include "compression_utils.h"
#include "decompressor.h"
#include "sequential_decompressor.h"

namespace lazybgen {
namespace bgen {
namespace decompress {

/**
 * Create a sequential decompressor
 *
 * This factory function creates a SequentialDecompressor with the provided
 * configuration. It requires a FileReader to be specified in the config.
 *
 * @param file_reader FileReader instance to use
 * @param config Base decompressor configuration
 * @return Unique pointer to SequentialDecompressor
 */
std::unique_ptr<VariantDecompressor> create_sequential_decompressor(
    lazybgen::io::bgen::FileReader* file_reader,
    const VariantDecompressor::Config& config = VariantDecompressor::Config()) {
    if (!file_reader) {
        throw std::invalid_argument("create_sequential_decompressor: file_reader cannot be null");
    }

    // Initialize compression libraries
    static bool initialized = initialize_compression_libraries();
    if (!initialized) {
        throw std::runtime_error("Failed to initialize compression libraries");
    }

    // Create sequential configuration
    SequentialDecompressor::SequentialConfig seq_config;

    // Copy base configuration
    seq_config.validate_size = config.validate_size;
    seq_config.max_decompressed_size = config.max_decompressed_size;

    // Set sequential-specific configuration
    seq_config.file_reader = file_reader;

    return std::unique_ptr<SequentialDecompressor>(new SequentialDecompressor(seq_config));
}

}  // namespace decompress
}  // namespace bgen
}  // namespace lazybgen