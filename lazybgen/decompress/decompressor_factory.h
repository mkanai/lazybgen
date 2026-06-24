#ifndef LAZYBGEN_DECOMPRESS_DECOMPRESSOR_FACTORY_H
#define LAZYBGEN_DECOMPRESS_DECOMPRESSOR_FACTORY_H

#include <memory>

#include "../io/reader_interface.h"
#include "decompressor.h"

namespace lazybgen {
namespace bgen {
namespace decompress {

/**
 * Factory functions for creating different types of decompressors
 */

/**
 * Create a sequential (single-block) decompressor
 *
 * @param file_reader FileReader instance to use
 * @param config Base decompressor configuration
 * @return Unique pointer to SequentialDecompressor
 */
std::unique_ptr<VariantDecompressor> create_sequential_decompressor(
    lazybgen::io::bgen::FileReader* file_reader,
    const VariantDecompressor::Config& config = VariantDecompressor::Config());

}  // namespace decompress
}  // namespace bgen
}  // namespace lazybgen

#endif  // LAZYBGEN_DECOMPRESS_DECOMPRESSOR_FACTORY_H