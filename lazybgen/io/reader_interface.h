#ifndef LAZYBGEN_BGEN_READER_INTERFACE_H
#define LAZYBGEN_BGEN_READER_INTERFACE_H

#include <cstddef>
#include <cstdint>
#include <string>

namespace lazybgen {
namespace io {
namespace bgen {

/**
 * FileReader - Abstract interface for file reading operations
 *
 * This interface allows different file reading implementations (regular files,
 * memory-mapped files, compressed files, etc.) to be used interchangeably.
 */
class FileReader {
   public:
    virtual ~FileReader() = default;

    /**
     * Read data from current position
     *
     * @param buffer Buffer to read into
     * @param size Number of bytes to read
     * @return Number of bytes actually read
     */
    virtual size_t read(uint8_t* buffer, size_t size) = 0;

    /**
     * Read data from specific offset (without changing current position)
     *
     * @param offset File offset to read from
     * @param buffer Buffer to read into
     * @param size Number of bytes to read
     * @return Number of bytes actually read
     */
    virtual size_t read_at(uint64_t offset, uint8_t* buffer, size_t size) = 0;

    /**
     * Seek to specific position
     *
     * @param offset File offset to seek to
     */
    virtual void seek(uint64_t offset) = 0;

    /**
     * Get current file position
     *
     * @return Current offset in file
     */
    virtual uint64_t tell() const = 0;

    /**
     * Get file size
     *
     * @return Total size of file in bytes
     */
    virtual uint64_t size() const = 0;

    /**
     * Check if file is open
     *
     * @return true if file is open and readable
     */
    virtual bool is_open() const = 0;

    /**
     * Close the file
     */
    virtual void close() = 0;

    /**
     * Get filename (for error messages)
     *
     * @return Filename or description
     */
    virtual const std::string& filename() const = 0;


    /**
     * Hint the per-read fetch (readahead) block size in bytes.
     *
     * For remote (fsspec) readers this sets the byte-range readahead block used
     * by random-access reads, so a scattered selection fetches close to one
     * variant's worth of bytes per request instead of a large fixed block. The
     * caller sizes this to the typical variant record once the header is known.
     * For local readers this is a no-op.
     *
     * @param block_size Suggested readahead block size in bytes
     */
    virtual void set_read_block_size(size_t block_size) {
        // Default implementation is a no-op
        (void)block_size;
    }
};

}  // namespace bgen
}  // namespace io
}  // namespace lazybgen

#endif  // LAZYBGEN_BGEN_READER_INTERFACE_H