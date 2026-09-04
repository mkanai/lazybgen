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
     * Read several byte ranges in one call
     *
     * Range i fills buffers[i] with sizes[i] bytes taken from offsets[i], and
     * out_read[i] receives the number of bytes actually read (short, as for
     * read_at, when the range runs past the end of the file).
     *
     * The default issues the ranges one at a time. A reader whose cost is
     * dominated by per-request latency rather than by bytes moved (a remote
     * object store) should override this to put them in flight together. Called
     * from the same thread that would have called read_at, so the same handle
     * and GIL rules apply.
     *
     * @param offsets File offset of each range
     * @param sizes Byte count of each range
     * @param buffers Destination buffer for each range
     * @param out_read Receives the bytes actually read for each range
     * @param count Number of ranges
     */
    virtual void read_many(const uint64_t* offsets, const size_t* sizes, uint8_t* const* buffers,
                           size_t* out_read, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            out_read[i] = read_at(offsets[i], buffers[i], sizes[i]);
        }
    }

    /**
     * Borrow a read-only view of [offset, offset + size) without copying
     *
     * A memory-mapped reader can hand back a pointer into its mapping; a
     * streaming or remote reader cannot and returns nullptr, in which case the
     * caller must read_at() into a buffer it owns. The returned pointer stays
     * valid until the reader is closed, and the bytes are safe to read from any
     * thread.
     *
     * @param offset File offset the view starts at
     * @param size Number of bytes the view must cover
     * @return Pointer to the bytes, or nullptr if this reader cannot provide one
     */
    virtual const uint8_t* view_at(uint64_t offset, size_t size) const {
        (void)offset;
        (void)size;
        return nullptr;
    }

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