#ifndef LAZYBGEN_BGEN_FSSPEC_FILE_READER_H
#define LAZYBGEN_BGEN_FSSPEC_FILE_READER_H

#include <Python.h>

#include <memory>
#include <utility>
#include <vector>

#include "reader_interface.h"

namespace lazybgen {
namespace io {
namespace bgen {

/**
 * FsspecFileReader - File reader implementation for fsspec-backed remote stores
 *
 * Uses a Python fsspec backend (gcsfs / s3fs) to read BGEN files directly from
 * remote object storage without downloading the entire file. Supports efficient
 * range requests and buffering for sequential access patterns.
 */
class FsspecFileReader : public FileReader {
   public:
    /**
     * Constructor
     * @param filename remote path (gs:// or s3://)
     * @param storage_options borrowed dict of kwargs for the FileSystem ctor (may be NULL)
     * @param buffer_size Internal sequential read buffer in bytes (default 1MB).
     *        This buffer backs only the one-time header/sample-id parse at open
     *        (read()); all genotype I/O is random-access read_at that bypasses it.
     *        A larger buffer just over-fetches at open. read() loops fill_buffer,
     *        so a sample block larger than the buffer still reads correctly (a few
     *        extra one-time GETs).
     * @param block_size fsspec readahead block size in bytes for random-access
     *        reads (default 1MB). Tuned per file via set_read_block_size once the
     *        header is known; a large block over-fetches on scattered selections.
     */
    explicit FsspecFileReader(const std::string& filename, PyObject* storage_options = nullptr,
                              size_t buffer_size = 1024 * 1024,
                              size_t block_size = 1024 * 1024);

    ~FsspecFileReader() override;

    // FileReader interface implementation
    size_t read(uint8_t* buffer, size_t size) override;
    size_t read_at(uint64_t offset, uint8_t* buffer, size_t size) override;
    void read_many(const uint64_t* offsets, const size_t* sizes, uint8_t* const* buffers,
                   size_t* out_read, size_t count) override;
    void seek(uint64_t offset) override;
    uint64_t tell() const override;
    uint64_t size() const override;
    bool is_open() const override;
    void close() override;
    void set_read_block_size(size_t block_size) override;
    const std::string& filename() const override {
        return filename_;
    }

   private:
    // Initialize Python and import required modules
    void initialize_python();

    // Create fsspec filesystem, then fetch file size and open the handle
    void open_file();

    // Open (or reopen) the fsspec file handle using the current block_size_.
    // Assumes file_obj_ is NULL; sets file_obj_ and is_open_ on success.
    void open_file_handle();

    // Read data using the fsspec backend
    size_t read_internal(uint64_t offset, uint8_t* buffer, size_t size);

    // Fetch one group of ranges through fs.cat_ranges, which puts them in
    // flight together. Sized by the caller so the bytes it materializes stay
    // bounded. Requires the GIL to be free (it takes it itself).
    void fetch_ranges(const uint64_t* offsets, const size_t* sizes, uint8_t* const* buffers,
                      size_t* out_read, size_t count);

    // Buffer management for sequential reads
    void fill_buffer(uint64_t offset);
    size_t read_from_buffer(uint8_t* buffer, size_t size);

    // Python objects (owned references)
    PyObject* fs_module_;        // fsspec backend module (gcsfs / s3fs)
    PyObject* fs_;               // FileSystem instance
    PyObject* file_obj_;         // file handle from fs.open()
    PyObject* storage_options_;  // borrowed; kwargs for the FileSystem ctor (may be NULL)
    bool has_cat_ranges_;        // filesystem exposes the batched range API

    // File information
    std::string filename_;
    uint64_t file_size_;
    uint64_t current_pos_;
    bool is_open_;

    // Buffering for sequential reads
    std::vector<uint8_t> buffer_;
    size_t buffer_size_;
    size_t block_size_;  // fsspec readahead block size for random-access reads
    uint64_t buffer_start_;  // File offset where buffer starts
    size_t buffer_valid_;    // Number of valid bytes in buffer

    // Error handling. Raises a std::runtime_error describing the pending Python
    // exception (if any) for ``operation``. The GIL must be held by the caller.
    void raise_python_error(const std::string& operation);

    // Disable copy operations
    FsspecFileReader(const FsspecFileReader&) = delete;
    FsspecFileReader& operator=(const FsspecFileReader&) = delete;
};

}  // namespace bgen
}  // namespace io
}  // namespace lazybgen

#endif  // LAZYBGEN_BGEN_FSSPEC_FILE_READER_H