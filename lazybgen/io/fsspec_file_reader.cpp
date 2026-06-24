#include "fsspec_file_reader.h"

#include <algorithm>
#include <cstring>
#include <sstream>
#include <stdexcept>

#include "retry_wrapper.h"

namespace lazybgen {
namespace io {
namespace bgen {

namespace {
struct Backend { const char* module; const char* class_name; };
// Map URL scheme -> (fsspec module, FileSystem class).
bool backend_for(const std::string& f, Backend* out) {
    if (f.rfind("gs://", 0) == 0) { *out = {"gcsfs", "GCSFileSystem"}; return true; }
    if (f.rfind("s3://", 0) == 0) { *out = {"s3fs", "S3FileSystem"}; return true; }
    return false;
}

// RAII guard for the Python GIL. Acquires on construction, releases on
// destruction (including during stack unwinding from a C++ throw), so the GIL is
// never leaked across an exception. Per-frame: each method declares its own.
class GilGuard {
   public:
    GilGuard() : state_(PyGILState_Ensure()) {}
    ~GilGuard() { PyGILState_Release(state_); }
    GilGuard(const GilGuard&) = delete;
    GilGuard& operator=(const GilGuard&) = delete;

   private:
    PyGILState_STATE state_;
};
}  // namespace

FsspecFileReader::FsspecFileReader(const std::string& filename, PyObject* storage_options,
                                   size_t buffer_size, size_t block_size)
    : fs_module_(nullptr),
      fs_(nullptr),
      file_obj_(nullptr),
      storage_options_(storage_options),
      filename_(filename),
      file_size_(0),
      current_pos_(0),
      is_open_(false),
      buffer_size_(buffer_size),
      block_size_(block_size),
      buffer_start_(0),
      buffer_valid_(0) {
    Backend backend;
    if (!backend_for(filename, &backend)) {
        throw std::runtime_error("FsspecFileReader: unsupported URL scheme: " + filename);
    }
    buffer_.resize(buffer_size_);
    // If initialize_python / open_file throws mid-construction, ~FsspecFileReader
    // does NOT run (the object was never fully constructed), so any owned Python
    // refs acquired so far would leak. Release them here before rethrowing.
    try {
        initialize_python();
        open_file();
    } catch (...) {
        GilGuard gil;
        Py_CLEAR(file_obj_);
        Py_CLEAR(fs_);
        Py_CLEAR(fs_module_);
        // storage_options_ is borrowed; do NOT decref.
        throw;
    }
}

FsspecFileReader::~FsspecFileReader() {
    close();

    // Release Python objects
    GilGuard gil;
    Py_XDECREF(file_obj_);
    Py_XDECREF(fs_);
    Py_XDECREF(fs_module_);
    // storage_options_ is borrowed; do NOT decref.
}

void FsspecFileReader::initialize_python() {
    GilGuard gil;

    Backend backend{};
    backend_for(filename_, &backend);  // validated in ctor

    fs_module_ = PyImport_ImportModule(backend.module);
    if (!fs_module_) {
        throw std::runtime_error(std::string("Failed to import ") + backend.module +
                                 " module. Please install: pip install " + backend.module);
    }

    PyObject* fs_class = PyObject_GetAttrString(fs_module_, backend.class_name);
    if (!fs_class) {
        raise_python_error("get FileSystem class");
    }

    PyObject* args = PyTuple_New(0);
    if (!args) {
        Py_DECREF(fs_class);
        raise_python_error("allocate args tuple");
    }

    bool own_kwargs = false;
    PyObject* kwargs;
    if (storage_options_ && storage_options_ != Py_None && PyDict_Check(storage_options_)) {
        kwargs = storage_options_;  // borrowed
    } else {
        kwargs = PyDict_New();  // owned
        own_kwargs = true;
        if (!kwargs) {
            Py_DECREF(args);
            Py_DECREF(fs_class);
            raise_python_error("allocate kwargs dict");
        }
    }

    fs_ = PyObject_Call(fs_class, args, kwargs);
    Py_DECREF(args);
    if (own_kwargs) {
        Py_DECREF(kwargs);
    }
    Py_DECREF(fs_class);

    if (!fs_) {
        raise_python_error("create FileSystem");
    }
}

void FsspecFileReader::open_file() {
    GilGuard gil;

    // Get file info to determine size
    PyObject* info_method = PyObject_GetAttrString(fs_, "info");
    if (!info_method) {
        raise_python_error("get info method");
    }

    PyObject* path_arg = PyUnicode_FromString(filename_.c_str());
    if (!path_arg) {
        Py_DECREF(info_method);
        raise_python_error("encode path");
    }
    PyObject* info_result = PyObject_CallFunctionObjArgs(info_method, path_arg, NULL);
    Py_DECREF(info_method);
    Py_DECREF(path_arg);

    if (!info_result) {
        raise_python_error("get file info");
    }

    // Extract file size from info dict (borrowed reference)
    PyObject* size_obj = PyDict_GetItemString(info_result, "size");
    if (!size_obj) {
        Py_DECREF(info_result);
        throw std::runtime_error("Failed to get file size from remote info dict");
    }

    file_size_ = PyLong_AsUnsignedLongLong(size_obj);
    // PyLong_AsUnsignedLongLong returns (unsigned long long)-1 and sets an error
    // on a non-int / None / overflow. Check immediately so we never proceed with a
    // corrupt size or a dangling exception.
    if (file_size_ == static_cast<unsigned long long>(-1) && PyErr_Occurred()) {
        Py_DECREF(info_result);
        raise_python_error("parse file size");
    }
    Py_DECREF(info_result);

    // Open the file handle (uses block_size_ for the fsspec readahead block).
    open_file_handle();
}

void FsspecFileReader::open_file_handle() {
    GilGuard gil;

    // Open file for reading
    PyObject* open_method = PyObject_GetAttrString(fs_, "open");
    if (!open_method) {
        raise_python_error("get open method");
    }

    // Build all open() arguments, checking every allocation. On any NULL we
    // release whatever we already own (XDECREF tolerates NULL) and throw.
    PyObject* path_arg = PyUnicode_FromString(filename_.c_str());
    PyObject* mode_arg = PyUnicode_FromString("rb");
    PyObject* kwargs = PyDict_New();
    PyObject* block_size = PyLong_FromSize_t(block_size_);
    PyObject* args = nullptr;

    if (!path_arg || !mode_arg || !kwargs || !block_size) {
        Py_XDECREF(block_size);
        Py_XDECREF(kwargs);
        Py_XDECREF(mode_arg);
        Py_XDECREF(path_arg);
        Py_DECREF(open_method);
        raise_python_error("allocate open() arguments");
    }

    // Set block_size for efficient reading
    if (PyDict_SetItemString(kwargs, "block_size", block_size) < 0) {
        Py_DECREF(block_size);
        Py_DECREF(kwargs);
        Py_DECREF(mode_arg);
        Py_DECREF(path_arg);
        Py_DECREF(open_method);
        raise_python_error("set block_size");
    }
    Py_DECREF(block_size);

    args = PyTuple_Pack(2, path_arg, mode_arg);
    if (!args) {
        Py_DECREF(kwargs);
        Py_DECREF(mode_arg);
        Py_DECREF(path_arg);
        Py_DECREF(open_method);
        raise_python_error("pack open() args");
    }

    file_obj_ = PyObject_Call(open_method, args, kwargs);

    Py_DECREF(args);
    Py_DECREF(kwargs);
    Py_DECREF(mode_arg);
    Py_DECREF(path_arg);
    Py_DECREF(open_method);

    if (!file_obj_) {
        raise_python_error("open file");
    }

    is_open_ = true;
}

size_t FsspecFileReader::read(uint8_t* buffer, size_t size) {
    if (!is_open_) {
        throw std::runtime_error("FsspecFileReader: file is not open");
    }

    size_t total_read = 0;

    while (total_read < size && current_pos_ < file_size_) {
        // Try to read from buffer first
        size_t from_buffer = read_from_buffer(buffer + total_read, size - total_read);
        total_read += from_buffer;
        current_pos_ += from_buffer;

        // If we need more data and haven't reached EOF, refill buffer
        if (total_read < size && current_pos_ < file_size_) {
            fill_buffer(current_pos_);
            // Guard against a backend that returns no bytes despite not being at
            // EOF: without this the loop would spin forever re-issuing the read.
            if (buffer_valid_ == 0) {
                break;
            }
        }
    }

    return total_read;
}

size_t FsspecFileReader::read_at(uint64_t offset, uint8_t* buffer, size_t size) {
    if (!is_open_) {
        throw std::runtime_error("FsspecFileReader: file is not open");
    }

    // For read_at, we bypass the buffer and read directly
    return read_internal(offset, buffer, size);
}

void FsspecFileReader::seek(uint64_t offset) {
    if (offset > file_size_) {
        throw std::runtime_error("FsspecFileReader: seek beyond end of file");
    }
    current_pos_ = offset;
}

uint64_t FsspecFileReader::tell() const {
    return current_pos_;
}

uint64_t FsspecFileReader::size() const {
    return file_size_;
}

bool FsspecFileReader::is_open() const {
    return is_open_;
}

void FsspecFileReader::close() {
    if (!is_open_) {
        return;
    }

    GilGuard gil;

    if (file_obj_) {
        PyObject* close_method = PyObject_GetAttrString(file_obj_, "close");
        if (close_method) {
            PyObject* result = PyObject_CallObject(close_method, NULL);
            Py_XDECREF(result);
            Py_DECREF(close_method);
        }
        // Swallow any close() error: close() must not throw (called from dtor).
        PyErr_Clear();
    }

    is_open_ = false;
}

void FsspecFileReader::set_read_block_size(size_t block_size) {
    // Clamp to a sane range: large enough to coalesce a variant's metadata +
    // genotype into one range request, small enough to avoid the fixed-block
    // over-fetch that otherwise dominates scattered remote reads.
    const size_t kMinBlock = 256 * 1024;
    const size_t kMaxBlock = 16 * 1024 * 1024;
    block_size = std::min(std::max(block_size, kMinBlock), kMaxBlock);

    if (block_size == block_size_) {
        return;  // nothing to do
    }
    if (!is_open_ || !file_obj_) {
        block_size_ = block_size;  // takes effect at next open
        return;
    }

    // Reopen the handle with the new block size. Open the new handle first and
    // only release the old one on success, so a transient failure leaves the
    // existing (working) handle in place.
    GilGuard gil;

    PyObject* old_file_obj = file_obj_;
    file_obj_ = nullptr;
    block_size_ = block_size;
    try {
        open_file_handle();  // sets file_obj_ using the new block_size_
    } catch (...) {
        file_obj_ = old_file_obj;  // keep the working handle
        throw;
    }

    // Success: close and release the superseded handle.
    PyObject* close_method = PyObject_GetAttrString(old_file_obj, "close");
    if (close_method) {
        PyObject* result = PyObject_CallObject(close_method, NULL);
        Py_XDECREF(result);
        Py_DECREF(close_method);
    }
    PyErr_Clear();
    Py_DECREF(old_file_obj);

    // The sequential read buffer was filled by the old handle; invalidate it so
    // the next read() refills through the reopened handle.
    buffer_valid_ = 0;
}

size_t FsspecFileReader::read_internal(uint64_t offset, uint8_t* buffer, size_t size) {
    // Wrap the entire read operation with retry logic
    return RetryWrapper<size_t>::execute_with_retry(
        [this, offset, buffer, size]() -> size_t {
            GilGuard gil;

            // Seek to position
            PyObject* seek_method = PyObject_GetAttrString(file_obj_, "seek");
            if (!seek_method) {
                raise_python_error("get seek method");
            }

            PyObject* offset_arg = PyLong_FromUnsignedLongLong(offset);
            if (!offset_arg) {
                Py_DECREF(seek_method);
                raise_python_error("encode seek offset");
            }
            PyObject* seek_result = PyObject_CallFunctionObjArgs(seek_method, offset_arg, NULL);
            Py_DECREF(offset_arg);
            Py_DECREF(seek_method);

            if (!seek_result) {
                raise_python_error("seek");
            }
            Py_DECREF(seek_result);

            // Read data
            PyObject* read_method = PyObject_GetAttrString(file_obj_, "read");
            if (!read_method) {
                raise_python_error("get read method");
            }

            PyObject* size_arg = PyLong_FromSize_t(size);
            if (!size_arg) {
                Py_DECREF(read_method);
                raise_python_error("encode read size");
            }
            PyObject* data = PyObject_CallFunctionObjArgs(read_method, size_arg, NULL);
            Py_DECREF(size_arg);
            Py_DECREF(read_method);

            if (!data) {
                raise_python_error("read");
            }

            // Extract bytes from Python object
            char* py_buffer;
            Py_ssize_t py_size;

            if (PyBytes_AsStringAndSize(data, &py_buffer, &py_size) < 0) {
                Py_DECREF(data);
                raise_python_error("extract bytes");
            }

            // Clamp to the requested size: the destination holds only `size`
            // bytes, so never copy more even if a (custom or buggy) filesystem
            // returns an over-long read. fsspec's contract is read(n) <= n bytes.
            size_t bytes_read = std::min(static_cast<size_t>(py_size), size);
            std::memcpy(buffer, py_buffer, bytes_read);

            Py_DECREF(data);

            return bytes_read;
        },
        "remote read",
        3  // max retries
    );
}

void FsspecFileReader::fill_buffer(uint64_t offset) {
    buffer_start_ = offset;
    buffer_valid_ = read_internal(offset, buffer_.data(), buffer_size_);
}

size_t FsspecFileReader::read_from_buffer(uint8_t* buffer, size_t size) {
    // Check if current position is within buffer
    if (current_pos_ < buffer_start_ || current_pos_ >= buffer_start_ + buffer_valid_) {
        return 0;  // Not in buffer
    }

    uint64_t buffer_offset = current_pos_ - buffer_start_;
    size_t available = buffer_valid_ - buffer_offset;
    size_t to_copy = std::min(size, available);

    std::memcpy(buffer, buffer_.data() + buffer_offset, to_copy);

    return to_copy;
}

void FsspecFileReader::raise_python_error(const std::string& operation) {
    // The GIL is held by the caller (via a GilGuard frame). Build a message from
    // the pending Python exception, if any, then throw UNCONDITIONALLY: this is
    // only reached after a Python C-API call returned NULL / a failure code, so a
    // NULL-without-set-exception must not silently fall through to a NULL deref.
    std::string error_msg = "Python error in " + operation;

    if (PyErr_Occurred()) {
        PyObject *type, *value, *traceback;
        PyErr_Fetch(&type, &value, &traceback);

        error_msg += ": ";
        if (value) {
            PyObject* str_value = PyObject_Str(value);
            if (str_value) {
                const char* error_str = PyUnicode_AsUTF8(str_value);
                if (error_str) {
                    error_msg += error_str;
                }
                Py_DECREF(str_value);
            }
        }

        Py_XDECREF(type);
        Py_XDECREF(value);
        Py_XDECREF(traceback);
    } else {
        error_msg += " (call failed without setting a Python exception)";
    }

    throw std::runtime_error(error_msg);
}

}  // namespace bgen
}  // namespace io
}  // namespace lazybgen
