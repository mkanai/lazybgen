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
// Map (URL scheme, transport) -> (module, FileSystem class). The obstore
// transport serves every scheme from one adapter class, which keys its own
// store off the URL; the fsspec transport has one class per scheme. Adding a
// scheme is one more line here plus one entry in lazybgen/remote.py.
bool backend_for(const std::string& f, const std::string& transport, Backend* out) {
    if (transport == "obstore") {
        if (f.rfind("gs://", 0) == 0 || f.rfind("s3://", 0) == 0) {
            *out = {"lazybgen.obstore_backend", "ObstoreFileSystem"};
            return true;
        }
        return false;
    }
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
                                   const std::string& transport, size_t buffer_size,
                                   size_t block_size)
    : fs_module_(nullptr),
      fs_(nullptr),
      file_obj_(nullptr),
      storage_options_(storage_options),
      has_cat_ranges_(false),
      filename_(filename),
      transport_(transport.empty() ? std::string("fsspec") : transport),
      file_size_(0),
      current_pos_(0),
      is_open_(false),
      buffer_size_(buffer_size),
      block_size_(block_size),
      buffer_start_(0),
      buffer_valid_(0) {
    Backend backend;
    if (!backend_for(filename, transport_, &backend)) {
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
    backend_for(filename_, transport_, &backend);  // validated in ctor

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

    // Batched range reads go through fs.cat_ranges, which older fsspec releases
    // do not have. Probe once here so read_many can fall back silently.
    has_cat_ranges_ = PyObject_HasAttrString(fs_, "cat_ranges") == 1;

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

namespace {
// One cat_ranges call puts every range in it in flight at once, so a call's
// range count is the concurrency and its byte total is what the returned Python
// objects hold at that moment. 128 matches fsspec's own default gather batch.
constexpr size_t kMaxRangesPerCall = 128;
constexpr size_t kMaxBytesPerCall = 64u * 1024u * 1024u;

// Ranges closer together than this are fetched as one request. A separate
// request costs a round trip (milliseconds); pulling a few unwanted kilobytes
// alongside the wanted ones costs far less. Variants selected from a contiguous
// region sit back to back, so they collapse into larger requests, while a
// scattered selection is spread across megabytes and never merges.
constexpr uint64_t kMaxMergeGap = 64u * 1024u;

// Upper bound on a single merged request, so a long contiguous run becomes
// several requests that go out together instead of one that goes out alone.
// This has to stay well under kMaxBytesPerCall: a merged run is bandwidth-bound
// rather than latency-bound, and if one merged request could fill a whole call
// then a contiguous read would issue them one at a time and reach a fraction of
// the available throughput.
constexpr uint64_t kMaxMergedBytes = 4u * 1024u * 1024u;

// At least this many merged requests must fit in one call, or the merging above
// would serialize exactly the reads it is meant to speed up.
constexpr size_t kMinMergedPerCall = 8;
static_assert(kMaxMergedBytes * kMinMergedPerCall <= kMaxBytesPerCall,
              "kMaxMergedBytes is too close to kMaxBytesPerCall: merged requests "
              "would be issued one per cat_ranges call, with no concurrency");
static_assert(kMinMergedPerCall <= kMaxRangesPerCall,
              "kMaxRangesPerCall cannot admit kMinMergedPerCall merged requests");

// One request covering [offset, offset + size), serving ranges order[first,last).
struct FetchGroup {
    uint64_t offset;
    size_t size;
    size_t first;
    size_t last;
};
}  // namespace

void FsspecFileReader::read_many(const uint64_t* offsets, const size_t* sizes,
                                 uint8_t* const* buffers, size_t* out_read, size_t count) {
    if (!is_open_) {
        throw std::runtime_error("FsspecFileReader: file is not open");
    }
    if (count == 0) {
        return;
    }
    // A single range has nothing to overlap or overlap with, and without
    // cat_ranges there is no batched path to take. Both keep the buffered
    // read_at behavior they had before.
    if (count == 1 || !has_cat_ranges_) {
        for (size_t i = 0; i < count; ++i) {
            out_read[i] = read_internal(offsets[i], buffers[i], sizes[i]);
        }
        return;
    }

    // Group in file order so neighbours can be merged, whatever order the
    // caller listed them in.
    std::vector<size_t> order(count);
    for (size_t i = 0; i < count; ++i) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(),
              [offsets](size_t a, size_t b) { return offsets[a] < offsets[b]; });

    std::vector<FetchGroup> groups;
    for (size_t k = 0; k < count; ++k) {
        const size_t i = order[k];
        const uint64_t begin = offsets[i];
        const uint64_t end = begin + sizes[i];
        if (!groups.empty()) {
            FetchGroup& g = groups.back();
            const uint64_t group_end = g.offset + g.size;
            // Overlapping or nested ranges have no gap at all, so max() keeps a
            // shorter follower from shrinking the group.
            const uint64_t gap = begin > group_end ? begin - group_end : 0;
            const uint64_t merged = std::max(end, group_end) - g.offset;
            if (gap <= kMaxMergeGap && merged <= kMaxMergedBytes) {
                g.size = static_cast<size_t>(merged);
                g.last = k + 1;
                continue;
            }
        }
        groups.push_back(FetchGroup{begin, sizes[i], k, k + 1});
    }

    // Issue the groups in batches, bounded by request count and by the bytes a
    // batch materializes at once.
    std::vector<uint64_t> batch_offsets;
    std::vector<size_t> batch_sizes;
    std::vector<std::vector<uint8_t>> fetched;
    size_t g0 = 0;
    while (g0 < groups.size()) {
        size_t g1 = g0;
        size_t bytes = 0;
        while (g1 < groups.size() && (g1 - g0) < kMaxRangesPerCall &&
               (g1 == g0 || bytes + groups[g1].size <= kMaxBytesPerCall)) {
            bytes += groups[g1].size;
            ++g1;
        }

        batch_offsets.clear();
        batch_sizes.clear();
        for (size_t g = g0; g < g1; ++g) {
            batch_offsets.push_back(groups[g].offset);
            batch_sizes.push_back(groups[g].size);
        }
        fetched.assign(g1 - g0, std::vector<uint8_t>());
        std::vector<uint8_t*> dests(g1 - g0);
        std::vector<size_t> got(g1 - g0, 0);
        for (size_t g = g0; g < g1; ++g) {
            fetched[g - g0].resize(groups[g].size);
            dests[g - g0] = fetched[g - g0].data();
        }
        fetch_ranges(batch_offsets.data(), batch_sizes.data(), dests.data(), got.data(), g1 - g0);

        // Hand each range its slice of the group it landed in. A group short of
        // its requested length (a range past the end of the file) yields
        // correspondingly short reads, which the caller checks.
        for (size_t g = g0; g < g1; ++g) {
            const FetchGroup& group = groups[g];
            const std::vector<uint8_t>& data = fetched[g - g0];
            const size_t available = got[g - g0];
            for (size_t k = group.first; k < group.last; ++k) {
                const size_t i = order[k];
                const size_t start = static_cast<size_t>(offsets[i] - group.offset);
                const size_t take =
                    start >= available ? 0 : std::min(sizes[i], available - start);
                if (take > 0) {
                    std::memcpy(buffers[i], data.data() + start, take);
                }
                out_read[i] = take;
            }
        }
        g0 = g1;
    }
}

void FsspecFileReader::fetch_ranges(const uint64_t* offsets, const size_t* sizes,
                                    uint8_t* const* buffers, size_t* out_read, size_t count) {
    RetryWrapper<size_t>::execute_with_retry(
        [this, offsets, sizes, buffers, out_read, count]() -> size_t {
            GilGuard gil;

            // cat_ranges takes parallel lists: one path per range, plus start and
            // end offsets. Every range here is on the same file.
            PyObject* paths = PyList_New(static_cast<Py_ssize_t>(count));
            PyObject* starts = PyList_New(static_cast<Py_ssize_t>(count));
            PyObject* ends = PyList_New(static_cast<Py_ssize_t>(count));
            if (!paths || !starts || !ends) {
                Py_XDECREF(paths);
                Py_XDECREF(starts);
                Py_XDECREF(ends);
                raise_python_error("allocate cat_ranges arguments");
            }
            for (size_t i = 0; i < count; ++i) {
                PyObject* path = PyUnicode_FromString(filename_.c_str());
                PyObject* begin = PyLong_FromUnsignedLongLong(offsets[i]);
                PyObject* stop = PyLong_FromUnsignedLongLong(
                    static_cast<unsigned long long>(offsets[i]) + sizes[i]);
                if (!path || !begin || !stop) {
                    Py_XDECREF(path);
                    Py_XDECREF(begin);
                    Py_XDECREF(stop);
                    Py_DECREF(paths);
                    Py_DECREF(starts);
                    Py_DECREF(ends);
                    raise_python_error("encode cat_ranges arguments");
                }
                // PyList_SET_ITEM steals the reference.
                PyList_SET_ITEM(paths, static_cast<Py_ssize_t>(i), path);
                PyList_SET_ITEM(starts, static_cast<Py_ssize_t>(i), begin);
                PyList_SET_ITEM(ends, static_cast<Py_ssize_t>(i), stop);
            }

            PyObject* result =
                PyObject_CallMethod(fs_, "cat_ranges", "OOO", paths, starts, ends);
            Py_DECREF(paths);
            Py_DECREF(starts);
            Py_DECREF(ends);
            if (!result) {
                raise_python_error("cat_ranges");
            }

            PyObject* seq = PySequence_Fast(result, "cat_ranges did not return a sequence");
            if (!seq) {
                Py_DECREF(result);
                raise_python_error("read cat_ranges result");
            }
            if (static_cast<size_t>(PySequence_Fast_GET_SIZE(seq)) != count) {
                const Py_ssize_t got = PySequence_Fast_GET_SIZE(seq);
                Py_DECREF(seq);
                Py_DECREF(result);
                throw std::runtime_error("FsspecFileReader: cat_ranges returned " +
                                         std::to_string(got) + " results for " +
                                         std::to_string(count) + " ranges");
            }

            for (size_t i = 0; i < count; ++i) {
                // Borrowed reference into the sequence.
                PyObject* item = PySequence_Fast_GET_ITEM(seq, static_cast<Py_ssize_t>(i));
                // Any object exposing a contiguous buffer is acceptable: fsspec
                // returns bytes, the obstore adapter returns a view of the Rust
                // allocation, and taking the buffer rather than the bytes keeps
                // the latter copy-free. An async filesystem (gcsfs, s3fs) hands a
                // failed range back as the exception object in the result list
                // rather than raising, and an exception has no buffer, so this
                // check catches that too.
                Py_buffer view;
                if (PyObject_GetBuffer(item, &view, PyBUF_SIMPLE) < 0) {
                    PyErr_Clear();
                    std::string detail = "result does not expose a buffer";
                    PyObject* text = PyObject_Repr(item);
                    if (text) {
                        const char* utf8 = PyUnicode_AsUTF8(text);
                        if (utf8) {
                            detail = utf8;
                        }
                        Py_DECREF(text);
                    }
                    PyErr_Clear();
                    Py_DECREF(seq);
                    Py_DECREF(result);
                    throw std::runtime_error("FsspecFileReader: cat_ranges failed for range at " +
                                             std::to_string(offsets[i]) + ": " + detail);
                }
                // Clamp exactly as read_internal does: the destination holds only
                // sizes[i] bytes whatever the filesystem hands back.
                const size_t copied = std::min(static_cast<size_t>(view.len), sizes[i]);
                if (copied > 0) {
                    std::memcpy(buffers[i], view.buf, copied);
                }
                PyBuffer_Release(&view);
                out_read[i] = copied;
            }

            Py_DECREF(seq);
            Py_DECREF(result);
            return count;
        },
        "cat_ranges");
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
