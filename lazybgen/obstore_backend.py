"""obstore-backed transport for remote (gs:// / s3://) reads.

``ObstoreFileSystem`` presents the small filesystem surface the C++ reader uses
(``info`` / ``cat_ranges`` / ``open``) on top of `obstore
<https://developmentseed.org/obstore/>`_, which runs HTTP and TLS in Rust with
the GIL released.

Why this exists: fsspec drives every request in a process through one asyncio
event loop on one thread, so a single reader's remote throughput is capped by one
core of Python-side HTTP/TLS work no matter how many requests are in flight.
obstore has no such loop, so ``cat_ranges`` here splits a batch across a thread
pool and the requests genuinely overlap. obstore's own ``get_ranges`` caps
in-flight requests at ten per call, which is why the fan-out is done here rather
than left to it.

``storage_options`` stay in fsspec spelling whichever transport serves them, so
callers write one dict. :func:`translate_storage_options` maps that dict onto
obstore's constructor keywords and raises :class:`UnsupportedStorageOption` for
anything it cannot express; the caller then falls back to fsspec.

obstore is a dependency, but this module never assumes it is importable: it is
imported inside the functions that use it, and :func:`is_available` reports
whether a given environment actually has it.
"""

import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "ObstoreFileSystem",
    "UnsupportedStorageOption",
    "is_available",
    "translate_storage_options",
]


class UnsupportedStorageOption(ValueError):
    """A storage_options entry that the obstore transport cannot express."""


# Requests are split across this many threads. Measured on a same-region GCS
# object, 16 is where a batch stops getting faster; 32 is no better on a
# contiguous read and worse on a scattered one.
_DEFAULT_THREADS = 16

# Splitting costs a thread hand-off per chunk, which is only worth it once a
# batch has enough ranges to overlap. Below this a batch is fetched inline.
_MIN_RANGES_TO_SPLIT = 4

_pool: Optional[ThreadPoolExecutor] = None
_pool_lock = threading.Lock()


def _thread_count() -> int:
    """Fan-out width for a range batch, overridable per process."""
    raw = os.environ.get("LAZYBGEN_OBSTORE_THREADS")
    if not raw:
        return _DEFAULT_THREADS
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_THREADS
    return max(1, value)


def _shared_pool() -> ThreadPoolExecutor:
    """Process-wide fetch pool, created on first use.

    One pool for the process rather than one per reader, so opening many readers
    cannot multiply the number of sockets in flight.
    """
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ThreadPoolExecutor(max_workers=_thread_count(), thread_name_prefix="lazybgen-obstore")
    return _pool


def is_available() -> bool:
    """Return True if the obstore package can be imported."""
    try:
        import obstore  # noqa: F401
    except ImportError:
        return False
    return True


def split_url(url: str) -> Tuple[str, str, str]:
    """Split ``scheme://bucket/key`` into its three parts."""
    scheme, sep, rest = url.partition("://")
    if not sep:
        raise ValueError(f"not a remote URL: {url}")
    bucket, _, key = rest.partition("/")
    if not bucket or not key:
        raise ValueError(f"remote URL needs a bucket and a key: {url}")
    return scheme, bucket, key


def _billing_project(options: Dict[str, Any]) -> str:
    """Resolve the project billed for a GCS requester-pays read.

    ``requester_pays`` may name the project directly; ``requester_pays=True``
    means "bill the environment's default project", which is resolved the same
    way the Google client libraries resolve it.
    """
    requester_pays = options.get("requester_pays")
    if isinstance(requester_pays, str):
        return requester_pays

    project = options.get("project")
    if isinstance(project, str) and project:
        return project

    for name in ("GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT", "GCP_PROJECT", "CLOUDSDK_CORE_PROJECT"):
        value = os.environ.get(name)
        if value:
            return value

    try:
        import google.auth

        _credentials, default_project = google.auth.default()
    except Exception:
        default_project = None
    if default_project:
        return str(default_project)

    raise UnsupportedStorageOption(
        "requester_pays=True needs a billing project and none could be resolved; "
        'pass requester_pays="<project-id>" or set GOOGLE_CLOUD_PROJECT'
    )


def _translate_gcs(options: Dict[str, Any]) -> Dict[str, Any]:
    """Map gcsfs storage_options onto GCSStore keyword arguments."""
    kwargs: Dict[str, Any] = {}
    headers: Dict[str, str] = {}

    for key, value in options.items():
        if key == "anon":
            if value:
                kwargs["skip_signature"] = True
        elif key == "token":
            if value in (None, "google_default", "cloud", "default"):
                # The default credential chain, which is what obstore already does.
                continue
            if value == "anon":
                kwargs["skip_signature"] = True
            elif isinstance(value, str) and os.path.exists(value):
                kwargs["service_account"] = value
            elif isinstance(value, dict):
                kwargs["service_account_key"] = json.dumps(value)
            else:
                raise UnsupportedStorageOption(f"token={value!r} is not supported by the obstore transport")
        elif key in ("requester_pays", "project"):
            # Handled together below, since either can carry the billing project.
            continue
        else:
            raise UnsupportedStorageOption(f"{key!r} is not supported by the obstore transport")

    if options.get("requester_pays"):
        # GCS requester-pays has no dedicated obstore setting. The XML API reads
        # the billing project from this header, which object_store sends on every
        # request when it is given as a default header.
        headers["x-goog-user-project"] = _billing_project(options)

    if headers:
        kwargs["client_options"] = {"default_headers": headers}
    return kwargs


def _translate_s3(options: Dict[str, Any]) -> Dict[str, Any]:
    """Map s3fs storage_options onto S3Store keyword arguments."""
    kwargs: Dict[str, Any] = {}

    for key, value in options.items():
        if key == "anon":
            if value:
                kwargs["skip_signature"] = True
        elif key == "key":
            kwargs["access_key_id"] = value
        elif key == "secret":
            kwargs["secret_access_key"] = value
        elif key == "token":
            kwargs["token"] = value
        elif key == "requester_pays":
            if value:
                kwargs["request_payer"] = True
        elif key == "endpoint_url":
            kwargs["endpoint"] = value
        elif key == "region_name":
            kwargs["region"] = value
        elif key == "client_kwargs":
            if not isinstance(value, dict):
                raise UnsupportedStorageOption("client_kwargs must be a dict")
            for sub_key, sub_value in value.items():
                if sub_key == "region_name":
                    kwargs["region"] = sub_value
                elif sub_key == "endpoint_url":
                    kwargs["endpoint"] = sub_value
                else:
                    raise UnsupportedStorageOption(
                        f"client_kwargs[{sub_key!r}] is not supported by the obstore transport"
                    )
        else:
            raise UnsupportedStorageOption(f"{key!r} is not supported by the obstore transport")

    return kwargs


_TRANSLATORS = {"gs": _translate_gcs, "gcs": _translate_gcs, "s3": _translate_s3}


def translate_storage_options(scheme: str, storage_options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return obstore store keyword arguments for fsspec-style options.

    Raises :class:`UnsupportedStorageOption` when an option has no obstore
    equivalent, so a caller that merely prefers obstore can fall back to fsspec
    instead of reading with the wrong credentials.
    """
    translate = _TRANSLATORS.get(scheme)
    if translate is None:
        raise UnsupportedStorageOption(f"scheme {scheme!r} has no obstore store")
    return translate(dict(storage_options or {}))


def can_handle(scheme: str, storage_options: Optional[Dict[str, Any]]) -> bool:
    """Return True if obstore is installed and can serve these options."""
    if not is_available():
        return False
    try:
        translate_storage_options(scheme, storage_options)
    except UnsupportedStorageOption:
        return False
    return True


_store_cache: Dict[str, Any] = {}
_store_cache_lock = threading.Lock()


def _store_cache_key(scheme: str, bucket: str, kwargs: Dict[str, Any]) -> str:
    """Identity of a store: its scheme, bucket and every setting that shapes it.

    The settings are folded in as a digest rather than kept verbatim, so a cache
    that lives for the life of the process does not also hold a second copy of a
    credential.
    """
    material = json.dumps(kwargs, sort_keys=True, default=str)
    return f"{scheme}\x00{bucket}\x00{hashlib.sha256(material.encode('utf-8')).hexdigest()}"


def clear_store_cache() -> None:
    """Drop the cached stores. Mainly useful in tests."""
    with _store_cache_lock:
        _store_cache.clear()


def _build_store(scheme: str, bucket: str, kwargs: Dict[str, Any]) -> Any:
    if scheme in ("gs", "gcs"):
        from obstore.store import GCSStore

        return GCSStore(bucket, **kwargs)
    if scheme == "s3":
        from obstore.store import S3Store

        return S3Store(bucket, **kwargs)
    raise UnsupportedStorageOption(f"scheme {scheme!r} has no obstore store")


class ObstoreFileSystem:
    """Filesystem-shaped adapter over obstore, keyed by URL scheme and bucket.

    Only the operations the BGEN reader performs are implemented: size lookup,
    batched byte-range reads, and a sequential handle for the one-time header
    parse. Every method takes a full ``scheme://bucket/key`` URL, so one instance
    serves whatever paths it is handed.

    Constructed with fsspec-style ``storage_options`` (see
    :func:`translate_storage_options`).
    """

    def __init__(self, **storage_options: Any):
        self.storage_options = storage_options
        # Object sizes seen by this instance. The reader asks for the size at
        # open and again when it opens the sequential handle, and a remote HEAD
        # is a full round trip; the file cannot change under an open reader
        # anyway, so one lookup serves both.
        self._sizes: Dict[str, int] = {}
        self._sizes_lock = threading.Lock()

    def _store(self, url: str) -> Tuple[Any, str]:
        """Return (store, key) for a URL, reusing a store built earlier.

        Stores are cached for the process, not per instance: a store holds the
        access token it fetched, so building a fresh one per reader pays for
        another credential round trip on every open. fsspec caches its filesystem
        instances the same way, which keeps the two transports comparable.
        """
        scheme, bucket, key = split_url(url)
        kwargs = translate_storage_options(scheme, self.storage_options)
        cache_key = _store_cache_key(scheme, bucket, kwargs)
        with _store_cache_lock:
            store = _store_cache.get(cache_key)
            if store is None:
                store = _build_store(scheme, bucket, kwargs)
                _store_cache[cache_key] = store
        return store, key

    # --- filesystem surface used by the reader --------------------------------

    def info(self, url: str) -> Dict[str, Any]:
        """Return object metadata, including ``size`` in bytes."""
        return {"name": url, "size": self._size(url), "type": "file"}

    def _size(self, url: str) -> int:
        """Object size in bytes, fetched once per URL per instance."""
        import obstore

        with self._sizes_lock:
            size = self._sizes.get(url)
        if size is None:
            store, key = self._store(url)
            size = int(obstore.head(store, key)["size"])
            with self._sizes_lock:
                self._sizes[url] = size
        return size

    def cat_ranges(
        self,
        paths: Sequence[str],
        starts: Sequence[int],
        ends: Sequence[int],
    ) -> List[Any]:
        """Fetch byte ranges, returning one buffer per range in input order.

        The buffers implement the buffer protocol (obstore hands back a view of
        the Rust allocation), so the caller copies out of them without a
        round trip through ``bytes``.

        Ranges are split across a thread pool because obstore releases the GIL
        for the whole request, so several calls overlap in Rust. Splitting is
        skipped for a batch too small to benefit.
        """
        import obstore

        count = len(paths)
        if not (count == len(starts) == len(ends)):
            raise ValueError("cat_ranges needs paths, starts and ends of equal length")
        if count == 0:
            return []

        # Group by path so a batch spanning several objects still works; the
        # reader only ever asks for one, which keeps this to a single group.
        by_path: Dict[str, List[int]] = {}
        for index, path in enumerate(paths):
            by_path.setdefault(path, []).append(index)

        results: List[Any] = [None] * count

        def fetch(path: str, indices: Sequence[int]) -> None:
            store, key = self._store(path)
            fetched = obstore.get_ranges(
                store,
                key,
                starts=[starts[i] for i in indices],
                ends=[ends[i] for i in indices],
                # The caller has already merged what is worth merging; more
                # merging here would fetch bytes nobody asked for.
                coalesce=0,
            )
            for index, buffer in zip(indices, fetched):
                results[index] = buffer

        jobs: List[Tuple[str, Sequence[int]]] = []
        workers = _thread_count()
        for path, indices in by_path.items():
            if len(indices) < _MIN_RANGES_TO_SPLIT or workers == 1:
                jobs.append((path, indices))
                continue
            # Chunks stay in file order so each request covers a contiguous
            # stretch of the object.
            step = (len(indices) + workers - 1) // workers
            for start in range(0, len(indices), step):
                jobs.append((path, indices[start : start + step]))

        if len(jobs) == 1:
            fetch(*jobs[0])
        else:
            pool = _shared_pool()
            # list() forces every future, so an exception in any chunk propagates.
            list(pool.map(lambda job: fetch(*job), jobs))

        return results

    def open(self, url: str, mode: str = "rb", block_size: Optional[int] = None, **_kwargs: Any) -> "ObstoreFile":
        """Open a read-only sequential handle. Only ``"rb"`` is supported."""
        if mode not in ("rb", "br"):
            raise ValueError(f"ObstoreFileSystem.open supports mode 'rb', not {mode!r}")
        store, key = self._store(url)
        return ObstoreFile(store, key, self._size(url), block_size or 1024 * 1024)

    def download(self, url: str, dest_path: str) -> None:
        """Write a whole object to a local path."""
        import obstore

        store, key = self._store(url)
        response = obstore.get(store, key)
        with open(dest_path, "wb") as handle:
            for chunk in response.stream():
                handle.write(chunk)


class ObstoreFile:
    """Minimal seekable read handle over one object.

    Backs the reader's one-time header and sample-ID parse, which is sequential;
    genotype I/O goes through :meth:`ObstoreFileSystem.cat_ranges` instead. A read
    that misses the resident window fetches what it needs plus ``block_size``
    beyond it, so a sequential walk costs one request per block rather than one
    per read.
    """

    def __init__(self, store: Any, key: str, size: int, block_size: int):
        self._store = store
        self._key = key
        self._size = size
        self._block_size = max(1, block_size)
        self._pos = 0
        self._buffer = b""
        self._buffer_start = 0
        self.closed = False

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            target = offset
        elif whence == 1:
            target = self._pos + offset
        elif whence == 2:
            target = self._size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        self._pos = max(0, min(int(target), self._size))
        return self._pos

    def tell(self) -> int:
        return self._pos

    def read(self, size: int = -1) -> bytes:
        import obstore

        if self.closed:
            raise ValueError("read on a closed ObstoreFile")
        remaining = self._size - self._pos
        if size is None or size < 0:
            size = remaining
        size = min(size, remaining)
        if size <= 0:
            return b""

        buffer_end = self._buffer_start + len(self._buffer)
        if not (self._buffer_start <= self._pos and self._pos + size <= buffer_end):
            # Refill from the current position, taking a block beyond what was
            # asked for so a run of sequential reads costs one request rather
            # than one each. Reading exactly what was asked for doubles the
            # round trips on a large header, since the caller reads in
            # block-sized pieces and every piece would miss.
            start = self._pos
            end = min(self._size, start + size + self._block_size)
            self._buffer = bytes(obstore.get_range(self._store, self._key, start=start, end=end))
            self._buffer_start = start

        offset = self._pos - self._buffer_start
        data = self._buffer[offset : offset + size]
        self._pos += len(data)
        return data

    def close(self) -> None:
        self.closed = True
        self._buffer = b""

    def __enter__(self) -> "ObstoreFile":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
