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
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

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

# obstore drives its requests on a process-global tokio runtime that does NOT
# survive fork(): a child that inherits a runtime the parent has already used
# hangs on its first request instead of failing, with no error to diagnose it by.
# _did_io records whether this process ever reached obstore, and the at-fork hook
# turns that into "obstore is unusable in this child" so a read can fall back to
# fsspec rather than deadlock. Clearing our own caches is not enough; the runtime
# itself is what does not survive.
_did_io = False
_broken_by_fork = False


def _note_io() -> None:
    """Record that this process has driven at least one obstore request."""
    global _did_io
    _did_io = True


def _reset_after_fork() -> None:
    """Drop everything a forked child cannot inherit safely.

    The pool's worker threads do not exist in the child, and a cached store holds
    an access token the parent fetched. Neither lock is taken here: a lock held by
    another thread at fork time is inherited locked, so taking one would deadlock.
    Both are replaced outright instead.
    """
    global _pool, _pool_lock, _store_cache_lock, _broken_by_fork
    _pool = None
    _pool_lock = threading.Lock()
    _store_cache_lock = threading.Lock()
    _store_cache.clear()
    if _did_io:
        _broken_by_fork = True


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


def fork_broken() -> bool:
    """True if obstore cannot be used here because this process was forked.

    The parent had already driven a request, so the tokio runtime this child
    inherited is dead and any obstore call would hang.
    """
    return _broken_by_fork


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
    """Return True if obstore can be imported AND used in this process."""
    if _broken_by_fork:
        return False
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


def _load_json_file(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError) as exc:
        raise UnsupportedStorageOption(f"could not read the credentials at {path!r}: {exc}") from exc


def _require_service_account(payload: Any) -> None:
    """Reject credentials obstore's GCS store cannot use.

    gcsfs also accepts an ``authorized_user`` (gcloud user ADC) document, which
    obstore's ``service_account`` / ``service_account_key`` settings cannot load.
    Rejecting it sends the read back to fsspec rather than failing at open.
    """
    if not isinstance(payload, dict) or payload.get("type") != "service_account":
        kind = payload.get("type") if isinstance(payload, dict) else type(payload).__name__
        raise UnsupportedStorageOption(
            f"the obstore transport needs a service-account key; got credentials of type {kind!r}"
        )


# gcsfs resolves credentials through google_default -> cache -> cloud -> anon, so
# on a machine with NO GCP credentials it still reads a public bucket. obstore has
# no anonymous step: it fails the token request instead. These are the fragments
# of that failure, used to retry once unsigned and restore the 0.1.0 behavior for
# the only case where the two transports differ on credentials.
# Deliberately narrow. A broader match (a bare "credential", say) also catches a
# REJECTED credential - an expired or malformed ADC file - and retrying that
# unsigned would quietly read a public object as anonymous instead of reporting
# that the caller's credentials are broken.
_MISSING_CREDENTIAL_HINTS = ("computemetadata", "no credentials", "not find default credentials")

# Options that leave the credential choice entirely to the transport. Anything
# else means the caller named a credential, and a silent anonymous retry would
# not be what they asked for.
_CREDENTIAL_FREE_OPTIONS = frozenset({"token"})
_EXPLICIT_CREDENTIAL_KWARGS = ("service_account", "service_account_key", "skip_signature", "bearer_token")


def _looks_like_missing_credentials(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(hint in text for hint in _MISSING_CREDENTIAL_HINTS)


# Credential documents object_store's GCS store can load. google.auth accepts
# more: "external_account" (Workload Identity Federation, e.g. GitHub Actions
# OIDC) and "impersonated_service_account" both work on gcsfs and are a hard
# error on obstore.
_OBSTORE_ADC_TYPES = frozenset({"service_account", "authorized_user"})


def _default_credentials_path() -> Optional[Tuple[str, bool]]:
    """Where google.auth would find an ADC file, and whether obstore looks there.

    Returns (path, obstore_looks_here) for the first candidate that exists, or
    None when there is no ADC file at all (in which case both transports fall
    through to the metadata server and agree).
    """
    explicit = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if explicit:
        # object_store reads this variable too, so both transports load the same
        # file and only its type can differ between them.
        return (explicit, True) if os.path.exists(explicit) else None

    # gcloud's config dir can be relocated. google.auth honors CLOUDSDK_CONFIG;
    # object_store does not, and silently falls through to the metadata server,
    # which on a GCE VM means reading as a DIFFERENT PRINCIPAL rather than
    # failing. That is worse than an error, so it is refused below.
    sdk_config = os.environ.get("CLOUDSDK_CONFIG")
    if sdk_config:
        path = os.path.join(sdk_config, "application_default_credentials.json")
        if os.path.exists(path):
            return path, False

    if os.name == "nt":  # pragma: no cover - exercised only on Windows
        appdata = os.environ.get("APPDATA")
        home_path = os.path.join(appdata, "gcloud", "application_default_credentials.json") if appdata else None
    else:
        home_path = os.path.join(os.path.expanduser("~"), ".config", "gcloud", "application_default_credentials.json")
    if home_path and os.path.exists(home_path):
        return home_path, True
    return None


def _require_adc_obstore_can_use() -> None:
    """Refuse the default credential chain when the two transports would differ.

    Only called when the caller named no credential of their own, i.e. when both
    transports would resolve one themselves. Raising sends the read to fsspec,
    which keeps 0.1.0 behavior, instead of failing at open or reading as another
    principal.
    """
    found = _default_credentials_path()
    if found is None:
        return
    path, obstore_looks_here = found
    if not obstore_looks_here:
        raise UnsupportedStorageOption(
            f"the credentials at {path!r} are found through CLOUDSDK_CONFIG, which the "
            "obstore transport does not consult; it would read as a different principal"
        )
    try:
        with open(path, "r", encoding="utf-8") as handle:
            kind = json.load(handle).get("type")
    except (OSError, ValueError):
        # Unreadable or not JSON: let the transport report it rather than
        # guessing, since fsspec would fail on it too.
        return
    if kind not in _OBSTORE_ADC_TYPES:
        raise UnsupportedStorageOption(
            f"the credentials at {path!r} are of type {kind!r}, which the obstore " "transport cannot load"
        )


def _translate_gcs(options: Dict[str, Any]) -> Dict[str, Any]:
    """Map gcsfs storage_options onto GCSStore keyword arguments."""
    kwargs: Dict[str, Any] = {}
    headers: Dict[str, str] = {}

    for key, value in options.items():
        if key == "anon":
            # gcsfs has no `anon` parameter: it lands in **kwargs and is silently
            # discarded, so a gs:// read with anon=True is AUTHENTICATED there.
            # Mapping it to skip_signature here would quietly send a different
            # credential than 0.1.0 did, so it is treated as untranslatable and
            # the read falls back to fsspec. token="anon" is the spelling gcsfs
            # actually honors, and it is handled below.
            if value:
                raise UnsupportedStorageOption(
                    "anon is not honored for gs:// by gcsfs, so the obstore transport "
                    'refuses it rather than reading anonymously; use token="anon" to '
                    "read a public bucket without credentials"
                )
        elif key == "token":
            if value in (None, "google_default"):
                # The default credential chain, which is what obstore already does.
                # "cloud" and "default" are NOT equivalent and are refused below:
                # gcsfs's "cloud" is the metadata server ONLY, where obstore's
                # default chain prefers an ADC file, so the two can read as
                # different principals; and "default" is not a gcsfs method at
                # all, so gcsfs signs with the literal string and fails. Making
                # it work here would hide a bad option instead of surfacing it.
                continue
            if value == "anon":
                kwargs["skip_signature"] = True
            elif isinstance(value, str) and os.path.exists(value):
                _require_service_account(_load_json_file(value))
                kwargs["service_account"] = value
            elif isinstance(value, dict):
                _require_service_account(value)
                kwargs["service_account_key"] = json.dumps(value)
            else:
                raise UnsupportedStorageOption(f"token={value!r} is not supported by the obstore transport")
        elif key == "requester_pays":
            # Consumed below, together with any "project" that names the billing
            # project for it.
            continue
        elif key == "project":
            if not options.get("requester_pays"):
                # Outside requester-pays, gcsfs uses `project` as an assertion:
                # its google_default path raises when it does not match the ADC's
                # own project. obstore has no equivalent, so honoring the option
                # here would mean dropping a check the caller asked for and
                # turning their error into a silent success.
                raise UnsupportedStorageOption(
                    "project is only supported by the obstore transport alongside "
                    "requester_pays; on its own it asserts the credential's project, "
                    "which obstore cannot check"
                )
            continue
        else:
            raise UnsupportedStorageOption(f"{key!r} is not supported by the obstore transport")

    if not any(name in kwargs for name in _EXPLICIT_CREDENTIAL_KWARGS):
        # No credential was named, so obstore would resolve the default chain,
        # and that chain is narrower than google.auth's.
        _require_adc_obstore_can_use()

    if options.get("requester_pays"):
        # GCS requester-pays has no dedicated obstore setting. The XML API reads
        # the billing project from this header, which object_store sends on every
        # request when it is given as a default header.
        headers["x-goog-user-project"] = _billing_project(options)

    if headers:
        kwargs["client_options"] = {"default_headers": headers}
    return kwargs


def _conflicts(options: Dict[str, Any], sub_key: str) -> bool:
    """True if a client_kwargs entry is also set at the top level."""
    top_level = {"endpoint_url": "endpoint_url", "region_name": "region_name"}[sub_key]
    return top_level in options


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
        elif key == "region_name":  # noqa: SIM114 - kept distinct from client_kwargs below
            kwargs["region"] = value
        elif key == "client_kwargs":
            if value is None:
                continue  # s3fs treats a missing client_kwargs and None alike
            if not isinstance(value, dict):
                raise UnsupportedStorageOption("client_kwargs must be a dict")
            for sub_key, sub_value in value.items():
                if sub_key in ("endpoint_url", "region_name") and _conflicts(options, sub_key):
                    # s3fs raises on this pair rather than picking one, so
                    # resolving it by dict order here would invent a behavior.
                    raise UnsupportedStorageOption(f"{sub_key!r} is given both at the top level and in client_kwargs")
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

    endpoint = kwargs.get("endpoint")
    if isinstance(endpoint, str) and endpoint.startswith("http://"):
        # object_store refuses plain HTTP unless told otherwise, where s3fs allows
        # it. Local S3 stand-ins (MinIO, localstack) are the reason anyone sets a
        # plaintext endpoint at all.
        kwargs["client_options"] = {"allow_http": True}

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
        # Translated options, cached per scheme by _store(). The cache is the
        # point: _store runs on every fan-out chunk, and for requester_pays=True
        # translation can reach google.auth.default(), which off GCE shells out
        # to gcloud once per call. The single-threaded size lookup or download
        # that opens a file always populates this before any fan-out, so the
        # gcloud call happens once and never inside a worker thread.
        self._store_kwargs: Dict[str, Dict[str, Any]] = {}
        # Object sizes seen by this instance. The reader asks for the size at
        # open and again when it opens the sequential handle, and a remote HEAD
        # is a full round trip; the file cannot change under an open reader
        # anyway, so one lookup serves both.
        self._sizes: Dict[str, int] = {}
        self._sizes_lock = threading.Lock()
        # At most one anonymous retry per filesystem (see _call_with_anon_retry),
        # so a fan-out cannot turn one missing credential into N retries.
        self._anon_retry_lock = threading.Lock()
        self._anon_retry_done = False

    def _store(self, url: str) -> Tuple[Any, str]:
        """Return (store, key) for a URL, reusing a store built earlier.

        Stores are cached for the process, not per instance: a store holds the
        access token it fetched, so building a fresh one per reader pays for
        another credential round trip on every open. fsspec caches its filesystem
        instances the same way, which keeps the two transports comparable.
        """
        scheme, bucket, key = split_url(url)
        kwargs = self._store_kwargs.get(scheme)
        if kwargs is None:
            kwargs = translate_storage_options(scheme, self.storage_options)
            self._store_kwargs[scheme] = kwargs
        cache_key = _store_cache_key(scheme, bucket, kwargs)
        with _store_cache_lock:
            store = _store_cache.get(cache_key)
            if store is None:
                store = _build_store(scheme, bucket, kwargs)
                _store_cache[cache_key] = store
        return store, key

    def _may_retry_anonymously(self, scheme: str) -> bool:
        """True if an unsigned retry is the right response to a credential failure.

        Only when the caller named no credential of their own and is not paying
        for the read: a requester-pays fetch is billed to a project and can never
        be anonymous.
        """
        if scheme not in ("gs", "gcs"):
            return False
        if self.storage_options.get("requester_pays"):
            return False
        if not set(self.storage_options) <= _CREDENTIAL_FREE_OPTIONS:
            return False
        if self.storage_options.get("token") not in (None, "google_default"):
            return False
        kwargs = self._store_kwargs.get(scheme, {})
        return not any(name in kwargs for name in _EXPLICIT_CREDENTIAL_KWARGS)

    def _call_with_anon_retry(self, url: str, operation):
        """Run one obstore call, falling back to an unsigned read once.

        A public bucket read from a machine with no GCP credentials succeeds on
        gcsfs, whose credential ladder ends at anonymous access. Without this it
        would fail on obstore, which is a regression for the most basic remote
        call there is.
        """
        try:
            return operation()
        except Exception as exc:
            scheme = split_url(url)[0]
            with self._anon_retry_lock:
                if (
                    self._anon_retry_done
                    or not self._may_retry_anonymously(scheme)
                    or not _looks_like_missing_credentials(exc)
                ):
                    raise
                self._anon_retry_done = True
                kwargs = dict(self._store_kwargs.get(scheme, {}))
                kwargs["skip_signature"] = True
                self._store_kwargs[scheme] = kwargs
                logger.warning(
                    "No GCP credentials found; reading %s anonymously. Set credentials, "
                    'or pass storage_options={"token": "anon"} to make this explicit.',
                    url,
                )
            return operation()

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

            def head() -> int:
                store, key = self._store(url)
                _note_io()
                return int(obstore.head(store, key)["size"])

            # The reader's first obstore call for a file, so this is where a
            # missing credential surfaces and where the retry belongs: it runs on
            # one thread, before any fan-out.
            size = self._call_with_anon_retry(url, head)
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

        _note_io()

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

        def fetch() -> None:
            store, key = self._store(url)
            _note_io()
            response = obstore.get(store, key)
            with open(dest_path, "wb") as handle:
                for chunk in response.stream():
                    handle.write(chunk)

        # Downloading a remote .bgi is the first obstore call of a whole read, so
        # it needs the same anonymous fallback as the size lookup.
        self._call_with_anon_retry(url, fetch)


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
            _note_io()
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
