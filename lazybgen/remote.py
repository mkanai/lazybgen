"""Utilities for BGEN file handling."""

import hashlib
import json
import logging
import os
import tempfile
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# All storage_options participate in the remote .bgi cache identity, because any
# of them (credential scope, billing project, anon-vs-authed, custom endpoint)
# can change WHICH bytes a fetch returns. To avoid under-invalidating, every
# option value is discriminated on. Two value-handling policies keep secrets out
# of the (non-secret) cache-key material:
#   - cache-relevant, plainly-non-secret keys are folded in by value (readable);
#   - all other keys, including secret-looking ones (token/key/secret/...), are
#     folded in by a short sha256 digest of their value, so a changed credential
#     scope or endpoint still busts the cache WITHOUT the raw value ever appearing
#     verbatim in the discriminator.
_CACHE_RELEVANT_OPTION_KEYS = ("anon", "requester_pays", "project")
_SECRET_OPTION_KEY_HINTS = ("token", "key", "secret", "credential", "password", "cred")

# Single source of truth for "what counts as a remote (cloud) path".
REMOTE_SCHEMES = ("gs://", "s3://")

# fsspec registers gcsfs under "gs"/"gcs" and s3fs under "s3"; map scheme -> pip pkg.
_PACKAGE_FOR_SCHEME = {"gs": "gcsfs", "s3": "s3fs"}

# Transports that can serve a remote read. "obstore" runs HTTP and TLS in Rust
# with the GIL released, which lets one process overlap several range requests
# instead of funnelling them through fsspec's single event loop; "fsspec" is
# gcsfs / s3fs. "auto" prefers obstore when it is importable and can express the
# caller's storage_options, and falls back to fsspec otherwise. obstore ships as
# a dependency, but the fallback stays supported: a user can deselect it, and
# some storage_options have no obstore equivalent.
REMOTE_BACKENDS = ("auto", "fsspec", "obstore")
_DEFAULT_REMOTE_BACKEND = "auto"


def is_remote_path(path: str) -> bool:
    """Return True if ``path`` is a cloud URL handled by a remote backend."""
    return path.startswith(REMOTE_SCHEMES)


def resolve_remote_backend(path: str, storage_options: Optional[Dict] = None, backend: Optional[str] = None) -> str:
    """Pick the transport for a remote path: ``"fsspec"`` or ``"obstore"``.

    ``backend`` is the caller's preference (``"auto"``, ``"fsspec"`` or
    ``"obstore"``); when it is None the ``LAZYBGEN_REMOTE_BACKEND`` environment
    variable is consulted, then the default of ``"auto"``.

    ``"auto"`` chooses obstore only when it is installed AND every entry in
    ``storage_options`` has an obstore equivalent, so an option that would
    otherwise be silently dropped (and change which bytes are read) sends the
    read back to fsspec instead. Asking for ``"obstore"`` explicitly raises
    rather than falling back.
    """
    if backend is None:
        backend = os.environ.get("LAZYBGEN_REMOTE_BACKEND") or _DEFAULT_REMOTE_BACKEND
    backend = backend.lower()
    if backend not in REMOTE_BACKENDS:
        raise ValueError(f"remote_backend must be one of {REMOTE_BACKENDS}, got {backend!r}")
    if backend == "fsspec":
        return "fsspec"

    from . import obstore_backend

    scheme = path.split("://", 1)[0]
    if backend == "obstore":
        if not obstore_backend.is_available():
            raise ImportError(
                "remote_backend='obstore' requires the obstore package, which lazybgen "
                "normally installs: pip install obstore"
            )
        # Raises UnsupportedStorageOption naming the offending key.
        obstore_backend.translate_storage_options(scheme, storage_options)
        return "obstore"

    return "obstore" if obstore_backend.can_handle(scheme, storage_options) else "fsspec"


def choose_remote_block_size(nsamples: int, offsets) -> int:
    """Pick a remote readahead block size (bytes) for a variant selection.

    Random-access remote reads over-fetch by one readahead block per seek. A
    large block is desirable for a dense/contiguous selection (it coalesces
    neighboring variants into shared range GETs) but pure waste for a scattered
    one (each isolated seek drags a full block). Given the variants' file
    offsets, return a multi-variant block when they are packed and a one-variant
    block when they are scattered. The C++ reader clamps the value to a sane
    range, so this returns the raw estimate.

    Returns 0 when there is nothing to read (caller should leave the block
    unchanged). Cost is dominated by sorting the offsets once, negligible next
    to the per-variant network reads the block size governs.
    """
    n = len(offsets)
    if n == 0:
        return 0
    # Estimate one variant's on-disk record from the sample count: ~2 bytes per
    # sample for a biallelic 8-bit genotype payload, plus the per-variant
    # metadata read window.
    vsize = int(nsamples) * 2 + 128 * 1024
    if n == 1:
        return vsize

    # Classify density by the median gap between position-adjacent variants.
    ordered = sorted(offsets)
    gaps = sorted(ordered[i + 1] - ordered[i] for i in range(n - 1))
    med_gap = gaps[len(gaps) // 2]
    if med_gap <= vsize + vsize // 2:
        # Dense: variants are essentially adjacent; coalesce several per block.
        return vsize * 8
    # Scattered: fetch ~one variant per seek instead of a large block.
    return vsize


def _bgi_cache_dir() -> str:
    """Directory used to cache downloaded remote .bgi indexes.

    Honors the ``LAZYBGEN_BGI_CACHE_DIR`` environment variable; otherwise uses a
    dedicated subdirectory of the system temp dir. Never the current working
    directory.
    """
    override = os.environ.get("LAZYBGEN_BGI_CACHE_DIR")
    if override:
        return override
    return os.path.join(tempfile.gettempdir(), "lazybgen-bgi-cache")


def _storage_options_discriminator(storage_options: Optional[Dict]) -> str:
    """Stable, non-secret discriminator string derived from storage_options.

    The remote .bgi cache identity is (URL, this discriminator): an index fetched
    under one credential scope / billing project / anon-vs-authed / endpoint
    setting is never reused under another. The discriminator is built from a
    NORMALIZED, sorted view of the options so dict ordering does not matter and
    identical options always collapse to the same value, while DIFFERENT values
    always diverge (no under-invalidation).

    Raw secret material never appears verbatim in the result. Cache-relevant,
    plainly-non-secret keys (anon/requester_pays/project) are included by value
    for readability; every other key, including secret-looking ones
    (token/key/secret/credential/password/cred), is folded in as a short sha256
    digest of its value. So two different credentials (or endpoints) yield two
    different cache keys, but the credential itself is not embedded.
    """
    if not storage_options:
        return ""

    normalized = {}
    for key in sorted(storage_options):
        lowered = key.lower()
        if key in _CACHE_RELEVANT_OPTION_KEYS and not any(hint in lowered for hint in _SECRET_OPTION_KEY_HINTS):
            # Plainly-non-secret, cache-relevant option: include by value.
            normalized[key] = storage_options[key]
        else:
            # Everything else (secret-looking keys and any other option that can
            # affect which bytes are read, e.g. an S3 endpoint or profile): fold in
            # a short digest of the value. This discriminates on a changed value
            # without ever embedding the raw value (which could be a credential).
            raw = json.dumps(storage_options[key], sort_keys=True, default=str)
            normalized[key] = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    return json.dumps(normalized, sort_keys=True, default=str)


def _local_bgi_cache_path(bgi_path: str, storage_options: Optional[Dict] = None) -> str:
    """Local cache path for a remote .bgi, keyed by the full URL and storage_options.

    Keying by the full URL (not just the basename) avoids collisions between
    same-named indexes in different buckets and avoids silently reusing an
    unrelated local file that happens to share the basename. Folding in a
    non-secret discriminator of storage_options means an index fetched under one
    credential / billing project / anon-vs-authed setting is not reused under a
    different one. See ``_storage_options_discriminator`` for what is (and is not)
    hashed.
    """
    key = bgi_path + "\x00" + _storage_options_discriminator(storage_options)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return os.path.join(_bgi_cache_dir(), f"{digest}-{os.path.basename(bgi_path)}")


# Global BGI memory cache to avoid repeated downloads. Keyed by (remote URL,
# storage_options discriminator) so an index fetched under different credentials
# / billing project / anon-vs-authed is not reused; value is the local path.
_bgi_memory_cache: Dict[str, str] = {}


def ensure_local_bgi(bgi_path: str, storage_options: Optional[Dict] = None, backend: Optional[str] = None) -> str:
    """
    Ensure BGI file is available locally, downloading from a remote URL if needed.

    Downloads a remote (gs:// or s3://) index into a temp cache directory (NOT the
    current working directory; see ``_bgi_cache_dir``) and reuses it on later
    calls. The cache identity is (remote URL, a non-secret discriminator of
    ``storage_options``), so an index fetched under one credential / billing
    project / anon-vs-authed setting is not reused under a different one. The
    download is atomic: bytes land in a unique temp file in the cache directory
    and are then ``os.replace``-d into place, so a concurrent reader never sees a
    partial file and a failed download is not cached.

    Parameters
    ----------
    bgi_path : str
        Path to BGI file (local or gs:// / s3://)
    storage_options : dict, optional
        Extra keyword arguments forwarded to ``fsspec.filesystem`` (e.g.
        ``{"requester_pays": True}`` for GCS requester-pays buckets, or
        ``{"anon": True}`` for public buckets).
    backend : str, optional
        Transport preference, as for :func:`resolve_remote_backend`.

    Returns
    -------
    str
        Local path to BGI file
    """
    # If already local, return as-is
    if not is_remote_path(bgi_path):
        return bgi_path

    # The in-process cache key folds in the storage_options discriminator so an
    # index fetched under different credentials/billing is not reused.
    mem_key = bgi_path + "\x00" + _storage_options_discriminator(storage_options)

    # Check memory cache first
    if mem_key in _bgi_memory_cache:
        cached_path = _bgi_memory_cache[mem_key]
        # Verify the cached file still exists
        if os.path.exists(cached_path):
            logger.debug(f"Using BGI from memory cache: {cached_path}")
            return cached_path
        else:
            # File was removed, clear from cache
            del _bgi_memory_cache[mem_key]

    # Cache under a dedicated directory keyed by a hash of (full URL,
    # storage_options), so same-named indexes in different buckets do not collide,
    # an unrelated local file is never silently reused, and an index fetched under
    # different credentials is not reused.
    local_path = _local_bgi_cache_path(bgi_path, storage_options)
    cache_dir = os.path.dirname(local_path)
    os.makedirs(cache_dir, exist_ok=True)

    # Check if already cached locally
    if os.path.exists(local_path):
        logger.info(f"Using cached BGI file: {local_path}")
        # Add to memory cache
        _bgi_memory_cache[mem_key] = local_path
        return local_path

    scheme = bgi_path.split("://", 1)[0]
    pkg = _PACKAGE_FOR_SCHEME.get(scheme, f"{scheme}fs")

    max_retries = 3
    retry_delay = 1.0

    # The index is downloaded through the same transport the genotype reads will
    # use, so a caller that has only one of the two backends installed still works.
    use_obstore = resolve_remote_backend(bgi_path, storage_options, backend) == "obstore"

    for attempt in range(max_retries):
        try:
            if use_obstore:
                from .obstore_backend import ObstoreFileSystem

                download = ObstoreFileSystem(**(storage_options or {})).download
            else:
                import fsspec

                download = fsspec.filesystem(scheme, **(storage_options or {})).get

            logger.info(f"Downloading BGI index from {bgi_path} to {local_path}...")
            # Download to a unique temp file in the SAME directory, then atomically
            # rename into place. os.replace is atomic on a single filesystem, so a
            # concurrent process never observes a partially written .bgi and a
            # failed/truncated download is never cached at the final path.
            fd, tmp_path = tempfile.mkstemp(dir=cache_dir, prefix=".tmp-", suffix=".bgi")
            os.close(fd)
            try:
                download(bgi_path, tmp_path)
                os.replace(tmp_path, local_path)
            except BaseException:
                # Clean up the partial temp file on any failure.
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
                raise
            logger.info("BGI index downloaded successfully")
            _bgi_memory_cache[mem_key] = local_path
            return local_path
        except ImportError as e:
            raise ImportError(f"{pkg} is required for {scheme}:// support. Install with: pip install {pkg}") from e
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"BGI download attempt {attempt + 1} failed: {e}. Retrying...")
                import time

                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise RuntimeError(f"Failed to download BGI file from {bgi_path} after {max_retries} attempts: {e}")

    # Unreachable for max_retries >= 1 (the loop always returns or raises), but
    # guards the type contract and a misconfigured max_retries <= 0.
    raise RuntimeError(f"Failed to download BGI file from {bgi_path}")


def clear_bgi_cache() -> None:
    """Clear the BGI memory cache."""
    # The dict is mutated in place (cleared), never rebound, so no `global` needed.
    _bgi_memory_cache.clear()
    logger.debug("BGI memory cache cleared")


def get_bgi_cache_info() -> Dict[str, str]:
    """Get information about cached BGI files."""
    return _bgi_memory_cache.copy()
