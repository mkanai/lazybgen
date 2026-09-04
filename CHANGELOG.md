# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

A performance release, plus a faster default transport for `gs://` reads. The API
is unchanged and every read returns byte-identical dosages to 0.1.0.

### Removed

- **Python 3.9 is no longer supported**; the floor is now 3.10. This is what
  lets obstore, whose current releases require 3.10, be a plain dependency.

### Added

- **obstore is the default transport for `gs://` reads.** It runs
  HTTP and TLS in Rust with the GIL released, so many byte ranges are genuinely
  in flight at once, where fsspec drives every request in a process through a
  single asyncio event loop that pins one CPU core. The fetch itself is 2.5x
  (contiguous) to 3.8x (scattered) faster than gcsfs, and a full read is
  1.1x to 3.5x depending on the read: most for small latency-bound ones, least
  for a full decode, where the remaining time is decode and output allocation. It
  is installed as a dependency and needs no code change.

  `s3://` keeps using s3fs. obstore's S3 store does not resolve a bucket's region
  and its credential chain skips `~/.aws/credentials`, `AWS_PROFILE` and SSO, so
  making it the S3 default would break working code; `remote_backend="obstore"`
  selects it explicitly for anyone who wants it.

  **Multiprocessing note**: obstore's runtime does not survive `fork()`. A process
  that reads and then forks (the Linux default start method) leaves its children
  unable to use obstore, so they fall back to fsspec automatically. Use the
  `spawn` or `forkserver` start method to keep the faster transport in workers.
- **`remote_backend`** on `load_bgen()` and `BgenReader`: `"auto"` (default),
  `"obstore"` or `"fsspec"`, also settable per process with
  `LAZYBGEN_REMOTE_BACKEND`. `"auto"` falls back to fsspec when obstore is
  unavailable or cannot express an entry in `storage_options`, so an option that
  decides which bytes come back is never silently dropped. `storage_options` keep
  their fsspec spelling on either transport, and requester-pays works on both.
- **Zero-copy local reads**: a local BGEN is memory-mapped, so a block decode
  reads each variant record in place instead of copying it. This removes the
  serial bottleneck that previously capped multi-threaded scaling.

### Changed

- **libdeflate replaces zlib-ng** for DEFLATE decode (1.4-1.7x on inflate).
  `get_build_info()` now reports the backend under the keys `type`, `deflate`,
  `zstd` and `note`. Nothing shells out to CMake on Unix any more; it is needed
  only for the Windows zstd build.
- **Decode workers claim runs of variants** rather than one at a time, which
  fixes page-fault serialization on a freshly allocated output matrix.
- **Sample IDs are materialized on first access**, so a read that never asks for
  them does not pay to build hundreds of thousands of Python strings.
- **Remote block reads are batched and coalesced**: a block's records go out in
  one call, and records less than 64 KB apart are fetched as one request.
- Opening a file no longer scales with the index's total variant count, and a
  sample-filtered read indexes only the samples it was asked for.

## [0.1.0] - 2026-06-24

Initial release. lazybgen is a high-performance BGEN reader with random-access,
region- and variant-filtered loading from local files, Google Cloud Storage
(`gs://`), and Amazon S3 (`s3://`), fetching only the data you ask for via
byte-range reads. It was extracted from
[ldcov](https://github.com/mkanai/ldcov) into a standalone, reusable package.

### Highlights

- `load_bgen()`: read alt-allele dosages into `(ndarray, DataFrame, sample_ids)`,
  with optional region (`chr:start-end`), variant filter, and sample subsetting.
- `BgenReader`: lower-level reader with context-manager support, plus
  `iter_variants()` for memory-bounded streaming of arbitrarily large files.
- **Cloud partial reads**: stream only the requested variants/regions directly
  from GCS and S3 over random-access byte ranges, with no full-file download and a
  `storage_options` passthrough (anonymous, requester-pays, or custom project).
- **Parallel decode by default**: blocks are inflated and decoded across
  auto-detected CPU cores, several times faster on multi-core machines and
  byte-identical to single-threaded decoding; opt out with `num_threads=1`.
- Format coverage: BGEN layout v1.2 / v1.3 (best-effort v1.1), 8/16/32-bit, zlib
  and zstd compression, for biallelic diploid (phased or unphased) variants.
  Multi-allelic, non-diploid, and uncompressed inputs raise a clear error rather
  than returning wrong dosages.
- Vendored, statically linked zlib-ng and zstd for consistent cross-platform
  decompression; ships a `py.typed` marker and prebuilt wheels
  (manylinux/musllinux x86_64, macOS arm64). `get_build_info()` reports the
  compression backend the package was built against.

[0.2.0]: https://github.com/mkanai/lazybgen/releases/tag/v0.2.0
[0.1.0]: https://github.com/mkanai/lazybgen/releases/tag/v0.1.0
