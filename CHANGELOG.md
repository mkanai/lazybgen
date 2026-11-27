# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-11-27

### Added

- **`BgenReader.has_embedded_sample_ids` and `BgenReader.sample_id_source`.** The header flag (bit 31), and `"sample_file"`, `"embedded"` or `"placeholder"` for where `samples` comes from; neither materializes the IDs. Embedded IDs can themselves be named `sample_<i>`, so `samples` alone could not tell them from placeholders.

### Changed

- **Breaking: a `.sample` file must have one ID_2 per BGEN sample.** Neither the row count nor one-field rows were checked, so a mismatched file silently mislabeled samples. Such a file now raises `ValueError` at the first `samples` access, and so in `load_bgen`; reads that never touch `samples` do not check. Blank lines are still skipped.

## [0.3.0] - 2026-11-21

A performance and correctness release, and the first since 0.2.0 (0.2.1 was
never published; its fix is included here). On a 16 vCPU Cascade Lake host
reading 8-bit zlib files, full-decode and region reads are 1.1-1.2x faster
than 0.2.0 at 5k samples and 1.4x at 500k, and 23x faster than the `bgen`
package at 500k (see [benchmarks/README.md](benchmarks/README.md)). Every full-sample
read of a v1.2 (layout 2) genotype block returns bitwise the same dosages as
0.2.0. The outputs that do change are listed under **Fixed**: each was a bug,
and the new value is the one the BGEN specification defines or a full read
already returned.

### Removed

- **Windows is no longer a build target.** The 0.2.0 README said a Windows
  build needed CMake, but the reader uses POSIX `mmap` and could not be built
  there, and no Windows wheel was published. The Windows branches of the build
  and its conditional CMake build requirement are gone. Linux and macOS are the
  supported platforms.

### Fixed

- **Importing no longer crashes on x86-64 CPUs without AVX2** (Sandy Bridge,
  Ivy Bridge and older; #8). The extension was compiled with AVX2 enabled
  throughout, so `import lazybgen` died with SIGILL on those CPUs. It is now
  built for baseline x86-64, and the AVX2 kernels run only on CPUs that report
  AVX2 and FMA. An AVX2 CPU still runs AVX2 kernels, with the same results.
  `LAZYBGEN_DISABLE_SIMD=1` forces the baseline kernels, for testing.
- **Region coordinates above 2^31 - 1 were bound as negative numbers.** The
  index query took positions as 32-bit signed integers. `load_variants()` and
  `iter_variants()` given `region_chrom` but no `region_end` (or only a start)
  ask for positions up to 4294967295, so they silently read zero variants; they
  now read the rest of the chromosome. An explicit end at or above 2^31, and a
  `variant_filter` position there, were affected the same way.
- **Sample-filtered 8- and 16-bit reads match the full read exactly.** The SIMD
  lanes of the filtered decode rounded some dosages differently from the full
  decode, so a sample's value could differ by one float ULP (at most 1.2e-7)
  depending on its position in `sample_indices`. Those values now equal what a
  full read returns.
- **BGEN v1.1 dosages follow the specification.** v1.1 stores the three genotype
  probabilities as 16-bit integers that the spec divides by 32768. lazybgen
  divided by each sample's stored sum instead, which differs whenever the stored
  triplet does not sum to exactly 32768 (by up to 6.1e-5 in a quarter of the
  values of the bundled example file). Dosage is now (P(AB) + 2 P(BB)) / 32768,
  as the `bgen` package and qctool compute it. A layout-2 file whose blocks are
  decoded as v1.1 (the fallback for blocks of exactly 6 bytes per sample) changes
  the same way. Rounding can put a conforming file one unit past 2
  (65537/32768), which `load_bgen` accepts for every file; anything larger still
  raises. The check sees the values at the output dtype, so a `float16` read
  cannot tell them apart from 2. A sample-filtered v1.1 read is now bitwise the
  full read; it rounded differently before.
- **zlib blocks written with a window smaller than 32 KiB failed to decompress.**
  Only the four zlib headers of a 32 KiB window were recognized, and any other
  valid zlib stream was misread as raw deflate. Every valid dictionary-free zlib
  header (RFC 1950) is now accepted. Raw deflate, which some non-conforming
  writers produce, is still read, including a stream whose first bytes happen to
  look like a zlib header.
- **A file whose sample-ID block ends at end of file opens.** The block was read
  4 bytes too long, into the variant data that normally follows it. Open now
  also checks that the block ends before the variant data, holds a length for
  every ID, and is no longer than N IDs can be (8 + 65537 N bytes), in place of
  a fixed 100 MB size limit.
- **Forked workers no longer hang on `gs://` reads.** Since gcsfs 2026.10, the
  filesystem gcsfs exports looks up each bucket's type over gRPC whenever
  credentials are available, and in a child forked after the parent read
  remotely that lookup hangs or crashes. It broke the fsspec fallback that
  forked children rely on, so a `multiprocessing` pool using the `fork` start
  method could stall with no error on a GCE VM or any machine with
  application-default credentials. A forked child that reads `gs://` through
  fsspec (which every child of a parent that has read remotely does, since
  obstore cannot survive the fork) now uses gcsfs's core (HTTP-only) filesystem. A filesystem class you registered for
  `gs` yourself is still used for the index download (genotype reads have always
  gone through gcsfs). The child is recognized by a fork hook, so import lazybgen
  before forking workers. Tested with gcsfs 2026.10.1 and 2025.7.0.

### Changed

- **Faster decode.** Three changes together give the speedups above. The
  full-sample decode of unphased 8- and 16-bit data runs many samples per step
  (AVX2 on x86-64, NEON on arm64, elsewhere a scalar loop that looks 8-bit
  dosages up in a table) and computes the block's `last_dosage_stats` in the same
  pass instead of rereading the output. On x86-64 Linux, libdeflate is
  assembled so that no jump crosses a 32-byte boundary, the workaround for the
  Skylake-family JCC erratum: its inflate speed had depended on where the linker
  happened to place it. On a Cascade Lake host this cut multithreaded zlib full
  and region reads by about 11%, consistent with the erratum, while
  single-threaded reads were unchanged (under 1%); it was not measured on CPUs
  without the erratum. And
  embedded sample IDs are no longer turned into strings at open (below), which
  is most of the gain on single-variant reads. Unphased 16-bit full reads are
  1.1-1.5x faster than 0.2.0.
- **The default thread count follows the CPUs the job may use.** With
  `num_threads=0`, Linux now starts up to one worker per CPU in the decoding
  thread's affinity mask (read at each decode), capped by the tightest cgroup CPU
  quota on the process's cgroup or its ancestors (cgroup v1 or v2, located
  through `/proc/self/mountinfo`, read once per process and rounded up). A SLURM
  job, `taskset`, or a container limited to 4 CPUs gets 4 workers rather than one
  per core of the machine. Elsewhere the default is still the core count. An
  explicit `num_threads` is an upper bound, as before: a block never starts more
  workers than it has variants, and a single-variant read is serial.
- **Embedded sample IDs are parsed on first use.** Opening a file still checks
  the sample-ID block's shape and sample count, but reads and builds the ID
  strings only when `samples` is first read, so a reader given a `.sample` file
  never builds them. The IDs remain readable after `close()`: a local reader
  whose IDs were never read keeps the file mapped (without holding a file
  descriptor) until they are, or until it is garbage-collected, while a reader
  given a `.sample` file releases the file at `close()`. A malformed ID string
  now raises at that first read instead of at open.
- **Forked children cannot read zonal ("Rapid") GCS buckets**, which need gRPC.
  Standard buckets are unaffected, as are `spawn` and `forkserver` workers and
  single-process use. Most forked readers of a zonal bucket hung before this
  release anyway; the exception is a child whose parent had not yet read
  remotely, which could read one before.

## [0.2.0] - 2026-09-04

A performance release, plus a faster default transport for `gs://` reads. Every
read returns byte-identical dosages to 0.1.0, and the reading API is unchanged;
the one signature change is to `get_build_info()`, which now reports the DEFLATE
backend under a `deflate` key where 0.1.0 used `zlib`.

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
- `BgenReader.remote_backend` reports the transport a reader resolved to, and
  `BgenReader.last_dosage_stats` the `(min, max, has_nan)` summary a block decode
  produced, or `None` when no path computed one.
- **Zero-copy local reads**: a local BGEN is memory-mapped, so a block decode
  reads each variant record in place instead of copying it. This removes the
  serial bottleneck that previously capped multi-threaded scaling.

### Changed

- **libdeflate replaces zlib-ng** for DEFLATE decode (1.4-1.7x on inflate).
  `get_build_info()` now reports the backend under the keys `type`, `deflate`,
  `zstd` and `note`; code reading the old `zlib` key must be updated. Nothing shells out to CMake on Unix any more; it is needed
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

[0.4.0]: https://github.com/mkanai/lazybgen/releases/tag/v0.4.0
[0.3.0]: https://github.com/mkanai/lazybgen/releases/tag/v0.3.0
[0.2.0]: https://github.com/mkanai/lazybgen/releases/tag/v0.2.0
[0.1.0]: https://github.com/mkanai/lazybgen/releases/tag/v0.1.0
