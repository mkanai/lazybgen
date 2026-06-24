# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.0]: https://github.com/mkanai/lazybgen/releases/tag/v0.1.0
