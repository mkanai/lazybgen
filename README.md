# lazybgen

[![CI](https://github.com/mkanai/lazybgen/actions/workflows/ci.yml/badge.svg)](https://github.com/mkanai/lazybgen/actions/workflows/ci.yml)
[![Wheels](https://github.com/mkanai/lazybgen/actions/workflows/wheels.yml/badge.svg)](https://github.com/mkanai/lazybgen/actions/workflows/wheels.yml)
[![PyPI](https://img.shields.io/pypi/v/lazybgen.svg)](https://pypi.org/project/lazybgen/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

High-performance [BGEN](https://www.chg.ox.ac.uk/~gav/bgen_format/) reader with
**Google Cloud Storage and Amazon S3 partial-read support**. lazybgen reads only the variants
or regions you ask for, fetching them directly from local files, GCS, or S3 via
random-access byte-range reads, so there is no need to download the whole file.

It is a Cython/C++ implementation with vendored, optimized compression backends
(libdeflate and zstd) compiled from source for consistent cross-platform behavior
and speed (SIMD genotype parsing, parallel block decompression).

## Install

```bash
pip install lazybgen
```

For Amazon S3 support, install the `s3` extra:

```bash
pip install lazybgen[s3]
```

### From source

```bash
git clone --recursive https://github.com/mkanai/lazybgen.git
cd lazybgen
pip install .
```

Building from source requires a C/C++ compiler (CMake is needed on Windows only).
The vendored `libdeflate` and `zstd` are git submodules, so clone with `--recursive`
(or run `git submodule update --init --recursive`).

## Usage

```python
from lazybgen import load_bgen

# Local file
genotypes, variant_info, sample_ids = load_bgen(
    "chr1.bgen",
    region="chr1:1000000-2000000",   # partial read: only this region is fetched
)

# GCS (default credentials)
load_bgen("gs://bucket/file.bgen", index_path="gs://bucket/file.bgen.bgi")

# GCS requester-pays bucket. True bills the default project from your environment
load_bgen("gs://bucket/file.bgen", storage_options={"requester_pays": True})
# ...or pass a project id string to bill a specific project
load_bgen("gs://bucket/file.bgen", storage_options={"requester_pays": "my-billing-project"})

# Public S3 bucket (anonymous, no credentials)
load_bgen("s3://bucket/file.bgen", storage_options={"anon": True})
```

`load_bgen` returns `(genotypes, variant_info, sample_ids)`, where `genotypes`
is an `(n_samples, n_variants)` `np.ndarray`, `variant_info` is a `pd.DataFrame`
with columns `chrom, pos, rsid, ref, alt`, and `sample_ids` is a `list[str]`.

A `.bgi` index is required; create one with `bgenix -g file.bgen`.

### Parameters

- `file_path`: path, `gs://`, or `s3://` URL to the BGEN file
- `index_path`: `.bgi` index (defaults to `file_path + ".bgi"`)
- `sample_path`: optional `.sample` file
- `region`: `"chr:start-end"` to read a genomic interval
- `variant_filter`: variant subset as a dict; build it with
  `load_variant_filter("variants.z")`, which reads variant IDs/positions from a
  `.z` file (`from lazybgen import load_variant_filter`)
- `sample_ids`: subset of samples to load
- `dtype`: dosage dtype (default `float64`). `np.float32` decodes ~18% faster
  and uses half the memory; use it when single precision is sufficient (dosages
  are computed in single precision regardless, so `float64` output is the exact
  widening of the `float32` result)
- `show_progress`: show a progress bar while loading (default `False`)
- `nan_action`: how to handle missing dosages: `"error"` (default, raise),
  `"mean"` (impute with the per-variant mean), `"omit"` (drop affected samples),
  or `"warn"` (keep NaNs and log a warning)
- `num_threads`: worker threads for decoding. `0` (default) auto-detects the CPU
  core count and decodes blocks in parallel; `1` forces single-threaded decoding;
  `N > 1` uses N threads (see [Parallel decode](#parallel-decode))
- `storage_options`: backend kwargs forwarded to the fsspec filesystem (e.g. `{"anon": True}` for public S3, `{"requester_pays": True}` to bill the env default project, or `{"requester_pays": "billing-project-id"}` for GCS requester-pays buckets)

### Supported BGEN features

lazybgen computes alt-allele dosages and targets the most common BGEN profile:

| Feature | Support |
| --- | --- |
| Layout v1.2 / v1.3 | Yes |
| Layout v1.1 | Best-effort |
| Compression: zlib, zstd | Yes |
| Compression: none (uncompressed) | No |
| Biallelic, diploid (phased or unphased) | Yes |
| Multi-allelic (>2 alleles) | No |
| Non-diploid (ploidy != 2) | No |

Unsupported inputs raise a clear error rather than returning wrong dosages.
Compress uncompressed files with `bgenix` or `qctool2` first.

Layout v1.1 is an older format with a different probability encoding; lazybgen
decodes it through a separate, less-exercised path, so it is best-effort. Prefer
v1.2 / v1.3 (re-encode with `qctool2` if needed) for production use.

### Build info

`from lazybgen import get_build_info` returns the compression backend the package
was built against (vendored libdeflate / zstd, or system libraries).

### Remote `.bgi` index caching

For a `gs://`/`s3://` BGEN, the genotype data is read in place via byte ranges,
but the `.bgi` index is downloaded once to a local cache. The cache lives in a
dedicated directory (a `lazybgen-bgi-cache` subdirectory of the system temp dir)
and each entry is keyed by a hash of the full URL, so same-named indexes in
different buckets never collide. Override the location with the
`LAZYBGEN_BGI_CACHE_DIR` environment variable.

### Streaming large files

`load_bgen` materializes the whole `(n_samples, n_variants)` matrix. For files
too large to hold at once, `BgenReader.iter_variants()` streams variants in
memory-bounded blocks (peak memory `O(n_samples × block_size)`). It yields
`(info, dosage)` per variant, where `info` is a `dict` with keys
`chrom, pos, rsid, ref, alt` (access as `info["chrom"]`) and `dosage` is a 1-D
array of per-sample dosages (NaN for missing):

```python
from lazybgen.reader import BgenReader

with BgenReader("chr1.bgen") as reader:
    for info, dosage in reader.iter_variants():
        ...  # info["pos"]; dosage.shape == (reader.nsamples,)
```

It accepts the same `region_chrom/region_start/region_end`, `variant_filter`,
`sample_indices`, and `dtype` selection as `load_variants`. By default
`block_size` auto-scales to keep each block near a fixed memory budget (it
shrinks as the sample count grows); pass an explicit `block_size` to override.

### Parallel decode

Decoding runs in parallel across CPU cores by default: each block is inflated and
decoded across worker threads, byte-identical to single-threaded decoding. This
applies to `load_bgen`, `BgenReader.load_variants`, and `iter_variants`, for both
all-samples and sample-filtered (cohort) reads, and scales with the sample and
core count (e.g. ~5x faster on a 100K-sample load on a 16-core machine, ~3.7x on a
cohort).

Control the worker count with `num_threads`: `0` (default) auto-detects the core
count, `1` forces single-threaded decoding, and `N > 1` uses N threads.

```python
# Parallel by default (auto-detected cores)
genotypes, variant_info, sample_ids = load_bgen("chr1.bgen")

# Force single-threaded decoding
load_bgen("chr1.bgen", num_threads=1)

# BgenReader takes the same num_threads control
with BgenReader("chr1.bgen", num_threads=8) as reader:
    dosages, info = reader.load_variants(region_chrom="chr1", region_start=1, region_end=1_000_000)
```

## Performance

lazybgen vs the [`bgen`](https://pypi.org/project/bgen/) package reading the same
local files (16 vCPU / 125 GB n2 VM, median of 3 page-cache-warm runs, 2026-09-02).
Variant count fixed at 10k, samples scaling to biobank size; speedup is lazybgen
vs bgen, parenthetical is lazybgen's wall time:

| Workload                 | 5k x 10k (94 MB) | 50k x 10k (931 MB) | 500k x 10k (9.1 GB) |
|--------------------------|------------------|--------------------|---------------------|
| Full decode              | 3.3x (265 ms)    | 8.3x (980 ms)      | 14.5x (5.36 s)      |
| Region (500 variants)    | 5.7x (8 ms)      | 6.6x (68 ms)       | 10.4x (417 ms)      |
| Scattered (200 variants) | 3.3x (7 ms)      | 5.0x (38 ms)       | 6.9x (254 ms)       |

A local file is memory-mapped and read in place, so a partial read holds only the
matrix it returns: at 500k samples the region and scattered reads above peak at
178 MB and 187 MB, against 2.1 GB and 933 MB for the same reads through `bgen`.
A single-variant lookup is the one workload lazybgen does not win, because it is
dominated by opening the reader; hold a `BgenReader` open across lookups instead
of calling `load_bgen` per variant.

### Remote: lazy partial reads at biobank scale

A local-only reader must download the whole file before reading a byte; lazybgen
fetches only the variants you ask for, directly from `gs://`, in time that depends
on the sample count and the number of variants requested - **not** the file size.
So as a file grows toward biobank-scale variant counts, lazybgen's partial-read
time stays flat while the download baseline grows. At 500k samples:

| Read (500k samples)      | lazybgen `gs://` | 10k var (9.1 GB) | 50k var (45 GB) | 100k var (91 GB) |
|--------------------------|------------------|------------------|-----------------|------------------|
| One variant              | **0.50 s**       | ~16x (8 s)       | ~67x (33 s)     | ~172x (85 s)     |
| Region (500 contiguous)  | **3.9 s**        | ~3x (12 s)       | ~10x (37 s)     | ~23x (89 s)      |
| Scattered (200 random)   | **1.5 s**        | ~6x (10 s)       | ~23x (35 s)     | ~57x (87 s)      |

Each cell is the end-to-end speedup, with the download-then-read baseline time in
parentheses (whole-file `gcloud storage cp` + bgen's local read). lazybgen's
partial-read times are measured and size-invariant; the download times they are
compared against are carried forward from an earlier same-region measurement
(~1.2 GB/s), since they time the transfer rather than either reader. This is
**best-case for the download** (fast same-region link, free egress), so a
laptop/cross-region/metered link widens every gap. See
[benchmarks/README.md](benchmarks/README.md#lazybgen-vs-the-bgen-package) for the
full size ladder, wall times, invariance check, and methodology.

## License

MIT

## Citation

Kanai, M. et al. [Population-scale multiome immune cell atlas reveals complex disease drivers](https://doi.org/10.1101/2025.11.25.25340489). medRxiv (2025)

## Contact

Masahiro Kanai (<mkanai@broadinstitute.org>)
