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

`gs://` reads go through [obstore](https://developmentseed.org/obstore/) by
default, which is 1.1x to 3.5x faster than gcsfs and needs no code change. See
[Remote transports](#remote-transports).

`load_bgen` returns `(genotypes, variant_info, sample_ids)`, where `genotypes`
is an `(n_samples, n_variants)` `np.ndarray`, `variant_info` is a `pd.DataFrame`
with columns `chrom, pos, rsid, ref, alt`, and `sample_ids` is a `list[str]`.

A `.bgi` index is required; create one with `bgenix -g file.bgen`.

### Parameters

- `file_path`: path, `gs://`, or `s3://` URL to the BGEN file
- `index_path`: `.bgi` index (defaults to `file_path + ".bgi"`)
- `sample_path`: optional `.sample` file
- `region`: `"chr:start-end"` to read a genomic interval
- `variant_filter`: variant subset as a dict with keys `chromosome`, `positions`,
  `allele1` and `allele2`, the three lists aligned element by element and the
  alleles matching the file exactly (see
  [Reading many variants](#reading-many-variants)); build it with
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
- `storage_options`: cloud backend kwargs, in fsspec spelling whichever transport
  serves them (e.g. `{"anon": True}` for public S3, `{"requester_pays": True}` to
  bill the env default project, or `{"requester_pays": "billing-project-id"}` for
  GCS requester-pays buckets)
- `remote_backend`: transport for `gs://` and `s3://` reads: `"auto"` (default),
  `"obstore"`, or `"fsspec"` (see [Remote transports](#remote-transports))

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

### Reading many variants

Ask for the variants together rather than one call each. Every call returns a
variant-info DataFrame, and pandas charges about 100 us to build one whatever its
row count, so a loop of point lookups spends more time in pandas than in reading.
For 500 variants the batched forms below are ~17x faster than the loop:

```python
from lazybgen import load_bgen
from lazybgen.reader import BgenReader

# Your variants: positions and their alleles, aligned element by element.
positions = [10_001, 25_500, 91_200]  # ...hundreds more
ref_alleles = ["A", "C", "G"]
alt_alleles = ["G", "T", "A"]

# Slow: one call, one DataFrame, per variant.
for pos in positions:
    load_bgen("chr1.bgen", region=f"chr1:{pos}-{pos}")

# Fast: one call for the whole selection.
genotypes, variant_info, sample_ids = load_bgen(
    "chr1.bgen",
    variant_filter={
        "chromosome": "chr1",
        "positions": positions,
        "allele1": ref_alleles,
        "allele2": alt_alleles,
    },
)

# Also fast, and memory-bounded: stream a contiguous range.
with BgenReader("chr1.bgen") as reader:
    for info, dosage in reader.iter_variants(region_chrom="chr1", region_start=10_000, region_end=100_000):
        ...
```

All four keys are required, and the alleles must match the file exactly: a
variant whose alleles differ is not matched, and neither is one whose `ref`/`alt`
are the other way round, so a filter that matches nothing raises rather than
returning an empty result. `load_variant_filter("variants.z")` builds the dict
from a `.z` file if your variants come from one.

### Memory

A local file is memory-mapped and read in place, so the compressed bytes are
never copied on the way and a read costs about the matrix it returns. lazybgen
peaks around 1.2x the `bgen` package for the same read (9.7 GB against 7.8 GB for
a 2000-variant region at 500k samples).

Ask for `float32` when single precision is enough. Dosages are computed in single
precision either way, so a `float64` result is the exact widening of the same
numbers, and asking for it costs twice the memory and ~18% more decode time:

```python
import numpy as np

genotypes, _, _ = load_bgen("chr1.bgen", region="chr1:1-1000000", dtype=np.float32)
```

To avoid materializing the matrix at all, stream it: `iter_variants` is
`O(n_samples x block_size)` whatever the file holds (see
[Streaming large files](#streaming-large-files)).

### Streaming large files

`load_bgen` materializes the whole `(n_samples, n_variants)` matrix. For files
too large to hold at once, `BgenReader.iter_variants()` streams variants in
memory-bounded blocks (peak memory `O(n_samples x block_size)`). It yields
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
core count: on a 16-core machine at 100k samples, a full decode is 6.8x faster
than `num_threads=1` and a 2000-variant region 6.6x.

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

### Remote transports

`gs://` reads go through **obstore**, which does HTTP and TLS in Rust with the GIL
released, so many range requests are genuinely in flight at once. `s3://` goes
through **fsspec** (`s3fs`), because obstore's S3 store assumes the `us-east-1`
region and does not read `~/.aws/credentials`, `AWS_PROFILE` or SSO. obstore and
gcsfs are installed by default; `s3://` additionally needs
`pip install lazybgen[s3]`. `storage_options` are spelled the same way for either
transport, and requester-pays works on both.

`remote_backend` overrides the choice per reader (`"auto"`, `"obstore"`,
`"fsspec"`), and `LAZYBGEN_REMOTE_BACKEND` does it for a process.
`"auto"` falls back to fsspec whenever obstore is unavailable or cannot express
one of your `storage_options`, so an option that decides which bytes come back is
never silently dropped.

Against gcsfs, obstore is worth 1.1x to 3.5x end to end depending on the read:
most for small ones, which are mostly connection and request latency, and least
for a full decode, which is bandwidth plus decode.

**Multiprocessing**: obstore's runtime does not survive `fork()`, the default
start method on Linux. Children of a process that has already read fall back to
fsspec automatically rather than hanging; use the `spawn` or `forkserver` start
method to keep the faster transport in workers.

### Remote `.bgi` index caching

For a `gs://`/`s3://` BGEN, the genotype data is read in place via byte ranges,
but the `.bgi` index is downloaded once to a local cache. The cache lives in a
dedicated directory (a `lazybgen-bgi-cache` subdirectory of the system temp dir)
and each entry is keyed by a hash of the full URL, so same-named indexes in
different buckets never collide. Override the location with the
`LAZYBGEN_BGI_CACHE_DIR` environment variable.

### Build info

`from lazybgen import get_build_info` returns the compression backend the package
was built against (vendored libdeflate / zstd, or system libraries).

## Performance

Against the [`bgen`](https://pypi.org/project/bgen/) package on the same local
files, 10k variants and samples scaling to biobank size (16 vCPU n2 VM, median of
3 warm runs, lazybgen's wall time in parentheses):

| Workload                  | 5k samples (94 MB) | 50k (931 MB)   | 500k (9.1 GB)  |
|---------------------------|--------------------|----------------|----------------|
| Full decode               | 10.7x (75 ms)      | 15.3x (483 ms) | 15.5x (4.80 s) |
| Region (2000 variants)    | 8.8x (18 ms)       | 15.3x (99 ms)  | 16.5x (961 ms) |
| Scattered (1000 variants) | 6.0x (15 ms)       | 13.5x (57 ms)  | 15.5x (501 ms) |

### Remote: lazy partial reads at biobank scale

This is the point of the package. A local-only reader must download the whole file
before reading a byte, so its cost tracks file size; lazybgen fetches only the
variants you ask for, so its cost tracks the slice and **does not grow with the
file**. Each cell is the end-to-end speedup over downloading the file first
(whole-file `gcloud storage cp` + a local read), with that baseline in
parentheses:

| Read (500k samples)       | lazybgen `gs://` | 10k var (9.1 GB) | 50k var (45 GB) | 100k var (91 GB) |
|---------------------------|------------------|------------------|-----------------|------------------|
| One variant               | **341 ms**       | ~15x (5.2 s)     | ~76x (26 s)     | ~151x (52 s)     |
| Region (2000 contiguous)  | **5.93 s**       | 0.9x (5.2 s)     | ~4x (26 s)      | ~9x (52 s)       |
| Scattered (1000 random)   | **3.12 s**       | ~2x (5.2 s)      | ~8x (26 s)      | ~17x (52 s)      |

Only the 9.1 GB download is measured (5.2 s, median of 5); the 26 s and 52 s
figures scale it by byte size, since download time depends on bytes and not on
what is in them.

The lazybgen column is one number per row because it does not change with the
file: the same read costs the same at 10k variants and at 100k. The 0.9x is the
honest edge of that: at 10k variants a 2000-variant region is a fifth of the file,
and fetching a fifth of an object costs about what fetching all of it does on a
fast same-region link. Partial reads pay when the slice is genuinely a slice,
which is the regime biobank-scale files are in. The baseline is also best-case
for the download - same-region, free egress, and not very repeatable run to run -
so a laptop, cross-region or metered link widens every gap.

These are single batched reads. A loop of one-variant calls is ~17x slower for
the same variants, and a read costs about the matrix it returns; see
[Reading many variants](#reading-many-variants) and [Memory](#memory).

[benchmarks/README.md](benchmarks/README.md#lazybgen-vs-the-bgen-package) has the
full size ladder, remote and transport comparisons, peak-memory tables, and the
methodology.

## License

MIT

## Citation

Kanai, M. et al. [Population-scale multiome immune cell atlas reveals complex disease drivers](https://doi.org/10.1101/2025.11.25.25340489). medRxiv (2025)

## Contact

Masahiro Kanai (<mkanai@broadinstitute.org>)
