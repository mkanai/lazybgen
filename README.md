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
default, which is ~1.1-1.5x faster than gcsfs and needs no code change. See
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

### Build info

`from lazybgen import get_build_info` returns the compression backend the package
was built against (vendored libdeflate / zstd, or system libraries).

### Remote transports

Remote reads go through one of two transports, chosen per reader by
`remote_backend`:

- **obstore**, installed by default and used by default **for `gs://`**. It runs
  HTTP and TLS in Rust with the GIL released, so one process can have many range
  requests genuinely in flight.
- **fsspec** (`gcsfs` / `s3fs`), always available, the fallback, and still the
  default for `s3://`. fsspec drives every request in a process through a single
  asyncio event loop, which pins one CPU core and caps throughput there.

`s3://` stays on s3fs because obstore's S3 store does not resolve a bucket's
region (it assumes `us-east-1`) and its credential chain does not read
`~/.aws/credentials`, `AWS_PROFILE` or SSO. Pass `remote_backend="obstore"` to
use it for S3 anyway, with an explicit region and credentials.

**Multiprocessing**: obstore's runtime does not survive `fork()`, which is the
default start method on Linux. A process that reads and then forks leaves its
children unable to use obstore, and they fall back to fsspec automatically rather
than hanging. Use the `spawn` or `forkserver` start method if you want workers to
keep the faster transport.

`remote_backend="auto"` (the default) uses obstore when it is importable **and**
every entry in `storage_options` has an obstore equivalent, and falls back to
fsspec otherwise, so an option that decides which bytes come back is never
silently dropped. Reads keep working if obstore is deselected at install time.
`LAZYBGEN_REMOTE_BACKEND` sets the transport for a whole process;
`remote_backend="fsspec"` or `"obstore"` pins one reader. The number of threads a
range batch is split across is `LAZYBGEN_OBSTORE_THREADS` (default 16).

`storage_options` are written the same way for both, in fsspec spelling
(`anon`, `requester_pays`, `key`, `secret`, `endpoint_url`, ...); the obstore
transport translates them. Requester-pays works on both: GCS by way of the
`x-goog-user-project` header, S3 by way of the native request-payer setting.

Measured on a same-region GCS bucket from a `us-central1` VM (fsspec -> obstore,
same reader, both verified byte-identical):

| Read (500k samples, 9.1 GB file) | fsspec | obstore |
|---|---|---|
| Region (500 contiguous) | 2.98 s | **2.79 s** (1.07x) |
| Scattered (500 spread)  | 1.49 s | **1.26 s** (1.18x) |
| One variant             | 503 ms | **345 ms** (1.46x) |
| Full decode             | 53.7 s | **46.1 s** (1.16x) |

The fetch itself is 2.5x (contiguous) to 3.8x (scattered) faster; the end-to-end
figure is smaller because roughly half of a remote read is decoding and
allocating the output, which the transport does not touch. The spread across
workloads and machines is wide (a second run on a smaller file put every read
between 1.2x and 1.4x, and one scattered read came out slightly below 1.0), so
treat these as "somewhat faster, never slower in aggregate" rather than a
constant. `benchmarks/compare_remote_backends.py` runs the comparison against
your own bucket.

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
local files (16 vCPU / 125 GB n2 VM, median of 3 page-cache-warm runs, 2026-09-04).
Variant count fixed at 10k, samples scaling to biobank size; speedup is lazybgen
vs bgen, parenthetical is lazybgen's wall time:

| Workload                 | 5k x 10k (94 MB) | 50k x 10k (931 MB) | 500k x 10k (9.1 GB) |
|--------------------------|------------------|--------------------|---------------------|
| Full decode              | 9.0x (97 ms)     | 14.4x (578 ms)     | 14.4x (5.37 s)      |
| Region (500 variants)    | 7.1x (6 ms)      | 13.5x (35 ms)      | 13.7x (286 ms)      |
| Scattered (200 variants) | 4.2x (5 ms)      | 9.4x (21 ms)       | 13.7x (127 ms)      |

A local file is memory-mapped and read in place, so the compressed bytes are never
copied into a buffer on the way. What a read costs in memory is then essentially
the matrix it returns: the 500-variant region above is 1.9 GB of float64 at 500k
samples, and the read peaks at 2.6 GB, against 2.1 GB for the same read through
`bgen`. Ask for float32, or fewer samples, if that matters more than precision.
A single-variant lookup is the one workload lazybgen does not win. Ask for the
variants together rather than one call each: every call builds a variant-info
DataFrame, and pandas charges about 100 us for that however few rows it holds, so
a loop of 500 single-variant calls spends more time in pandas than in reading.
Passing the positions as one `variant_filter`, or streaming the range with
`iter_variants`, is ~17x faster for the same 500 variants and is what the
scattered row above measures.

### Remote: lazy partial reads at biobank scale

A local-only reader must download the whole file before reading a byte; lazybgen
fetches only the variants you ask for, directly from `gs://`, in time that depends
on the sample count and the number of variants requested - **not** the file size.
So as a file grows toward biobank-scale variant counts, lazybgen's partial-read
time stays flat while the download baseline grows. At 500k samples:

| Read (500k samples)      | lazybgen `gs://` | 10k var (9.1 GB) | 50k var (45 GB) | 100k var (91 GB) |
|--------------------------|------------------|------------------|-----------------|------------------|
| One variant              | **345 ms**       | ~31x (10.6 s)    | ~152x (52 s)    | ~305x (105 s)    |
| Region (500 contiguous)  | **2.79 s**       | ~4x (10.6 s)     | ~19x (52 s)     | ~38x (105 s)     |
| Scattered (200 random)   | **1.26 s**       | ~8x (10.6 s)     | ~41x (52 s)     | ~83x (105 s)     |

The numbers above are for the fsspec transport. With **pure-Python** transports,
remote throughput is bounded per process and not by the link: the limit is one CPU
core's worth of Python-side HTTP and TLS work, and every pure-Python transport we
measured pins at exactly one core and lands within about 15% of the others,
whether it goes through fsspec or straight at aiohttp. Running several event loops
on several threads is *worse*, not better, since they contend for one GIL.

The default obstore transport lifts that: it does the HTTP and TLS in Rust with
the GIL released, which takes a byte-range fetch from ~350 MB/s to ~1 GB/s in one
process. End to end that is worth roughly 1.1-1.5x on the reads above, since about
half of a remote read is decoding and allocating the output rather than moving
bytes (see [Remote transports](#remote-transports)). Beyond that, shard the
variants across **processes**; threads on a pure-Python transport will not do it.

Each cell is the end-to-end speedup, with the download-then-read baseline time in
parentheses (whole-file `gcloud storage cp` + bgen's local read). Both halves are
measured same-region and in the same run: the 9.1 GB download ran five times at
7.6 / 8.8 / 10.6 / 11.9 / 18.2 s, median 10.6 s or 0.86 GB/s, and the larger sizes
scale from that rate. Download speed is machine-dependent and not very
repeatable - the same measurement gave 1.62 GB/s on an identical VM a day earlier
and 1.06 GB/s on the dev box - so treat the ratios as being for this class of
host, not as constants. This is
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
