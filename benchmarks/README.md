# lazybgen benchmarking

Tools for measuring and tracking the lazybgen reader's performance across git
commits. Benchmarks the reader directly: the measured workload is the
**reader itself** (decode + parse + range queries), not a downstream CLI. The
reader was always the I/O bottleneck downstream, so timing it directly is what
we want when optimizing lazybgen.

## Components

- `run_benchmark.py` - times the reader on a matrix of synthetic BGEN files
  (local, page-cache-warm; isolates decode cost).
- `compare_libraries.py` - head-to-head benchmark of lazybgen vs the `bgen`
  package across file sizes (local), plus lazybgen-only remote (`gs://`) reads.
  See "lazybgen vs the `bgen` package" below.
- `run_remote_benchmark.py` - the remote (`gs://`) regime: cold/warm wall time,
  byte-range GET count, bytes fetched, read amplification, a stream
  vs download-then-read vs warm-local comparison, and a prefetch-knob sweep.
  Network-bound, not decode-bound. Quick wiring check (public bucket, via ADC):
  `python benchmarks/run_remote_benchmark.py --smoke`. Full run (uploads the
  fixtures to a GCS bucket first, then measures):
  `python benchmarks/run_remote_benchmark.py --upload --strategies stream,download_then_read,warm_local --prefetch-sweep`.
- `Dockerfile.benchmark` - builds lazybgen from any commit (vendored compression
  submodules + in-place extension build). It still installs CMake, because
  commits from before the switch to libdeflate build their vendored zlib with
  it; current commits compile every vendored library directly.
- `benchmark_commits.sh` - builds an image per commit and runs the benchmark.
- `compare_results.py` - tables, CSV, plots, and regression detection across runs.
- `analyze_profiles.py` - turns `--profile` output into reports / call graphs.
- `generate_test_bgen.py` - regenerates the `test_data/` fixtures.

## Workloads (per file)

| Workload               | What it measures                                           |
|------------------------|-----------------------------------------------------------|
| `load_full`            | `load_bgen()` decoding every variant x sample             |
| `load_region`          | `load_bgen(region=)` over the middle slice (indexed range)|
| `load_sample_filtered` | `load_bgen(sample_ids=)` over a 50% sample subset         |
| `iter_variants`        | `BgenReader.iter_variants()` streaming (memory-bounded)   |

Each workload runs a warmup pass plus N measured runs; the **median** is
reported (page caches are warm, so this isolates decode/parse cost). Peak RSS is
sampled in a background thread. Workloads a reader lacks (e.g. `iter_variants`
on the initial extraction commit) are skipped, not failed.

`nan_action="mean"` is used so that 8-bit quantization rounding never aborts a
run. Uncompressed (`nocomp`) configs are excluded: lazybgen rejects uncompressed
BGEN by design.

## lazybgen vs the `bgen` package

`compare_libraries.py` is a head-to-head benchmark against the
[`bgen`](https://pypi.org/project/bgen/) package (1.9.0), the other actively
maintained random-access BGEN reader on PyPI. Four workloads are run through each
library's own idiomatic API, so the numbers reflect how each one is actually
used:

| Workload      | lazybgen                              | bgen                                          |
|---------------|---------------------------------------|-----------------------------------------------|
| `full_decode` | `load_bgen(path)`                     | iterate all variants, stack `alt_dosage`      |
| `region`      | `load_bgen(region="chr:start-end")`   | `fetch(chrom, start, stop)` + `alt_dosage`    |
| `scattered`   | `load_bgen(variant_filter=...)`       | `at_position(pos)` loop (200 variants)        |
| `single`      | `load_bgen(region=)` (one variant)    | `at_position(pos)` (one variant)              |

Both libraries populate a float64 dosage matrix of the same element count and
dtype: lazybgen materializes its `(samples, variants)` output in one C++ call;
the `bgen` package exposes one variant at a time, so its equivalent fills the
matrix one variant at a time. Wall time and peak RSS are therefore comparable.

**Setup.** Measured 2026-09-02 on a 16 vCPU / 125 GB n2 VM (Xeon 2.80 GHz), so
the 37 GB full-decode materialization at 500k x 10k fits without hitting a RAM
ceiling. Page-cache-warm; median of 3 runs after 1 warmup; synthetic zlib / 8-bit
fixtures from `generate_test_bgen.py`; `bgen` 1.9.0. lazybgen uses its default
parallel decode (auto-detected cores); the `bgen` package decodes one variant at
a time in Rust. Reproduce with:

```bash
python benchmarks/compare_libraries.py \
    --data-dir benchmarks/test_data --output benchmarks/compare_results.json \
    --max-full-gb 64
```

### Local: lazybgen vs bgen (speedup)

Variant count fixed at 10k; sample count scales from 500 to 500k (biobank scale).
Speedup is bgen time / lazybgen time (>1 means lazybgen is faster):

| samples (x 10k var) | file    | full_decode | region | scattered | single |
|---------------------|---------|-------------|--------|-----------|--------|
| 500                 | 11 MB   | 2.5x        | 2.0x   | 1.3x      | 0.3x   |
| 1k                  | 20 MB   | 3.3x        | 2.5x   | 1.8x      | 0.5x   |
| 5k                  | 94 MB   | 3.3x        | 5.7x   | 3.3x      | 0.8x   |
| 10k                 | 187 MB  | 3.7x        | 3.0x   | 4.6x      | 1.0x   |
| 50k                 | 931 MB  | 8.3x        | 6.6x   | 5.0x      | 1.2x   |
| 100k                | 1.86 GB | 10.9x       | 8.5x   | 5.8x      | 1.2x   |
| 500k                | 9.1 GB  | 14.5x       | 10.4x  | 6.9x      | 1.2x   |

lazybgen wall time at the same points (median of 3):

| samples | full_decode | region (500 var) | scattered (200 var) | single |
|---------|-------------|------------------|---------------------|--------|
| 500     | 59 ms       | 5 ms             | 5 ms                | 2 ms   |
| 1k      | 85 ms       | 6 ms             | 5 ms                | 2 ms   |
| 5k      | 265 ms      | 8 ms             | 7 ms                | 3 ms   |
| 10k     | 498 ms      | 29 ms            | 9 ms                | 4 ms   |
| 50k     | 980 ms      | 68 ms            | 38 ms               | 16 ms  |
| 100k    | 1.46 s      | 104 ms           | 62 ms               | 31 ms  |
| 500k    | 5.36 s      | 417 ms           | 254 ms              | 150 ms |

The lead grows with sample count, from ~2.5x at 500 samples to 14.5x at 500k: the
parallel decode has more to chew on, and the per-read fixed costs stop mattering.
Two exceptions worth knowing:

- **`single` is not a win.** One variant is a few milliseconds of real work, so it
  is dominated by opening the reader, and `load_bgen` builds the sample-ID list
  before it returns. Below ~10k samples the `bgen` package's lighter open beats
  us; above it we are level. If you want single-variant lookups to be cheap, hold
  one `BgenReader` open across them rather than calling `load_bgen` per variant.
- **The 500-sample fixtures** are small enough that all four workloads are a few
  milliseconds end to end, so treat that row as noise-adjacent.

**Peak memory.** A local file is memory-mapped and read in place, so a partial
read holds only the matrix it returns - at 500k samples, 178 MB for the
500-variant region and 187 MB for the 200-variant scattered read, against 2.1 GB
and 933 MB for the same reads through `bgen`. `full_decode` is the opposite case:
it materializes the entire `(samples, variants)` float64 array, 37 GB at
500k x 10k (lazybgen peaks at 47.6 GB including per-thread decode buffers, `bgen`
at 38.3 GB), so it needs a host with comparable RAM. On a memory-constrained box
read a region or stream with `iter_variants`; those paths are memory-bounded and
are where the realistic large-file workloads live anyway.

### Remote (`gs://`): lazy partial reads at biobank scale

The `bgen` package has no remote-read path, so a non-lazy workflow must download
the whole file before it can read a byte. lazybgen issues byte-range requests and
fetches only the variants you ask for, so its partial-read time depends on the
**sample count** and how many variants you request - **not** the total file size.
Measured at 500k samples (lazybgen reading directly from a same-region GCS
bucket, median of 3):

| Read                     | lazybgen, direct from `gs://` |
|--------------------------|-------------------------------|
| One variant              | 0.50 s                        |
| Region (500 contiguous)  | 3.91 s                        |
| Scattered (200 random)   | 1.52 s                        |

Because lazybgen fetches only the requested variants, these times are independent
of the total variant count (file size). A block decode asks for a whole block's
records in one batched request and merges the ones that sit next to each other,
so a scattered read is 200 *concurrent* range requests rather than 200 sequential
round-trips. That is why the contiguous region is the slower of the two here: at
500k samples its 500 records are ~500 MB to move, so it is bound by bandwidth
rather than by latency. The download-then-read baseline, by contrast, grows
linearly with the whole file:

| Read (500k samples)      | lazybgen `gs://` | 10k var (9.1 GB) | 50k var (45 GB) | 100k var (91 GB) |
|--------------------------|------------------|------------------|-----------------|------------------|
| One variant              | 0.50 s           | ~16x             | ~67x            | ~172x            |
| Region (500 contiguous)  | 3.91 s           | ~3x              | ~10x            | ~23x             |
| Scattered (200 random)   | 1.52 s           | ~6x              | ~23x            | ~57x             |

lazybgen's partial-read times were measured 2026-09-02 and are size-invariant.
The whole-file download times they are compared against (9.1 / 45 / 91 GB in
8 / 33 / 85 s, ~1.2 GB/s same-region; the 45 and 91 GB objects built by composing
the 9.1 GB file, since download time depends only on byte size) were measured
2026-06-23 and carried forward, because they time the transfer rather than either
reader. Each baseline cell is that download plus bgen's local read of the same
selection.

lazybgen now wins every partial read at every size on this list, and by more as
the file grows. That was not true before the batched range fetch: a scattered
read of 200 variants used to cost 200 sequential round-trips and roughly broke
even with a bulk download of a 9 GB file.

Two caveats, both in the baseline's favor (so the real-world advantage is
larger): this is **best-case for the download** (~1.2 GB/s, same-region, free
egress); a laptop, cross-region, or metered link makes the whole-file download far
slower while lazybgen's byte-range cost is unchanged. And lazybgen always
transfers far fewer bytes (a few MB vs the whole file), so the data-egress win
holds even where the wall-time race is close. Reproduce with:

```bash
python benchmarks/compare_libraries.py --skip-local \
    --remote-bucket gs://your-bucket/lazybgen-bench \
    --output benchmarks/compare_results_remote.json
```

The default 24 GB `--max-full-gb` cap deliberately skips the remote `full_decode`
at 500k x 10k: it would refetch 9.1 GB per run to materialize a 37 GB matrix, and
none of the tables above use it.

## Quick start

Run from the repository root.

```bash
# Local, in-process (fast; no Docker). Point at a built source tree.
python benchmarks/run_benchmark.py \
    --data-dir benchmarks/test_data --output-dir /tmp/bench \
    --mode quick --num-runs 3

# initial-extraction baseline vs HEAD, reproducibly, via Docker
./benchmarks/benchmark_commits.sh dd01e34 07505b6
./benchmarks/compare_results.py benchmark_results/benchmark_*.json --baseline dd01e34
cat comparison_results/benchmark_summary.md
```

### Comparing readers locally without Docker

`run_benchmark.py` can import any reader, so you can A/B two source trees in one
process model (most directly comparable for in-process timing/memory):

```bash
# a baseline reader module (any package exposing load_bgen / BgenReader)
GIT_COMMIT=baseline python benchmarks/run_benchmark.py \
    --reader-module your_package.reader --reader-path /path/to/baseline \
    --data-dir benchmarks/test_data --output-dir /tmp/bench --mode quick

# lazybgen HEAD (built in place)
GIT_COMMIT=lazybgen-head python benchmarks/run_benchmark.py \
    --data-dir benchmarks/test_data --output-dir /tmp/bench --mode quick

./benchmarks/compare_results.py /tmp/bench/benchmark_*.json --baseline baseline
```

## Modes

`--mode` selects the config matrix (see `get_test_configurations`):

- `quick` - tiny / small / medium (seconds; smoke test)
- `standard` - square + shape configs up to 100k samples (default)
- `compression` - medium across zstd / 16-bit / 32-bit
- `scaling` - sample- and variant-scaling sweeps
- `comprehensive` - everything

`--region-size {small,medium,large}` sets the `load_region` slice (10/50/90%).

## Profiling

```bash
# cProfile (+ perf for C++ frames, needs SYS_ADMIN) for one commit
./benchmarks/benchmark_commits.sh --profile HEAD
./benchmarks/analyze_profiles.py benchmark_results/profiles/

# or locally, in-process
python benchmarks/run_benchmark.py --profile --mode quick \
    --data-dir benchmarks/test_data --output-dir /tmp/prof
```

Outputs per workload: `*_python.prof` + `*_python_stats.txt` (cProfile) and, when
`perf` is available, `*_perf.data` + `*_perf_report.txt`. `analyze_profiles.py`
adds call graphs (gprof2dot) and a summary. Profiling perturbs timing; do not
compare profile runs against regular benchmark runs.

## Test data

`test_data/` holds synthetic BGENs across sample counts (500-100000), variant
counts (500-20000), compression (zlib / zstd / nocomp), and bit depth (8/16/32).
Total ~5 GB. It is git-ignored. Regenerate with
`generate_test_bgen.py --mode <mode>` (needs the `bgen` writer library and
`bgenix`).

## Notes

- Per-commit Docker images build lazybgen from each checkout, fetching the
  submodule commits pinned by that checkout.
- `compare_results.py` flags any (config, workload) pair >10% slower than the
  baseline (default baseline: `master` if present, else earliest; override with
  `--baseline`).
