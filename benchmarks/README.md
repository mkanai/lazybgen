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
- `compare_remote_backends.py` - times the two remote transports (fsspec vs
  obstore) on the same `gs://` / `s3://` BGEN and checks that they return
  identical dosages:
  `python benchmarks/compare_remote_backends.py --url gs://bucket/big.bgen`.
  obstore is a default dependency, so both rows run out of the box.
- `peak_memory.py` - peak RSS of each `compare_libraries.py` read, lazybgen and
  `bgen`, each in a fresh process (Linux; the peak-memory table below).
- `compare_results.py` - tables, CSV, plots, and regression detection across runs.
- `analyze_profiles.py` - turns `--profile` output into reports / call graphs.
- `generate_test_bgen.py` - regenerates the `test_data/` fixtures.

## Workloads (per file)

**Remote comparisons must be interleaved.** `--interleave` applies only to the
local run; the remote run has none, and two remote configurations measured as
separate runs are not comparable - this link has shown the same read drift by up
to 7x across a day, which is larger than most of the effects being measured.
Alternate the configurations rep by rep, warm each shape before timing it, and
report the spread. Repeated reads of one object are served warm, so warm numbers
compare fairly with each other but are not cold streaming rates.

The slice workloads read `--region-variants` (default 2000) contiguous variants
and `--scattered-variants` (default 1000) spread-out ones. These are the ladder's
constant: a run that changes them is not comparable with one that does not, and
the column headers below say which sizes produced each table. Each result also
carries `spread`, the range of the
measured runs over their median, so a speedup smaller than the noise is visible
rather than assumed.

| Workload               | What it measures                                           |
|------------------------|-----------------------------------------------------------|
| `load_full`            | `load_bgen()` decoding every variant x sample             |
| `load_region`          | `load_bgen(region=)` over the middle slice (indexed range)|
| `load_sample_filtered` | `load_bgen(sample_ids=)` over a 50% sample subset         |
| `iter_variants`        | `BgenReader.iter_variants()` streaming (memory-bounded)   |

Each workload runs a warmup pass plus N measured runs; the **median** is
reported (page caches are warm, so this isolates decode/parse cost). Peak RSS is
the kernel's high-water mark (`VmHWM`, reset before each run), with a
background-thread sample only where that is unavailable. Workloads a reader lacks (e.g. `iter_variants`
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
| `scattered`   | `load_bgen(variant_filter=...)`       | `at_position(pos)` loop (1000 variants)       |
| `single`      | `load_bgen(region=)` (one variant)    | `at_position(pos)` (one variant)              |

Both libraries populate a float64 dosage matrix of the same element count and
dtype: lazybgen materializes its `(samples, variants)` output in one C++ call;
the `bgen` package exposes one variant at a time, so its equivalent fills the
matrix one variant at a time. Wall time and peak RSS are therefore comparable.

**Setup.** lazybgen 0.3.0, measured 2026-11-21 on a 16 vCPU / 125 GB n2 VM (Xeon
2.80 GHz, Cascade Lake), so the 37 GiB full-decode materialization at 500k x 10k
fits without hitting a RAM ceiling. Page-cache-warm; median of 3 runs after 1
warmup, averaged over two rounds that alternated with the previous release;
synthetic zlib / 8-bit
fixtures from `generate_test_bgen.py`; `bgen` 1.9.0. lazybgen uses its default
parallel decode (one worker per CPU); the `bgen` package decodes one variant at
a time (a Cython wrapper around C++). Reproduce with:

```bash
# The seven 10k-variant ladder files (500 to 500k samples, ~12 GB)
python benchmarks/generate_test_bgen.py --mode ladder
python benchmarks/compare_libraries.py \
    --data-dir benchmarks/test_data --output benchmarks/compare_results.json \
    --max-full-gb 64
python benchmarks/peak_memory.py --data-dir benchmarks/test_data
```

### Local: lazybgen vs bgen (speedup)

Variant count fixed at 10k; sample count scales from 500 to 500k (biobank scale).
Speedup is bgen time / lazybgen time (>1 means lazybgen is faster):

| samples (x 10k var) | file    | full_decode | region | scattered | single |
|---------------------|---------|-------------|--------|-----------|--------|
| 500                 | 11 MiB  | 3.8x | 3.0x | 3.6x | 0.6x |
| 1k                  | 20 MiB  | 6.3x | 5.7x | 4.7x | 0.8x |
| 5k                  | 94 MiB  | 12.5x | 10.6x | 6.8x | 2.4x |
| 10k                 | 187 MiB | 16.6x | 14.7x | 10.1x | 3.4x |
| 50k                 | 931 MiB | 21.7x | 21.1x | 18.1x | 9.0x |
| 100k                | 1.8 GiB | 21.9x | 21.2x | 19.9x | 12.1x |
| 500k                | 9.1 GiB | 22.7x | 22.5x | 21.9x | 17.3x |

lazybgen wall time at the same points:

| samples | full_decode | region (2000 var) | scattered (1000 var) | single |
|---------|-------------|-------------------|----------------------|--------|
| 500     | 41 ms | 11 ms | 7.9 ms | 1.2 ms |
| 1k      | 43 ms | 10 ms | 8.5 ms | 1.2 ms |
| 5k      | 68 ms | 17 ms | 15 ms | 1.3 ms |
| 10k     | 104 ms | 23 ms | 19 ms | 1.5 ms |
| 50k     | 372 ms | 77 ms | 46 ms | 2.3 ms |
| 100k    | 707 ms | 147 ms | 81 ms | 3.4 ms |
| 500k    | 3.42 s | 697 ms | 366 ms | 12 ms |

Against 0.2.0 in the same run (the two builds alternated round by round), 0.3.0
is 1.1-1.2x faster at 5k samples, 1.2-1.3x at 10k, and 1.3-1.4x from 50k samples
up on `full_decode`, `region` and `scattered`, and up to 1.8x on `single`; at 500
and 1k samples it is 0.98-1.07x, within the run-to-run noise. 0.2.0 was in
turn 1.6-5.2x faster than 0.1.0 on `full_decode`.

The lead over `bgen` grows with sample count, from ~3.8x at 500 samples to 23x at 500k: the
parallel decode has more to chew on, and the per-read fixed costs stop mattering.
Two exceptions worth knowing:

- **`single` depends on what the process has already read, so read that row
  carefully.** The ladder runs all four workloads for a size in one process, so
  by the time `single` runs the `.sample` file has been parsed and cached, and
  one variant at 500k samples costs 12 ms. The same call in a *fresh*
  process costs about 190 ms (186 ms, opening the file plus the read, with the
  library already imported), because it parses 500k sample IDs first, against
  221 ms for the `bgen` package measured the same way (median of 5 fresh
  processes each; `benchmarks/readme_figures.py`). Reading many variants
  amortizes that; reading exactly one does not.
- **A loop of point queries is slower than it looks, whatever the reader state.**
  Every call pays fixed costs whatever its row count, building a variant-info
  DataFrame among them. Measured on 500 lookups at 5000 samples
  (`benchmarks/readme_figures.py`, with a `.sample` file): one `load_variants`
  call per position on an open reader is 300 us each, and one `load_bgen` call
  per position, which also reopens the file, 1.0 ms; the same 500 positions
  passed as a single `variant_filter` are 15 us each, 20x and 67x faster.
  `iter_variants` over the 9,981 variants the positions span costs 11 us per
  variant. Batch the lookups.
- **The small fixtures** are fast enough that every workload is a few to tens of
  milliseconds end to end, so treat the 500- to 10k-sample rows as
  noise-adjacent (up to 22% spread across runs for lazybgen and 25% for `bgen`).
  From 50k samples up, every cell measured within 8% except one lazybgen full
  decode at 500k, at 19% in one round and 0.2% in the other.

**Peak memory.** A local file is memory-mapped and read in place, so the
compressed bytes are never copied into a buffer on the way, and what a read costs
is essentially the matrix it returns. Peak RSS in MiB, lazybgen against `bgen`,
each read alone in a fresh process (median of 3; `benchmarks/peak_memory.py`), so
the figure is that read's own cost rather than what an earlier workload left
mapped. Both sides return the same float64 matrix:

| samples | full_decode | region | scattered |
|---------|-------------|--------|-----------|
| 5k   | 571 / 421 | 177 / 116 | 203 / 78 |
| 50k  | 4845 / 3858 | 1036 / 806 | 671 / 425 |
| 100k | 9594 / 7676 | 1991 / 1573 | 1150 / 810 |
| 500k | 47583 / 38225 | 9628 / 7708 | 4993 / 3893 |

At 500k samples the 2000-variant region is 7.5 GiB of float64 and peaks at 9.4
GiB, against 7.5 GiB for the same read through `bgen`: lazybgen peaks about a
quarter above the matrix it returns, and `bgen` barely above it, so lazybgen
does not undercut the other reader. The ratio is larger at small sample counts, where fixed
overhead dominates (2.6x for the 5k scattered read). `full_decode` is the extreme
of the same rule: it materializes the entire `(samples, variants)` array, 37 GiB
at 500k x 10k, so it needs a host with comparable RAM. On a memory-constrained box
read a region or stream with `iter_variants`; those paths are memory-bounded and
are where the realistic large-file workloads live anyway.

Peak RSS is read from the kernel's own high-water mark (`VmHWM`).
Sampling it from a Python thread does not work here: a decode holds the GIL for
its whole duration, so the sampler never runs while the memory is actually in
use, and the figure comes back near the pre-run baseline, understating peak
memory by more than 10x in some cases.

### Remote (`gs://`): lazy partial reads at biobank scale

The `bgen` package has no remote-read path, so a non-lazy workflow must download
the whole file before it can read a byte. lazybgen issues byte-range requests and
fetches only the variants you ask for, so its partial-read time depends on the
**sample count** and how many variants you request - **not** the total file size.

**Remote comparisons must be interleaved.** A remote link drifts far more over
minutes than the effects being measured (one read has ranged over 180 s, 81 s
and 24 s across a day on one machine), so two configurations measured as separate
runs are not comparable: the drift is more than enough to invent or hide a 2x.
`compare_remote_builds.py` alternates every configuration read by read; the table
below comes from it, medians of 5 after a warmup.

Measured 2026-09-04, same-region GCS bucket, 500k samples x 10k variants:

| Read                      | 0.1.0 (fsspec) | 0.2.0 fsspec | 0.2.0 obstore |
|---------------------------|----------------|--------------|---------------|
| Full decode               | 37.49 s        | 33.76 s      | **31.81 s**   |
| Region (2000 contiguous)  | 9.01 s         | 7.54 s       | **5.93 s**    |
| Scattered (1000 random)   | 45.24 s        | 4.38 s       | **3.12 s**    |
| One variant               | 0.79 s         | 0.79 s       | **0.34 s**    |

0.3.0 reads remotely at about 0.2.0's speed (the decode speedups do not show over
the network). Measured 2026-11-21 against the 0.2.0 decoder, interleaved in two
passes of 5 reads, medians of all 10 (0.2.0 first): full decode 38.5 s against
36.3 s, region 7.32 s against 6.79 s, scattered 4.64 s against 3.79 s, one
variant 0.35 s against 0.37 s. Only the scattered read was faster in both passes
(3.62 and 4.14 s against 4.50 and 4.83 s), and the single read slower in both,
by 1-6%; the full and region reads changed order between the passes. All are
slower than the 2026-09-04 table for both builds alike, which is the link's
day-to-day drift described above.

Splitting reader changes from transport changes: the 0.2.0 reader is worth
1.11x (full decode), 1.20x (region), **10.3x** (scattered) and 1.00x (single),
and obstore a further 1.06x, 1.27x, 1.40x and **2.31x** on top. The scattered win
is the batched range fetch, which turns one round trip per variant into a handful
of concurrent requests. The single-variant row is untouched by the reader work
and moved only by the transport, because that read is almost entirely open, auth
and one small GET.

At 100k samples the same comparison gives reader gains of 1.61x / 2.22x / 17.2x /
1.00x and transport gains of 1.29x / 1.53x / 1.83x / 3.46x: obstore helps most
where a read is latency-bound rather than bandwidth-bound.

**Against downloading the file first.** lazybgen's partial-read time is
independent of the total variant count, so the comparison improves with file
size. Measured 2026-11-21 on 0.3.0: the baseline is a whole-file
`gcloud storage cp` (9.1 GiB in 8.28 s, 1.10 GiB/s same-region, median of 5) plus
the same read from the local copy with lazybgen (12 ms, 0.70 s and 0.37 s from
the local ladder above); the 45 and 91 GiB rows scale the download by byte size,
since download time depends only on that. The lazybgen times are the medians of
10 reads above:

| Read (500k samples)       | lazybgen `gs://` | 10k var (9.1 GiB) | 50k var (45 GiB) | 100k var (91 GiB) |
|---------------------------|------------------|------------------|-----------------|------------------|
| One variant               | 0.37 s           | ~22x             | ~112x           | ~224x            |
| Region (2000 contiguous)  | 6.79 s           | 1.3x             | ~6x             | ~12x             |
| Scattered (1000 random)   | 3.79 s           | ~2x              | ~11x            | ~22x             |

The 1.3x is worth reading rather than explaining away: at 10k variants a
2000-variant region is a fifth of the file, and on a fast same-region link
fetching a fifth of an object costs nearly what fetching all of it does. Partial
reads pay when the slice is genuinely a slice, which is the regime biobank-scale
files are in - the same read is ~12x ahead at 100k variants. Reading the local
copy with the `bgen` package instead would replace lazybgen's local-read term in
each baseline with its own (0.20 s, 15.7 s and 8.0 s), widening each gap. The ratios are higher
than in the 0.2.0 README because the download measured slower that day (1.10
against 1.73 GiB/s), not because lazybgen's remote reads changed.

The download baseline is not a constant: the identical measurement has given
0.84, 1.10, 1.58 and 1.73 GiB/s on the same VM type on different days. It is also
**best-case for the download** (same-region, free egress), so a laptop,
cross-region or metered link widens every gap. lazybgen also transfers far fewer
bytes, so the egress win holds even where the wall-time race is close.

Reproduce with:

```bash
# Interleaved, and the way the table above was produced. Each --build is
# name=path with optional env overrides, so one checkout can serve as both the
# fsspec and the obstore arm.
python benchmarks/compare_remote_builds.py \
    --build 0.1.0=~/lz-master \
    --build 0.2.0-fsspec=~/lz-head:LAZYBGEN_REMOTE_BACKEND=fsspec \
    --build 0.2.0-obstore=~/lz-head:LAZYBGEN_REMOTE_BACKEND=obstore \
    --bucket gs://your-bucket/lazybgen-bench --reps 5 --max-full-gb 64 \
    --output remote_interleaved.json
```

`compare_libraries.py --skip-local --remote-bucket ...` also measures the remote
workloads, but only for the build it is run from, so use it to profile one build
rather than to compare two. `--max-full-gb` must be raised past 37 GB for either
to include the remote `full_decode` at 500k x 10k, which refetches 9.1 GiB per run.

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

## Upscaled workloads (`compare_upscaled.py`)

The `compare_libraries.py` ladder sweeps one axis, sample count, with everything
else pinned: 10k variants in the file, all samples read, zlib, 8-bit, float64,
and reads that are either tiny (200-500 variants) or the whole file. Those are
the numbers in the tables above and they stay as they are.

`compare_upscaled.py` pushes the axes that ladder holds constant, which is where
a bottleneck the current numbers cannot see would have to live. It shares the
harness with `compare_libraries.py` (it imports `measure`, the peak-RSS sampler,
and the fixture and position helpers from it), so the conventions are the same:
warmup pass, median of `--num-runs`, peak RSS from `VmHWM`, JSON out, and the `bgen` package as the comparison wherever it can express the
workload.

### Suites

| Suite | What it varies | Why it can find something new |
|---|---|---|
| `region_scaling` | region width 50 to 5000 variants | Separates fixed per-call cost from per-variant cost, and at 500k samples pushes one read's output into the tens of GB, where output first-touch rather than decode is the ceiling |
| `cohort` | sample subset 1% / 10% / 50% | A different decode kernel (row gather), a different memory profile, and an ID-to-index mapping the all-samples workloads never pay. Uncovered by the existing ladder |
| `stream` | `iter_variants` over a whole file | The memory-bounded path the README recommends for large files, never benchmarked head to head, and the one workload where both libraries use the same shape of API |
| `encoding` | zlib / zstd, 8 / 16 / 32-bit | Bit depth doubles the bytes per genotype and changes the unpack kernel; the codec decides which decompressor runs |
| `index` | 20k vs 500k variants in the file | Index and metadata cost (BGI query, variant DataFrame) at a variant count where it stops being a rounding error |
| `point_loop` | many point queries, one open reader | The per-query cost with the fixed open cost amortized, which the ladder's `single` workload cannot show |
| `threads` | `num_threads` 1 to 16 on a heavy region | Whether a much larger output still scales to every core |

Two workloads the `bgen` package cannot express: it has no sample-filtered read
(the comparison decodes all samples and gathers the cohort's rows, which is what
a user has to do), and no thread control, so `threads` is lazybgen only.

One asymmetry to keep in mind when reading `point_loop` and `index`: lazybgen's
point query also returns a one-row pandas DataFrame of variant metadata, which
the `bgen` package does not build. A loss there is the cost of the richer return
value, not of the decode.

### Fixtures

Most of the new workloads run over the fixtures `test_data/` already holds:
region width, cohort fraction, streaming, thread count and point-query loops are
all heavier *uses* of the existing 500 to 500k sample files. Four fixtures do
not exist and cannot be substituted, because no current fixture reaches the
regime they probe:

| Fixture | Size | Answers |
|---|---|---|
| `test_2000s_500000v_zlib_8bit.bgen` | ~1.9 GB (plus ~78 MB `.bgi`) | Index and metadata cost. Nothing in `test_data/` exceeds 20k variants, and a real per-chromosome BGEN holds a million or more. The sample count is kept low so the file grows with the index, not with the decode |
| `test_100000s_2000v_zlib_8bit.bgen` | ~370 MB | The matched 8-bit control for the two below. Without it, comparing them against the 100k x 10k file confounds encoding with variant count |
| `test_100000s_2000v_zlib_16bit.bgen` | ~745 MB | Bit depth where it matters. The only 16-bit fixture today is 5000 x 5000, and at 5k samples the decode kernel is a few percent of the read, so it cannot answer whether 16-bit changes the picture at biobank sample counts |
| `test_100000s_2000v_zstd_8bit.bgen` | ~370 MB | The same question for the codec |

About 3.5 GB in total. Generate them with:

```bash
python benchmarks/generate_test_bgen.py --mode upscale
```

`index` is skipped when the 500k-variant fixture is absent, and `encoding` falls
back to the 5000 x 5000 set. Everything else runs on the existing fixtures.

### Measurement integrity

The harness reports the regime it measured rather than assuming one.

- **Warm loop versus one-shot.** The reported time is a median over a warm loop:
  page cache hot, and the allocator likely handing back the block the previous
  run freed. glibc's mmap threshold is dynamic (it starts at 128 KB and grows to
  at most 32 MB as freed mappings are seen), so an output under 32 MB is usually
  recycled and never pays first-touch, while a larger one gets a fresh mapping
  and faults in every page. Which side of that line a call lands on depends on
  what ran before it in the same process. Every result therefore also carries
  `first_call` (the first pass, timed separately) and `minor_faults` (the pages
  the run first-touched), and `--malloc-regime mmap-always` pins the allocator so
  every run pays first-touch and the one-shot cost is measured repeatably. That
  regime is not neutral in a head-to-head, since a reader that allocates once per
  matrix and one that allocates once per variant are charged very differently, so
  use it for lazybgen-versus-lazybgen work and leave the default on for the
  library comparison tables.
- **The parsed `.sample` cache.** The reader keeps the most recently parsed
  `.sample` file keyed on path, mtime and size, so a loop that reopens one file
  pays the parse once and every later run reads a warm cache, which flatters any
  repeated-call workload. The cases that open a reader clear it before each run;
  the JSON records `sample_cache_cleared` so a run on a build without the cache
  is not mistaken for one with it cleared.
- **Warm page cache is a claim, not a fact.** At these sizes the fixture and the
  output stop fitting in RAM together. Each case records `disk_read_mb` actually
  read from disk during its runs and prints a note when that is more than
  trivial, so a partly-cold run is visible instead of being read as a decode
  regression. A warning is printed up front when the largest selected output
  exceeds half of RAM.
- **Drift over a long sweep.** `measure` finishes every run of one library
  before starting the other, so a machine that drifts charges the drift to
  whichever ran later. This script interleaves them by default
  (`--no-interleave` restores the sequential order); `compare_libraries.py`
  keeps the sequential order by default so its published tables stay
  reproducible, with `--interleave` available.
- **Cohort setup is not part of the workload.** The cohort ID list is built once
  outside the timer, since a real caller already has it. What is inside the timer
  is the mapping from IDs to file columns, for both libraries, and two extra
  lazybgen-only cases isolate it: `id_map (reader open, IDs built)` times the
  mapping alone, `id_map (fresh reader)` also carries opening the file and
  materializing the IDs.
- **Known blind spot, not addressed here.** The synthetic fixtures contain no
  missing calls, so `nan_action="mean"` never actually imputes and the
  imputation path is unmeasured at every scale. Closing that needs a generator
  option for a missing-call rate and a fixture to go with it.

### Run plan

Tier A fits a 64 GB host. The `--max-output-gb` default of 8 keeps any single
output well inside it.

```bash
# roughly 35-50 min: everything that fits in 64 GB
python benchmarks/compare_upscaled.py --data-dir benchmarks/test_data \
    --suite region_scaling,cohort,point_loop,threads,encoding,index \
    --output benchmarks/upscaled_tierA.json

# streaming: cap the biggest point, since a whole-file 500k stream decodes 37 GiB
python benchmarks/compare_upscaled.py --data-dir benchmarks/test_data \
    --suite stream --stream-variants 2000 \
    --output benchmarks/upscaled_stream.json
```

Tier B needs the high-memory host the published tables were measured on
(16 vCPU / 125 GB). It is the same script with the cap lifted, which admits the
5000-variant region at 500k samples (18.6 GB of output) and the whole-file
stream.

```bash
# roughly 45-70 min
python benchmarks/compare_upscaled.py --data-dir benchmarks/test_data \
    --suite all --max-output-gb 40 --output benchmarks/upscaled_tierB.json
```

`--dtype float32` halves every output, which is the lever that brings a Tier B
workload onto a Tier A host. It is also somewhat faster: on 0.3.0 at 100k
samples, a float32 full decode took 601 ms against 707 ms for float64, and a
2000-variant region 127 ms against 146 ms (`benchmarks/readme_figures.py`), so
compare float32 runs only with float32 runs.

### Reading the result

Still scaling fine:

- `region_scaling`: time roughly linear in region width, with the constant
  improving as fixed cost amortizes, and the speedup over the `bgen` package
  holding within about 25% across the width sweep.
- `cohort`: time roughly proportional to cohort size, with `id_map` a small part
  of a 10% cohort read.
- `stream`: within about 1.3x of the per-variant cost of the equivalent
  `load_bgen` read, and still ahead of the `bgen` package.
- `encoding`: 16-bit near 2x the 8-bit time (it is 2x the bytes), zstd within
  roughly 0.8-1.3x of zlib.
- `index`: open, point and scattered on the 500k-variant file close to the same
  on the 20k-variant file, since only the index grew.
- `threads`: monotone improvement out to 16 threads at every region width.

A new bottleneck:

- `region_scaling` superlinear in width, or the speedup falling by more than
  about 25% from 500 to 5000 variants, at 500k samples. That points at output
  first-touch, which a wider read cannot amortize because at biobank sample
  counts a single column already exceeds the decode's per-worker run target.
- `cohort` time barely falling between a 50% and a 1% cohort, or `id_map (fresh
  reader)` exceeding roughly 30% of a 10% cohort read at 500k. That points at
  the ID-to-index mapping rather than the decode. Separately, a filtered read
  costing more than about 1.5x an unfiltered read of the same output size points
  at the row-gather kernel, and comparing `--cohort-order shuffled` against
  `sorted` says whether it is gather locality.
- `stream` more than about 2x the per-variant cost of the batch read, or losing
  to the `bgen` package at any size. Streaming yields one contiguous array per
  variant (a view when the block's column already is one, else a copy), so it
  can pay a per-variant allocation the batch path does not.
- `encoding` 16-bit worse than about 2.5x of 8-bit (the unpack kernel), or zstd
  worse than about 1.5x of zlib. zlib decode is libdeflate now, so zstd becoming
  the slow branch would be a new finding rather than an old one.
- `index` point or scattered queries more than about 2x slower on the
  500k-variant file than on the 20k-variant one, or metadata for a large
  selection running into the seconds.
- `threads` saturating at 8 on a wide region while a narrow one still scales to
  16. That is the same first-touch ceiling seen from the other side.
