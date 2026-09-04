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
the published tables below were measured at the older 500 / 200, which is what
their column headers say. Re-running with the current defaults reproduces the
ratios, not the wall times. Each result also carries `spread`, the range of the
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

**Setup.** Measured 2026-09-03 on a 16 vCPU / 125 GB n2 VM (Xeon 2.80 GHz), so
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
| 500                 | 11 MB   | 3.6x | 2.4x | 1.6x | 0.5x |
| 1k                  | 20 MB   | 5.8x | 3.2x | 2.3x | 0.7x |
| 5k                  | 94 MB   | 10.6x | 7.9x | 4.5x | 1.8x |
| 10k                 | 187 MB  | 13.0x | 6.8x | 7.1x | 2.9x |
| 50k                 | 931 MB  | 15.5x | 13.3x | 10.4x | 5.7x |
| 100k                | 1.86 GB | 16.8x | 15.2x | 13.0x | 6.6x |
| 500k                | 9.1 GB  | 15.6x | 15.8x | 14.4x | 7.9x |

lazybgen wall time at the same points (median of 3):

| samples | full_decode | region (500 var) | scattered (200 var) | single |
|---------|-------------|------------------|---------------------|--------|
| 500     | 33 ms | 3 ms | 3 ms | 1 ms |
| 1k      | 39 ms | 4 ms | 3 ms | 1 ms |
| 5k      | 75 ms | 5 ms | 4 ms | 1 ms |
| 10k     | 119 ms | 11 ms | 5 ms | 1 ms |
| 50k     | 471 ms | 30 ms | 16 ms | 3 ms |
| 100k    | 929 ms | 55 ms | 27 ms | 5 ms |
| 500k    | 4.77 s | 257 ms | 114 ms | 19 ms |

The lead grows with sample count, from ~3.6x at 500 samples to 15.6x at 500k: the
parallel decode has more to chew on, and the per-read fixed costs stop mattering.
Two exceptions worth knowing:

- **`single` depends on what the process has already read, so read that row
  carefully.** The ladder runs all four workloads for a size in one process, so
  by the time `single` runs the `.sample` file has been parsed and cached, and
  one variant at 500k samples costs 21 ms. The same call in a *fresh* process
  costs about 236 ms, because it parses 500k sample IDs first, which loses to the
  `bgen` package's 181 ms. Reading many variants amortizes that; reading exactly
  one does not.
- **A loop of point queries is slower than it looks, whatever the reader state.**
  Every call returns a variant-info DataFrame, and pandas charges about 100 us to
  build one whatever its row count. Measured on 500 lookups at 5000 samples: one
  call per position is 249 us each, the same 500 positions passed as a single
  `variant_filter` are 14.4 us each, and `iter_variants` over the range 14.1 us.
  Batch the lookups.
- **The 500-sample fixtures** are small enough that all four workloads are a few
  milliseconds end to end, so treat that row as noise-adjacent.

**Peak memory.** A local file is memory-mapped and read in place, so the
compressed bytes are never copied into a buffer on the way, and what a read costs
is essentially the matrix it returns. At 500k samples the 500-variant region is
1.9 GB of float64 and peaks at 2.6 GB, against 2.1 GB for the same read through
`bgen`; lazybgen carries a modest overhead above the matrix, it does not undercut
the other reader. `full_decode` is the extreme of the
same rule: it materializes the entire `(samples, variants)` array, 37 GB at
500k x 10k, so it needs a host with comparable RAM. On a memory-constrained box
read a region or stream with `iter_variants`; those paths are memory-bounded and
are where the realistic large-file workloads live anyway.

Peak RSS is read from the kernel's own high-water mark (`VmHWM`), reset per run.
Sampling it from a Python thread does not work here: a decode holds the GIL for
its whole duration, so the sampler never runs while the memory is actually in
use, and the figure comes back near the pre-run baseline. Numbers published
before 2026-09-03 were collected that way and understated peak memory, in some
cases by more than 10x.

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

These are the fsspec (gcsfs) transport, which is now the fallback rather than the
default. The same reads run roughly 1.1-1.5x faster on obstore, because it does the
HTTP and TLS in Rust
with the GIL released and several range requests can be genuinely in flight at
once; see "Remote transports" in the root README, and
`compare_remote_backends.py` to reproduce it on your own bucket.

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

## Upscaled workloads (`compare_upscaled.py`)

The `compare_libraries.py` ladder sweeps one axis, sample count, with everything
else pinned: 10k variants in the file, all samples read, zlib, 8-bit, float64,
and reads that are either tiny (200-500 variants) or the whole file. Those are
the numbers in the tables above and they stay as they are.

`compare_upscaled.py` pushes the axes that ladder holds constant, which is where
a bottleneck the current numbers cannot see would have to live. It shares the
harness with `compare_libraries.py` (it imports `measure`, the peak-RSS sampler,
and the fixture and position helpers from it), so the conventions are the same:
warmup pass, median of `--num-runs`, peak RSS sampled in a background thread,
JSON out, and the `bgen` package as the comparison wherever it can express the
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

# streaming: cap the biggest point, since a whole-file 500k stream decodes 37 GB
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
workload onto a Tier A host. It is a memory lever and not expected to be a speed
one: decoding to float32 has measured the same wall time as float64 at 100k
samples, because bytes written are not what limits the decode.

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
  to the `bgen` package at any size. Streaming yields one contiguous copy per
  variant, so it can pay a per-variant allocation the batch path does not.
- `encoding` 16-bit worse than about 2.5x of 8-bit (the unpack kernel), or zstd
  worse than about 1.5x of zlib. zlib decode is libdeflate now, so zstd becoming
  the slow branch would be a new finding rather than an old one.
- `index` point or scattered queries more than about 2x slower on the
  500k-variant file than on the 20k-variant one, or metadata for a large
  selection running into the seconds.
- `threads` saturating at 8 on a wide region while a narrow one still scales to
  16. That is the same first-touch ceiling seen from the other side.
