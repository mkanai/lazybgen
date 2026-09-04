#!/usr/bin/env python3
"""Upscaled benchmark: the workload shapes the size ladder in ``compare_libraries.py`` does not cover.

``compare_libraries.py`` sweeps one axis (sample count, 500 to 500k) with the
variant count pinned at 10k and four fixed workloads. That ladder answers "how
does a slice read scale with cohort size", and nothing else: every read is
all-samples, zlib, 8-bit, float64, and either tiny (200-500 variants) or the
whole file. This script pushes the axes that ladder holds constant, because that
is where a bottleneck that the current numbers hide would have to live.

Suites (select with ``--suite``, list them with ``--list``):

  region_scaling  region reads from 50 to 5000 variants. Separates fixed
                  per-call cost from per-variant cost, and at biobank sample
                  counts pushes one read's output into the tens of GB.
  cohort          sample-filtered reads (a 1% / 10% / 50% subset of the file's
                  samples). A different decode kernel, a different memory
                  profile, and an ID-to-index mapping step the all-samples
                  workloads never pay. Not covered by the existing ladder at all.
  stream          ``iter_variants`` over a whole file versus the `bgen` package's
                  own per-variant iteration. The memory-bounded path, and the
                  one workload where both libraries use the same shape of API.
  encoding        the same read across zlib / zstd and 8 / 16 / 32-bit
                  probabilities. Bit depth changes bytes per genotype and the
                  unpack kernel; the codec changes which decompressor runs.
  index           a file with 500k variants versus one with 20k at the same
                  sample count. Isolates index and metadata cost (BGI query,
                  variant DataFrame) from decode cost.
  point_loop      many point queries through one open reader, rather than one
                  query per fresh ``load_bgen`` call. Shows the per-query cost
                  once the fixed open cost is amortized.
  threads         a worker-thread sweep on a heavy region read.

Harness conventions match ``compare_libraries.py``: a warmup pass, then the
median of ``--num-runs`` measured runs, peak RSS sampled in a background thread,
JSON output, and the `bgen` package as the comparison wherever it can express
the workload. Two workloads it cannot: `bgen` has no sample-filtered read (the
comparison decodes all samples and gathers the cohort rows, which is what a user
must do), and no thread control.

Reading the output: `single`-style point queries through lazybgen also build a
one-row pandas DataFrame of variant metadata, which the `bgen` package does not,
so a point-query loss is a cost of the richer return value rather than of the
decode.
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_libraries import (  # noqa: E402
    CHROM,
    environment_snapshot,
    measure,
    measure_interleaved,
    region_bounds,
    scattered_positions,
    set_malloc_regime,
    single_position,
)

# Sample-count ladder shared by the suites that sweep size. Variants are fixed at
# 10k, matching the fixtures the existing ladder uses, so a result here can be
# read against the published table.
SIZES = [
    ("5k x 10k", 5000, 10000),
    ("50k x 10k", 50000, 10000),
    ("100k x 10k", 100000, 10000),
    ("500k x 10k", 500000, 10000),
]

# Region widths for region_scaling. 500 is the existing ladder's width, so it is
# the anchor; 5000 is a chromosome-arm-sized read.
REGION_LADDER = [50, 500, 2000, 5000]

# (compression, bit_depth) fixture variants for the encoding suite.
ENCODINGS = [("zlib", 8), ("zstd", 8), ("zlib", 16), ("zlib", 32)]

# Shapes to try for the encoding suite. 5000 x 5000 exists for every encoding
# today; the 100k shape needs the fixtures listed in the README, and is skipped
# when they are absent.
ENCODING_SHAPES = [(5000, 5000), (100000, 2000)]

# (samples, variants) pairs for the index suite: the same sample count with two
# very different index sizes, so the difference is the index and not the decode.
INDEX_SHAPES = [(2000, 20000), (2000, 500000)]

SCATTERED_VARIANTS = 200

SUITES = ("region_scaling", "cohort", "stream", "encoding", "index", "point_loop", "threads")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def fixture_stem(samples, variants, compression="zlib", bit_depth=8):
    return f"test_{samples}s_{variants}v_{compression}_{bit_depth}bit"


def fixture_for(data_dir, samples, variants, compression="zlib", bit_depth=8):
    """Return (bgen, sample) paths, or (None, None) when the .bgen is absent."""
    stem = fixture_stem(samples, variants, compression, bit_depth)
    bgen_path = Path(data_dir) / f"{stem}.bgen"
    if not bgen_path.exists():
        return None, None
    sample_path = Path(data_dir) / f"{stem}.sample"
    return bgen_path, (sample_path if sample_path.exists() else None)


def output_gb(rows, cols, dtype):
    return rows * cols * np.dtype(dtype).itemsize / (1024**3)


# ---------------------------------------------------------------------------
# Shared state a workload needs but should not be timed for
# ---------------------------------------------------------------------------
def sample_ids_of(bgen_path, sample_path):
    """Read a file's sample IDs once, outside any timer.

    A cohort read starts from a list of IDs the caller already has (a phenotype
    table, say), so materializing that list is setup, not part of the workload.
    """
    from lazybgen.reader import BgenReader

    with BgenReader(str(bgen_path), sample_path=(str(sample_path) if sample_path else None)) as r:
        return list(r.samples)


def pick_cohort(all_ids, fraction, order, seed=20260903):
    """Choose a reproducible subset of sample IDs.

    ``shuffled`` is what a real cohort list looks like: the requested order has
    nothing to do with the file's order, so the decode gathers rows from all over
    each column. ``sorted`` keeps the file's order, which turns the gather into a
    forward scan. Comparing the two separates cohort size from gather locality.
    """
    n = max(1, int(round(len(all_ids) * fraction)))
    rng = random.Random(seed)
    picked = rng.sample(range(len(all_ids)), n)
    if order == "sorted":
        picked.sort()
    return [all_ids[i] for i in picked]


def sample_cache_clearer():
    """A callable that empties lazybgen's parsed-.sample cache, or None.

    The reader caches the most recently parsed .sample file keyed on path, mtime
    and size, so a loop that reopens one file pays the parse once and every later
    run reads a warm cache. Clearing it before each run restores the cost a
    caller sees on a fresh process. Returns None on a build without the cache.
    """
    try:
        import lazybgen.reader as reader_module
    except Exception:
        return None
    cache = getattr(reader_module, "_SAMPLE_ID_CACHE", None)
    if cache is None or not hasattr(cache, "clear"):
        return None

    def clear():
        try:
            cache.clear()
        except Exception:
            pass

    return clear


# ---------------------------------------------------------------------------
# Workload builders
# ---------------------------------------------------------------------------
def lz_region_fn(bgen_path, sample_path, region, dtype, num_threads=0, sample_ids=None):
    import lazybgen

    kwargs = {"nan_action": "mean", "dtype": dtype, "num_threads": num_threads}
    if sample_path is not None:
        kwargs["sample_path"] = str(sample_path)
    if sample_ids is not None:
        kwargs["sample_ids"] = sample_ids

    def run():
        return lazybgen.load_bgen(str(bgen_path), region=region, **kwargs)[0]

    return run


def bgen_region_fn(bgen_path, samples, rstart, rend, count, dtype, row_index=None):
    """Fill one matrix from the `bgen` package, one variant at a time.

    The output is preallocated rather than stacked from a list of per-variant
    arrays: at biobank scale a stack holds the whole result twice, which would
    measure the harness's own allocation rather than the reader.
    """
    from bgen import BgenReader

    n_rows = samples if row_index is None else len(row_index)

    def run():
        r = BgenReader(str(bgen_path))
        try:
            out = np.empty((count, n_rows), dtype=dtype)
            i = 0
            for v in r.fetch(CHROM, rstart, rend):
                if i >= count:
                    break
                d = v.alt_dosage
                out[i] = d if row_index is None else d[row_index]
                i += 1
            return out
        finally:
            r.close()

    return run


def bgen_cohort_fn(bgen_path, cohort_ids, rstart, rend, count, dtype):
    """The `bgen` package's equivalent of a cohort read.

    It has no sample-filtered API, so the work a user has to do is: map the
    requested IDs onto file columns, decode every sample, and keep the cohort's
    rows. The mapping is inside the timer because lazybgen's equivalent mapping
    is inside ``load_bgen``.
    """
    from bgen import BgenReader

    def run():
        r = BgenReader(str(bgen_path))
        try:
            index_of = {sid: i for i, sid in enumerate(r.samples)}
            idx = np.array([index_of[s] for s in cohort_ids if s in index_of], dtype=np.intp)
            out = np.empty((count, len(idx)), dtype=dtype)
            i = 0
            for v in r.fetch(CHROM, rstart, rend):
                if i >= count:
                    break
                out[i] = v.alt_dosage[idx]
                i += 1
            return out
        finally:
            r.close()

    return run


def lz_stream_fn(bgen_path, sample_path, selection, dtype):
    from lazybgen.reader import BgenReader

    def run():
        acc = 0.0
        n = 0
        with BgenReader(str(bgen_path), sample_path=(str(sample_path) if sample_path else None)) as r:
            for _info, dosage in r.iter_variants(dtype=dtype, **selection):
                # Touch one element so the dosage array is actually produced;
                # summing the column would add a second pass that has nothing to
                # do with either reader.
                acc += float(dosage[0])
                n += 1
        return n, acc

    return run


def bgen_stream_fn(bgen_path, rstart, rend):
    from bgen import BgenReader

    def run():
        acc = 0.0
        n = 0
        r = BgenReader(str(bgen_path))
        try:
            source = r if rstart is None else r.fetch(CHROM, rstart, rend)
            for v in source:
                acc += float(v.alt_dosage[0])
                n += 1
        finally:
            r.close()
        return n, acc

    return run


def lz_point_loop_fn(bgen_path, sample_path, positions, dtype):
    """Point queries through one already-open reader.

    ``load_bgen``'s point query reopens the file every call, so the ladder's
    `single` workload is dominated by open cost. Holding the reader open is what
    a variant-at-a-time driver actually does.
    """
    from lazybgen.reader import BgenReader

    reader = BgenReader(str(bgen_path), sample_path=(str(sample_path) if sample_path else None))

    def run():
        total = 0
        for p in positions:
            dosages, _info = reader.load_variants(region_chrom=CHROM, region_start=p, region_end=p, dtype=dtype)
            total += dosages.shape[1]
        return total

    run.close = reader.close
    return run


def bgen_point_loop_fn(bgen_path, positions):
    from bgen import BgenReader

    reader = BgenReader(str(bgen_path))

    def run():
        total = 0
        for p in positions:
            total += len(reader.at_position(p)[0].alt_dosage)
        return total

    run.close = reader.close
    return run


# ---------------------------------------------------------------------------
# Suites
# ---------------------------------------------------------------------------
def make_case(suite, name, axis, lazy_fn, bgen_fn=None, out_gb=0.0, note="", before=None, closers=()):
    return {
        "suite": suite,
        "name": name,
        "axis": axis,
        "lazybgen": lazy_fn,
        "bgen": bgen_fn,
        "output_gb": out_gb,
        "note": note,
        "before": before,
        "closers": closers,
    }


def suite_region_scaling(args, sizes):
    cases = []
    for label, samples, variants in sizes:
        bgen_path, sample_path = fixture_for(args.data_dir, samples, variants)
        if bgen_path is None:
            continue
        for count in args.region_sizes:
            if count > variants:
                continue
            rstart, rend = region_bounds(variants, count)
            region = f"{CHROM}:{rstart}-{rend}"
            cases.append(
                make_case(
                    "region_scaling",
                    f"{label} region={count}",
                    {"size": label, "samples": samples, "variants": variants, "region_variants": count},
                    lz_region_fn(bgen_path, sample_path, region, args.dtype),
                    bgen_region_fn(bgen_path, samples, rstart, rend, count, args.dtype),
                    out_gb=output_gb(samples, count, args.dtype),
                )
            )
    return cases


def suite_cohort(args, sizes):
    cases = []
    count = args.cohort_variants
    for label, samples, variants in sizes:
        bgen_path, sample_path = fixture_for(args.data_dir, samples, variants)
        if bgen_path is None:
            continue
        rstart, rend = region_bounds(variants, count)
        region = f"{CHROM}:{rstart}-{rend}"
        print(f"  reading sample IDs for {label} (setup, not timed)")
        all_ids = sample_ids_of(bgen_path, sample_path)
        for fraction in args.cohort_fractions:
            cohort = pick_cohort(all_ids, fraction, args.cohort_order)
            axis = {
                "size": label,
                "samples": samples,
                "cohort_size": len(cohort),
                "fraction": fraction,
                "region_variants": count,
                "order": args.cohort_order,
            }
            cases.append(
                make_case(
                    "cohort",
                    f"{label} cohort={len(cohort)} ({fraction:.0%}, {args.cohort_order})",
                    axis,
                    lz_region_fn(bgen_path, sample_path, region, args.dtype, sample_ids=cohort),
                    bgen_cohort_fn(bgen_path, cohort, rstart, rend, count, args.dtype),
                    out_gb=output_gb(len(cohort), count, args.dtype),
                )
            )
        # Isolate the ID-to-index mapping from the decode. The warm case reuses
        # one reader whose sample IDs are already built, so it times only the
        # mapping; the cold case opens a reader per run, so it also carries
        # materializing the IDs. Their difference is what a cohort read pays for
        # sample identity before a single genotype is decoded.
        mid_cohort = pick_cohort(all_ids, args.cohort_fractions[len(args.cohort_fractions) // 2], args.cohort_order)
        cases.append(_cohort_map_warm_case(label, samples, bgen_path, sample_path, mid_cohort))
        cases.append(_cohort_map_cold_case(label, samples, bgen_path, sample_path, mid_cohort))
    return cases


def _cohort_map_warm_case(label, samples, bgen_path, sample_path, cohort):
    from lazybgen.reader import BgenReader

    reader = BgenReader(str(bgen_path), sample_path=(str(sample_path) if sample_path else None))
    reader.samples  # materialize once, outside the timer

    def run():
        return reader.get_sample_indices(cohort)

    return make_case(
        "cohort",
        f"{label} id_map (reader open, IDs built)",
        {"size": label, "samples": samples, "cohort_size": len(cohort), "phase": "map_only"},
        run,
        note="lazybgen only: ID-to-index mapping with sample IDs already materialized",
        closers=(reader.close,),
    )


def _cohort_map_cold_case(label, samples, bgen_path, sample_path, cohort):
    from lazybgen.reader import BgenReader

    def run():
        with BgenReader(str(bgen_path), sample_path=(str(sample_path) if sample_path else None)) as r:
            return r.get_sample_indices(cohort)

    return make_case(
        "cohort",
        f"{label} id_map (fresh reader)",
        {"size": label, "samples": samples, "cohort_size": len(cohort), "phase": "open_and_map"},
        run,
        note="lazybgen only: open + materialize sample IDs + map",
        before=sample_cache_clearer(),
    )


def suite_stream(args, sizes):
    cases = []
    for label, samples, variants in sizes:
        bgen_path, sample_path = fixture_for(args.data_dir, samples, variants)
        if bgen_path is None:
            continue
        limit = args.stream_variants
        if limit and limit < variants:
            rstart, rend = region_bounds(variants, limit)
            selection = {"region_chrom": CHROM, "region_start": rstart, "region_end": rend}
            streamed = limit
        else:
            rstart = rend = None
            selection = {}
            streamed = variants
        cases.append(
            make_case(
                "stream",
                f"{label} stream={streamed}",
                {"size": label, "samples": samples, "streamed_variants": streamed},
                lz_stream_fn(bgen_path, sample_path, selection, args.dtype),
                bgen_stream_fn(bgen_path, rstart, rend),
                # Exempt from --max-output-gb: streaming is memory-bounded.
                # lazybgen decodes in blocks sized to a fixed budget and the
                # `bgen` package yields one variant at a time, so neither ever
                # holds the matrix a full read would materialize.
                out_gb=0.0,
                note=f"decodes {output_gb(samples, streamed, args.dtype):.1f} GB of dosages in bounded blocks",
            )
        )
    return cases


def suite_encoding(args, _sizes):
    cases = []
    for samples, variants in ENCODING_SHAPES:
        for compression, bit_depth in ENCODINGS:
            bgen_path, sample_path = fixture_for(args.data_dir, samples, variants, compression, bit_depth)
            if bgen_path is None:
                continue
            tag = f"{compression}/{bit_depth}bit"
            axis = {
                "samples": samples,
                "variants": variants,
                "compression": compression,
                "bit_depth": bit_depth,
                "file_size_mb": bgen_path.stat().st_size / (1024**2),
            }
            cases.append(
                make_case(
                    "encoding",
                    f"{samples}s x {variants}v {tag} full",
                    dict(axis, workload="full"),
                    lz_region_fn(bgen_path, sample_path, None, args.dtype),
                    bgen_region_fn(bgen_path, samples, 0, 0xFFFFFFFF, variants, args.dtype),
                    out_gb=output_gb(samples, variants, args.dtype),
                )
            )
            count = min(500, variants)
            rstart, rend = region_bounds(variants, count)
            cases.append(
                make_case(
                    "encoding",
                    f"{samples}s x {variants}v {tag} region={count}",
                    dict(axis, workload="region"),
                    lz_region_fn(bgen_path, sample_path, f"{CHROM}:{rstart}-{rend}", args.dtype),
                    bgen_region_fn(bgen_path, samples, rstart, rend, count, args.dtype),
                    out_gb=output_gb(samples, count, args.dtype),
                )
            )
    return cases


def suite_index(args, _sizes):
    cases = []
    for samples, variants in INDEX_SHAPES:
        bgen_path, sample_path = fixture_for(args.data_dir, samples, variants)
        if bgen_path is None:
            continue
        bgi = Path(str(bgen_path) + ".bgi")
        axis = {
            "samples": samples,
            "variants": variants,
            "bgi_mb": (bgi.stat().st_size / (1024**2)) if bgi.exists() else None,
        }
        label = f"{samples}s x {variants}v"
        cases.append(_index_open_case(label, axis, bgen_path, sample_path))

        spt = single_position(variants)
        cases.append(
            make_case(
                "index",
                f"{label} point",
                dict(axis, workload="point"),
                lz_region_fn(bgen_path, sample_path, f"{CHROM}:{spt}-{spt}", args.dtype),
                bgen_region_fn(bgen_path, samples, spt, spt, 1, args.dtype),
                out_gb=output_gb(samples, 1, args.dtype),
            )
        )

        positions = scattered_positions(variants, SCATTERED_VARIANTS)
        cases.append(
            make_case(
                "index",
                f"{label} scattered={len(positions)}",
                dict(axis, workload="scattered"),
                _lz_scattered_fn(bgen_path, sample_path, positions, args.dtype),
                _bgen_scattered_fn(bgen_path, samples, positions, args.dtype),
                out_gb=output_gb(samples, len(positions), args.dtype),
            )
        )

        # Same number of decoded variants on both files, so any difference is
        # index and metadata cost rather than decode cost.
        for count in sorted({min(10000, variants), min(args.index_variants, variants)}):
            rstart, rend = region_bounds(variants, count)
            cases.append(
                make_case(
                    "index",
                    f"{label} region={count}",
                    dict(axis, workload="region", region_variants=count),
                    lz_region_fn(bgen_path, sample_path, f"{CHROM}:{rstart}-{rend}", args.dtype),
                    bgen_region_fn(bgen_path, samples, rstart, rend, count, args.dtype),
                    out_gb=output_gb(samples, count, args.dtype),
                )
            )
    return cases


def _index_open_case(label, axis, bgen_path, sample_path):
    from bgen import BgenReader as PkgReader
    from lazybgen.reader import BgenReader

    def lz_open():
        with BgenReader(str(bgen_path), sample_path=(str(sample_path) if sample_path else None)) as r:
            return r.nvariants

    def bg_open():
        r = PkgReader(str(bgen_path))
        try:
            return len(r)
        finally:
            r.close()

    return make_case(
        "index",
        f"{label} open",
        dict(axis, workload="open"),
        lz_open,
        bg_open,
        note="open the file and read the variant count; no genotypes decoded",
        before=sample_cache_clearer(),
    )


def _lz_scattered_fn(bgen_path, sample_path, positions, dtype):
    import lazybgen

    vfilter = {
        "chromosome": CHROM,
        "positions": positions,
        "allele1": ["A"] * len(positions),
        "allele2": ["G"] * len(positions),
    }
    kwargs = {"nan_action": "mean", "dtype": dtype}
    if sample_path is not None:
        kwargs["sample_path"] = str(sample_path)

    def run():
        return lazybgen.load_bgen(str(bgen_path), variant_filter=vfilter, **kwargs)[0]

    return run


def _bgen_scattered_fn(bgen_path, samples, positions, dtype):
    from bgen import BgenReader

    def run():
        r = BgenReader(str(bgen_path))
        try:
            out = np.empty((len(positions), samples), dtype=dtype)
            for i, p in enumerate(positions):
                out[i] = r.at_position(p)[0].alt_dosage
            return out
        finally:
            r.close()

    return run


def suite_point_loop(args, sizes):
    cases = []
    for label, samples, variants in sizes:
        bgen_path, sample_path = fixture_for(args.data_dir, samples, variants)
        if bgen_path is None:
            continue
        positions = scattered_positions(variants, args.point_queries)
        lz = lz_point_loop_fn(bgen_path, sample_path, positions, args.dtype)
        bg = bgen_point_loop_fn(bgen_path, positions)
        cases.append(
            make_case(
                "point_loop",
                f"{label} {len(positions)} point queries",
                {"size": label, "samples": samples, "queries": len(positions)},
                lz,
                bg,
                # One column at a time, freed before the next: resident output is
                # a single column, not the loop's total.
                out_gb=output_gb(samples, 1, args.dtype),
                note="one reader held open across all queries",
                closers=(lz.close, bg.close),
            )
        )
    return cases


def suite_threads(args, sizes):
    cases = []
    count = args.thread_region_variants
    for label, samples, variants in sizes:
        bgen_path, sample_path = fixture_for(args.data_dir, samples, variants)
        if bgen_path is None or count > variants:
            continue
        rstart, rend = region_bounds(variants, count)
        region = f"{CHROM}:{rstart}-{rend}"
        for nt in args.thread_counts:
            cases.append(
                make_case(
                    "threads",
                    f"{label} region={count} nt={nt}",
                    {"size": label, "samples": samples, "region_variants": count, "num_threads": nt},
                    lz_region_fn(bgen_path, sample_path, region, args.dtype, num_threads=nt),
                    note="lazybgen only: the `bgen` package decodes on the calling thread",
                    out_gb=output_gb(samples, count, args.dtype),
                )
            )
    return cases


SUITE_BUILDERS = {
    "region_scaling": suite_region_scaling,
    "cohort": suite_cohort,
    "stream": suite_stream,
    "encoding": suite_encoding,
    "index": suite_index,
    "point_loop": suite_point_loop,
    "threads": suite_threads,
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _disk_read_mb(proc):
    try:
        return proc.io_counters().read_bytes / (1024**2)
    except Exception:
        return None


def _fmt(result):
    if result is None:
        return "-"
    if not result.get("ok"):
        return "FAIL"
    return f"{result['median_time'] * 1000:.1f}ms"


def _fmt_cold(result):
    if result is None or not result.get("ok"):
        return ""
    first = result.get("first_call")
    if not first:
        return ""
    ratio = first["time"] / result["median_time"] if result["median_time"] > 0 else 0
    return f" [cold {first['time'] * 1000:.1f}ms, {ratio:.2f}x warm]"


def run_cases(cases, args, proc):
    rows = []
    for case in cases:
        record = {
            "suite": case["suite"],
            "name": case["name"],
            "axis": case["axis"],
            "output_gb": case["output_gb"],
            "note": case["note"],
        }
        if case["output_gb"] > args.max_output_gb:
            print(f"  skip {case['name']}: {case['output_gb']:.1f} GB output > {args.max_output_gb:.1f} GB cap")
            record["skipped"] = f"output {case['output_gb']:.1f} GB over the --max-output-gb cap"
            rows.append(record)
            _close_all(case)
            continue

        fns = {"lazybgen": case["lazybgen"], "bgen": case["bgen"]}
        read_before = _disk_read_mb(proc)
        if args.interleave:
            results = measure_interleaved(fns, args.num_runs, args.warmup, proc, before=case["before"])
        else:
            results = {
                name: measure(fn, args.num_runs, args.warmup, proc, before=case["before"])
                for name, fn in fns.items()
                if fn is not None
            }
        read_after = _disk_read_mb(proc)
        _close_all(case)

        lz, bg = results.get("lazybgen"), results.get("bgen")
        record["results"] = results
        if read_before is not None and read_after is not None:
            # A warm-cache claim is only true if the timed runs read (almost)
            # nothing from disk. A large number here means the fixture no longer
            # fits alongside the output and the run was partly I/O bound.
            record["disk_read_mb"] = read_after - read_before
        speedup = ""
        if lz and bg and lz.get("ok") and bg.get("ok") and lz["median_time"] > 0:
            record["speedup"] = bg["median_time"] / lz["median_time"]
            speedup = f"  ({record['speedup']:.2f}x)"
        print(f"  {case['name']:44s} lazybgen {_fmt(lz):>10s}   bgen {_fmt(bg):>10s}{speedup}{_fmt_cold(lz)}")
        if record.get("disk_read_mb", 0) > 64:
            print(
                f"      note: {record['disk_read_mb']:.0f} MB read from disk during this case; not a warm-cache number"
            )
        rows.append(record)
    return rows


def _close_all(case):
    for closer in case.get("closers") or ():
        try:
            closer()
        except Exception:
            pass


def check_headroom(cases, args):
    """Warn when a case's fixture plus its output cannot both stay resident."""
    available = psutil.virtual_memory().total / (1024**3)
    worst = max((c["output_gb"] for c in cases), default=0.0)
    if worst > 0.5 * available:
        print(
            f"WARNING: the largest selected output is {worst:.1f} GB on a {available:.0f} GB host. "
            "Above roughly half of RAM the output competes with the page cache holding the fixture, "
            "and the timings become noisy rather than warm. Lower --max-output-gb or use --dtype float32."
        )


def parse_float_list(text):
    return [float(x) for x in text.split(",") if x.strip()]


def parse_int_list(text):
    return [int(x) for x in text.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser(description="Upscaled lazybgen vs bgen benchmark (heavier workload shapes)")
    p.add_argument("--data-dir", default="benchmarks/test_data")
    p.add_argument("--output", default="benchmarks/compare_upscaled_results.json")
    p.add_argument("--suite", default="region_scaling,cohort,stream", help=f"comma-separated; one of {SUITES} or 'all'")
    p.add_argument("--list", action="store_true", help="list the suites and exit")
    p.add_argument("--num-runs", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--sizes", default=None, help="comma-separated size labels to keep (default: all in the ladder)")
    p.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    p.add_argument(
        "--max-output-gb",
        type=float,
        default=8.0,
        help="Skip a case whose output matrix exceeds this. The default suits a 64 GB host; raise it "
        "on a high-memory host to reach the largest region and full reads.",
    )
    p.add_argument("--region-sizes", default=",".join(str(v) for v in REGION_LADDER))
    p.add_argument("--cohort-fractions", default="0.01,0.1,0.5")
    p.add_argument("--cohort-variants", type=int, default=1000, help="variants read per cohort case")
    p.add_argument(
        "--cohort-order",
        choices=["shuffled", "sorted"],
        default="shuffled",
        help="Order of the requested cohort IDs. 'shuffled' gathers rows from all over each column "
        "(what a real cohort list does); 'sorted' makes the gather a forward scan.",
    )
    p.add_argument("--stream-variants", type=int, default=0, help="variants to stream; 0 streams the whole file")
    p.add_argument("--index-variants", type=int, default=100000, help="large region width for the index suite")
    p.add_argument("--point-queries", type=int, default=500)
    p.add_argument("--thread-counts", default="1,2,4,8,16")
    p.add_argument("--thread-region-variants", type=int, default=2000)
    p.add_argument(
        "--no-interleave",
        dest="interleave",
        action="store_false",
        help="Finish every run of one library before starting the other. Interleaved by default here, "
        "so drift over a long sweep is not charged to whichever library ran later.",
    )
    p.add_argument(
        "--malloc-regime",
        choices=["default", "mmap-always"],
        default="default",
        help="Allocator regime for output matrices; see compare_libraries.set_malloc_regime.",
    )
    p.set_defaults(interleave=True)
    args = p.parse_args()

    if args.list:
        for name in SUITES:
            print(name)
        return

    args.dtype = np.float64 if args.dtype == "float64" else np.float32
    args.region_sizes = parse_int_list(args.region_sizes)
    args.cohort_fractions = parse_float_list(args.cohort_fractions)
    args.thread_counts = parse_int_list(args.thread_counts)

    requested = SUITES if args.suite == "all" else tuple(s.strip() for s in args.suite.split(",") if s.strip())
    unknown = [s for s in requested if s not in SUITE_BUILDERS]
    if unknown:
        p.error(f"unknown suite(s): {', '.join(unknown)}; choose from {', '.join(SUITES)}")

    sizes = SIZES
    if args.sizes:
        keep = {s.strip() for s in args.sizes.split(",")}
        sizes = [s for s in SIZES if s[0] in keep]
        if not sizes:
            p.error(f"--sizes matched nothing; ladder labels are {[s[0] for s in SIZES]}")

    malloc_note = set_malloc_regime(args.malloc_regime)
    cache_clear = sample_cache_clearer()
    print(f"allocator: {malloc_note}")
    print(
        "sample-file cache: "
        + (
            "present; cleared before each run of the cases that open a reader"
            if cache_clear
            else "not present in this build"
        )
    )

    proc = psutil.Process()
    out = {
        "timestamp": datetime.now().isoformat(),
        "environment": environment_snapshot(),
        "num_runs": args.num_runs,
        "warmup": args.warmup,
        "dtype": np.dtype(args.dtype).name,
        "max_output_gb": args.max_output_gb,
        "interleaved": args.interleave,
        "malloc_regime": malloc_note,
        "sample_cache_cleared": cache_clear is not None,
        "suites": {},
    }

    for name in requested:
        print(f"\n=== {name} ===")
        cases = SUITE_BUILDERS[name](args, sizes)
        if not cases:
            print("  no fixtures found for this suite; skipped")
            out["suites"][name] = []
            continue
        check_headroom(cases, args)
        out["suites"][name] = run_cases(cases, args, proc)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
