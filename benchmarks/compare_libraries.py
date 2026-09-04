#!/usr/bin/env python3
"""Head-to-head benchmark: lazybgen vs the `bgen` package on the same files.

Times four reader workloads that both libraries can serve through their own
idiomatic API, so the comparison reflects how each one is actually used:

  full_decode   every variant x every sample
  region        a contiguous position range (one locus)
  scattered     random-access lookup of variants spread across the file
  single        a point query of one variant

lazybgen materializes a (samples, variants) matrix via ``load_bgen``; the `bgen`
package exposes one variant at a time, so its equivalent fills a matrix of the
same element count and dtype one variant at a time from ``alt_dosage``. Both
therefore allocate and populate the same number of float64 dosages, which keeps
wall time and peak RSS comparable.

The same harness also benchmarks lazybgen on remote (``gs://``) copies of the
files (the `bgen` package has no remote-read path), to show that the partial-read
workloads stay cheap when only a slice is fetched over the network.

Local timings are page-cache-warm (a warmup pass precedes the measured runs), so
they isolate decode/parse cost rather than first-touch disk I/O. The median of
``--num-runs`` runs is reported; peak RSS is sampled in a background thread.
"""

import argparse
import gc
import json
import statistics
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil

# Synthetic fixture layout (generate_test_bgen.py): one chromosome, evenly
# spaced positions, biallelic A/G.
CHROM = "chr1"
POS_STRIDE = 1000

# Size ladder (label, samples, variants); all zlib / 8-bit. Variants fixed at
# 10k; samples scale from 500 to 500k (biobank-scale headline at 500k x 10k).
SIZES = [
    ("500 x 10k", 500, 10000),
    ("1k x 10k", 1000, 10000),
    ("5k x 10k", 5000, 10000),
    ("10k x 10k", 10000, 10000),
    ("50k x 10k", 50000, 10000),
    ("100k x 10k", 100000, 10000),
    ("500k x 10k", 500000, 10000),
]

# Variants read by the slice workloads. Sized so the measured read stays well
# clear of timer and scheduler noise now that the reader is fast: at 500 and 200
# a 5k-sample region read was ~6 ms, where a few hundred microseconds of jitter
# is a visible "regression". These land the same read at tens of ms small and
# ~1 s at 500k samples. They are the ladder's constant, so a run that changes
# them is not comparable with one that does not; --region-variants and
# --scattered-variants exist for quick local runs, not for published numbers.
REGION_VARIANTS = 2000  # contiguous block for the `region` workload
SCATTERED_VARIANTS = 1000  # spread-out lookups for the `scattered` workload

# Skip the full_decode workload when its (variants x samples) f64 output matrix
# would exceed this; full materialization is not a realistic workload at that
# scale (500k x 10k = 40 GB), and the slice workloads are what people run there.
MAX_FULL_GB = 24.0


def full_matrix_gb(samples, variants):
    return variants * samples * 8 / (1024**3)


def fixture_paths(data_dir, samples, variants):
    stem = f"test_{samples}s_{variants}v_zlib_8bit"
    return Path(data_dir) / f"{stem}.bgen", Path(data_dir) / f"{stem}.sample"


def region_bounds(variants, count):
    count = max(1, min(count, variants))
    start_idx = (variants - count) // 2
    start_pos = (start_idx + 1) * POS_STRIDE
    end_pos = (start_idx + count) * POS_STRIDE
    return start_pos, end_pos


def scattered_positions(variants, count):
    count = max(1, min(count, variants))
    step = max(1, variants // count)
    idxs = list(range(0, variants, step))[:count]
    return [(i + 1) * POS_STRIDE for i in idxs]


def single_position(variants):
    return ((variants // 2) + 1) * POS_STRIDE


def _vm_hwm_bytes():
    """This process's peak RSS as the kernel recorded it, or None if unavailable."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _reset_vm_hwm():
    """Reset the kernel's peak-RSS mark so the next reading covers one run only."""
    try:
        with open("/proc/self/clear_refs", "w") as fh:
            fh.write("5")
        return True
    except OSError:
        return False


class PeakRSS:
    """Peak RSS (MiB) over the enclosed block.

    Prefers the kernel's own high-water mark, reset on entry, because sampling
    from a Python thread cannot see a peak that happens while the measured code
    holds the GIL: a decode that never yields keeps the sampler off the CPU for
    its whole duration, and the reading comes back near the pre-run baseline. On
    a platform without /proc, this falls back to sampling and is subject to that.
    """

    def __init__(self, proc, interval=0.005):
        self.proc = proc
        self.interval = interval
        self.peak = 0
        self._stop = False
        self._t = None
        self._kernel = False

    def __enter__(self):
        self.peak = self.proc.memory_info().rss
        if _reset_vm_hwm() and _vm_hwm_bytes() is not None:
            self._kernel = True
            return self
        self._stop = False
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        while not self._stop:
            try:
                self.peak = max(self.peak, self.proc.memory_info().rss)
            except Exception:
                pass
            time.sleep(self.interval)

    def __exit__(self, *exc):
        if self._kernel:
            hwm = _vm_hwm_bytes()
            if hwm is not None:
                self.peak = max(self.peak, hwm)
            return False
        self._stop = True
        if self._t is not None:
            self._t.join(timeout=1.0)
        try:
            self.peak = max(self.peak, self.proc.memory_info().rss)
        except Exception:
            pass
        return False

    @property
    def peak_mb(self):
        return self.peak / (1024**2)


# ---------------------------------------------------------------------------
# lazybgen workloads (one materialized matrix per call)
# ---------------------------------------------------------------------------
def lazybgen_workloads(bf, sf, variants):
    import lazybgen

    rstart, rend = region_bounds(variants, REGION_VARIANTS)
    region = f"{CHROM}:{rstart}-{rend}"
    spos = scattered_positions(variants, SCATTERED_VARIANTS)
    vfilter = {
        "chromosome": CHROM,
        "positions": spos,
        "allele1": ["A"] * len(spos),
        "allele2": ["G"] * len(spos),
    }
    spt = single_position(variants)
    single = f"{CHROM}:{spt}-{spt}"

    def full():
        return lazybgen.load_bgen(bf, sample_path=sf, nan_action="mean")[0]

    def region_w():
        return lazybgen.load_bgen(bf, sample_path=sf, region=region, nan_action="mean")[0]

    def scattered():
        return lazybgen.load_bgen(bf, sample_path=sf, variant_filter=vfilter, nan_action="mean")[0]

    def single_w():
        return lazybgen.load_bgen(bf, sample_path=sf, region=single, nan_action="mean")[0]

    return {"full_decode": full, "region": region_w, "scattered": scattered, "single": single_w}


# ---------------------------------------------------------------------------
# bgen-package workloads (fill the same matrix row by row from alt_dosage)
# ---------------------------------------------------------------------------
def bgen_workloads(bf, samples, variants):
    from bgen import BgenReader

    rstart, rend = region_bounds(variants, REGION_VARIANTS)
    spos = scattered_positions(variants, SCATTERED_VARIANTS)
    spt = single_position(variants)

    def full():
        r = BgenReader(str(bf))
        try:
            out = np.empty((variants, samples), dtype=np.float64)
            for i, v in enumerate(r):
                out[i] = v.alt_dosage
            return out
        finally:
            r.close()

    def region_w():
        r = BgenReader(str(bf))
        try:
            rows = [v.alt_dosage for v in r.fetch(CHROM, rstart, rend)]
            return np.asarray(rows)
        finally:
            r.close()

    def scattered():
        r = BgenReader(str(bf))
        try:
            out = np.empty((len(spos), samples), dtype=np.float64)
            for i, p in enumerate(spos):
                out[i] = r.at_position(p)[0].alt_dosage
            return out
        finally:
            r.close()

    def single_w():
        r = BgenReader(str(bf))
        try:
            return np.asarray(r.at_position(spt)[0].alt_dosage)
        finally:
            r.close()

    return {"full_decode": full, "region": region_w, "scattered": scattered, "single": single_w}


def minor_faults():
    """This process's cumulative minor page-fault count, or None if unavailable.

    Differencing it across a run counts the pages the run first-touched, which is
    what separates "the allocator handed back a warm block" from "this run faulted
    in a fresh mapping". A large output matrix is first-touched by whichever
    thread writes each page, so the delta tracks a cost that wall time alone
    attributes to decoding.
    """
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_minflt
    except Exception:
        return None


def _timed_call(fn, proc):
    """Run fn once, returning (wall seconds, peak RSS MiB, minor-fault delta)."""
    faults_before = minor_faults()
    with PeakRSS(proc) as rss:
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
    faults_after = minor_faults()
    faults = None if faults_before is None or faults_after is None else faults_after - faults_before
    return dt, rss.peak_mb, faults


def _summarize(times, mems, faults, first):
    median = statistics.median(times)
    out = {
        "ok": True,
        "median_time": median,
        "min_time": min(times),
        "max_time": max(times),
        # Spread of the measured runs as a fraction of the median. A speedup
        # smaller than this is not a result, and recording it means that can be
        # checked afterwards instead of assumed.
        "spread": (max(times) - min(times)) / median if median > 0 else None,
        "peak_memory_mb": statistics.median(mems),
    }
    known = [f for f in faults if f is not None]
    if known:
        out["minor_faults"] = statistics.median(known)
    if first is not None:
        out["first_call"] = first
    return out


def measure(fn, num_runs, warmup, proc, before=None):
    """Time fn: `warmup` untimed-for-the-median passes, then `num_runs` measured.

    The reported time is the median of the measured runs, which is a warm-loop
    number: the page cache is hot and the allocator is likely handing back the
    block the previous run freed. The first warmup pass is timed separately and
    reported as ``first_call`` so that regime is visible rather than assumed.

    ``before`` runs before each pass and outside the timer, for per-run setup
    such as clearing a cache the workload would otherwise reuse.
    """
    first = None
    for i in range(warmup):
        try:
            if before is not None:
                before()
            dt, peak, faults = _timed_call(fn, proc)
            if i == 0:
                first = {"time": dt, "peak_memory_mb": peak, "minor_faults": faults}
        except Exception as e:
            return {"ok": False, "error": repr(e)}
        gc.collect()
    times, mems, faults_seen = [], [], []
    for _ in range(num_runs):
        try:
            if before is not None:
                before()
            dt, peak, faults = _timed_call(fn, proc)
            times.append(dt)
            mems.append(peak)
            faults_seen.append(faults)
            gc.collect()
        except Exception as e:
            return {"ok": False, "error": repr(e)}
    return _summarize(times, mems, faults_seen, first)


def measure_interleaved(fns, num_runs, warmup, proc, before=None):
    """Time several callables round-robin, one pass each per repetition.

    ``measure`` finishes every run of one callable before starting the next, so a
    machine that drifts over the course of a sweep charges the drift to whichever
    callable ran later. Alternating them spreads any drift across all of them,
    which matters when two libraries are being compared on the same workload.

    Takes ``{name: callable}`` (a None callable is skipped) and returns
    ``{name: result}`` in the same shape ``measure`` returns.
    """
    names = [n for n, f in fns.items() if f is not None]
    state = {n: {"times": [], "mems": [], "faults": [], "first": None, "error": None} for n in names}
    for rep in range(warmup + num_runs):
        measured = rep >= warmup
        for n in names:
            st = state[n]
            if st["error"] is not None:
                continue
            try:
                if before is not None:
                    before()
                dt, peak, faults = _timed_call(fns[n], proc)
            except Exception as e:
                st["error"] = repr(e)
                continue
            if rep == 0:
                st["first"] = {"time": dt, "peak_memory_mb": peak, "minor_faults": faults}
            if measured:
                st["times"].append(dt)
                st["mems"].append(peak)
                st["faults"].append(faults)
            gc.collect()
    results = {}
    for n in names:
        st = state[n]
        if st["error"] is not None or not st["times"]:
            results[n] = {"ok": False, "error": st["error"] or "no measured runs"}
        else:
            results[n] = _summarize(st["times"], st["mems"], st["faults"], st["first"])
    return results


# ---------------------------------------------------------------------------
# Allocator regime
# ---------------------------------------------------------------------------
# glibc mallopt parameter numbers (malloc.h). M_MMAP_THRESHOLD decides when a
# large allocation becomes its own mmap instead of coming out of the heap.
M_TRIM_THRESHOLD = -1
M_MMAP_THRESHOLD = -3


def set_malloc_regime(regime):
    """Pin how large output matrices are allocated. Returns a description.

    glibc's mmap threshold is dynamic: it starts at 128 KB and grows, up to
    32 MB, as freed mmap'd blocks are seen. So in a warm loop an output under
    32 MB is usually recycled from the heap and never pays first-touch, while a
    larger one gets a fresh mapping and faults in every page. Which side of that
    line a workload lands on depends on the process's allocation history, so the
    same call can be timed in two different regimes depending on what ran before
    it.

    ``default`` leaves glibc alone (what an application sees). ``mmap-always``
    sets the threshold and the trim threshold to 0, so every large output is a
    fresh mapping and every run pays first-touch: the one-shot cost, measured
    repeatably. Returns a note describing what was applied.

    ``mmap-always`` is not neutral in a head-to-head. It charges every allocation
    to the allocator, so a reader that allocates once per matrix and one that
    allocates once per variant are penalized very differently. Use it to compare
    lazybgen against itself across sizes or builds; use ``default`` for the
    library-vs-library tables.
    """
    if regime == "default":
        return "default (glibc dynamic mmap threshold; warm loops may recycle heap blocks)"
    if regime != "mmap-always":
        raise ValueError(f"unknown malloc regime: {regime}")
    import ctypes

    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        ok_mmap = libc.mallopt(M_MMAP_THRESHOLD, 0)
        ok_trim = libc.mallopt(M_TRIM_THRESHOLD, 0)
    except Exception as e:
        return f"mmap-always requested but mallopt is unavailable ({e!r}); running under the default regime"
    if not ok_mmap or not ok_trim:
        return "mmap-always requested but mallopt refused it; running under the default regime"
    return "mmap-always (every large output is a fresh mapping; each run pays first-touch)"


def environment_snapshot():
    """Machine facts that change what a benchmark number means."""
    vm = psutil.virtual_memory()
    snap = {
        "cpu_count": psutil.cpu_count(),
        "memory_total_gb": vm.total / (1024**3),
        "memory_available_gb": vm.available / (1024**3),
        "page_cache_gb": (getattr(vm, "cached", 0) or 0) / (1024**3),
    }
    try:
        import os

        snap["load_avg_1m"] = os.getloadavg()[0]
    except Exception:
        pass
    return snap


def run_local(data_dir, num_runs, warmup, max_full_gb=MAX_FULL_GB, interleave=False):
    proc = psutil.Process()
    results = []
    for label, samples, variants in SIZES:
        bf, sf = fixture_paths(data_dir, samples, variants)
        if not bf.exists():
            print(f"  skip {label}: {bf.name} not found")
            continue
        size_mb = bf.stat().st_size / (1024**2)
        print(f"\n[{label}] {samples}s x {variants}v ({size_mb:.0f} MB)")
        lz = lazybgen_workloads(str(bf), str(sf), variants)
        bg = bgen_workloads(bf, samples, variants)
        row = {"size": label, "samples": samples, "variants": variants, "file_size_mb": size_mb, "workloads": {}}
        workloads = ("full_decode", "region", "scattered", "single")
        if full_matrix_gb(samples, variants) > max_full_gb:
            workloads = ("region", "scattered", "single")
            print(f"  skip full_decode: {full_matrix_gb(samples, variants):.0f} GB matrix > {max_full_gb:.0f} GB")
        for w in workloads:
            if interleave:
                paired = measure_interleaved({"lazybgen": lz[w], "bgen": bg[w]}, num_runs, warmup, proc)
                lz_m, bg_m = paired["lazybgen"], paired["bgen"]
            else:
                lz_m = measure(lz[w], num_runs, warmup, proc)
                bg_m = measure(bg[w], num_runs, warmup, proc)
            row["workloads"][w] = {"lazybgen": lz_m, "bgen": bg_m}
            lt = f"{lz_m['median_time']*1000:.1f}ms" if lz_m["ok"] else "FAIL"
            bt = f"{bg_m['median_time']*1000:.1f}ms" if bg_m["ok"] else "FAIL"
            spd = ""
            if lz_m["ok"] and bg_m["ok"] and lz_m["median_time"] > 0:
                spd = f"  ({bg_m['median_time']/lz_m['median_time']:.2f}x)"
            print(f"  {w:14s} lazybgen {lt:>10s}   bgen {bt:>10s}{spd}")
        results.append(row)
    return results


def run_remote(bucket, files, num_runs, warmup, max_full_gb=MAX_FULL_GB):
    """lazybgen-only remote workloads (region / scattered / single / full)."""
    import lazybgen

    proc = psutil.Process()
    results = []
    for label, samples, variants in files:
        stem = f"test_{samples}s_{variants}v_zlib_8bit"
        url = f"{bucket.rstrip('/')}/{stem}.bgen"
        print(f"\n[remote {label}] {url}")
        rstart, rend = region_bounds(variants, REGION_VARIANTS)
        region = f"{CHROM}:{rstart}-{rend}"
        spos = scattered_positions(variants, SCATTERED_VARIANTS)
        vfilter = {
            "chromosome": CHROM,
            "positions": spos,
            "allele1": ["A"] * len(spos),
            "allele2": ["G"] * len(spos),
        }
        spt = single_position(variants)
        single = f"{CHROM}:{spt}-{spt}"

        wl = {
            "full_decode": lambda: lazybgen.load_bgen(url, nan_action="mean")[0],
            "region": lambda: lazybgen.load_bgen(url, region=region, nan_action="mean")[0],
            "scattered": lambda: lazybgen.load_bgen(url, variant_filter=vfilter, nan_action="mean")[0],
            "single": lambda: lazybgen.load_bgen(url, region=single, nan_action="mean")[0],
        }
        row = {"size": label, "samples": samples, "variants": variants, "url": url, "workloads": {}}
        workloads = ("full_decode", "region", "scattered", "single")
        if full_matrix_gb(samples, variants) > max_full_gb:
            workloads = ("region", "scattered", "single")
            print(f"  skip full_decode: {full_matrix_gb(samples, variants):.0f} GB matrix > {max_full_gb:.0f} GB")
        for w in workloads:
            m = measure(wl[w], num_runs, warmup, proc)
            row["workloads"][w] = {"lazybgen": m}
            mt = f"{m['median_time']*1000:.1f}ms" if m["ok"] else f"FAIL ({m.get('error','')[:60]})"
            print(f"  {w:14s} lazybgen {mt:>12s}")
        results.append(row)
    return results


def main():
    # Rebound from the CLI below; declared here because the argparse defaults
    # read the module-level values first.
    global REGION_VARIANTS, SCATTERED_VARIANTS
    p = argparse.ArgumentParser(description="lazybgen vs bgen head-to-head benchmark")
    p.add_argument("--data-dir", default="benchmarks/test_data")
    p.add_argument("--output", default="benchmarks/compare_results.json")
    p.add_argument("--num-runs", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--remote-bucket", default=None, help="gs:// prefix holding uploaded fixtures (enables remote run)")
    p.add_argument(
        "--remote-warmup", type=int, default=0, help="warmup passes for the remote run (default 0: measure cold)"
    )
    p.add_argument("--skip-local", action="store_true")
    p.add_argument(
        "--region-variants",
        type=int,
        default=REGION_VARIANTS,
        help=f"variants in the contiguous `region` read (default {REGION_VARIANTS})",
    )
    p.add_argument(
        "--scattered-variants",
        type=int,
        default=SCATTERED_VARIANTS,
        help=f"variants in the spread-out `scattered` read (default {SCATTERED_VARIANTS})",
    )
    p.add_argument(
        "--max-full-gb",
        type=float,
        default=MAX_FULL_GB,
        help="Skip the full_decode workload when its output matrix exceeds this many GB "
        f"(default {MAX_FULL_GB:.0f}). Raise it on a high-memory host to measure the "
        "large full materializations that would otherwise hit the RAM ceiling.",
    )
    p.add_argument(
        "--interleave",
        action="store_true",
        help="Alternate the two libraries run by run instead of finishing one before starting the "
        "other, so machine drift over a long sweep is not charged to whichever ran later. Off by "
        "default: the published tables were measured without it.",
    )
    p.add_argument(
        "--malloc-regime",
        choices=["default", "mmap-always"],
        default="default",
        help="Allocator regime for output matrices. 'default' leaves glibc alone, so a warm loop "
        "may recycle a heap block and skip first-touch. 'mmap-always' makes every large output a "
        "fresh mapping, so every run pays first-touch (the one-shot cost).",
    )
    args = p.parse_args()

    malloc_note = set_malloc_regime(args.malloc_regime)
    print(f"allocator: {malloc_note}")

    out = {
        "timestamp": datetime.now().isoformat(),
        "system": {"cpu": psutil.cpu_count(), "memory_gb": psutil.virtual_memory().total / (1024**3)},
        "environment": environment_snapshot(),
        "num_runs": args.num_runs,
        "max_full_gb": args.max_full_gb,
        "interleaved": args.interleave,
        "malloc_regime": malloc_note,
        "local": [],
        "remote": [],
    }

    REGION_VARIANTS = args.region_variants
    SCATTERED_VARIANTS = args.scattered_variants
    out["workload_variants"] = {"region": REGION_VARIANTS, "scattered": SCATTERED_VARIANTS}

    if not args.skip_local:
        print("=== LOCAL: lazybgen vs bgen ===")
        out["local"] = run_local(args.data_dir, args.num_runs, args.warmup, args.max_full_gb, args.interleave)

    if args.remote_bucket:
        print("\n=== REMOTE: lazybgen only ===")
        # Remote ladder: a large and a biobank-scale file (partial reads stay cheap
        # while the whole-file download grows with sample count).
        remote_files = [("100k x 10k", 100000, 10000), ("500k x 10k", 500000, 10000)]
        out["remote"] = run_remote(
            args.remote_bucket, remote_files, args.num_runs, args.remote_warmup, args.max_full_gb
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
