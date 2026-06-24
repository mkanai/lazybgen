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

REGION_VARIANTS = 500  # contiguous block for the `region` workload
SCATTERED_VARIANTS = 200  # spread-out lookups for the `scattered` workload

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


class PeakRSS:
    """Sample this process's RSS in a daemon thread; report the peak (MiB)."""

    def __init__(self, proc, interval=0.005):
        self.proc = proc
        self.interval = interval
        self.peak = 0
        self._stop = False
        self._t = None

    def __enter__(self):
        self.peak = self.proc.memory_info().rss
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


def measure(fn, num_runs, warmup, proc):
    for _ in range(warmup):
        try:
            fn()
        except Exception as e:
            return {"ok": False, "error": repr(e)}
        gc.collect()
    times, mems = [], []
    for _ in range(num_runs):
        try:
            with PeakRSS(proc) as rss:
                t0 = time.perf_counter()
                fn()
                dt = time.perf_counter() - t0
            times.append(dt)
            mems.append(rss.peak_mb)
            gc.collect()
        except Exception as e:
            return {"ok": False, "error": repr(e)}
    return {
        "ok": True,
        "median_time": statistics.median(times),
        "min_time": min(times),
        "peak_memory_mb": statistics.median(mems),
    }


def run_local(data_dir, num_runs, warmup, max_full_gb=MAX_FULL_GB):
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
        "--max-full-gb",
        type=float,
        default=MAX_FULL_GB,
        help="Skip the full_decode workload when its output matrix exceeds this many GB "
        f"(default {MAX_FULL_GB:.0f}). Raise it on a high-memory host to measure the "
        "large full materializations that would otherwise hit the RAM ceiling.",
    )
    args = p.parse_args()

    out = {
        "timestamp": datetime.now().isoformat(),
        "system": {"cpu": psutil.cpu_count(), "memory_gb": psutil.virtual_memory().total / (1024**3)},
        "num_runs": args.num_runs,
        "max_full_gb": args.max_full_gb,
        "local": [],
        "remote": [],
    }

    if not args.skip_local:
        print("=== LOCAL: lazybgen vs bgen ===")
        out["local"] = run_local(args.data_dir, args.num_runs, args.warmup, args.max_full_gb)

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
