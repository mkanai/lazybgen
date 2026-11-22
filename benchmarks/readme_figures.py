#!/usr/bin/env python3
"""The README figures that the library ladder does not produce.

1. float32 against float64: a full decode and a 2000-variant region at 100k
   samples x 10k variants (each the fastest of 3 warm calls in a fresh process,
   median over 3 rounds that alternate the order).
2. 500 point lookups at 5000 samples, per lookup: one ``load_bgen`` call per
   position (which reopens the file each time), one ``load_variants`` call per
   position on a single open ``BgenReader``, the same 500 positions as one
   ``variant_filter``, and ``iter_variants`` over the range they span (per
   variant streamed). Fastest of 3 after a warmup, one process.
3. A one-variant read in a fresh process at 500k samples, lazybgen against the
   `bgen` package. Each library is imported and its reader function built
   before the timer starts, so the figure is open plus read, and the median of
   5 processes each is reported.

Run from the repository root with lazybgen built in place and the ladder
fixtures generated (``generate_test_bgen.py --mode ladder``):

    python benchmarks/readme_figures.py --data-dir benchmarks/test_data
"""

import argparse
import statistics
import subprocess
import sys

HDR = r"""
import sys, time, numpy as np
sys.path.insert(0, "benchmarks")
import lazybgen
from compare_libraries import fixture_paths, region_bounds, scattered_positions, single_position, CHROM
"""

DTYPE = HDR + r"""
bf, sf = fixture_paths(sys.argv[1], 100000, 10000)
kw = {"sample_path": str(sf), "nan_action": "mean", "dtype": np.float32 if sys.argv[2] == "f32" else np.float64}
if sys.argv[3] == "region":
    s, e = region_bounds(10000, 2000)
    kw["region"] = f"{CHROM}:{s}-{e}"
lazybgen.load_bgen(str(bf), **kw)
ts = []
for _ in range(3):
    t = time.perf_counter(); lazybgen.load_bgen(str(bf), **kw); ts.append(time.perf_counter() - t)
print(min(ts))
"""

POINTS = HDR + r"""
from lazybgen.reader import BgenReader
bf, sf = fixture_paths(sys.argv[1], 5000, 10000)
pos = scattered_positions(10000, 500)
vf = {"chromosome": CHROM, "positions": pos, "allele1": ["A"] * len(pos), "allele2": ["G"] * len(pos)}
def load_bgen_loop():
    for p in pos:
        lazybgen.load_bgen(str(bf), sample_path=str(sf), region=f"{CHROM}:{p}-{p}", nan_action="mean")
def open_reader_loop():
    with BgenReader(str(bf), sample_path=str(sf)) as r:
        for p in pos:
            r.load_variants(region_chrom=CHROM, region_start=p, region_end=p)
def batch():
    lazybgen.load_bgen(str(bf), sample_path=str(sf), variant_filter=vf, nan_action="mean")
N = [0]
def stream():
    N[0] = 0
    with BgenReader(str(bf), sample_path=str(sf)) as r:
        for _ in r.iter_variants(region_chrom=CHROM, region_start=min(pos), region_end=max(pos)):
            N[0] += 1
res = {}
fns = {"load_bgen_loop": load_bgen_loop, "open_reader_loop": open_reader_loop, "batch": batch, "stream": stream}
for name, f in fns.items():
    f(); ts = []
    for _ in range(3):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    res[name] = min(ts)
per_lookup = [res[k] / len(pos) * 1e6 for k in ("load_bgen_loop", "open_reader_loop", "batch")]
print(*per_lookup, res["stream"] / N[0] * 1e6, N[0])
"""

FRESH = HDR + r"""
import pathlib
from compare_libraries import bgen_workloads, lazybgen_workloads
bf, sf = fixture_paths(sys.argv[1], 500000, 10000)
if sys.argv[2] == "lazybgen":
    fn = lazybgen_workloads(str(bf), str(sf), 10000)["single"]
else:
    import bgen  # noqa: F401  (imported before the timer, as lazybgen is)
    fn = bgen_workloads(pathlib.Path(bf), 500000, 10000)["single"]
t = time.perf_counter(); fn(); print(time.perf_counter() - t)
"""


def run(code, *args):
    proc = subprocess.run([sys.executable, "-c", code, *args], capture_output=True, text=True)
    if proc.returncode:
        raise SystemExit(proc.stderr[-800:])
    return proc.stdout.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", default="benchmarks/test_data")
    args = ap.parse_args()
    d = args.data_dir

    times = {}
    for rnd in range(3):
        for w in ("full", "region"):
            for dt in ("f64", "f32") if rnd % 2 == 0 else ("f32", "f64"):
                times.setdefault(f"{w}|{dt}", []).append(float(run(DTYPE, d, dt, w)))
    for w in ("full", "region"):
        f64, f32 = statistics.median(times[f"{w}|f64"]), statistics.median(times[f"{w}|f32"])
        print(f"dtype {w} @100k: float64 {f64 * 1e3:.0f} ms, float32 {f32 * 1e3:.0f} ms", flush=True)

    loop, reader_loop, batch, stream, n = map(float, run(POINTS, d).split())
    print(
        f"500 point lookups @5k, per lookup: load_bgen loop {loop:.0f} us, open-reader loop {reader_loop:.0f} us, "
        f"one variant_filter {batch:.1f} us ({loop / batch:.0f}x and {reader_loop / batch:.0f}x faster); "
        f"iter_variants {stream:.1f} us per variant over {n:.0f}",
        flush=True,
    )

    fresh = {}
    for _ in range(5):
        for lib in ("lazybgen", "bgen"):
            fresh.setdefault(lib, []).append(float(run(FRESH, d, lib)))
    print(
        "fresh-process single @500k (open + read, median of 5): "
        + ", ".join(f"{lib} {statistics.median(v) * 1e3:.0f} ms" for lib, v in fresh.items()),
        flush=True,
    )


if __name__ == "__main__":
    main()
