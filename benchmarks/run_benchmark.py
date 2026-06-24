#!/usr/bin/env python3
"""
Benchmark the lazybgen reader across a matrix of synthetic BGEN files.

Measures the reader directly, not a downstream CLI. The reader is the I/O
bottleneck in downstream tasks, so timing it in isolation is what we want for
optimizing lazybgen itself.

Each configuration runs a set of reader workloads spanning the documented BGEN
IO paths:

  load_full / load_full_f32         all variants x all samples (f64 / f32)
  load_region / load_region_small   contiguous slice (50% / a few of many)
  load_variant_filter               scattered .z-style variant lookups
  load_single_variant               point query (one variant)
  load_sample_filtered[_small]      all variants x 50% / ~1% sample cohort
  iter_variants / iter_sample_filtered  streaming (all / 50% cohort)
  serial_load / parallel_load_ntN   parallel-decode thread sweep (--threads)

Full-materialization workloads whose output matrix exceeds --max-matrix-gb are
skipped so a large config (e.g. 500K x 5000 = 20 GB) measures the realistic
subset/streaming paths instead of OOMing.

Timing uses a warmup pass plus N measured runs, reporting the median (page
caches are warm, so this isolates decode/parse cost, not first-touch disk I/O).
Peak RSS is sampled in a background thread around each call.

The reader under test is selected by --reader-module / --reader-path, so the
same harness benchmarks lazybgen HEAD, an older lazybgen commit (e.g. the initial
extraction commit as a baseline), or another reader module without code changes.

JSON output is compatible with compare_results.py.
"""

import argparse
import cProfile
import io
import json
import os
import pstats
import shutil
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil


# ---------------------------------------------------------------------------
# Test configuration matrix (mirrors generate_test_bgen.py filenames)
# ---------------------------------------------------------------------------
def get_test_configurations(mode="standard"):
    """Return list of (samples, variants, compression, bits, name) tuples.

    NOTE: lazybgen rejects uncompressed BGEN, so 'nocomp' configs are excluded.
    """
    core_configs = [
        (500, 500, "zlib", 8, "tiny"),
        (1000, 1000, "zlib", 8, "small"),
        (5000, 5000, "zlib", 8, "medium"),
        (10000, 10000, "zlib", 8, "large"),
        (50000, 10000, "zlib", 8, "xlarge"),
        (100000, 10000, "zlib", 8, "xxlarge"),
        (10000, 5000, "zlib", 8, "wide"),
        (5000, 10000, "zlib", 8, "tall"),
        (20000, 2000, "zlib", 8, "extreme_wide"),
        (2000, 20000, "zlib", 8, "extreme_tall"),
    ]
    # Compression / bit-depth impact (fixed 5K x 5K). nocomp omitted on purpose.
    compression_configs = [
        (5000, 5000, "zstd", 8, "medium_zstd"),
        (5000, 5000, "zlib", 16, "medium_16bit"),
        (5000, 5000, "zlib", 32, "medium_32bit"),
    ]
    # UKBB-shaped: many samples, few variants. Nobody loads *full* UKBB (500K x N
    # x 8B is tens of GB); the realistic workload is a region/variant slice over
    # all samples. These are where parallel decompression + the scalar decode
    # kernel dominate - pair with --threads to exercise the parallel load path.
    large_configs = [
        (200000, 500, "zlib", 8, "ukbb_200k"),
        (500000, 300, "zlib", 8, "ukbb_500k"),
        (500000, 300, "zstd", 8, "ukbb_500k_zstd"),
        # Many samples AND many variants: load_full is memory-gated off here, so
        # this measures the realistic subset/streaming/sample-filter paths on a
        # big file (see --max-matrix-gb).
        (500000, 5000, "zlib", 8, "ukbb_wide"),
    ]
    scaling_configs = [
        (1000, 5000, "zlib", 8, "scale_1k"),
        (2000, 5000, "zlib", 8, "scale_2k"),
        (5000, 5000, "zlib", 8, "scale_5k"),
        (10000, 5000, "zlib", 8, "scale_10k"),
        (20000, 5000, "zlib", 8, "scale_20k"),
        (5000, 1000, "zlib", 8, "vscale_1k"),
        (5000, 2000, "zlib", 8, "vscale_2k"),
        (5000, 5000, "zlib", 8, "vscale_5k"),
        (5000, 10000, "zlib", 8, "vscale_10k"),
        (5000, 20000, "zlib", 8, "vscale_20k"),
    ]

    if mode == "quick":
        configs = [c for c in core_configs if c[4] in ("tiny", "small", "medium")]
    elif mode == "standard":
        configs = [c for c in core_configs if not c[4].startswith("extreme")]
    elif mode == "compression":
        configs = [c for c in core_configs if c[4] == "medium"] + compression_configs
    elif mode == "scaling":
        configs = scaling_configs
    elif mode == "large":
        configs = large_configs
    elif mode == "comprehensive":
        configs = core_configs + compression_configs + scaling_configs
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return configs


REGION_PROPORTIONS = {"small": 0.1, "medium": 0.5, "large": 0.9}

# Variant layout produced by generate_test_bgen.py: all chr1, pos = (idx+1)*1000.
CHROM = "chr1"
POS_STRIDE = 1000


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


def load_reader(module_name, reader_path):
    """Import and return the reader module (e.g. lazybgen, or another package
    exposing load_bgen / BgenReader)."""
    if reader_path:
        sys.path.insert(0, reader_path)
    mod = __import__(module_name, fromlist=["load_bgen", "BgenReader"])
    return mod


class BenchmarkRunner:
    def __init__(
        self,
        reader,
        data_dir,
        output_dir,
        num_runs=3,
        warmup=1,
        profile=False,
        mode="standard",
        region_size="medium",
        reader_module="lazybgen",
        reader_path=None,
        threads=None,
        max_matrix_gb=16.0,
    ):
        self.reader = reader
        # Thread counts for the parallel-load sweep (None => sweep disabled).
        self.threads = threads
        # Skip any full-materialization workload whose output matrix would exceed
        # this many GB (e.g. load_full at 500K x 5000 f64 = 20 GB). Subset and
        # streaming workloads, which are the realistic large-scale paths, run
        # regardless.
        self.max_matrix_gb = max_matrix_gb
        self.reader_module = reader_module
        self.reader_path = reader_path
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.num_runs = num_runs
        self.warmup = warmup
        self.profile = profile
        self.mode = mode
        self.region_proportion = REGION_PROPORTIONS.get(region_size, 0.5)
        self.commit = os.environ.get("GIT_COMMIT", "unknown")
        self.process = psutil.Process()
        if self.profile:
            self.profile_dir = self.output_dir / "profiles"
            self.profile_dir.mkdir(parents=True, exist_ok=True)

    # -- system / metadata ---------------------------------------------------
    def get_system_info(self):
        return {
            "cpu": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "platform": os.uname().sysname,
            "reader_module": self.reader_module,
            "reader_file": getattr(self.reader, "__file__", "?"),
        }

    # -- workload construction ----------------------------------------------
    def _region_for(self, num_variants):
        """Middle slice covering region_proportion of the variants."""
        count = max(1, int(num_variants * self.region_proportion))
        start_idx = (num_variants - count) // 2
        end_idx = start_idx + count - 1
        start_pos = (start_idx + 1) * POS_STRIDE
        end_pos = (end_idx + 1) * POS_STRIDE
        return f"{CHROM}:{start_pos}-{end_pos}", count

    def _region_for_count(self, num_variants, count):
        """Middle slice covering exactly `count` variants (clamped)."""
        count = max(1, min(count, num_variants))
        start_idx = (num_variants - count) // 2
        end_idx = start_idx + count - 1
        start_pos = (start_idx + 1) * POS_STRIDE
        end_pos = (end_idx + 1) * POS_STRIDE
        return f"{CHROM}:{start_pos}-{end_pos}", count

    def _variant_filter(self, num_variants, count):
        """A .z-style filter selecting `count` variants spread across the file.

        Mirrors generate_test_bgen.py's layout: all CHROM, pos=(idx+1)*POS_STRIDE,
        alleles A/G. Exercises the scattered BGI find_variants_by_filter path.
        """
        count = max(1, min(count, num_variants))
        step = max(1, num_variants // count)
        idxs = list(range(0, num_variants, step))[:count]
        positions = [(i + 1) * POS_STRIDE for i in idxs]
        return {
            "chromosome": CHROM,
            "positions": positions,
            "allele1": ["A"] * len(positions),
            "allele2": ["G"] * len(positions),
        }, len(positions)

    def _sample_subset(self, sample_file, fraction=0.5):
        """First `fraction` of sample IDs from the .sample file (rows 2+)."""
        with open(sample_file) as f:
            lines = f.readlines()
        ids = [ln.split()[0] for ln in lines[2:] if ln.strip()]
        n = max(1, int(len(ids) * fraction))
        return ids[:n]

    def _matrix_gb(self, n_variants_sel, n_samples_sel, dtype_bytes):
        return n_variants_sel * n_samples_sel * dtype_bytes / (1024**3)

    def build_workloads(self, bgen_file, sample_file, num_variants, num_samples):
        """Return list of (name, callable, worker_snippet, effective_variants).

        Covers the documented BGEN IO paths. Full-materialization
        workloads whose output matrix would exceed --max-matrix-gb are skipped
        (logged) so a large config measures the realistic subset/streaming paths
        instead of OOMing. worker_snippet is a self-contained python -c body used
        for perf recording; callable is used for timing/cProfile.
        """
        load_bgen = self.reader.load_bgen
        BgenReader = self.reader.BgenReader
        bf, sf = str(bgen_file), str(sample_file)
        region, region_count = self._region_for(num_variants)
        small_region, small_region_count = self._region_for_count(num_variants, min(100, num_variants))
        single_region, _ = self._region_for_count(num_variants, 1)
        vfilter, vfilter_count = self._variant_filter(num_variants, min(1000, num_variants))
        subset = self._sample_subset(sample_file)
        subset_small = self._sample_subset(sample_file, fraction=0.01)

        imp = self.reader_module
        path_setup = f"import sys; sys.path.insert(0, {self.reader_path!r})\n" if self.reader_path else ""

        workloads = []
        skipped = []

        def add_matrix(name, n_var_sel, n_samp_sel, dtype_bytes, fn, snippet, eff):
            # Gate full-materialization workloads on the memory budget.
            gb = self._matrix_gb(n_var_sel, n_samp_sel, dtype_bytes)
            if gb > self.max_matrix_gb:
                skipped.append((name, gb))
                return
            workloads.append((name, fn, snippet, eff))

        # --- full materialization (memory-gated) -------------------------------
        # load_full: all variants x all samples, float64.
        add_matrix(
            "load_full",
            num_variants,
            num_samples,
            8,
            lambda: load_bgen(bf, sample_path=sf, nan_action="mean"),
            f"{path_setup}from {imp} import load_bgen\n" f"load_bgen({bf!r}, sample_path={sf!r}, nan_action='mean')\n",
            num_variants,
        )

        # load_full_f32: same but half the output bytes (dtype dimension).
        add_matrix(
            "load_full_f32",
            num_variants,
            num_samples,
            4,
            lambda: load_bgen(bf, sample_path=sf, dtype=np.float32, nan_action="mean"),
            f"{path_setup}import numpy as np\nfrom {imp} import load_bgen\n"
            f"load_bgen({bf!r}, sample_path={sf!r}, dtype=np.float32, nan_action='mean')\n",
            num_variants,
        )

        # load_region: contiguous middle slice (region_proportion).
        add_matrix(
            "load_region",
            region_count,
            num_samples,
            8,
            lambda: load_bgen(bf, sample_path=sf, region=region, nan_action="mean"),
            f"{path_setup}from {imp} import load_bgen\n"
            f"load_bgen({bf!r}, sample_path={sf!r}, region={region!r}, nan_action='mean')\n",
            region_count,
        )

        # load_region_small: a few variants out of many (the realistic large-file
        # "read a subset of variants" path).
        add_matrix(
            "load_region_small",
            small_region_count,
            num_samples,
            8,
            lambda: load_bgen(bf, sample_path=sf, region=small_region, nan_action="mean"),
            f"{path_setup}from {imp} import load_bgen\n"
            f"load_bgen({bf!r}, sample_path={sf!r}, region={small_region!r}, nan_action='mean')\n",
            small_region_count,
        )

        # load_variant_filter: scattered .z-style lookups (BGI find_by_filter).
        add_matrix(
            "load_variant_filter",
            vfilter_count,
            num_samples,
            8,
            lambda: load_bgen(bf, sample_path=sf, variant_filter=vfilter, nan_action="mean"),
            f"{path_setup}from {imp} import load_bgen\n"
            f"load_bgen({bf!r}, sample_path={sf!r}, variant_filter={vfilter!r}, nan_action='mean')\n",
            vfilter_count,
        )

        # load_single_variant: point query (one variant, all samples).
        add_matrix(
            "load_single_variant",
            1,
            num_samples,
            8,
            lambda: load_bgen(bf, sample_path=sf, region=single_region, nan_action="mean"),
            f"{path_setup}from {imp} import load_bgen\n"
            f"load_bgen({bf!r}, sample_path={sf!r}, region={single_region!r}, nan_action='mean')\n",
            1,
        )

        # load_sample_filtered: all variants x 50% sample subset (filtered SIMD).
        add_matrix(
            "load_sample_filtered",
            num_variants,
            len(subset),
            8,
            lambda: load_bgen(bf, sample_path=sf, sample_ids=subset, nan_action="mean"),
            f"{path_setup}from {imp} import load_bgen\n"
            f"load_bgen({bf!r}, sample_path={sf!r}, sample_ids={subset!r}, nan_action='mean')\n",
            num_variants,
        )

        # load_sample_filtered_small: all variants x ~1% cohort (UKBB-like).
        add_matrix(
            "load_sample_filtered_small",
            num_variants,
            len(subset_small),
            8,
            lambda: load_bgen(bf, sample_path=sf, sample_ids=subset_small, nan_action="mean"),
            f"{path_setup}from {imp} import load_bgen\n"
            f"load_bgen({bf!r}, sample_path={sf!r}, sample_ids={subset_small!r}, nan_action='mean')\n",
            num_variants,
        )

        # --- streaming (memory-bounded; not gated) -----------------------------
        if hasattr(BgenReader, "iter_variants"):

            def _iter():
                r = BgenReader(bf, sample_path=sf)
                try:
                    n = 0
                    for _info, _dos in r.iter_variants(block_size=1000):
                        n += 1
                    return n
                finally:
                    close = getattr(r, "close", None)
                    if close:
                        close()

            workloads.append(
                (
                    "iter_variants",
                    _iter,
                    f"{path_setup}from {imp} import BgenReader\n"
                    f"r = BgenReader({bf!r}, sample_path={sf!r})\n"
                    f"sum(1 for _ in r.iter_variants(block_size=1000))\n",
                    num_variants,
                )
            )

            def _iter_sf():
                r = BgenReader(bf, sample_path=sf)
                try:
                    idx = np.arange(0, r.nsamples, 2, dtype=np.int32)  # 50% cohort
                    n = 0
                    for _info, _dos in r.iter_variants(sample_indices=idx, block_size=1000):
                        n += 1
                    return n
                finally:
                    close = getattr(r, "close", None)
                    if close:
                        close()

            workloads.append(
                (
                    "iter_sample_filtered",
                    _iter_sf,
                    f"{path_setup}import numpy as np\nfrom {imp} import BgenReader\n"
                    f"r = BgenReader({bf!r}, sample_path={sf!r})\n"
                    f"idx = np.arange(0, r.nsamples, 2, dtype=np.int32)\n"
                    f"sum(1 for _ in r.iter_variants(sample_indices=idx, block_size=1000))\n",
                    num_variants,
                )
            )

        # Parallel-load thread sweep. Drive BgenReader.load_variants directly so
        # the sweep can pin num_threads per case (all samples, no validation).
        # "serial" is the single-threaded baseline (num_threads=1); "parallel_ntN"
        # varies the worker count. Needs a reader exposing load_variants.
        sweep_gb = self._matrix_gb(num_variants, num_samples, 8)
        if self.threads and hasattr(BgenReader, "load_variants") and sweep_gb <= self.max_matrix_gb:

            def _make_load(nt):
                def _run():
                    r = BgenReader(bf, sample_path=sf, num_threads=nt)
                    try:
                        return r.load_variants()  # all samples, float64
                    finally:
                        close = getattr(r, "close", None)
                        if close:
                            close()

                return _run

            # num_threads is the sole knob: 1 = sequential baseline, N = parallel.
            sweep = [("serial_load", 1)] + [(f"parallel_load_nt{nt}", nt) for nt in self.threads]
            for wname, nt in sweep:
                snippet = (
                    f"{path_setup}from {imp} import BgenReader\n"
                    f"r = BgenReader({bf!r}, sample_path={sf!r}, num_threads={nt})\n"
                    f"r.load_variants()\n"
                )
                workloads.append((wname, _make_load(nt), snippet, num_variants))
        elif self.threads and sweep_gb > self.max_matrix_gb:
            skipped.append(("parallel_load sweep", sweep_gb))

        # Cohort thread sweep: same all-variants load but a ~1% sample subset,
        # exercising the parallel FILTERED decode path (cohort extraction). Its
        # output matrix (num_variants x 1% samples) is much smaller, so this runs
        # at scales where the all-samples sweep is memory-gated off.
        cohort_n = max(1, num_samples // 100)
        cohort_gb = self._matrix_gb(num_variants, cohort_n, 8)
        if self.threads and hasattr(BgenReader, "load_variants") and cohort_gb <= self.max_matrix_gb:

            def _make_cohort(nt):
                def _run():
                    r = BgenReader(bf, sample_path=sf, num_threads=nt)
                    try:
                        idx = np.arange(0, r.nsamples, 100, dtype=np.int32)  # ~1% cohort
                        return r.load_variants(sample_indices=idx)
                    finally:
                        close = getattr(r, "close", None)
                        if close:
                            close()

                return _run

            cohort_sweep = [("serial_cohort", 1)] + [(f"parallel_cohort_nt{nt}", nt) for nt in self.threads]
            for wname, nt in cohort_sweep:
                snippet = (
                    f"{path_setup}import numpy as np\nfrom {imp} import BgenReader\n"
                    f"r = BgenReader({bf!r}, sample_path={sf!r}, num_threads={nt})\n"
                    f"idx = np.arange(0, r.nsamples, 100, dtype=np.int32)\n"
                    f"r.load_variants(sample_indices=idx)\n"
                )
                workloads.append((wname, _make_cohort(nt), snippet, num_variants))

        for name, gb in skipped:
            print(f"  skip {name}: output matrix {gb:.1f} GB > --max-matrix-gb {self.max_matrix_gb:.0f}")

        return workloads

    # -- measurement ---------------------------------------------------------
    def measure(self, name, fn, worker_snippet):
        metrics = {"name": name, "runs": []}
        if self.profile:
            self._profile_one(name, fn, worker_snippet, metrics)
            return metrics

        # warmup (not recorded)
        for _ in range(self.warmup):
            try:
                fn()
            except Exception as e:
                metrics["runs"].append({"success": False, "error": repr(e)})
                metrics["aggregate"] = {"success_rate": 0.0}
                return metrics

        for run in range(self.num_runs):
            try:
                with PeakRSS(self.process) as rss:
                    t0 = time.perf_counter()
                    fn()
                    dt = time.perf_counter() - t0
                metrics["runs"].append(
                    {
                        "run": run + 1,
                        "success": True,
                        "total_time": dt,
                        "peak_memory_mb": rss.peak_mb,
                    }
                )
            except Exception as e:
                metrics["runs"].append({"run": run + 1, "success": False, "error": repr(e)})

        ok = [r for r in metrics["runs"] if r.get("success")]
        if ok and len(ok) == len(metrics["runs"]):
            times = [r["total_time"] for r in ok]
            mems = [r["peak_memory_mb"] for r in ok]
            metrics["aggregate"] = {
                "median_time": statistics.median(times),
                "mean_time": statistics.mean(times),
                "std_time": statistics.pstdev(times) if len(times) > 1 else 0.0,
                "min_time": min(times),
                "max_time": max(times),
                "median_memory_mb": statistics.median(mems),
                "success_rate": 1.0,
            }
        else:
            metrics["aggregate"] = {"success_rate": len(ok) / len(metrics["runs"]) if metrics["runs"] else 0.0}
        return metrics

    def _profile_one(self, name, fn, worker_snippet, metrics):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        prof = self.profile_dir / f"{name}_{ts}_python.prof"
        stats_txt = self.profile_dir / f"{name}_{ts}_python_stats.txt"
        # warmup once so we profile steady-state (caches warm, lazy imports done)
        try:
            fn()
        except Exception as e:
            metrics["runs"].append({"success": False, "error": repr(e)})
            metrics["aggregate"] = {"success_rate": 0.0}
            return

        pr = cProfile.Profile()
        t0 = time.perf_counter()
        pr.enable()
        fn()
        pr.disable()
        dt = time.perf_counter() - t0
        pr.dump_stats(str(prof))
        st = pstats.Stats(str(prof))
        st.strip_dirs().sort_stats("cumulative", "time")
        buf = io.StringIO()
        st.stream = buf
        st.print_stats(50)
        st.print_callers(20)
        stats_txt.write_text(buf.getvalue())

        files = {"python_profile": str(prof), "python_stats": str(stats_txt)}

        # C++-level perf record (needs perf + privileges; best effort)
        if shutil.which("perf"):
            perf_data = self.profile_dir / f"{name}_{ts}_perf.data"
            perf_report = self.profile_dir / f"{name}_{ts}_perf_report.txt"
            worker = self.profile_dir / f"_worker_{name}_{ts}.py"
            worker.write_text(worker_snippet)
            rec = subprocess.run(
                ["perf", "record", "-F", "999", "-g", "-o", str(perf_data), "--", sys.executable, str(worker)],
                capture_output=True,
                text=True,
            )
            if perf_data.exists():
                rep = subprocess.run(
                    ["perf", "report", "--stdio", "--no-children", "-i", str(perf_data)],
                    capture_output=True,
                    text=True,
                )
                perf_report.write_text(rep.stdout)
                files["perf_data"] = str(perf_data)
                files["perf_report"] = str(perf_report)
            else:
                files["perf_error"] = rec.stderr[-2000:]
            try:
                worker.unlink()
            except OSError:
                pass

        metrics["runs"].append({"run": 1, "success": True, "total_time": dt})
        metrics["aggregate"] = {
            "median_time": dt,
            "mean_time": dt,
            "std_time": 0.0,
            "min_time": dt,
            "max_time": dt,
            "success_rate": 1.0,
        }
        metrics["profile_files"] = files

    # -- driver --------------------------------------------------------------
    def run(self):
        results = {
            "commit": self.commit,
            "timestamp": datetime.now().isoformat(),
            "system": self.get_system_info(),
            "mode": self.mode,
            "region_proportion": self.region_proportion,
            "benchmarks": [],
        }
        commit_info_file = Path("/app/COMMIT_INFO.txt")
        if commit_info_file.exists():
            results["commit_info"] = commit_info_file.read_text().strip()

        for samples, variants, compression, bits, name in get_test_configurations(self.mode):
            pattern = f"test_{samples}s_{variants}v_{compression}_{bits}bit"
            bgen = self.data_dir / f"{pattern}.bgen"
            sample = self.data_dir / f"{pattern}.sample"
            if not bgen.exists():
                print(f"  skip {name}: {bgen.name} not found")
                continue

            print(
                f"\n[{name}] {samples}s x {variants}v {compression} {bits}bit "
                f"({bgen.stat().st_size / (1024 ** 2):.0f} MB)"
            )
            cfg = {
                "config": name,
                "samples": samples,
                "variants": variants,
                "compression": compression,
                "bits": bits,
                "file": bgen.name,
                "file_size_mb": bgen.stat().st_size / (1024**2),
                "workflows": {},
            }
            for wname, fn, snippet, eff in self.build_workloads(bgen, sample, variants, samples):
                print(f"  {wname} ...", end=" ", flush=True)
                m = self.measure(wname, fn, snippet)
                agg = m.get("aggregate", {})
                if agg.get("success_rate", 0) >= 1.0:
                    mt = agg["median_time"]
                    m["derived"] = {
                        "variants_per_second": eff / mt if mt > 0 else 0,
                        "mb_per_second": cfg["file_size_mb"] / mt if mt > 0 else 0,
                        "effective_variants": eff,
                    }
                    print(f"{mt:.3f}s")
                else:
                    err = next((r.get("error") for r in m["runs"] if not r.get("success")), "")
                    print(f"FAILED ({err[:80]})")
                cfg["workflows"][wname] = m
            results["benchmarks"].append(cfg)
        return results

    def save(self, results):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        short = self.commit[:8]
        prefix = "profile" if self.profile else "benchmark"
        out = self.output_dir / f"{prefix}_{short}_{ts}.json"
        out.write_text(json.dumps(results, indent=2))
        if not self.profile:
            (self.output_dir / f"benchmark_{short}_latest.json").write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to: {out}")
        return out


def main():
    p = argparse.ArgumentParser(description="Benchmark the lazybgen reader")
    p.add_argument("--data-dir", default="/data/test_data")
    p.add_argument("--output-dir", default="/results")
    p.add_argument("--num-runs", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--profile", action="store_true", help="cProfile (+ perf if available); single measured run")
    p.add_argument(
        "--mode",
        default="standard",
        choices=["quick", "standard", "comprehensive", "compression", "scaling", "large"],
    )
    p.add_argument("--region-size", default="medium", choices=["small", "medium", "large"])
    p.add_argument(
        "--threads",
        default=None,
        help="Comma-separated worker-thread counts for the parallel-load sweep "
        "(e.g. '4,8,16'). Adds serial_load + parallel_load_ntN workloads that "
        "drive BgenReader.load_variants directly. Most informative with --mode large.",
    )
    p.add_argument(
        "--max-matrix-gb",
        type=float,
        default=16.0,
        help="Skip any full-materialization workload whose output matrix exceeds "
        "this many GB (default 16). Subset/streaming workloads always run. Lets a "
        "large config (e.g. 500K x 5000 = 20 GB) measure realistic paths without OOM.",
    )
    p.add_argument(
        "--reader-module",
        default="lazybgen",
        help="Reader package to import (must expose load_bgen / BgenReader)",
    )
    p.add_argument(
        "--reader-path",
        default=None,
        help="Path to prepend to sys.path before importing the reader " "(for benchmarking a checked-out source tree)",
    )
    args = p.parse_args()

    threads = None
    if args.threads:
        threads = [int(x) for x in args.threads.split(",") if x.strip()]

    reader = load_reader(args.reader_module, args.reader_path)
    mode = "profiling" if args.profile else "benchmarking"
    print(
        f"lazybgen {mode}: commit={os.environ.get('GIT_COMMIT', 'unknown')} "
        f"reader={args.reader_module} ({getattr(reader, '__file__', '?')})"
    )
    print(
        f"mode={args.mode} region={args.region_size} "
        f"({REGION_PROPORTIONS[args.region_size] * 100:.0f}%) "
        f"runs={args.num_runs} warmup={args.warmup}"
    )

    runner = BenchmarkRunner(
        reader=reader,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_runs=args.num_runs,
        warmup=args.warmup,
        profile=args.profile,
        mode=args.mode,
        region_size=args.region_size,
        reader_module=args.reader_module,
        reader_path=args.reader_path,
        threads=threads,
        max_matrix_gb=args.max_matrix_gb,
    )
    results = runner.run()
    runner.save(results)

    print("\n=== Summary ===")
    for cfg in results["benchmarks"]:
        print(f"\n{cfg['config']} ({cfg['samples']}x{cfg['variants']} " f"{cfg['compression']} {cfg['bits']}bit):")
        for wname, m in cfg["workflows"].items():
            agg = m.get("aggregate", {})
            if agg.get("success_rate", 0) >= 1.0:
                d = m.get("derived", {})
                print(
                    f"  {wname:22s} {agg['median_time']:.3f}s  "
                    f"{d.get('variants_per_second', 0):.0f} var/s  "
                    f"{d.get('mb_per_second', 0):.1f} MB/s"
                )
            else:
                print(f"  {wname:22s} FAILED")


if __name__ == "__main__":
    main()
