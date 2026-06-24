"""Remote (gs://) performance benchmark for the lazybgen reader.

Measures the network/latency regime that run_benchmark.py (warm-cache decode)
does not: cold/warm byte-range reads, GET count, bytes fetched, read
amplification, prefetch knobs, and stream-vs-download strategy comparison.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

DEFAULT_BUCKET = "gs://your-bucket/lazybgen-bench"
# Small public bgen fixtures for --smoke (correctness bucket; tiny, low egress).
SMOKE_URL = "gs://gcs-anndata-test/lazybgen/example.16bits.bgen"


@dataclass(frozen=True)
class RemoteFixture:
    key: str
    local_name: str
    size_class: str


FIXTURES = [
    RemoteFixture("small", "ukbb_500k_300.bgen", "small"),
    RemoteFixture("wide", "test_500000s_5000v_zlib_8bit.bgen", "wide"),
]


def remote_url(bucket: str, name: str) -> str:
    return bucket.rstrip("/") + "/" + name.lstrip("/")


def _coerce(value: str):
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(value)
    except ValueError:
        return value


def parse_storage_options(items, requester_pays):
    opts = {}
    for item in items or []:
        key, _, value = item.partition("=")
        opts[key] = _coerce(value)
    if requester_pays is not None:
        opts["requester_pays"] = requester_pays
    return opts


class IOCounter:
    def __init__(self):
        self.gets = 0
        self.fetched_bytes = 0
        self.read_bytes = 0

    @property
    def amplification(self) -> float:
        return self.fetched_bytes / self.read_bytes if self.read_bytes else 0.0


# Module-level counter the installed wrappers write to. count_gcs_io swaps a
# fresh IOCounter in for its duration; None means "not counting".
_ACTIVE_IO = None
_INSTRUMENTED = False


def _note_fetch(data):
    """Record one range GET and its byte count against the active counter."""
    c = _ACTIVE_IO
    if c is not None:
        c.gets += 1
        c.fetched_bytes += len(data) if data is not None else 0


def _note_read(data):
    """Record bytes the reader logically consumed against the active counter."""
    c = _ACTIVE_IO
    if c is not None and data is not None:
        c.read_bytes += len(data)


def install_gcs_instrumentation():
    """Wrap gcsfs / fsspec once so reads route through the active IOCounter.

    fsspec binds a file's range fetcher (GCSFile._fetch_range) into its cache at
    OPEN time, so a class patch only takes effect for files opened AFTER it is
    installed. This must therefore run before any BgenReader opens a remote file
    (main / _run_smoke call it up front). Idempotent. Fails loud if either hook
    is missing (so a gcsfs/fsspec rename surfaces instead of silently counting
    zero). The wrappers only do work while a counter is active, so leaving them
    installed for the process lifetime is cheap.
    """
    global _INSTRUMENTED
    if _INSTRUMENTED:
        return
    import fsspec.spec
    import gcsfs.core

    if not hasattr(gcsfs.core.GCSFile, "_fetch_range"):
        raise RuntimeError("gcsfs.core.GCSFile._fetch_range missing; instrumentation needs updating")
    if not hasattr(fsspec.spec.AbstractBufferedFile, "read"):
        raise RuntimeError("fsspec AbstractBufferedFile.read missing; instrumentation needs updating")

    orig_fetch = gcsfs.core.GCSFile._fetch_range
    orig_read = fsspec.spec.AbstractBufferedFile.read

    def fetch_range(self, start=None, end=None, *a, **k):
        data = orig_fetch(self, start, end, *a, **k)
        _note_fetch(data)
        return data

    def read(self, length=-1, *a, **k):
        data = orig_read(self, length, *a, **k)
        _note_read(data)
        return data

    gcsfs.core.GCSFile._fetch_range = fetch_range
    fsspec.spec.AbstractBufferedFile.read = read
    _INSTRUMENTED = True


@contextlib.contextmanager
def count_gcs_io():
    """Count object-store GETs / bytes / logical reads around a block of work.

    Installs the gcsfs/fsspec wrappers (idempotent) and makes a fresh IOCounter
    the active one for the duration, restoring the previous counter on exit so
    nesting is safe. NOTE: any remote file must be OPENED while instrumentation
    is installed (see install_gcs_instrumentation) for its range GETs to be
    counted; the benchmark installs it before opening any reader.
    """
    global _ACTIVE_IO
    install_gcs_instrumentation()
    counter = IOCounter()
    prev = _ACTIVE_IO
    _ACTIVE_IO = counter
    try:
        yield counter
    finally:
        _ACTIVE_IO = prev


def needs_upload(fs, url, local_size):
    if not fs.exists(url):
        return True
    try:
        return int(fs.info(url)["size"]) != int(local_size)
    except Exception:
        return True


def upload_fixtures(bucket, data_dir, storage_options):
    import fsspec

    fs = fsspec.filesystem("gs", **(storage_options or {}))
    uploaded = []
    for fixture in FIXTURES:
        for name in (fixture.local_name, fixture.local_name + ".bgi"):
            local = os.path.join(data_dir, name)
            if not os.path.exists(local):
                raise FileNotFoundError(f"fixture not found locally: {local}")
            url = remote_url(bucket, name)
            if needs_upload(fs, url, os.path.getsize(local)):
                print(f"  upload {name} -> {url}")
                fs.put(local, url)
                uploaded.append(url)
            else:
                print(f"  skip {name} (already present, size matches)")
    return uploaded


def _run_once(do_read, reader):
    with count_gcs_io() as counter:
        t0 = time.perf_counter()
        do_read(reader)
        dt = time.perf_counter() - t0
    return dt, counter


def _summarize(samples):
    # samples: list of (time, counter); report counts from the median-time run.
    times = [s[0] for s in samples]
    ordered = sorted(range(len(times)), key=lambda i: times[i])
    median_idx = ordered[len(ordered) // 2]
    c = samples[median_idx][1]
    return {
        "median_time": statistics.median(times),
        "min_time": min(times),
        "runs": times,
        "gets": c.gets,
        "fetched_bytes": c.fetched_bytes,
        "read_bytes": c.read_bytes,
        "amplification": c.amplification,
    }


CHROM = "chr1"
POS_STRIDE = 1000
# Variant counts per fixture (match generate_test_bgen.py output).
_NVAR = {"small": 300, "wide": 5000}


def build_remote_url_map(bucket):
    return {f.key: remote_url(bucket, f.local_name) for f in FIXTURES}


def _region_middle(num_variants, count):
    """(chrom, start_pos, end_pos) for `count` contiguous middle variants."""
    count = max(1, min(count, num_variants))
    start_idx = (num_variants - count) // 2
    return CHROM, (start_idx + 1) * POS_STRIDE, (start_idx + count) * POS_STRIDE


def _scattered_filter(num_variants, count):
    count = max(1, min(count, num_variants))
    step = max(1, num_variants // count)
    idxs = list(range(0, num_variants, step))[:count]
    positions = [(i + 1) * POS_STRIDE for i in idxs]
    return {
        "chromosome": CHROM,
        "positions": positions,
        "allele1": ["A"] * len(positions),
        "allele2": ["G"] * len(positions),
    }


def _sample_subset(sample_path, fraction):
    with open(sample_path) as fh:
        lines = fh.readlines()
    ids = [ln.split()[0] for ln in lines[2:] if ln.strip()]
    n = max(1, int(len(ids) * fraction))
    return ids[:n]


def _open_reader(reader_mod, url, sample_path, storage_options):
    return reader_mod.BgenReader(url, bgi_path=url + ".bgi", sample_path=sample_path, storage_options=storage_options)


def _make_open_only(reader_mod, urls, sample_path, storage_options):
    # The measured work IS the open (.bgi fetch + header read), so it happens
    # inside do_read; make_reader holds nothing persistent.
    url = urls["small"]

    def make_reader():
        return None

    def do_read(_):
        reader = _open_reader(reader_mod, url, sample_path, storage_options)
        _ = reader.samples

    return make_reader, do_read


def _region_load_build(fixture_key, count):
    def build(reader_mod, urls, sample_path, storage_options):
        url = urls[fixture_key]
        chrom, start, end = _region_middle(_NVAR[fixture_key], count)

        def make_reader():
            return _open_reader(reader_mod, url, sample_path, storage_options)

        def do_read(reader):
            reader.load_variants(region_chrom=chrom, region_start=start, region_end=end)

        return make_reader, do_read

    return build


def _filter_load_build(fixture_key, count):
    def build(reader_mod, urls, sample_path, storage_options):
        url = urls[fixture_key]
        vfilter = _scattered_filter(_NVAR[fixture_key], count)

        def make_reader():
            return _open_reader(reader_mod, url, sample_path, storage_options)

        def do_read(reader):
            reader.load_variants(variant_filter=vfilter)

        return make_reader, do_read

    return build


def _cohort_build(reader_mod, urls, sample_path, storage_options):
    url = urls["wide"]
    subset = _sample_subset(sample_path, 0.01)

    def make_reader():
        return _open_reader(reader_mod, url, sample_path, storage_options)

    def do_read(reader):
        # Mapping ids->indices is O(n_samples) and trivial next to the network
        # read; computing it per call keeps the workload self-contained.
        indices, _ = reader.get_sample_indices(subset)
        import numpy as np

        reader.load_variants(sample_indices=np.asarray(indices, dtype=np.int32))

    return make_reader, do_read


def _iter_stream_build(reader_mod, urls, sample_path, storage_options):
    url = urls["wide"]

    def make_reader():
        return _open_reader(reader_mod, url, sample_path, storage_options)

    def do_read(reader):
        for _ in reader.iter_variants():
            pass

    return make_reader, do_read


WORKLOADS = [
    {"name": "open_only", "fixture_key": "small", "build": _make_open_only},
    {"name": "single_variant", "fixture_key": "small", "build": _region_load_build("small", 1)},
    {"name": "region_small", "fixture_key": "small", "build": _region_load_build("small", 100)},
    {"name": "variant_filter_scattered", "fixture_key": "wide", "build": _filter_load_build("wide", 200)},
    {"name": "cohort_small", "fixture_key": "wide", "build": _cohort_build},
    {"name": "iter_stream", "fixture_key": "wide", "build": _iter_stream_build},
]


STRATEGIES = ("stream", "download_then_read", "warm_local")


def download_file(url, dest_dir, storage_options):
    import fsspec

    fs = fsspec.filesystem("gs", **(storage_options or {}))
    os.makedirs(dest_dir, exist_ok=True)
    local_bgen = os.path.join(dest_dir, os.path.basename(url))
    t0 = time.perf_counter()
    fs.get(url, local_bgen)
    fs.get(url + ".bgi", local_bgen + ".bgi")
    return local_bgen, time.perf_counter() - t0


def run_strategy(strategy, workload, reader_mod, urls, sample_path, storage_options, data_dir, dest_dir, num_runs):
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy}")
    key = workload["fixture_key"]
    url = urls[key]

    if strategy == "stream":
        make_reader, do_read = workload["build"](reader_mod, urls, sample_path, storage_options)
        return {"strategy": strategy, **measure_cold_warm(make_reader, do_read, num_runs)}

    if strategy == "download_then_read":
        local_bgen, dl_time = download_file(url, dest_dir, storage_options)
        local_urls = dict(urls, **{key: local_bgen})
        make_reader, do_read = workload["build"](reader_mod, local_urls, sample_path, None)
        out = measure_cold_warm(make_reader, do_read, num_runs)
        return {"strategy": strategy, "download_time": dl_time, **out}

    # warm_local: read the original local fixture in data_dir, no network.
    local_bgen = os.path.join(data_dir, os.path.basename(url))
    local_urls = dict(urls, **{key: local_bgen})
    make_reader, do_read = workload["build"](reader_mod, local_urls, sample_path, None)
    return {"strategy": strategy, **measure_cold_warm(make_reader, do_read, num_runs)}


def measure_cold_warm(make_reader, do_read, num_runs):
    # Make instrumentation-before-open structural, not convention: make_reader
    # opens the remote file, and a file's range fetcher is bound at open time, so
    # the wrappers must already be installed. main/_run_smoke also install up
    # front; this idempotent call protects any direct caller of measure_cold_warm.
    install_gcs_instrumentation()
    cold = []
    for _ in range(num_runs):
        reader = make_reader()
        cold.append(_run_once(do_read, reader))
    # Warm pass: one fresh reader (its block cache starts cold but warms up on the
    # first call); then repeat do_read num_runs times with the cache hot.
    warm_reader = make_reader()
    # Prime the cache with one read before recording warm samples.
    _run_once(do_read, warm_reader)
    warm = [_run_once(do_read, warm_reader) for _ in range(num_runs)]
    return {"cold": _summarize(cold), "warm": _summarize(warm)}


PREFETCH_VARIANTS = [
    {"label": "default"},
    {"label": "cache_none", "default_cache_type": "none"},
    {"label": "block_1mb", "default_block_size": 1 << 20},
    {"label": "block_16mb", "default_block_size": 16 << 20},
]


def format_summary(results):
    lines = [
        "",
        "=== Remote benchmark summary ===",
        f"{'workload':26s} {'strat':18s} {'cold_s':>8s} {'warm_s':>8s} {'GETs':>6s} {'fetMB':>8s} {'ampl':>6s}",
    ]
    for wname, strategies in results.get("workloads", {}).items():
        for sname, sides in strategies.items():
            if "cold" not in sides:  # e.g. the prefetch_sweep block (label -> {cold,warm})
                continue
            cold, warm = sides.get("cold", {}), sides.get("warm", {})
            lines.append(
                f"{wname:26s} {sname:18s} "
                f"{cold.get('median_time', 0):8.3f} {warm.get('median_time', 0):8.3f} "
                f"{cold.get('gets', 0):6d} {cold.get('fetched_bytes', 0) / 1048576:8.1f} "
                f"{cold.get('amplification', 0):6.2f}"
            )
    return "\n".join(lines)


def _build_argparser():
    p = argparse.ArgumentParser(description="Remote (gs://) performance benchmark for lazybgen")
    p.add_argument("--bucket", default=DEFAULT_BUCKET)
    p.add_argument("--data-dir", default="benchmarks/test_data")
    p.add_argument("--output-dir", default="/tmp/remote_bench")
    p.add_argument("--num-runs", type=int, default=5)
    p.add_argument("--workloads", default=None, help="comma-separated subset; default all")
    p.add_argument("--strategies", default="stream", help=f"comma-separated subset of {STRATEGIES}; default stream")
    p.add_argument("--prefetch-sweep", action="store_true")
    p.add_argument("--upload", action="store_true", help="push local fixtures to the bucket, then continue")
    p.add_argument("--smoke", action="store_true", help="single run against the small public bgen")
    p.add_argument("--requester-pays", nargs="?", const=True, default=None)
    p.add_argument("--storage-options", action="append", default=None, help="k=v, repeatable")
    return p


def main(argv=None):
    args = _build_argparser().parse_args(argv)
    storage_options = parse_storage_options(args.storage_options, args.requester_pays)
    reader_mod = __import__("lazybgen", fromlist=["load_bgen", "BgenReader"])
    # Install before any reader opens a remote file, so range GETs are counted
    # (fsspec binds a file's fetcher at open time).
    install_gcs_instrumentation()

    if args.upload:
        print("Uploading fixtures...")
        upload_fixtures(args.bucket, args.data_dir, storage_options)

    if args.smoke:
        return _run_smoke(reader_mod, storage_options)

    urls = build_remote_url_map(args.bucket)
    selected = args.workloads.split(",") if args.workloads else [w["name"] for w in WORKLOADS]
    strategies = [s for s in args.strategies.split(",") if s.strip()]
    results = {
        "timestamp": datetime.now().isoformat(),
        "bucket": args.bucket,
        "num_runs": args.num_runs,
        "workloads": {},
    }

    for w in WORKLOADS:
        if w["name"] not in selected:
            continue
        sample_path = _sample_path_for(args.data_dir, w["fixture_key"])
        dest_dir = os.path.join(args.output_dir, "download", w["fixture_key"])
        per_strategy = {}
        for strat in strategies:
            print(f"  {w['name']} [{strat}] ...", flush=True)
            per_strategy[strat] = run_strategy(
                strat, w, reader_mod, urls, sample_path, storage_options, args.data_dir, dest_dir, args.num_runs
            )
        if args.prefetch_sweep and w["name"] == "variant_filter_scattered":
            per_strategy["prefetch_sweep"] = _prefetch_sweep(
                w, reader_mod, urls, sample_path, storage_options, args.num_runs
            )
        results["workloads"][w["name"]] = per_strategy

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.output_dir) / f"remote_benchmark_{ts}.json"
    out.write_text(json.dumps(results, indent=2))
    print(format_summary(results))
    print(f"\nResults saved to: {out}")
    return 0


def _sample_path_for(data_dir, fixture_key):
    fixture = next(f for f in FIXTURES if f.key == fixture_key)
    return os.path.join(data_dir, fixture.local_name.replace(".bgen", ".sample"))


def _prefetch_sweep(workload, reader_mod, urls, sample_path, base_opts, num_runs):
    sweep = {}
    for variant in PREFETCH_VARIANTS:
        opts = dict(base_opts)
        opts.update({k: v for k, v in variant.items() if k != "label"})
        make_reader, do_read = workload["build"](reader_mod, urls, sample_path, opts)
        sweep[variant["label"]] = measure_cold_warm(make_reader, do_read, num_runs)
    return sweep


def _run_smoke(reader_mod, storage_options):
    print(f"smoke: reading {SMOKE_URL}")
    with count_gcs_io() as c:
        reader = reader_mod.BgenReader(SMOKE_URL, bgi_path=SMOKE_URL + ".bgi", storage_options=storage_options)
        _ = reader.samples
        df = reader.load_variants()[1]
    print(f"smoke OK: {len(df)} variants, {c.gets} GETs, {c.fetched_bytes} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
