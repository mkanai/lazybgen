"""Interleaved remote benchmark across several builds and transports.

Why this exists: two lazybgen builds cannot share a process (an editable install's
meta-path finder wins over a `cd`), so the obvious approach is to run each build's
remote suite to completion and compare the results. That does not work. A remote
link drifts far more over minutes than the effects being measured -- the same read
has measured 180 s, 81 s and 24 s across one day on the same machine -- so two
suites run minutes apart differ by the weather, not by the code. A comparison
built that way once produced a confident "3.2x slower" that an interleaved re-run
turned into "2x faster".

So this driver alternates. Every configuration runs one timed read, round-robin,
rep after rep, and only then are medians taken. A drift that hits one arm hits
them all. Each read runs in a fresh subprocess with PYTHONPATH pointing at the
build under test; the worker below is written to a temp file, so every
configuration is measured by the SAME harness code even when the checkouts differ.

Example
-------
    python benchmarks/compare_remote_builds.py \
        --build master=~/lz-master --build branch=~/lz-libdeflate \
        --build branch-obstore=~/lz-libdeflate:LAZYBGEN_REMOTE_BACKEND=obstore \
        --bucket gs://my-bench-bucket --reps 5 --output remote.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_libraries import (  # noqa: E402
    CHROM,
    REGION_VARIANTS,
    SCATTERED_VARIANTS,
    full_matrix_gb,
    region_bounds,
    scattered_positions,
    single_position,
)

# Runs one timed read and prints one JSON line. Kept as source here, not as a file
# in the repo, so that every build is measured by this exact code regardless of
# what its own checkout contains.
WORKER = """
import json, sys, time
spec = json.loads(sys.argv[1])
import lazybgen

kwargs = {"nan_action": "mean"}
if spec.get("region"):
    kwargs["region"] = spec["region"]
if spec.get("variant_filter"):
    kwargs["variant_filter"] = spec["variant_filter"]

def peak_rss_mb():
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None

try:
    t0 = time.perf_counter()
    out = lazybgen.load_bgen(spec["url"], **kwargs)[0]
    elapsed = time.perf_counter() - t0
    print(json.dumps({"ok": True, "time": elapsed, "peak_mb": peak_rss_mb(),
                      "shape": list(out.shape), "backend": spec.get("backend")}))
except Exception as exc:
    print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
"""


def parse_build(raw):
    """`name=path` or `name=path:ENV=VALUE[,ENV=VALUE]` -> (name, path, env)."""
    name, _, rest = raw.partition("=")
    if not name or not rest:
        raise argparse.ArgumentTypeError(f"--build wants name=path, got {raw!r}")
    path, _, envs = rest.partition(":")
    env = {}
    for item in filter(None, envs.split(",")):
        key, _, value = item.partition("=")
        env[key] = value
    return name, str(Path(path).expanduser().resolve()), env


def workloads_for(samples, variants, max_full_gb):
    """The read specs for one fixture, as plain data the worker can take."""
    rstart, rend = region_bounds(variants, REGION_VARIANTS)
    spos = scattered_positions(variants, SCATTERED_VARIANTS)
    spt = single_position(variants)
    specs = {
        "region": {"region": f"{CHROM}:{rstart}-{rend}"},
        "scattered": {
            "variant_filter": {
                "chromosome": CHROM,
                "positions": spos,
                "allele1": ["A"] * len(spos),
                "allele2": ["G"] * len(spos),
            }
        },
        "single": {"region": f"{CHROM}:{spt}-{spt}"},
    }
    if full_matrix_gb(samples, variants) <= max_full_gb:
        specs = {"full_decode": {}, **specs}
    return specs


def run_one(worker_path, build_path, env_extra, spec, timeout):
    env = dict(os.environ)
    # The build under test goes FIRST so its lazybgen wins, but the caller's
    # PYTHONPATH is kept: replacing it outright hides anything else the
    # environment provides (an obstore installed to a target dir, say).
    inherited = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and p != build_path]
    env["PYTHONPATH"] = os.pathsep.join([build_path, *inherited])
    env.update(env_extra)
    proc = subprocess.run(
        [sys.executable, worker_path, json.dumps(spec)],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    return {"ok": False, "error": (proc.stderr or proc.stdout)[-300:]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--build", action="append", required=True, type=parse_build, metavar="NAME=PATH[:ENV=VAL]")
    p.add_argument("--bucket", required=True)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1, help="untimed passes per configuration per workload")
    p.add_argument("--max-full-gb", type=float, default=64.0)
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--output", default="remote_builds.json")
    p.add_argument(
        "--fixtures",
        default="100k x 10k:100000:10000,500k x 10k:500000:10000",
        help="label:samples:variants, comma separated",
    )
    p.add_argument(
        "--fixture-suffix",
        default="zlib_8bit",
        help="trailing part of the fixture stem, i.e. test_<samples>s_<variants>v_<suffix>.bgen",
    )
    args = p.parse_args()

    fixtures = []
    for item in args.fixtures.split(","):
        label, samples, variants = item.split(":")
        fixtures.append((label, int(samples), int(variants)))

    with tempfile.NamedTemporaryFile("w", suffix="_worker.py", delete=False) as fh:
        fh.write(WORKER)
        worker_path = fh.name

    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "builds": [{"name": n, "path": pth, "env": e} for n, pth, e in args.build],
        "reps": args.reps,
        "region_variants": REGION_VARIANTS,
        "scattered_variants": SCATTERED_VARIANTS,
        "interleaved": True,
        "results": {},
    }

    try:
        for label, samples, variants in fixtures:
            url = f"{args.bucket.rstrip('/')}/test_{samples}s_{variants}v_{args.fixture_suffix}.bgen"
            print(f"\n[{label}] {url}", flush=True)
            for workload, spec_kw in workloads_for(samples, variants, args.max_full_gb).items():
                spec = {"url": url, **spec_kw}
                samples_by_build = {name: [] for name, _, _ in args.build}

                for _ in range(args.warmup):
                    for name, path, env in args.build:
                        run_one(worker_path, path, env, {**spec, "backend": name}, args.timeout)

                # The point of this file: one rep of every build before the
                # second rep of any of them.
                failures = {}
                for _ in range(args.reps):
                    for name, path, env in args.build:
                        res = run_one(worker_path, path, env, {**spec, "backend": name}, args.timeout)
                        if res.get("ok"):
                            samples_by_build[name].append(res)
                        else:
                            failures[name] = res.get("error", "unknown")

                row = {}
                for name, runs in samples_by_build.items():
                    if not runs:
                        row[name] = {"ok": False, "error": failures.get(name, "no successful runs")}
                        continue
                    times = [r["time"] for r in runs]
                    median = statistics.median(times)
                    row[name] = {
                        "ok": True,
                        "median_time": median,
                        "min_time": min(times),
                        "max_time": max(times),
                        "spread": (max(times) - min(times)) / median if median else None,
                        "peak_memory_mb": statistics.median([r["peak_mb"] for r in runs if r["peak_mb"]] or [0]),
                        "times": times,
                    }
                out["results"].setdefault(label, {})[workload] = row

                parts = []
                base = None
                for name, _, _ in args.build:
                    cell = row[name]
                    if not cell["ok"]:
                        parts.append(f"{name} FAIL")
                        continue
                    txt = f"{name} {cell['median_time'] * 1000:.0f}ms (+-{cell['spread'] * 100:.0f}%)"
                    if base is None:
                        base = cell["median_time"]
                    else:
                        txt += f" [{base / cell['median_time']:.2f}x]"
                    parts.append(txt)
                print(f"  {workload:14s} " + "  ".join(parts), flush=True)
    finally:
        os.unlink(worker_path)

    with open(args.output, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
