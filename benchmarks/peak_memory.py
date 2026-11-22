#!/usr/bin/env python3
"""Peak memory of one read in a fresh process: lazybgen against the `bgen` package.

Each (library, fixture, workload) runs ``--reps`` times, each in its own Python
process that reads nothing before it, so the figure is that read's own cost
rather than whatever an earlier workload left mapped. Peak RSS is the kernel's
high-water mark (``VmHWM`` in ``/proc/self/status``) after the call, in MiB, so
this is Linux-only. Any failed run fails the script. The workloads are the
``compare_libraries.py`` ones over the same 10k-variant ladder fixtures
(``generate_test_bgen.py --mode ladder``).

Run from the repository root with lazybgen built in place:

    python benchmarks/peak_memory.py --data-dir benchmarks/test_data \\
        --output benchmarks/peak_memory.json
"""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

CHILD = r"""
import sys
sys.path.insert(0, "benchmarks")
from compare_libraries import fixture_paths, lazybgen_workloads, bgen_workloads
lib, samples, workload, data_dir = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
bf, sf = fixture_paths(data_dir, samples, 10000)
if lib == "lazybgen":
    fns = lazybgen_workloads(str(bf), str(sf), 10000)
else:
    fns = bgen_workloads(bf, samples, 10000)
fns[workload]()
for line in open("/proc/self/status"):
    if line.startswith("VmHWM"):
        print(int(line.split()[1]) / 1024)
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", default="benchmarks/test_data")
    ap.add_argument("--output", default="benchmarks/peak_memory.json")
    ap.add_argument("--samples", default="5000,50000,100000,500000")
    ap.add_argument("--workloads", default="full_decode,region,scattered")
    ap.add_argument("--libraries", default="lazybgen,bgen")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    results = {}
    failed = []
    for samples in [int(s) for s in args.samples.split(",")]:
        for workload in args.workloads.split(","):
            for _ in range(args.reps):
                for lib in args.libraries.split(","):
                    proc = subprocess.run(
                        [sys.executable, "-c", CHILD, lib, str(samples), workload, args.data_dir],
                        capture_output=True,
                        text=True,
                    )
                    if proc.returncode:
                        print(lib, samples, workload, proc.stderr[-400:], flush=True)
                        failed.append(f"{samples}|{workload}|{lib}")
                        continue
                    key = f"{samples}|{workload}|{lib}"
                    results.setdefault(key, []).append(float(proc.stdout.split()[-1]))
            row = {
                lib: (round(statistics.median(vals)), len(vals))
                for lib in args.libraries.split(",")
                if (vals := results.get(f"{samples}|{workload}|{lib}"))
            }
            print(f"{samples:>7d} {workload:12s} peak MiB (median, runs): {row}", flush=True)

    Path(args.output).write_text(json.dumps(results, indent=1))
    print(f"Saved: {args.output}")
    if failed:
        # A cell missing a run would otherwise publish a median of fewer runs
        # than it claims.
        sys.exit(f"{len(failed)} run(s) failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
