"""Compare the remote transports (fsspec vs obstore) on the same BGEN URL.

Reads the same variants through each transport and reports wall time and
effective throughput per workload, so the choice can be made from measurements
on your own bucket and machine rather than from someone else's.

Both transports read the same bytes, so the dosages are compared as well: a
transport that is fast and wrong is not faster.

Example
-------
    python benchmarks/compare_remote_backends.py \
        --url gs://my-bucket/big.bgen --reps 3

Requires obstore (``pip install lazybgen[obstore]``) for the obstore rows;
without it the script reports fsspec only.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import time

BACKENDS = ("fsspec", "obstore")


def _coerce(value: str):
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(value)
    except ValueError:
        return value


def parse_storage_options(items, requester_pays):
    options = {}
    for item in items or []:
        key, _, value = item.partition("=")
        options[key] = _coerce(value)
    if requester_pays is not None:
        options["requester_pays"] = requester_pays
    return options


def variant_rows(url, storage_options):
    """Read (chromosome, position, allele1, allele2, offset) from the .bgi."""
    from lazybgen.remote import ensure_local_bgi

    bgi_local = ensure_local_bgi(url + ".bgi", storage_options)
    con = sqlite3.connect(bgi_local)
    try:
        return con.execute(
            "SELECT chromosome, position, allele1, allele2, file_start_position "
            "FROM Variant ORDER BY file_start_position"
        ).fetchall()
    finally:
        con.close()


def build_workloads(rows, count):
    """Return {name: variant_filter} for the access patterns worth timing."""
    if len(rows) < count:
        count = len(rows)
    middle = len(rows) // 2 - count // 2
    region = rows[middle : middle + count]
    scattered = rows[:: max(1, len(rows) // count)][:count]

    def as_filter(picks):
        return {
            "chromosome": picks[0][0],
            "positions": [p[1] for p in picks],
            "allele1": [p[2] for p in picks],
            "allele2": [p[3] for p in picks],
        }

    return {
        "region": as_filter(region),
        "scattered": as_filter(scattered),
        "single": as_filter(rows[len(rows) // 2 : len(rows) // 2 + 1]),
    }


def time_read(url, storage_options, backend, variant_filter, reps):
    """Time one workload on one transport; returns (times, dosages)."""
    from lazybgen import load_bgen

    times = []
    dosages = None
    for _ in range(reps + 1):  # first pass warms auth and connections
        start = time.perf_counter()
        dosages, _info, _ids = load_bgen(
            url,
            variant_filter=variant_filter,
            nan_action="warn",
            storage_options=storage_options or None,
            remote_backend=backend,
        )
        times.append(time.perf_counter() - start)
    return times[1:], dosages


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True, help="gs:// or s3:// URL of a BGEN with a sibling .bgi")
    parser.add_argument("--variants", type=int, default=500, help="variants per region/scattered read")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--storage-option", action="append", metavar="KEY=VALUE")
    parser.add_argument("--requester-pays", default=None, help="billing project, or 'true' for the default one")
    parser.add_argument("--json", default=None, help="write the raw timings here")
    args = parser.parse_args()

    storage_options = parse_storage_options(
        args.storage_option, _coerce(args.requester_pays) if args.requester_pays else None
    )

    from lazybgen.obstore_backend import is_available

    backends = [b for b in BACKENDS if b != "obstore" or is_available()]
    if "obstore" not in backends:
        print("obstore is not installed; reporting fsspec only (pip install lazybgen[obstore])")

    rows = variant_rows(args.url, storage_options)
    workloads = build_workloads(rows, args.variants)
    print(f"{args.url}: {len(rows)} variants in the index")

    results = {}
    for name, variant_filter in workloads.items():
        print(f"\n{name} ({len(variant_filter['positions'])} variants)")
        reference = None
        for backend in backends:
            try:
                times, dosages = time_read(args.url, storage_options, backend, variant_filter, args.reps)
            except Exception as exc:  # noqa: BLE001 - report and keep going
                print(f"  {backend:8s} FAILED: {type(exc).__name__}: {exc}")
                results.setdefault(name, {})[backend] = {"error": f"{type(exc).__name__}: {exc}"}
                continue

            median = statistics.median(times)
            megabytes = dosages.nbytes / 1e6
            results.setdefault(name, {})[backend] = {
                "times_s": times,
                "median_s": median,
                "decoded_MB": megabytes,
            }
            print(f"  {backend:8s} median {median * 1000:8.1f} ms   {megabytes / median:7.1f} MB/s decoded")

            if reference is None:
                reference = (backend, times, dosages)
            else:
                import numpy as np

                base_backend, base_times, base_dosages = reference
                np.testing.assert_array_equal(np.isnan(dosages), np.isnan(base_dosages))
                np.testing.assert_array_equal(dosages[~np.isnan(dosages)], base_dosages[~np.isnan(base_dosages)])
                ratio = statistics.median(base_times) / median
                print(f"  {'':8s} {ratio:.2f}x {base_backend}, and byte-identical")
                results[name][backend]["speedup_vs_" + base_backend] = ratio

    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"url": args.url, "variants": args.variants, "results": results}, handle, indent=2)


if __name__ == "__main__":
    main()
