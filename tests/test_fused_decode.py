"""The fused 8/16-bit full-sample decode, against an independent reference.

Unphased v1.2 8- and 16-bit columns are decoded by kernels that handle many
samples per step (AVX2 on x86 when the CPU has it, NEON on arm64, a branch-free
scalar loop otherwise) and that return the column's min, max and missing-call
flag in the same pass. These tests write BGEN files byte by byte, so every
numerator, invalid probability sums and missing calls can be placed anywhere,
including in the scalar tail after the last full vector, and compare the result
bit for bit with the decode contract computed in NumPy:

    dosage = float32(2*M - 2*P(AA) - P(AB)) / float32(M)    (M = 255 or 65535)

with NaN for a missing call or for P(AA) + P(AB) > M. A float64 column must be
the exact widening of the float32 one, and the range the reader reports must
equal the range of the values it returned.
"""

import os
import sqlite3
import struct
import time
import zlib

import numpy as np
import pytest
from lazybgen.reader import BgenReader

NAN32 = np.float32(np.nan)


def _s2(x):
    return struct.pack("<H", len(x)) + x.encode()


def _s4(x):
    return struct.pack("<I", len(x)) + x.encode()


def _write_v12(path, columns, bits, compress=zlib.compress):
    """Write a zlib-compressed layout-2 BGEN with one variant per column.

    Each column is (aa, ab, missing): integer arrays of the stored P(AA) and
    P(AB) and a boolean missing flag per sample. Also writes the .bgi index.
    """
    n = len(columns[0][0])
    blocks = []
    word = "<B" if bits == 8 else "<H"
    for v, (aa, ab, miss) in enumerate(columns):
        ploidy = bytes(0x82 if m else 0x02 for m in miss)
        probs = b"".join(struct.pack(word, int(a)) + struct.pack(word, int(b)) for a, b in zip(aa, ab))
        geno = struct.pack("<IHBB", n, 2, 2, 2) + ploidy + struct.pack("<BB", 0, bits) + probs
        packed = compress(geno)
        pos = 1000 + v
        block = (
            _s2(f"v{v}")
            + _s2(f"rs{v}")
            + _s2("01")
            + struct.pack("<IH", pos, 2)
            + _s4("A")
            + _s4("G")
            + struct.pack("<II", len(packed) + 4, len(geno))
            + packed
        )
        blocks.append((pos, f"rs{v}", block))
    _write_file(path, n, layout=2, blocks=blocks)


def _write_file(path, n, layout, blocks):
    """Lay out a zlib-compressed BGEN with embedded sample IDs, plus its .bgi.

    blocks is a list of (position, rsid, variant block bytes) in file order.
    """
    samples = [f"s{i}" for i in range(n)]
    ids = b"".join(_s2(s) for s in samples)
    sample_block = struct.pack("<II", 8 + len(ids), n) + ids
    flags = 1 | (layout << 2) | (1 << 31)  # zlib, the layout, sample IDs present
    header = struct.pack("<II", len(blocks), n) + b"bgen" + struct.pack("<I", flags)
    header_length = 4 + len(header)  # L_H counts its own 4 bytes
    prefix = struct.pack("<II", header_length + len(sample_block), header_length) + header

    rows = []
    body = bytearray()
    start = len(prefix) + len(sample_block)
    for pos, rsid, block in blocks:
        rows.append(("01", pos, rsid, 2, "A", "G", start + len(body), len(block)))
        body += block

    with open(path, "wb") as f:
        f.write(prefix + sample_block + body)

    conn = sqlite3.connect(str(path) + ".bgi")
    try:
        conn.execute(
            "CREATE TABLE Metadata (filename TEXT NOT NULL, file_size INT NOT NULL, "
            "last_write_time INT NOT NULL, first_1000_bytes BLOB NOT NULL, index_creation_time INT NOT NULL)"
        )
        with open(path, "rb") as f:
            first = f.read(1000)
        now = int(time.time())
        conn.execute(
            "INSERT INTO Metadata VALUES (?,?,?,?,?)",
            (os.path.basename(str(path)), os.path.getsize(path), now, first, now),
        )
        conn.execute(
            "CREATE TABLE Variant (chromosome TEXT NOT NULL, position INT NOT NULL, rsid TEXT NOT NULL, "
            "number_of_alleles INT NOT NULL, allele1 TEXT NOT NULL, allele2 TEXT NULL, "
            "file_start_position INT NOT NULL, size_in_bytes INT NOT NULL)"
        )
        conn.executemany("INSERT INTO Variant VALUES (?,?,?,?,?,?,?,?)", rows)
        conn.commit()
    finally:
        conn.close()


def _reference(columns, bits):
    m = (1 << bits) - 1
    out = []
    for aa, ab, miss in columns:
        aa = np.asarray(aa, dtype=np.int64)
        ab = np.asarray(ab, dtype=np.int64)
        k = 2 * m - 2 * aa - ab
        val = np.float32(k) / np.float32(m)
        bad = np.asarray(miss) | (aa + ab > m)
        out.append(np.where(bad, NAN32, val).astype(np.float32))
    return np.stack(out, axis=1)


def _columns(n, bits, rng):
    """Columns that between them hit every case the kernels distinguish."""
    m = (1 << bits) - 1
    cols = []
    # Uniform over all stored pairs, valid or not, with scattered missing calls.
    aa = rng.integers(0, m + 1, size=n)
    ab = rng.integers(0, m + 1, size=n)
    cols.append((aa, ab, rng.random(n) < 0.05))
    # Valid only: P(AB) drawn within what P(AA) leaves.
    aa = rng.integers(0, m + 1, size=n)
    ab = (rng.random(n) * (m - aa + 1)).astype(np.int64)
    cols.append((aa, ab, np.zeros(n, bool)))
    # The extremes of the range: dosage exactly 0 and exactly 2.
    cols.append((np.full(n, m), np.zeros(n, np.int64), np.zeros(n, bool)))
    cols.append((np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(n, bool)))
    # Nothing valid: all missing, then all invalid sums.
    cols.append((np.zeros(n, np.int64), np.zeros(n, np.int64), np.ones(n, bool)))
    cols.append((np.full(n, m), np.full(n, m), np.zeros(n, bool)))
    # A single valid sample at the very end (the tail), the rest missing.
    miss = np.ones(n, bool)
    miss[-1] = False
    cols.append((np.full(n, 3), np.full(n, 1), miss))
    # A sum exactly at the limit (valid) next to one just past it (invalid).
    aa = np.full(n, m // 2)
    ab = np.where(np.arange(n) % 2 == 0, m - m // 2, m - m // 2 + 1)
    cols.append((aa, ab, np.zeros(n, bool)))
    return cols


def _check(path, columns, bits, num_threads):
    expected32 = _reference(columns, bits)
    for dtype in (np.float32, np.float64):
        with BgenReader(str(path), num_threads=num_threads) as reader:
            got, _ = reader.load_variants(dtype=dtype)
            stats = reader.last_dosage_stats
        want = expected32.astype(dtype)
        utype = np.uint32 if dtype == np.float32 else np.uint64
        diff = got.view(utype) != np.ascontiguousarray(want).view(utype).reshape(got.shape)
        assert not diff.any(), f"{int(diff.sum())} values differ ({np.dtype(dtype).name})"
        if stats is not None:
            finite = want[~np.isnan(want)]
            lo = float(finite.min()) if finite.size else float("inf")
            hi = float(finite.max()) if finite.size else float("-inf")
            assert stats == (lo, hi, bool(np.isnan(want).any()))


# Column lengths around the 8- and 16-lane vector widths, so the vector loop,
# its tail, and a column shorter than one vector are all exercised.
LENGTHS = [1, 2, 7, 8, 9, 15, 16, 17, 31, 33, 100, 1027]


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("n", LENGTHS)
def test_fused_decode_matches_reference(tmp_path, bits, n):
    rng = np.random.default_rng(n * 31 + bits)
    columns = _columns(n, bits, rng)
    path = tmp_path / f"f{bits}_{n}.bgen"
    _write_v12(path, columns, bits)
    for num_threads in (1, 4):
        _check(path, columns, bits, num_threads)


def test_every_8bit_numerator(tmp_path):
    """All 32,896 valid (P(AA), P(AB)) pairs, so all 511 numerators, in one column."""
    aa, ab = np.meshgrid(np.arange(256), np.arange(256), indexing="ij")
    keep = (aa + ab) <= 255
    aa, ab = aa[keep], ab[keep]
    rng = np.random.default_rng(1)
    order = rng.permutation(aa.size)
    columns = [(aa[order], ab[order], np.zeros(aa.size, bool))]
    path = tmp_path / "all8.bgen"
    _write_v12(path, columns, 8)
    _check(path, columns, 8, num_threads=2)
