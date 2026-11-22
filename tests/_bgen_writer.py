"""Byte-level BGEN writers for tests: every stored value is placed exactly.

The decode tests need files whose numerators, invalid sums, missing calls and
lengths they choose, which no writer library offers, so these lay the format out
by hand (zlib-compressed, embedded sample IDs s0..s{n-1}, single chromosome
"01", positions from 1000) and write the matching .bgi index.
"""

import os
import sqlite3
import struct
import time
import zlib


def s2(x):
    return struct.pack("<H", len(x)) + x.encode()


def s4(x):
    return struct.pack("<I", len(x)) + x.encode()


def write_v12(path, columns, bits, compress=zlib.compress, positions=None):
    """Write a zlib-compressed layout-2 BGEN with one variant per column.

    Each column is (aa, ab, missing): integer arrays of the stored P(AA) and
    P(AB) and a boolean missing flag per sample. Positions default to 1000 + v.
    Also writes the .bgi index.
    """
    n = len(columns[0][0])
    blocks = []
    word = "<B" if bits == 8 else "<H"
    for v, (aa, ab, miss) in enumerate(columns):
        ploidy = bytes(0x82 if m else 0x02 for m in miss)
        probs = b"".join(struct.pack(word, int(a)) + struct.pack(word, int(b)) for a, b in zip(aa, ab))
        geno = struct.pack("<IHBB", n, 2, 2, 2) + ploidy + struct.pack("<BB", 0, bits) + probs
        packed = compress(geno)
        pos = 1000 + v if positions is None else positions[v]
        block = (
            s2(f"v{v}")
            + s2(f"rs{v}")
            + s2("01")
            + struct.pack("<IH", pos, 2)
            + s4("A")
            + s4("G")
            + struct.pack("<II", len(packed) + 4, len(geno))
            + packed
        )
        blocks.append((pos, f"rs{v}", block))
    write_file(path, n, layout=2, blocks=blocks)


def write_file(path, n, layout, blocks):
    """Lay out a zlib-compressed BGEN with embedded sample IDs, plus its .bgi.

    blocks is a list of (position, rsid, variant block bytes) in file order.
    """
    samples = [f"s{i}" for i in range(n)]
    ids = b"".join(s2(s) for s in samples)
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


def write_v11(path, columns):
    """Write a zlib-compressed layout-1 BGEN; each column is (aa, ab, bb) integer arrays."""
    n = len(columns[0][0])
    blocks = []
    for v, (aa, ab, bb) in enumerate(columns):
        geno = b"".join(struct.pack("<HHH", int(a), int(b), int(c)) for a, b, c in zip(aa, ab, bb))
        packed = zlib.compress(geno)
        pos = 1000 + v
        block = (
            struct.pack("<I", n)
            + s2(f"v{v}")
            + s2(f"rs{v}")
            + s2("01")
            + struct.pack("<I", pos)
            + s4("A")
            + s4("G")
            + struct.pack("<I", len(packed))
            + packed
        )
        blocks.append((pos, f"rs{v}", block))
    write_file(path, n, layout=1, blocks=blocks)


def write_v12_with_v11_blocks(path, columns):
    """A layout-2 BGEN whose genotype blocks hold v1.1 triplets, as some writers produce.

    Each column is (aa, ab, bb). The variant records are layout 2; the decoded
    genotype block is 6*N bytes of 16-bit triplets, which the reader decodes as
    layout 1 when the block fails to parse as layout 2.
    """
    n = len(columns[0][0])
    blocks = []
    for v, (aa, ab, bb) in enumerate(columns):
        geno = b"".join(struct.pack("<HHH", int(a), int(b), int(c)) for a, b, c in zip(aa, ab, bb))
        packed = zlib.compress(geno)
        pos = 1000 + v
        block = (
            s2(f"v{v}")
            + s2(f"rs{v}")
            + s2("01")
            + struct.pack("<IH", pos, 2)
            + s4("A")
            + s4("G")
            + struct.pack("<II", len(packed) + 4, len(geno))
            + packed
        )
        blocks.append((pos, f"rs{v}", block))
    write_file(path, n, layout=2, blocks=blocks)
