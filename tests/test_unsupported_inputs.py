"""Rejection contract: inputs lazybgen must refuse, with clear errors.

lazybgen decodes biallelic, diploid, COMPRESSED (zlib/zstd) BGEN, phased or
unphased. Anything else must fail loudly rather than return silently-wrong
dosages. Phased decoding lives in test_phased.py (it needs the bgen writer);
this module covers multiallelic, non-diploid, and uncompressed, synthesizing the
fixtures directly so the test actually exercises the rejection path.
"""

import os
import sqlite3
import struct
import time

import numpy as np
import pytest

from lazybgen import load_bgen

try:
    import bgen as external_bgen

    HAS_EXTERNAL_BGEN = True
except ImportError:
    external_bgen = None
    HAS_EXTERNAL_BGEN = False


# ==================== Multi-allelic (n_alleles != 2) ====================


@pytest.mark.skipif(not HAS_EXTERNAL_BGEN, reason="bgen library required to write a multiallelic fixture")
def test_multiallelic_rejected(tmp_path):
    """A 3-allele variant must be rejected: dosage computation is biallelic-only.

    The dosage fast paths assume two stored probabilities per diploid sample.
    A multiallelic variant has a different genotype-probability layout, so
    decoding it as biallelic would yield silently wrong dosages.
    """
    from bgen import BgenWriter

    path = tmp_path / "multiallelic.bgen"
    # Diploid, 3 alleles => 6 genotype probabilities per sample.
    geno = np.array(
        [
            [1.0, 0, 0, 0, 0, 0],
            [0, 1.0, 0, 0, 0, 0],
            [0, 0, 1.0, 0, 0, 0],
        ]
    )
    with BgenWriter(str(path), n_samples=3, samples=["s0", "s1", "s2"]) as w:
        w.add_variant("v1", "rs1", "1", 1000, ["A", "B", "C"], geno, ploidy=2, phased=False, bit_depth=8)

    with pytest.raises((ValueError, RuntimeError), match="(?i)biallelic"):
        load_bgen(file_path=str(path), nan_action="warn")
    with pytest.raises((ValueError, RuntimeError), match="(?i)biallelic"):
        load_bgen(file_path=str(path), sample_ids=["s0"], nan_action="warn")


# ==================== Non-diploid (ploidy != 2) ====================


@pytest.mark.skipif(not HAS_EXTERNAL_BGEN, reason="bgen library required to write a non-diploid fixture")
def test_non_diploid_rejected(tmp_path):
    """Non-diploid (e.g. haploid) variants must be rejected with a clear error.

    The dosage fast paths assume diploid biallelic layout (two stored
    probabilities per sample); haploid/variable-ploidy data would otherwise be
    mis-decoded (or, with the SIMD filtered path, silently skip missingness).
    """
    from bgen import BgenWriter

    path = tmp_path / "haploid.bgen"
    # Haploid biallelic: P(A), P(B) per sample.
    geno = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    with BgenWriter(str(path), n_samples=3, samples=["s0", "s1", "s2"]) as w:
        w.add_variant("v1", "rs1", "1", 1000, ["A", "B"], geno, ploidy=1, phased=False, bit_depth=8)

    with pytest.raises((ValueError, RuntimeError), match="(?i)diploid"):
        load_bgen(file_path=str(path), nan_action="warn")
    with pytest.raises((ValueError, RuntimeError), match="(?i)diploid"):
        load_bgen(file_path=str(path), sample_ids=["s0"], nan_action="warn")


# ==================== Uncompressed (compression flag 0) ====================


def _write_uncompressed_v11_bgen(path):
    """Write a minimal valid UNCOMPRESSED v1.1 BGEN (compression flag 0).

    The bgen library writer always compresses, so we hand-build the smallest
    well-formed uncompressed file: one biallelic diploid variant, three samples.
    Returns (variant_file_start, variant_size_bytes, variant_info_dict) so a
    matching .bgi can be written.
    """
    samples = ["s0", "s1", "s2"]
    n_samples = len(samples)
    varid, rsid, chrom, pos, a1, a2 = "v1", "rs1", "01", 1000, "A", "G"

    # v1.1 uncompressed genotype block: 6 bytes/sample (3 little-endian uint16 probs).
    probs = [(65535, 0, 0), (0, 65535, 0), (0, 0, 65535)]
    geno = b"".join(struct.pack("<HHH", *p) for p in probs)

    # Sample identifier block: [4-byte total length][4-byte N][ (2-byte len + id) ... ].
    ids = b"".join(struct.pack("<H", len(s)) + s.encode() for s in samples)
    sample_block = struct.pack("<I", 8 + len(ids)) + struct.pack("<I", n_samples) + ids

    # Header block (length = header_length = 20): nvariants | nsamples | "bgen" | flags.
    # flags: layout 1 (v1.1) in bits 2-5, compression 0 (none), sample-ids-present bit 31.
    flags = (1 << 2) | (1 << 31)
    header_length = 20
    header_block = struct.pack("<I", 1) + struct.pack("<I", n_samples) + b"bgen" + struct.pack("<I", flags)
    offset_val = header_length + len(sample_block)
    prefix = struct.pack("<I", offset_val) + struct.pack("<I", header_length) + header_block

    def s2(x):
        return struct.pack("<H", len(x)) + x.encode()

    def s4(x):
        return struct.pack("<I", len(x)) + x.encode()

    # v1.1 variant block: nsamples | varid | rsid | chrom | pos | allele1 | allele2 | geno.
    vblock = (
        struct.pack("<I", n_samples)
        + s2(varid)
        + s2(rsid)
        + s2(chrom)
        + struct.pack("<I", pos)
        + s4(a1)
        + s4(a2)
        + geno
    )
    variant_file_start = len(prefix) + len(sample_block)
    with open(path, "wb") as f:
        f.write(prefix)
        f.write(sample_block)
        f.write(vblock)
    return variant_file_start, len(vblock), {"chrom": chrom, "pos": pos, "rsid": rsid, "a1": a1, "a2": a2}


def _write_bgi(bgen_path, file_start, size_bytes, vi):
    """Write a minimal .bgi (sqlite) for the single-variant fixture."""
    bgi_path = str(bgen_path) + ".bgi"
    conn = sqlite3.connect(bgi_path)
    try:
        conn.execute(
            "CREATE TABLE Metadata (filename TEXT NOT NULL, file_size INT NOT NULL, "
            "last_write_time INT NOT NULL, first_1000_bytes BLOB NOT NULL, index_creation_time INT NOT NULL)"
        )
        with open(bgen_path, "rb") as f:
            first = f.read(1000)
        now = int(time.time())
        conn.execute(
            "INSERT INTO Metadata VALUES (?,?,?,?,?)",
            (os.path.basename(str(bgen_path)), os.path.getsize(bgen_path), now, first, now),
        )
        conn.execute(
            "CREATE TABLE Variant (chromosome TEXT NOT NULL, position INT NOT NULL, rsid TEXT NOT NULL, "
            "number_of_alleles INT NOT NULL, allele1 TEXT NOT NULL, allele2 TEXT NULL, "
            "file_start_position INT NOT NULL, size_in_bytes INT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO Variant VALUES (?,?,?,?,?,?,?,?)",
            (vi["chrom"], vi["pos"], vi["rsid"], 2, vi["a1"], vi["a2"], file_start, size_bytes),
        )
        conn.commit()
    finally:
        conn.close()
    return bgi_path


def test_uncompressed_bgen_rejected(tmp_path):
    """An uncompressed (compression flag 0) BGEN must raise a clear error.

    The reader's header parser accepts CompressionType::None, but genotype decode
    rejects it explicitly, so this asserts the decode path (not just the header).
    """
    path = tmp_path / "uncompressed.bgen"
    file_start, size_bytes, vi = _write_uncompressed_v11_bgen(path)
    _write_bgi(path, file_start, size_bytes, vi)

    with pytest.raises((ValueError, RuntimeError), match="(?i)uncompressed"):
        load_bgen(file_path=str(path), nan_action="warn")


def test_uncompressed_fixture_is_well_formed_except_compression(tmp_path):
    """Guard: the synthesized fixture parses far enough to reach the compression check.

    If the hand-built header/variant layout were malformed, the rejection above
    could fire for the wrong reason (a parse error). Here we confirm the reader
    opens the file and reads samples/variant metadata cleanly; only the genotype
    decode (the compression branch) fails. This keeps the rejection test honest.
    """
    from lazybgen.reader import BgenReader

    path = tmp_path / "uncompressed2.bgen"
    file_start, size_bytes, vi = _write_uncompressed_v11_bgen(path)
    _write_bgi(path, file_start, size_bytes, vi)

    with BgenReader(str(path)) as reader:
        assert reader.nsamples == 3
        assert list(reader.samples) == ["s0", "s1", "s2"]
        assert reader.nvariants == 1
        # Metadata is readable; only decoding the genotype block must fail.
        # (BgenReader.load_variants returns raw dosages and takes no nan_action.)
        with pytest.raises((ValueError, RuntimeError), match="(?i)uncompressed"):
            reader.load_variants()
