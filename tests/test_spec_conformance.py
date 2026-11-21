"""Decoding follows the BGEN specification where the format leaves no choice.

* Layout 1 (v1.1) stores P(AA), P(AB) and P(BB) as 16-bit integers that the
  spec converts to probabilities by dividing by 32768, with all three zero
  marking a missing call. The dosage is (P(AB) + 2*P(BB)) / 32768 even when
  the stored triplet does not sum to 32768; the reader must not renormalize by
  the per-sample sum. (bgen 1.9.0 and qctool scale the same way.)
* Compressed blocks are zlib streams (RFC 1950). Any valid zlib header must be
  accepted, whatever window size the writer chose, and a raw deflate stream,
  which some non-conforming writers produce, still reads.
"""

import struct
import zlib

import numpy as np
import pytest
from lazybgen.reader import BgenReader
from test_fused_decode import _reference, _s2, _s4, _write_file, _write_v12

NAN32 = np.float32(np.nan)


def _write_v11(path, columns):
    """Write a zlib-compressed layout-1 BGEN; each column is (aa, ab, bb) integer arrays."""
    n = len(columns[0][0])
    blocks = []
    for v, (aa, ab, bb) in enumerate(columns):
        geno = b"".join(struct.pack("<HHH", int(a), int(b), int(c)) for a, b, c in zip(aa, ab, bb))
        packed = zlib.compress(geno)
        pos = 1000 + v
        block = (
            struct.pack("<I", n)
            + _s2(f"v{v}")
            + _s2(f"rs{v}")
            + _s2("01")
            + struct.pack("<I", pos)
            + _s4("A")
            + _s4("G")
            + struct.pack("<I", len(packed))
            + packed
        )
        blocks.append((pos, f"rs{v}", block))
    _write_file(path, n, layout=1, blocks=blocks)


def _v11_reference(columns):
    out = []
    for aa, ab, bb in columns:
        aa, ab, bb = (np.asarray(x, dtype=np.int64) for x in (aa, ab, bb))
        val = (ab + 2 * bb).astype(np.float32) / np.float32(32768)
        out.append(np.where((aa | ab | bb) == 0, NAN32, val).astype(np.float32))
    return np.stack(out, axis=1)


def _bits(a):
    a = np.ascontiguousarray(a)
    return a.view(np.uint32 if a.dtype == np.float32 else np.uint64)


@pytest.fixture
def v11_file(tmp_path):
    rng = np.random.default_rng(11)
    n = 37  # not a multiple of the 4- or 8-lane vector width
    columns = []
    # Triplets that sum to 32768, as a conforming writer rounds them.
    aa = rng.integers(0, 32769, n)
    ab = (rng.random(n) * (32768 - aa + 1)).astype(np.int64)
    columns.append((aa, ab, 32768 - aa - ab))
    # Triplets that do not sum to 32768 (rounding drift, or a lax writer).
    columns.append((rng.integers(0, 30000, n), rng.integers(0, 30000, n), rng.integers(0, 30000, n)))
    # Missing calls scattered between valid ones, including the scalar tail.
    aa, ab, bb = (rng.integers(0, 20000, n) for _ in range(3))
    for i in (0, 5, 17, n - 1):
        aa[i] = ab[i] = bb[i] = 0
    columns.append((aa, ab, bb))
    # The extremes: a stored 65535 and a sample with only P(AA) set.
    columns.append((np.zeros(n, np.int64), np.full(n, 65535), np.full(n, 65535)))
    columns.append((np.full(n, 32768), np.zeros(n, np.int64), np.zeros(n, np.int64)))
    path = tmp_path / "spec.v11.bgen"
    _write_v11(path, columns)
    return path, columns


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_v11_dosage_scales_by_32768(v11_file, dtype):
    path, columns = v11_file
    want = _v11_reference(columns).astype(dtype)
    with BgenReader(str(path)) as reader:
        got, _ = reader.load_variants(dtype=dtype)
    assert (_bits(got) == _bits(want)).all()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_v11_filtered_read_is_bitwise_the_full_read(v11_file, dtype):
    path, columns = v11_file
    n = len(columns[0][0])
    rng = np.random.default_rng(0)
    with BgenReader(str(path)) as reader:
        full, _ = reader.load_variants(dtype=dtype)
        for idx in (np.arange(n)[::-1], rng.permutation(n), rng.integers(0, n, 50), np.array([n - 1])):
            idx = idx.astype(np.int32)
            sub, _ = reader.load_variants(dtype=dtype, sample_indices=idx)
            assert (_bits(sub) == _bits(full[idx, :])).all()


def _deflate(wbits, level=6):
    def compress(data):
        c = zlib.compressobj(level, zlib.DEFLATED, wbits)
        return c.compress(data) + c.flush()

    return compress


# Every zlib window size (CINFO 1..7, i.e. wbits 9..15) and the four FLEVEL
# values that zlib writes into the header's second byte, then raw deflate.
COMPRESSORS = {f"zlib wbits={w} level={lv}": _deflate(w, lv) for w in range(9, 16) for lv in (1, 2, 6, 9)}
COMPRESSORS["raw deflate"] = _deflate(-15)


@pytest.mark.parametrize("name", list(COMPRESSORS))
def test_every_zlib_header_decodes(tmp_path, name):
    rng = np.random.default_rng(3)
    n = 300
    aa = rng.integers(0, 256, n)
    ab = (rng.random(n) * (255 - aa + 1)).astype(np.int64)
    columns = [(aa, ab, rng.random(n) < 0.05)]
    path = tmp_path / "z.bgen"
    _write_v12(path, columns, 8, compress=COMPRESSORS[name])
    with BgenReader(str(path)) as reader:
        got, _ = reader.load_variants(dtype=np.float32)
    assert (_bits(got) == _bits(_reference(columns, 8))).all()
