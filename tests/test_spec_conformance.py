"""Decoding follows the BGEN specification where the format leaves no choice.

* Layout 1 (v1.1) stores P(AA), P(AB) and P(BB) as 16-bit integers that the
  spec converts to probabilities by dividing by 32768, with all three zero
  marking a missing call. The dosage is (P(AB) + 2*P(BB)) / 32768 even when
  the stored triplet does not sum to 32768; the reader must not renormalize by
  the per-sample sum. (bgen 1.9.0 and qctool scale the same way.)
* Compressed blocks are zlib streams (RFC 1950). Any valid zlib header must be
  accepted, whatever window size the writer chose, and a raw deflate stream,
  which some non-conforming writers produce, still reads, including one whose
  first bytes happen to form a valid zlib header.
"""

import zlib

import numpy as np
import pytest
from _bgen_writer import write_v11, write_v12
from lazybgen.reader import BgenReader
from test_fused_decode import _bits, _reference

from lazybgen import load_bgen

NAN32 = np.float32(np.nan)


def _v11_reference(columns):
    out = []
    for aa, ab, bb in columns:
        aa, ab, bb = (np.asarray(x, dtype=np.int64) for x in (aa, ab, bb))
        val = (ab + 2 * bb).astype(np.float32) / np.float32(32768)
        out.append(np.where((aa | ab | bb) == 0, NAN32, val).astype(np.float32))
    return np.stack(out, axis=1)


def _v11_columns(n, rng):
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
    return columns


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("num_threads", [1, 2])
def test_v11_dosage_scales_by_32768(tmp_path, dtype, num_threads):
    """Full and sample-filtered reads both equal the spec's 1/32768 scaling."""
    n = 37  # not a multiple of the 4- or 8-lane vector width
    rng = np.random.default_rng(11)
    columns = _v11_columns(n, rng)
    path = tmp_path / "spec.v11.bgen"
    write_v11(path, columns)
    want = _v11_reference(columns).astype(dtype)
    with BgenReader(str(path), num_threads=num_threads) as reader:
        got, _ = reader.load_variants(dtype=dtype)
        assert (_bits(got) == _bits(want)).all()
        for idx in (rng.permutation(n), rng.integers(0, n, 50), np.array([n - 1])):
            idx = idx.astype(np.int32)
            sub, _ = reader.load_variants(dtype=dtype, sample_indices=idx)
            assert (_bits(sub) == _bits(want[idx, :])).all()


def test_v11_rounding_past_two_is_accepted_but_not_more(tmp_path):
    """load_bgen allows the one unit past 2 that rounding a valid triplet can give.

    (0, 1/65536, 65535/65536) rounds half up to (0, 1, 32768), a dosage of
    65537/32768. Anything larger cannot come from a conforming writer.
    """
    n = 3
    ok = tmp_path / "ok.v11.bgen"
    write_v11(ok, [(np.zeros(n, int), np.ones(n, int), np.full(n, 32768))] * 2)
    dosages, _, _ = load_bgen(str(ok))
    assert dosages.max() == 65537 / 32768

    bad = tmp_path / "bad.v11.bgen"
    write_v11(bad, [(np.zeros(n, int), np.full(n, 2), np.full(n, 32768))] * 2)
    with pytest.raises(ValueError, match="out of valid range"):
        load_bgen(str(bad))


def _deflate(wbits, level=6):
    def compress(data):
        c = zlib.compressobj(level, zlib.DEFLATED, wbits)
        return c.compress(data) + c.flush()

    return compress


def _zlib_cinfo0(data):
    """A zlib stream declaring a 256-byte window (CINFO 0), which Python's zlib never writes.

    Stored (level 0) blocks use no back-references, so any window is honest.
    """
    body = zlib.compress(data, 0)[2:]  # deflate stream + Adler-32
    cmf = 0x08
    flg = (31 - (cmf << 8) % 31) % 31
    return bytes([cmf, flg]) + body


def _raw_with_zlib_lookalike_header(data):
    """Raw deflate whose first two bytes, 0x08 0x1d, also form a valid zlib header.

    A non-final stored block of 29 bytes (BFINAL 0, BTYPE 00 in the low bits of
    0x08, then LEN 0x001d / NLEN 0xffe2), followed by the rest of the data as a
    final block. Read as raw deflate it is the data. Read as zlib, the stream
    after the header is another stored block whose LEN is 0xffe2 and whose NLEN
    is the data's first two bytes; when those are 1d 00 (a genotype block for
    29 samples starts with N = 29) that block is well formed and 65,506 bytes
    long, so zlib decoding runs out of output space instead of rejecting the
    data, and only a fallback that retries on any zlib failure reads it.
    """
    head = data[:29]
    first = bytes([0x08, 0x1D, 0x00, 0xE2, 0xFF]) + head
    assert ((first[0] << 8) | first[1]) % 31 == 0  # a valid zlib header too
    return first + _deflate(-15)(data[29:])


# Every zlib window size (CINFO 0 to 7) at one level, the other FLEVEL values
# at the default window, raw deflate, and raw deflate that looks like zlib.
COMPRESSORS = {f"zlib wbits={w}": _deflate(w) for w in range(9, 16)}
COMPRESSORS.update({f"zlib level={lv}": _deflate(15, lv) for lv in (1, 2, 9)})
COMPRESSORS["zlib CINFO=0"] = _zlib_cinfo0
COMPRESSORS["raw deflate"] = _deflate(-15)
COMPRESSORS["raw deflate, zlib-like header"] = _raw_with_zlib_lookalike_header


@pytest.mark.parametrize("name", list(COMPRESSORS))
def test_every_zlib_header_decodes(tmp_path, name):
    rng = np.random.default_rng(3)
    n = 29  # see _raw_with_zlib_lookalike_header
    aa = rng.integers(0, 256, n)
    ab = (rng.random(n) * (255 - aa + 1)).astype(np.int64)
    columns = [(aa, ab, rng.random(n) < 0.05)] * 2
    path = tmp_path / "z.bgen"
    write_v12(path, columns, 8, compress=COMPRESSORS[name])
    for num_threads in (1, 2):
        with BgenReader(str(path), num_threads=num_threads) as reader:
            got, _ = reader.load_variants(dtype=np.float32)
        assert (_bits(got) == _bits(_reference(columns, 8))).all()


@pytest.mark.parametrize("num_threads", [0, 1])
def test_v11_blocks_in_a_layout2_file_get_the_same_tolerance(tmp_path, num_threads):
    """A layout-2 file carrying v1.1 blocks decodes them as layout 1, rounding included.

    The first triplet, (0, 1, 32768), also keeps the block from parsing as
    layout 2 (its first four bytes are not N), which is what sends the reader to
    the v1.1 decoder. load_bgen must then accept the 65537/32768 that rounding
    gives, whatever the header's layout says, and still reject anything larger.
    """
    from _bgen_writer import write_v12_with_v11_blocks

    ok = tmp_path / "ok.bgen"
    aa, ab, bb = np.array([0, 32768, 0]), np.array([1, 0, 16384]), np.array([32768, 0, 8192])
    write_v12_with_v11_blocks(ok, [(aa, ab, bb)] * 2)
    dosages, _, _ = load_bgen(str(ok), num_threads=num_threads)
    assert dosages.max() == 65537 / 32768
    np.testing.assert_array_equal(dosages[:, 0], [65537 / 32768, 0.0, (16384 + 2 * 8192) / 32768])

    bad = tmp_path / "bad.bgen"
    write_v12_with_v11_blocks(bad, [(aa, np.array([1, 0, 2]), np.array([32768, 0, 32768]))] * 2)
    with pytest.raises(ValueError, match=r"out of valid range \[0, 2\.00003\]"):
        load_bgen(str(bad), num_threads=num_threads)
