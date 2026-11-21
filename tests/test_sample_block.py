"""The embedded sample-ID block: located at open, parsed on first use.

A reader checks the block's sample count when it opens the file but builds the
ID strings only when they are asked for, so a reader given a .sample file never
pays for them. The block is exactly L_SI bytes long (L_SI counts its own four
bytes), so it must parse even when nothing follows it in the file.
"""

import struct

import numpy as np
import pytest
from lazybgen.reader import BgenReader
from test_fused_decode import _s2, _write_v12


def _column(n):
    return (np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(n, bool))


def test_embedded_ids_round_trip(tmp_path):
    path = tmp_path / "ids.bgen"
    _write_v12(path, [_column(5)], 8)
    with BgenReader(str(path)) as reader:
        assert reader.samples == [f"s{i}" for i in range(5)]
        dosages, _ = reader.load_variants()
    assert dosages.shape == (5, 1)


def test_a_sidecar_replaces_the_embedded_ids(tmp_path):
    path = tmp_path / "ids.bgen"
    _write_v12(path, [_column(3)], 8)
    sample = tmp_path / "ids.sample"
    sample.write_text("ID_1 ID_2 missing\n0 0 0\na A 0\nb B 0\nc C 0\n")
    with BgenReader(str(path), sample_path=str(sample)) as reader:
        assert reader.samples == ["A", "B", "C"]


def test_block_ending_at_end_of_file(tmp_path):
    """A file whose sample block is its last byte (no variant data after it)."""
    path = tmp_path / "ids.bgen"
    _write_v12(path, [_column(4)], 8)
    raw = path.read_bytes()
    header_length = struct.unpack_from("<I", raw, 4)[0]
    block_size = struct.unpack_from("<I", raw, 4 + header_length)[0]
    end = 4 + header_length + block_size
    # Keep the variant in the index but cut the file after the sample block:
    # opening and listing samples must not need any byte past it.
    path.write_bytes(raw[:end])
    with BgenReader(str(path)) as reader:
        assert reader.samples == [f"s{i}" for i in range(4)]


def test_sample_count_mismatch_is_reported_at_open(tmp_path):
    path = tmp_path / "ids.bgen"
    _write_v12(path, [_column(4)], 8)
    raw = bytearray(path.read_bytes())
    header_length = struct.unpack_from("<I", raw, 4)[0]
    struct.pack_into("<I", raw, 4 + header_length + 4, 5)  # N in the sample block
    path.write_bytes(bytes(raw))
    with pytest.raises(RuntimeError, match="(?i)sample count mismatch"):
        BgenReader(str(path))
