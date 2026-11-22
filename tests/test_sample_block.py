"""The embedded sample-ID block: checked at open, parsed on first use.

Opening a file checks the block's shape (its sample count, and that its length
fits both N and the space before the variant data) but builds the ID strings
only when they are asked for, so a reader given a .sample file never pays for
them. The IDs stay readable after close(): a remote reader keeps the block's
bytes from open, and a local one keeps its mapping until the IDs are read,
unless a .sample file supplies them, in which case close() releases the file.
The block is exactly L_SI bytes long (L_SI counts its own four bytes), so it
must parse even when nothing follows it.
"""

import gc
import os
import struct
import sys

import numpy as np
import pytest
from _bgen_writer import write_file, write_v12
from lazybgen.reader import BgenReader


def _column(n):
    return (np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(n, bool))


def _block_offsets(raw):
    """(start of the sample block, its L_SI, start of the variant data)."""
    header_length = struct.unpack_from("<I", raw, 4)[0]
    start = 4 + header_length
    return start, struct.unpack_from("<I", raw, start)[0], struct.unpack_from("<I", raw, 0)[0] + 4


def test_a_sidecar_replaces_the_embedded_ids_without_parsing_them(tmp_path):
    """With a .sample file, a malformed embedded ID body is never parsed."""
    path = tmp_path / "ids.bgen"
    write_v12(path, [_column(3)] * 2, 8)
    raw = bytearray(path.read_bytes())
    start, _, _ = _block_offsets(raw)
    struct.pack_into("<H", raw, start + 8, 0xFFFF)  # first ID's length overruns the block
    path.write_bytes(bytes(raw))
    sample = tmp_path / "ids.sample"
    sample.write_text("ID_1 ID_2 missing\n0 0 0\na A 0\nb B 0\nc C 0\n")
    with BgenReader(str(path), sample_path=str(sample)) as reader:
        assert reader.samples == ["A", "B", "C"]
        dosages, _ = reader.load_variants()
    assert dosages.shape == (3, 2)


def test_a_malformed_id_raises_when_the_ids_are_first_read(tmp_path):
    path = tmp_path / "ids.bgen"
    write_v12(path, [_column(3)], 8)
    raw = bytearray(path.read_bytes())
    start, _, _ = _block_offsets(raw)
    struct.pack_into("<H", raw, start + 8, 0xFFFF)
    path.write_bytes(bytes(raw))
    with BgenReader(str(path)) as reader:
        reader.load_variants()  # the genotypes are unaffected
        with pytest.raises(RuntimeError, match="(?i)sample ID"):
            reader.samples


def test_block_ending_at_end_of_file(tmp_path):
    """A file with no variants, whose sample block is its last byte."""
    path = tmp_path / "ids.bgen"
    write_file(path, 4, layout=2, blocks=[])
    raw = path.read_bytes()
    start, size, variants = _block_offsets(raw)
    assert start + size == len(raw) == variants
    with BgenReader(str(path)) as reader:
        assert reader.samples == [f"s{i}" for i in range(4)]


def test_sample_count_mismatch_is_reported_at_open(tmp_path):
    path = tmp_path / "ids.bgen"
    write_v12(path, [_column(4)], 8)
    raw = bytearray(path.read_bytes())
    start, _, _ = _block_offsets(raw)
    struct.pack_into("<I", raw, start + 4, 5)  # N in the sample block
    path.write_bytes(bytes(raw))
    with pytest.raises(RuntimeError, match="(?i)sample count mismatch"):
        BgenReader(str(path))


@pytest.mark.parametrize("excess", [1, 1 << 20])
def test_a_block_overrunning_the_variant_data_is_reported_at_open(tmp_path, excess):
    """L_SI past the first variant (or the file) is caught at open, not at first use."""
    path = tmp_path / "ids.bgen"
    write_v12(path, [_column(4)], 8)
    raw = bytearray(path.read_bytes())
    start, size, _ = _block_offsets(raw)
    struct.pack_into("<I", raw, start, size + excess)
    path.write_bytes(bytes(raw))
    with pytest.raises(RuntimeError, match="(?i)invalid sample block"):
        BgenReader(str(path))


def test_a_block_longer_than_n_ids_can_be_is_reported_at_open(tmp_path):
    """L_SI may not exceed 8 + N * (2 + 65535), even when the file has room for it."""
    path = tmp_path / "ids.bgen"
    write_file(path, 1, layout=2, blocks=[])
    raw = bytearray(path.read_bytes())
    start, size, _ = _block_offsets(raw)
    pad = 8 + 65537 - size + 1  # one byte past the most a single ID can occupy
    struct.pack_into("<I", raw, 0, struct.unpack_from("<I", raw, 0)[0] + pad)  # the variant data moves too
    struct.pack_into("<I", raw, start, size + pad)
    path.write_bytes(bytes(raw) + bytes(pad))
    with pytest.raises(RuntimeError, match="(?i)invalid sample block"):
        BgenReader(str(path))


def _mapped(path):
    real = os.path.realpath(path)
    with open("/proc/self/maps") as f:
        return any(line.rstrip("\n").endswith(real) for line in f)


def _open_fds(path):
    real = os.path.realpath(path)
    fds = []
    for fd in os.listdir("/proc/self/fd"):
        try:
            if os.readlink(f"/proc/self/fd/{fd}") == real:
                fds.append(fd)
        except OSError:
            pass
    return fds


linux_only = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc/self")


@linux_only
def test_close_releases_the_file_when_a_sidecar_supplies_the_ids(tmp_path):
    path = tmp_path / "ids.bgen"
    write_v12(path, [_column(3)] * 2, 8)
    sample = tmp_path / "ids.sample"
    sample.write_text("ID_1 ID_2 missing\n0 0 0\na A 0\nb B 0\nc C 0\n")
    reader = BgenReader(str(path), sample_path=str(sample))
    reader.load_variants()
    assert _mapped(path) and not _open_fds(path)  # mapped, but no descriptor held
    reader.close()
    assert not _mapped(path)
    assert reader.samples == ["A", "B", "C"]


@linux_only
def test_unread_embedded_ids_keep_the_mapping_until_read(tmp_path):
    path = tmp_path / "ids.bgen"
    write_v12(path, [_column(3)] * 2, 8)
    reader = BgenReader(str(path))
    reader.close()
    assert _mapped(path) and not _open_fds(path)
    assert reader.samples == ["s0", "s1", "s2"]
    assert not _mapped(path)

    reader = BgenReader(str(path))
    reader.close()
    del reader
    gc.collect()
    assert not _mapped(path)


@linux_only
def test_ids_read_before_close_release_the_file_at_close(tmp_path):
    path = tmp_path / "ids.bgen"
    write_v12(path, [_column(3)] * 2, 8)
    reader = BgenReader(str(path))
    assert reader.samples == ["s0", "s1", "s2"]
    reader.close()
    assert not _mapped(path)
