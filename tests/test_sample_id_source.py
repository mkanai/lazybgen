"""Where `samples` comes from: a .sample file, the embedded block, or placeholders.

haplotypes.bgen shows embedded IDs can themselves be named sample_<i>.
"""

import struct
from pathlib import Path

import pytest
from lazybgen.reader import BgenReader

DATA_DIR = Path(__file__).parent / "data"

# Flag bit 31 of each bundled fixture, checked independently from the raw header.
FIXTURE_FLAGS = {
    "data.bgen": False,
    "example.v11.bgen": False,
    "example.8bits.bgen": True,
    "example.16bits.bgen": True,
    "example.32bits.bgen": True,
    "example.16bits.zstd.bgen": True,
    "haplotypes.bgen": True,
}


def _header_flag(path):
    """Bit 31 of the header flags, read straight from the file."""
    raw = Path(path).read_bytes()
    header_length = struct.unpack_from("<I", raw, 4)[0]
    flags = struct.unpack_from("<I", raw, header_length)[0]  # last field of the header block [4, 4 + L_H)
    return bool((flags >> 31) & 1)


def _write_sample_file(path, ids):
    lines = ["ID_1 ID_2 missing", "0 0 0"] + [f"fid{i} {sid} 0" for i, sid in enumerate(ids)]
    path.write_text("\n".join(lines) + "\n")


@pytest.mark.parametrize("name, expected", sorted(FIXTURE_FLAGS.items()))
def test_property_matches_the_header_flag(name, expected):
    path = DATA_DIR / name
    assert _header_flag(path) is expected  # the table above is the file's truth
    with BgenReader(str(path)) as reader:
        pass
    assert reader.has_embedded_sample_ids is expected  # read at open, valid after close
    assert reader.sample_id_source == ("embedded" if expected else "placeholder")


def test_placeholder_names_do_not_reveal_the_source():
    """haplotypes.bgen has real embedded IDs that look like placeholders."""
    with BgenReader(str(DATA_DIR / "haplotypes.bgen")) as embedded, BgenReader(str(DATA_DIR / "data.bgen")) as bare:
        assert embedded.samples[:2] == bare.samples[:2] == ["sample_0", "sample_1"]
        assert embedded.has_embedded_sample_ids and not bare.has_embedded_sample_ids


def test_a_relabelled_sample_file_overrides_differing_embedded_ids(tmp_path):
    """The .sample ID_2 wins even when the embedded block names samples otherwise."""
    path = DATA_DIR / "example.16bits.bgen"
    with BgenReader(str(path)) as reader:
        embedded = list(reader.samples)
    relabelled = list(reversed(embedded))
    assert relabelled != embedded
    sample = tmp_path / "relabelled.sample"
    _write_sample_file(sample, relabelled)

    with BgenReader(str(path), sample_path=str(sample)) as reader:
        assert reader.samples == relabelled
        assert reader.has_embedded_sample_ids
        assert reader.sample_id_source == "sample_file"


@pytest.mark.parametrize("n_rows", [0, 499, 501])
def test_a_sample_file_with_the_wrong_row_count_is_an_error(tmp_path, n_rows):
    """A .sample file must name every BGEN sample; a rejected file is not cached."""
    from lazybgen import reader as reader_module

    reader_module._SAMPLE_ID_CACHE.clear()
    path = DATA_DIR / "example.16bits.bgen"
    with BgenReader(str(path)) as reader:
        assert reader.nsamples == 500
    sample = tmp_path / f"rows{n_rows}.sample"
    _write_sample_file(sample, [f"S{i}" for i in range(n_rows)])
    with BgenReader(str(path), sample_path=str(sample)) as reader:
        assert reader.sample_id_source == "sample_file"
        for _ in range(2):  # the failure is not cached as a partial list
            with pytest.raises(ValueError, match=f"{n_rows} sample rows but the BGEN has 500 samples"):
                reader.samples
    assert not reader_module._SAMPLE_ID_CACHE


def test_a_cached_sample_file_is_rechecked_against_each_bgen(tmp_path):
    """The cache is keyed by the .sample file, so a hit is checked against the BGEN too."""
    big, small = DATA_DIR / "example.16bits.bgen", DATA_DIR / "data.bgen"
    with BgenReader(str(big)) as r_big, BgenReader(str(small)) as r_small:
        n_big, n_small = r_big.nsamples, r_small.nsamples
    assert n_big != n_small
    sample = tmp_path / "big.sample"
    _write_sample_file(sample, [f"S{i}" for i in range(n_big)])
    with BgenReader(str(big), sample_path=str(sample)) as reader:
        assert len(reader.samples) == n_big  # parsed and cached
    with BgenReader(str(small), sample_path=str(sample)) as reader:
        with pytest.raises(ValueError, match=f"{n_big} sample rows but the BGEN has {n_small} samples"):
            reader.samples


@pytest.mark.parametrize("name, expected", [("example.16bits.bgen", True), ("data.bgen", False)])
def test_the_properties_do_not_read_the_sample_file_rows(tmp_path, name, expected):
    """The .sample rows are read on the first `samples` access, not by the properties.

    The file is removed after open: the properties still answer, and only the
    first `samples` access fails, so neither property loaded the rows.
    """
    path = DATA_DIR / name
    with BgenReader(str(path)) as reader:
        n = reader.nsamples
    sample = tmp_path / "late.sample"
    _write_sample_file(sample, [f"L{i}" for i in range(n)])
    with BgenReader(str(path), sample_path=str(sample)) as reader:
        sample.unlink()
        assert reader.has_embedded_sample_ids is expected
        assert reader.sample_id_source == "sample_file"
        with pytest.raises(FileNotFoundError):
            reader.samples
    assert reader.has_embedded_sample_ids is expected  # and after close
    assert reader.sample_id_source == "sample_file"
