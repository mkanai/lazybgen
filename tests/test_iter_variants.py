"""Tests for BgenReader.iter_variants: memory-bounded streaming read.

iter_variants yields one (info, dosage) pair per variant, where `info` is a
dict with the variant_info fields (chrom, pos, rsid, ref, alt) and `dosage` is a
1-D ndarray of per-sample dosages. It must reproduce the dosages returned by the
batch load_variants() exactly, while never materializing the full
(n_samples x n_variants) matrix.
"""

from pathlib import Path

import numpy as np
import pytest
from lazybgen.reader import BgenReader

from lazybgen import load_bgen


@pytest.fixture(scope="module")
def bgen_path():
    return str(Path(__file__).parent / "data" / "data.bgen")


def test_iter_variants_reproduces_load_variants(bgen_path):
    """Streaming per-variant read equals the batch column-by-column."""
    with BgenReader(bgen_path) as reader:
        dosages, vinfo = reader.load_variants(dtype=np.float64)  # (n_samples, n_variants)
        rows = list(reader.iter_variants(dtype=np.float64))

    # one yield per variant, in file order
    assert len(rows) == dosages.shape[1]

    n_samples = dosages.shape[0]
    for j, (info, dosage) in enumerate(rows):
        assert isinstance(dosage, np.ndarray)
        assert dosage.shape == (n_samples,)
        # NaN/missing pattern matches the batch column
        np.testing.assert_array_equal(np.isnan(dosage), np.isnan(dosages[:, j]))
        finite = ~np.isnan(dosage)
        np.testing.assert_allclose(dosage[finite], dosages[finite, j])
        # info carries the variant fields for this column
        assert str(info["chrom"]) == str(vinfo.iloc[j]["chrom"])
        assert int(info["pos"]) == int(vinfo.iloc[j]["pos"])
        assert str(info["rsid"]) == str(vinfo.iloc[j]["rsid"])
        assert str(info["ref"]) == str(vinfo.iloc[j]["ref"])
        assert str(info["alt"]) == str(vinfo.iloc[j]["alt"])


def test_iter_variants_info_is_plain_dict(bgen_path):
    """`info` is yielded as a plain dict (not a pandas Series).

    Building a per-row pandas Series via ``info.iloc[j]`` dominated the streaming
    path (~40% of wall time). Yielding a dict from a per-block
    ``to_dict('records')`` preserves subscript access (``info['chrom']``) at a
    fraction of the cost. This pins the lighter contract.
    """
    with BgenReader(bgen_path) as reader:
        _dosages, vinfo = reader.load_variants(dtype=np.float64)
        first_info, _first_dosage = next(reader.iter_variants(dtype=np.float64))

    assert isinstance(first_info, dict)
    assert set(first_info.keys()) == {"chrom", "pos", "rsid", "ref", "alt"}
    # values still match the batch's first variant
    assert str(first_info["chrom"]) == str(vinfo.iloc[0]["chrom"])
    assert int(first_info["pos"]) == int(vinfo.iloc[0]["pos"])


def _stack(rows):
    """Stack streamed (info, dosage) pairs into an (n_samples, n_variants) array."""
    return np.column_stack([d for _, d in rows]) if rows else np.empty((0, 0))


def test_iter_variants_block_size_invariant(bgen_path):
    """Result is identical regardless of block_size (memory knob is transparent)."""
    with BgenReader(bgen_path) as reader:
        small = _stack(list(reader.iter_variants(dtype=np.float64, block_size=1)))
        big = _stack(list(reader.iter_variants(dtype=np.float64, block_size=100000)))
        auto = _stack(list(reader.iter_variants(dtype=np.float64)))  # block_size=None -> auto
    assert small.shape == big.shape == auto.shape
    np.testing.assert_array_equal(np.isnan(small), np.isnan(big))
    np.testing.assert_array_equal(np.isnan(small), np.isnan(auto))
    finite = ~np.isnan(small)
    np.testing.assert_array_equal(small[finite], big[finite])
    np.testing.assert_array_equal(small[finite], auto[finite])


@pytest.mark.parametrize("block_size", [0, -5])
def test_iter_variants_rejects_non_positive_block_size(bgen_path, block_size):
    """An explicit block_size < 1 is rejected (None means auto)."""
    with BgenReader(bgen_path) as reader:
        with pytest.raises(ValueError):
            next(reader.iter_variants(block_size=block_size))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_iter_variants_parallel_matches_serial(bgen_path, dtype):
    """Streaming under parallel decode (num_threads > 1) equals the serial stream.

    iter_variants decodes each block via the same path as load_variants, so a
    parallel reader streams blocks across worker threads; output must be
    byte-identical to the sequential stream.
    """
    with BgenReader(bgen_path, num_threads=1) as r:
        serial = _stack(list(r.iter_variants(dtype=dtype, block_size=7)))
    with BgenReader(bgen_path, num_threads=4) as r:
        par = _stack(list(r.iter_variants(dtype=dtype, block_size=7)))
    assert serial.shape == par.shape
    np.testing.assert_array_equal(np.isnan(serial), np.isnan(par))
    finite = ~np.isnan(serial)
    np.testing.assert_array_equal(serial[finite], par[finite])


def test_iter_variants_region_matches_batch(bgen_path):
    """Streaming a region equals the batch region load."""
    with BgenReader(bgen_path) as reader:
        _, vinfo = reader.load_variants(dtype=np.float64)
        chrom = str(vinfo.iloc[0]["chrom"])
        positions = sorted(int(p) for p in vinfo[vinfo["chrom"] == chrom]["pos"])
        start, end = positions[0], positions[len(positions) // 2]

        batch, _ = reader.load_variants(region_chrom=chrom, region_start=start, region_end=end, dtype=np.float64)
        rows = list(reader.iter_variants(region_chrom=chrom, region_start=start, region_end=end, dtype=np.float64))

    assert len(rows) == batch.shape[1]
    streamed = _stack(rows)
    np.testing.assert_array_equal(np.isnan(streamed), np.isnan(batch))
    finite = ~np.isnan(streamed)
    np.testing.assert_array_equal(streamed[finite], batch[finite])


@pytest.mark.parametrize("from_middle", [False, True])
def test_open_ended_region_returns_the_rest_of_the_chromosome(bgen_path, from_middle):
    """Omitting region_end selects through the end of the chromosome.

    Without an end, the reader asks the index for positions up to 0xFFFFFFFF.
    Bound as a 32-bit int, that became -1 and matched nothing.
    """
    with BgenReader(bgen_path) as reader:
        _, vinfo = reader.load_variants(dtype=np.float64)
        chrom = str(vinfo.iloc[0]["chrom"])
        on_chrom = vinfo[vinfo["chrom"].astype(str) == chrom]
        bounds = {}
        if from_middle:
            bounds["region_start"] = int(on_chrom["pos"].iloc[len(on_chrom) // 2])
        expected = on_chrom[on_chrom["pos"] >= bounds.get("region_start", 0)]

        batch, binfo = reader.load_variants(region_chrom=chrom, dtype=np.float64, **bounds)
        rows = list(reader.iter_variants(region_chrom=chrom, dtype=np.float64, **bounds))

    assert 0 < len(expected) < len(on_chrom) or not from_middle
    assert batch.shape[1] == len(expected)
    assert list(binfo["pos"]) == list(expected["pos"])
    assert [info["pos"] for info, _ in rows] == list(expected["pos"])


def test_iter_variants_sample_indices(bgen_path):
    """sample_indices restricts each yielded dosage to the chosen samples."""
    idx = np.array([0, 2, 4], dtype=np.int32)
    with BgenReader(bgen_path) as reader:
        batch, _ = reader.load_variants(sample_indices=idx, dtype=np.float64)
        rows = list(reader.iter_variants(sample_indices=idx, dtype=np.float64))
    streamed = _stack(rows)
    assert streamed.shape == batch.shape
    assert streamed.shape[0] == len(idx)
    finite = ~np.isnan(streamed)
    np.testing.assert_array_equal(streamed[finite], batch[finite])


def test_iter_variants_is_lazy_iterator(bgen_path):
    """iter_variants returns a lazy iterator advanced on demand, not a list."""
    with BgenReader(bgen_path) as reader:
        gen = reader.iter_variants(dtype=np.float64)
        # Lazy: an iterator (has __next__), not an eagerly-built list/tuple
        assert not isinstance(gen, (list, tuple))
        assert iter(gen) is gen
        first = next(gen)
        assert len(first) == 2  # (info, dosage)
        assert isinstance(first[1], np.ndarray)


def test_positions_past_two_to_the_31(tmp_path):
    """Positions at and above 2^31 match through every selection path.

    The index query bound its coordinates as 32-bit signed ints, so a region
    end or a filtered position from 2^31 up wrapped negative and matched
    nothing. Positions are unsigned 32-bit in the format, so 2^32 - 1 is legal.
    """
    from _bgen_writer import write_v12

    n = 4
    rng = np.random.default_rng(31)
    positions = [1000, 2**31 - 1, 2**31, 2**32 - 1]
    columns = [(rng.integers(0, 128, n), rng.integers(0, 128, n), np.zeros(n, bool)) for _ in positions]
    path = tmp_path / "big.bgen"
    write_v12(path, columns, 8, positions=positions)
    full, info, _ = load_bgen(str(path))
    assert list(info["pos"]) == positions

    for start, end, want in [(2**31, 2**32 - 1, [2, 3]), (2**31 - 1, 2**31, [1, 2]), (2**32 - 1, 2**32 - 1, [3])]:
        dos, info, _ = load_bgen(str(path), region=f"01:{start}-{end}")
        assert list(info["pos"]) == [positions[i] for i in want]
        np.testing.assert_array_equal(dos, full[:, want])

    vf = {"chromosome": "01", "positions": [2**31, 2**32 - 1], "allele1": ["A", "A"], "allele2": ["G", "G"]}
    dos, info, _ = load_bgen(str(path), variant_filter=vf)
    assert list(info["pos"]) == [2**31, 2**32 - 1]
    np.testing.assert_array_equal(dos, full[:, [2, 3]])

    with BgenReader(str(path)) as reader:
        rows = list(reader.iter_variants(region_chrom="01", region_start=2**31, region_end=2**32 - 1))
    assert [r[0]["pos"] for r in rows] == [2**31, 2**32 - 1]
