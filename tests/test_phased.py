"""Phased BGEN decoding: alt-allele dosage from phased biallelic diploid data.

lazybgen computes the alt-allele dosage for phased biallelic diploid variants.
Phased layout 2 stores, per sample, one probability per haplotype (P of the
first allele on that haplotype), so the alt dosage is the sum over the two
haplotypes of P(allele == alt):

    dosage = (max - h1) / max + (max - h2) / max = 2 - (h1 + h2) / max

The oracle here is the official Oxford ``haplotypes.bgen`` example file
(4 samples x 4 variants, phased, biallelic, diploid, 1-bit probabilities),
whose dosages were derived directly from the BGEN spec.
"""

import numpy as np
import pytest
from lazybgen.reader import BgenReader

from lazybgen import load_bgen

try:
    import bgen as external_bgen

    HAS_EXTERNAL_BGEN = True
except ImportError:
    external_bgen = None
    HAS_EXTERNAL_BGEN = False


# Ground truth for the official haplotypes.bgen, indexed [sample, variant]
# (RS1, RS2, RS3, RS4). Derived by hand-decoding the file per the BGEN spec.
HAPLOTYPES_EXPECTED = np.array(
    [
        [0.0, 1.0, 1.0, 2.0],
        [1.0, 1.0, 2.0, 0.0],
        [1.0, 2.0, 0.0, 1.0],
        [2.0, 0.0, 1.0, 1.0],
    ]
)


def test_phased_haplotypes_official_dosages(data_dir):
    """The official phased haplotypes.bgen decodes to the spec-derived dosages."""
    path = data_dir / "haplotypes.bgen"
    dosages, info, sample_ids = load_bgen(file_path=str(path), nan_action="warn")

    assert dosages.shape == (4, 4)
    assert list(info["rsid"]) == ["RS1", "RS2", "RS3", "RS4"]
    np.testing.assert_allclose(dosages, HAPLOTYPES_EXPECTED, rtol=0, atol=1e-6)


def test_phased_haplotypes_filtered_matches_full(data_dir):
    """Sample-filtered phased decode matches the corresponding full-decode rows."""
    path = data_dir / "haplotypes.bgen"
    full, _, sample_ids = load_bgen(file_path=str(path), nan_action="warn")

    requested = [sample_ids[i] for i in (0, 3)]
    subset, _, _ = load_bgen(file_path=str(path), sample_ids=requested, nan_action="warn")

    np.testing.assert_allclose(subset, full[[0, 3], :], rtol=0, atol=1e-6)


@pytest.mark.skipif(not HAS_EXTERNAL_BGEN, reason="bgen library required to write a phased fixture")
@pytest.mark.parametrize("bit_depth", [1, 2, 4, 8, 16, 32])
def test_phased_dosages_match_reference(tmp_path, bit_depth):
    """Phased dosages match the external bgen reference across bit depths.

    The reference alt dosage is P(hap1 == alt) + P(hap2 == alt), i.e. columns
    1 and 3 of the phased probability tensor (hap1[A,B], hap2[A,B]). Sweeping the
    bit depth exercises the general bit reader at non-byte-aligned widths (2, 4),
    the byte-aligned widths (8, 16, 32), and the 32-bit max-value branch.
    """
    from bgen import BgenWriter

    path = tmp_path / f"phased{bit_depth}.bgen"
    # hap1[A,B], hap2[A,B] per sample: A|B, B|B, A|A, B|A
    geno = np.array(
        [
            [0.8, 0.2, 0.1, 0.9],
            [0.3, 0.7, 0.4, 0.6],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
        ]
    )
    with BgenWriter(str(path), n_samples=4, samples=["s0", "s1", "s2", "s3"]) as w:
        w.add_variant("v1", "rs1", "1", 1000, ["A", "B"], geno, ploidy=2, phased=True, bit_depth=bit_depth)

    with external_bgen.BgenReader(str(path)) as ref:
        v = next(iter(ref))
        probs = np.asarray(v.probabilities)  # (samples, 4): hap1[A,B], hap2[A,B]
        ref_dosage = probs[:, 1] + probs[:, 3]

    full, _, sample_ids = load_bgen(file_path=str(path), nan_action="warn")
    np.testing.assert_allclose(full[:, 0], ref_dosage, rtol=1e-3, atol=1e-4)

    # Sample-filtered path must agree too.
    requested = [sample_ids[i] for i in (0, 2)]
    subset, _, _ = load_bgen(file_path=str(path), sample_ids=requested, nan_action="warn")
    np.testing.assert_allclose(subset[:, 0], ref_dosage[[0, 2]], rtol=1e-3, atol=1e-4)


@pytest.mark.skipif(not HAS_EXTERNAL_BGEN, reason="bgen library required to write a phased fixture")
def test_phased_data_decoded(tmp_path):
    """Phased biallelic diploid data decodes to the correct alt-allele dosage.

    Phased layout 2 stores one probability per haplotype (P of the first allele),
    so the alt dosage is P(hap1 == alt) + P(hap2 == alt) = columns 1 and 3 of the
    reference probability tensor. Decoding phased data as unphased would give
    silently wrong dosages, so this guards the dedicated phased path.
    """
    from bgen import BgenWriter

    path = tmp_path / "phased.bgen"
    # hap1[A,B], hap2[A,B] for each sample: A|B, B|B, A|A
    geno = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]])
    with BgenWriter(str(path), n_samples=3, samples=["s0", "s1", "s2"]) as w:
        w.add_variant("v1", "rs1", "1", 1000, ["A", "B"], geno, ploidy=2, phased=True, bit_depth=8)

    with external_bgen.BgenReader(str(path)) as ref:
        v = next(iter(ref))
        probs = np.asarray(v.probabilities)  # (samples, 4): hap1[A,B], hap2[A,B]
        ref_dosage = probs[:, 1] + probs[:, 3]

    dosages, _, _ = load_bgen(file_path=str(path), nan_action="warn")
    np.testing.assert_allclose(dosages[:, 0], ref_dosage, rtol=1e-3, atol=1e-4)


@pytest.mark.skipif(not HAS_EXTERNAL_BGEN, reason="bgen library required to write a phased fixture")
def test_phased_missing_is_nan(tmp_path):
    """A missing sample in phased data decodes to NaN, like the unphased path."""
    from bgen import BgenWriter

    path = tmp_path / "phased_missing.bgen"
    geno = np.array(
        [
            [1.0, 0.0, 0.0, 1.0],
            [np.nan, np.nan, np.nan, np.nan],
            [0.0, 1.0, 0.0, 1.0],
        ]
    )
    with BgenWriter(str(path), n_samples=3, samples=["s0", "s1", "s2"]) as w:
        w.add_variant("v1", "rs1", "1", 1000, ["A", "B"], geno, ploidy=2, phased=True, bit_depth=8)

    with BgenReader(str(path)) as reader:
        dosages, _ = reader.load_variants()

    assert np.isnan(dosages[1, 0])
    assert not np.isnan(dosages[0, 0])
    assert not np.isnan(dosages[2, 0])
