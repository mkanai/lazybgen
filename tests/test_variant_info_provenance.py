"""variant_info (rsid/alleles/pos/chrom) is sourced from the BGI, not the bgen record.

This is an intentional design choice: it
lets a caller override variant identity by editing the small ``.bgi`` instead of
rewriting hundreds of GB of bgen. The genotype decode is unaffected (still read
from the bgen at file_start_position), so dosages stay correct even when the BGI
metadata is edited.
"""

import shutil
import sqlite3
from pathlib import Path

import numpy as np

from lazybgen import load_bgen

LOCAL_DATA = Path(__file__).parent / "data"


def _copy_fixture(tmp_path, name="example.16bits.bgen"):
    bgen = tmp_path / name
    bgi = tmp_path / (name + ".bgi")
    shutil.copy(LOCAL_DATA / name, bgen)
    shutil.copy(LOCAL_DATA / (name + ".bgi"), bgi)
    return str(bgen)


def test_rsid_comes_from_bgi_not_bgen(tmp_path):
    """Editing an rsid in the .bgi changes the loaded variant_info rsid.

    The dosage for that variant is unchanged, proving the genotype is still
    decoded from the bgen while the identity column comes from the BGI.
    """
    bgen_path = _copy_fixture(tmp_path)

    # Baseline load (unedited BGI).
    dosages0, info0, _ = load_bgen(bgen_path, nan_action="warn")
    # Pick a target variant by its (position, rsid) in BGI order.
    target_pos = int(info0.iloc[0]["pos"])
    original_rsid = str(info0.iloc[0]["rsid"])
    sentinel = "EDITED_FROM_BGI_12345"
    assert original_rsid != sentinel

    # Edit the rsid for that one variant in the copied BGI.
    con = sqlite3.connect(bgen_path + ".bgi")
    con.execute(
        "UPDATE Variant SET rsid = ? WHERE position = ? AND rsid = ?",
        (sentinel, target_pos, original_rsid),
    )
    con.commit()
    n_changed = con.total_changes
    con.close()
    assert n_changed == 1

    dosages1, info1, _ = load_bgen(bgen_path, nan_action="warn")

    # The edited rsid now appears in variant_info (sourced from the BGI)...
    row = info1[info1["pos"] == target_pos]
    assert len(row) == 1
    assert str(row.iloc[0]["rsid"]) == sentinel

    # ...and the dosage column for that variant is byte-identical to baseline
    # (decode is independent of the BGI rsid edit).
    col = int(np.where(info1["pos"].values == target_pos)[0][0])
    col0 = int(np.where(info0["pos"].values == target_pos)[0][0])
    np.testing.assert_array_equal(np.isnan(dosages1[:, col]), np.isnan(dosages0[:, col0]))
    np.testing.assert_array_equal(
        dosages1[~np.isnan(dosages1[:, col]), col],
        dosages0[~np.isnan(dosages0[:, col0]), col0],
    )
