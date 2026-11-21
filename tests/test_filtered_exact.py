"""A sample-filtered read returns exactly the full read's values for those samples.

The filtered 8/16-bit kernels decode several samples per SIMD vector and finish
with a scalar tail. Each lane must round exactly like the scalar decode, so a
sample's dosage cannot depend on where it falls in the requested index list.
Before this was pinned, the vector lanes multiplied by a rounded 1/255 (or
1/65535) while the tail and the full decode divided, and a lane could differ
from the full read by one float ULP.
"""

from pathlib import Path

import numpy as np
import pytest
from lazybgen.reader import BgenReader

DATA = Path(__file__).parent / "data"
FIXTURES = [
    "example.8bits.bgen",
    "example.16bits.bgen",
    "example.16bits.zstd.bgen",
    "example.32bits.bgen",
    "data.bgen",
]
# example.v11.bgen is left out on purpose: the v1.1 filtered path rounds
# float(ab/sum) + 2*float(bb/sum) while the full path rounds (ab + 2*bb)/sum, a
# separate known difference (and the spec scales by 1/32768 rather than by the
# per-sample sum), to be settled on its own.


def _selections(n):
    rng = np.random.default_rng(0)
    yield np.arange(n, dtype=np.int32)
    yield np.arange(n, dtype=np.int32)[::-1].copy()
    yield rng.permutation(n).astype(np.int32)
    # Every length around the 4- and 8-lane vector widths and their tails.
    for length in range(1, 20):
        yield rng.choice(n, size=length, replace=False).astype(np.int32)
    yield rng.integers(0, n, size=101).astype(np.int32)  # duplicates


@pytest.mark.parametrize("fixture", FIXTURES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_filtered_read_is_bitwise_the_full_read(fixture, dtype):
    with BgenReader(str(DATA / fixture)) as reader:
        full, _ = reader.load_variants(dtype=dtype)
        for idx in _selections(full.shape[0]):
            sub, _ = reader.load_variants(dtype=dtype, sample_indices=idx)
            expected = full[idx, :]
            assert sub.shape == expected.shape
            mismatch = sub.view(np.uint32 if dtype == np.float32 else np.uint64) != expected.view(
                np.uint32 if dtype == np.float32 else np.uint64
            )
            assert not mismatch.any(), (
                f"{fixture} {np.dtype(dtype).name} len={len(idx)}: {int(mismatch.sum())} values differ, "
                f"max abs diff {np.nanmax(np.abs(sub - expected)):.3g}"
            )
