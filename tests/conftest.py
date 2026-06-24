"""Shared fixtures for the lazybgen test suite.

Test data (BGEN/BGI/sample fixtures) lives in tests/data/.
"""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Path to the bundled test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def bgen_file(data_dir) -> Path:
    """Path to the main test BGEN file."""
    return data_dir / "data.bgen"


@pytest.fixture(scope="session")
def bgi_file(data_dir) -> Path:
    """Path to the main test BGI index file."""
    return data_dir / "data.bgen.bgi"


@pytest.fixture(scope="session")
def sample_file(data_dir) -> Path:
    """Path to the main test sample file."""
    return data_dir / "data.sample"


@pytest.fixture(scope="session")
def test_bgen_files(data_dir) -> dict:
    """Dictionary of test BGEN files by format/compression type."""
    return {
        "basic": data_dir / "data.bgen",
        "8bit": data_dir / "example.8bits.bgen",
        "16bit": data_dir / "example.16bits.bgen",
        "32bit": data_dir / "example.32bits.bgen",
        "zstd": data_dir / "example.16bits.zstd.bgen",
        "v11": data_dir / "example.v11.bgen",
    }
