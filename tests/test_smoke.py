"""Smoke tests: the package imports and the compiled extensions load.

Read/round-trip correctness is covered by test_reader.py and
test_missing_genotypes.py; these tests just guard the standalone packaging
(imports, public API, native extension loading).
"""


def test_public_api_importable():
    import lazybgen

    assert hasattr(lazybgen, "load_bgen")
    assert hasattr(lazybgen, "BgenReader")
    assert hasattr(lazybgen, "load_variant_filter")
    assert hasattr(lazybgen, "get_build_info")
    assert set(lazybgen.__all__) == {
        "load_bgen",
        "BgenReader",
        "load_variant_filter",
        "get_build_info",
    }


def test_version_string():
    import lazybgen

    assert isinstance(lazybgen.__version__, str) and lazybgen.__version__


def test_native_extensions_load():
    # The compiled Cython/C++ reader extension; importing proves it built.
    from lazybgen import reader  # noqa: F401


def test_vendored_region_parser():
    from lazybgen.region import parse_region

    assert parse_region("chr1:1000-2000") == ("chr1", (1000, 2000))
