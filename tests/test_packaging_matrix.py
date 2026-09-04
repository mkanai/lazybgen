"""The supported Python versions are declared in three places; they must agree.

They are the classifiers, the cibuildwheel build list, and the CI test matrix.
Nothing enforced that, and the versions the wheels job built used to be
cibuildwheel's default set rather than a list of our own, so upgrading
cibuildwheel changed what a release shipped: 4.2.0 added CPython 3.15, whose
dependency wheels do not exist yet, and the release build failed trying to
compile them from source.

These tests are cheap and they fail the moment the three drift apart, which is
the failure they exist to prevent.
"""

import re
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - the floor is 3.10, which has no tomllib
    tomllib = pytest.importorskip("tomli", reason="needs tomllib or tomli to read pyproject")

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


@pytest.fixture(scope="module")
def config():
    if not PYPROJECT.exists():
        pytest.skip("pyproject.toml not present (installed package, not a checkout)")
    with open(PYPROJECT, "rb") as fh:
        return tomllib.load(fh)


def classifier_versions(config):
    prefix = "Programming Language :: Python :: 3."
    return sorted(c.rsplit(" ", 1)[1] for c in config["project"]["classifiers"] if c.startswith(prefix))


def cibuildwheel_versions(config):
    # "cp313-*" -> "3.13"
    return sorted(b.replace("cp3", "3.").replace("-*", "") for b in config["tool"]["cibuildwheel"]["build"])


def ci_matrix_versions():
    if not CI_WORKFLOW.exists():
        pytest.skip("ci.yml not present")
    match = re.search(r"python-version: \[(.*?)\]", CI_WORKFLOW.read_text())
    assert match, "could not find the python-version matrix in ci.yml"
    return sorted(item.strip().strip("'\"") for item in match.group(1).split(","))


def test_wheels_are_built_for_exactly_the_versions_we_advertise(config):
    assert cibuildwheel_versions(config) == classifier_versions(config)


def test_ci_tests_exactly_the_versions_we_advertise(config):
    assert ci_matrix_versions() == classifier_versions(config)


def test_the_wheel_build_list_is_explicit(config):
    """An implicit build set makes a cibuildwheel upgrade change the release."""
    assert config["tool"]["cibuildwheel"].get("build"), "cibuildwheel `build` must list versions explicitly"


def test_free_threaded_builds_are_skipped_by_pattern(config):
    """The extension re-enables the GIL on import, so no free-threaded wheels.

    By pattern rather than by naming one version: skipping only `cp314t-*` is
    what let `cp315t-*` into a release build.
    """
    skip = config["tool"]["cibuildwheel"]["skip"]
    assert any(
        "t-*" in entry and "*" in entry.split("t-")[0] for entry in skip
    ), f"no wildcard free-threaded skip in {skip}"


def test_requires_python_matches_the_lowest_supported_version(config):
    lowest = min(classifier_versions(config), key=lambda v: int(v.split(".")[1]))
    assert config["project"]["requires-python"] == f">={lowest}"
