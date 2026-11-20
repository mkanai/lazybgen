"""Network-free tests for the gcsfs class lazybgen uses after fork().

gcsfs 2026.10's exported extended filesystem can hang in a forked child (it looks
up bucket types over gRPC when credentials exist), so lazybgen hands every forked
child gcsfs's core HTTP-only class. The integration fork tests in test_gcs.py
cannot see a regression in credential-free CI, where that lookup fails fast and
either class reads fine; these tests check the class selection itself, so they
fail if the switch is removed.
"""

import os
import time

import pytest

gcsfs = pytest.importorskip("gcsfs")

from lazybgen import BgenReader, remote  # noqa: E402

pytestmark = pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")

DATA = os.path.join(os.path.dirname(__file__), "data")


def _in_forked_child(fn, timeout=30):
    """Run ``fn`` in a forked child and return the string it returns."""
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - runs only in the child
        os.close(read_fd)
        try:
            out = fn()
        except BaseException as exc:  # noqa: BLE001 - report anything to the parent
            out = f"ERROR {type(exc).__name__}: {exc}"
        os.write(write_fd, out.encode())
        os._exit(0)
    os.close(write_fd)
    deadline = time.time() + timeout
    while time.time() < deadline:
        done, _ = os.waitpid(pid, os.WNOHANG)
        if done:
            break
        time.sleep(0.05)
    else:
        os.kill(pid, 9)
        os.waitpid(pid, 0)
        os.close(read_fd)
        pytest.fail("forked child hung")
    chunks = []
    while True:
        chunk = os.read(read_fd, 65536)
        if not chunk:
            break
        chunks.append(chunk)
    os.close(read_fd)
    return b"".join(chunks).decode()


def _qualname(cls):
    return f"{cls.__module__}.{cls.__qualname__}"


def test_unforked_process_keeps_the_exported_class():
    assert remote.gcs_filesystem_class() is gcsfs.GCSFileSystem


def test_forked_child_gets_the_core_class():
    got = _in_forked_child(lambda: _qualname(remote.gcs_filesystem_class()))
    assert got == _qualname(gcsfs.core.GCSFileSystem)
    # The parent is unaffected by its child's fork.
    assert remote.gcs_filesystem_class() is gcsfs.GCSFileSystem


def test_forked_child_keeps_a_custom_registered_class():
    class CustomGCS:
        pass

    got = _in_forked_child(lambda: "same" if remote.gcs_filesystem_class(CustomGCS) is CustomGCS else "swapped")
    assert got == "same"


def test_cpp_reader_builds_gs_filesystems_through_the_factory(monkeypatch):
    """The C++ reader must resolve gs:// through lazybgen.remote.gcs_filesystem."""

    def sentinel(**_kwargs):
        raise RuntimeError("factory-was-called")

    monkeypatch.setattr(remote, "gcs_filesystem", sentinel)
    with pytest.raises(Exception, match="factory-was-called"):
        BgenReader(
            "gs://lazybgen-no-such-bucket/x.bgen",
            bgi_path=os.path.join(DATA, "example.16bits.bgen.bgi"),
            remote_backend="fsspec",
        )


def test_forked_child_downloads_a_bgi_with_the_core_class(tmp_path):
    """An uncached remote .bgi fetched in a forked child goes through core gcsfs."""

    def child():
        calls = []

        class RecordingCore:
            def __init__(self, **kwargs):
                pass

            def get(self, rpath, lpath):
                calls.append(rpath)
                raise OSError("stop after recording")

        os.environ["LAZYBGEN_BGI_CACHE_DIR"] = str(tmp_path)
        gcsfs.core.GCSFileSystem = RecordingCore  # child-only patch; the child exits
        time.sleep = lambda _s: None  # skip the retry backoff
        try:
            remote.ensure_local_bgi("gs://lazybgen-no-such-bucket/x.bgen.bgi", backend="fsspec")
        except Exception:  # noqa: BLE001 - the recording stub always fails the download
            pass
        return ",".join(calls) or "core-not-used"

    got = _in_forked_child(child)
    assert got.startswith("gs://lazybgen-no-such-bucket/x.bgen.bgi"), got
