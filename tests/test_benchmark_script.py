import pytest

# benchmarks/ is dev tooling and is not shipped in the sdist, so skip this whole
# module when it cannot be imported (e.g. running the test suite from an sdist).
rb = pytest.importorskip("benchmarks.run_remote_benchmark")


def test_remote_url_joins_with_single_slash():
    assert rb.remote_url("gs://b/pre", "f.bgen") == "gs://b/pre/f.bgen"
    assert rb.remote_url("gs://b/pre/", "f.bgen") == "gs://b/pre/f.bgen"


def test_fixtures_cover_small_and_wide():
    keys = {f.key for f in rb.FIXTURES}
    assert keys == {"small", "wide"}
    small = next(f for f in rb.FIXTURES if f.key == "small")
    assert small.local_name == "ukbb_500k_300.bgen"


def test_parse_storage_options_builds_dict():
    assert rb.parse_storage_options(["anon=true", "x=1"], None) == {"anon": True, "x": 1}
    assert rb.parse_storage_options(None, True) == {"requester_pays": True}
    assert rb.parse_storage_options(None, "proj-id") == {"requester_pays": "proj-id"}
    assert rb.parse_storage_options(None, None) == {}


def test_count_gcs_io_records_via_active_counter():
    # The installed gcsfs/fsspec wrappers call _note_fetch / _note_read against
    # whatever counter count_gcs_io has made active. Drive those helpers directly
    # so the mechanism is verified with no network and no real gcsfs file.
    with rb.count_gcs_io() as c:
        rb._note_fetch(b"x" * 100)
        rb._note_fetch(b"y" * 150)
        rb._note_read(b"z" * 120)

    assert c.gets == 2
    assert c.fetched_bytes == 250
    assert c.read_bytes == 120
    assert abs(c.amplification - (250 / 120)) < 1e-9


def test_note_helpers_are_noop_outside_context():
    # With no active counter, recording must do nothing (not raise).
    rb._note_fetch(b"x" * 999)
    rb._note_read(b"y" * 999)
    with rb.count_gcs_io() as c:
        pass
    rb._note_fetch(b"x" * 999)  # after the context closed
    assert c.gets == 0 and c.fetched_bytes == 0 and c.read_bytes == 0


def test_count_gcs_io_installs_gcsfs_wrapper():
    import gcsfs.core

    with rb.count_gcs_io():
        pass
    assert rb._INSTRUMENTED is True
    # The class method is now the instrumented wrapper, so any file opened from
    # here on routes its range GETs through the active counter.
    assert gcsfs.core.GCSFile._fetch_range.__name__ == "fetch_range"


def test_io_counter_zero_read_is_zero_amplification():
    c = rb.IOCounter()
    assert c.amplification == 0.0


class _FakeFS:
    def __init__(self, sizes):
        self._sizes = sizes  # url -> size, missing => not present
        self.put_calls = []

    def exists(self, url):
        return url in self._sizes

    def info(self, url):
        return {"size": self._sizes[url]}

    def put(self, local, url):
        self.put_calls.append((local, url))
        self._sizes[url] = 123


def test_needs_upload_true_when_missing():
    fs = _FakeFS({})
    assert rb.needs_upload(fs, "gs://b/f.bgen", 999) is True


def test_needs_upload_false_when_present_same_size():
    fs = _FakeFS({"gs://b/f.bgen": 999})
    assert rb.needs_upload(fs, "gs://b/f.bgen", 999) is False


def test_needs_upload_true_when_size_differs():
    fs = _FakeFS({"gs://b/f.bgen": 1})
    assert rb.needs_upload(fs, "gs://b/f.bgen", 999) is True


def test_workloads_cover_six_names_and_build_callables():
    names = [w["name"] for w in rb.WORKLOADS]
    assert names == [
        "open_only",
        "single_variant",
        "region_small",
        "variant_filter_scattered",
        "cohort_small",
        "iter_stream",
    ]
    for w in rb.WORKLOADS:
        assert w["fixture_key"] in ("small", "wide")
        assert callable(w["build"])


def test_build_remote_url_map():
    m = rb.build_remote_url_map("gs://b/pre")
    assert m["small"] == "gs://b/pre/ukbb_500k_300.bgen"
    assert m["wide"].endswith("test_500000s_5000v_zlib_8bit.bgen")


def test_measure_cold_warm_shapes_and_fresh_reader(monkeypatch):
    # Deterministic clock so medians are predictable.
    ticks = iter(range(0, 1000))
    monkeypatch.setattr(rb.time, "perf_counter", lambda: next(ticks))

    made = []

    def make_reader():
        r = {"id": len(made)}
        made.append(r)
        return r

    def do_read(reader):
        pass

    out = rb.measure_cold_warm(make_reader, do_read, num_runs=3)
    # 3 cold runs => 3 fresh readers, plus 1 dedicated warm reader (primed once,
    # then measured num_runs times) => 4 total constructions.
    assert len(made) == 4
    assert set(out) == {"cold", "warm"}
    for side in out.values():
        assert {"median_time", "min_time", "runs", "gets", "fetched_bytes", "read_bytes", "amplification"} <= set(side)
        assert len(side["runs"]) == 3


def test_run_strategy_rejects_unknown():
    with pytest.raises(ValueError):
        rb.run_strategy("bogus", rb.WORKLOADS[0], object(), {}, "s.sample", {}, "d", "/tmp", 1)


def test_format_summary_includes_workload_rows():
    results = {
        "workloads": {
            "open_only": {
                "stream": {
                    "cold": {"median_time": 0.5, "gets": 3, "fetched_bytes": 1048576, "amplification": 1.0},
                    "warm": {"median_time": 0.1, "gets": 0, "fetched_bytes": 0, "amplification": 0.0},
                }
            }
        }
    }
    text = rb.format_summary(results)
    assert "open_only" in text
    assert "GET" in text or "gets" in text.lower()


def test_prefetch_variants_have_labels():
    assert all("label" in v for v in rb.PREFETCH_VARIANTS)


@pytest.mark.integration
def test_smoke_against_public_bucket():
    # Hits the public gs://gcs-anndata-test bgen via ADC; verifies the full
    # wiring (instrumentation patch + reader open + read) end to end.
    rc = rb.main(["--smoke"])
    assert rc == 0
