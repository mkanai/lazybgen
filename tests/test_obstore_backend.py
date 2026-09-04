"""Tests for the obstore remote transport and how it is selected.

The translation and selection tests run without obstore installed; only the
tests that build a real store are skipped when it is missing.
"""

import os

import pytest

from lazybgen import obstore_backend
from lazybgen.obstore_backend import UnsupportedStorageOption, translate_storage_options
from lazybgen.remote import REMOTE_BACKENDS, resolve_remote_backend

requires_obstore = pytest.mark.skipif(not obstore_backend.is_available(), reason="obstore not installed")


@pytest.fixture(autouse=True)
def _clear_backend_env(monkeypatch):
    """Keep an operator's LAZYBGEN_REMOTE_BACKEND out of these tests."""
    monkeypatch.delenv("LAZYBGEN_REMOTE_BACKEND", raising=False)


# --- URL splitting ------------------------------------------------------------


def test_split_url_parts():
    assert obstore_backend.split_url("gs://bucket/a/b.bgen") == ("gs", "bucket", "a/b.bgen")
    assert obstore_backend.split_url("s3://bucket/key") == ("s3", "bucket", "key")


@pytest.mark.parametrize("url", ["bucket/key", "gs://bucket", "gs://"])
def test_split_url_rejects_incomplete_urls(url):
    with pytest.raises(ValueError):
        obstore_backend.split_url(url)


# --- storage_options translation ----------------------------------------------


def test_gcs_no_options_is_empty():
    assert translate_storage_options("gs", None) == {}
    assert translate_storage_options("gs", {}) == {}


def test_gcs_anon_maps_to_skip_signature():
    assert translate_storage_options("gs", {"anon": True}) == {"skip_signature": True}
    # A false flag is the default behavior, so it adds nothing.
    assert translate_storage_options("gs", {"anon": False}) == {}


def test_gcs_requester_pays_string_sets_the_billing_header():
    got = translate_storage_options("gs", {"requester_pays": "my-project"})
    assert got == {"client_options": {"default_headers": {"x-goog-user-project": "my-project"}}}


def test_gcs_requester_pays_true_uses_the_project_option(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    got = translate_storage_options("gs", {"requester_pays": True, "project": "billed-project"})
    assert got["client_options"]["default_headers"]["x-goog-user-project"] == "billed-project"


def test_gcs_requester_pays_true_falls_back_to_the_environment(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-project")
    got = translate_storage_options("gs", {"requester_pays": True})
    assert got["client_options"]["default_headers"]["x-goog-user-project"] == "env-project"


def test_gcs_project_alone_adds_no_header():
    # Without requester_pays the project does not change which bytes are read.
    assert translate_storage_options("gs", {"project": "some-project"}) == {}


def test_gcs_token_default_values_are_the_default_chain():
    for token in (None, "google_default", "cloud", "default"):
        assert translate_storage_options("gs", {"token": token}) == {}


def test_gcs_token_anon_maps_to_skip_signature():
    assert translate_storage_options("gs", {"token": "anon"}) == {"skip_signature": True}


def test_gcs_token_dict_becomes_a_service_account_key():
    key = {"type": "service_account", "project_id": "p"}
    got = translate_storage_options("gs", {"token": key})
    assert "service_account_key" in got
    assert '"project_id": "p"' in got["service_account_key"]


def test_gcs_token_path_becomes_a_service_account_file(tmp_path):
    key_file = tmp_path / "sa.json"
    key_file.write_text("{}")
    got = translate_storage_options("gs", {"token": str(key_file)})
    assert got == {"service_account": str(key_file)}


def test_gcs_unknown_option_is_rejected_by_name():
    with pytest.raises(UnsupportedStorageOption, match="some_future_option"):
        translate_storage_options("gs", {"some_future_option": 1})


def test_s3_credentials_and_flags():
    got = translate_storage_options(
        "s3",
        {"anon": True, "key": "AKIA", "secret": "s3cret", "requester_pays": True, "endpoint_url": "https://x"},
    )
    assert got == {
        "skip_signature": True,
        "access_key_id": "AKIA",
        "secret_access_key": "s3cret",
        "request_payer": True,
        "endpoint": "https://x",
    }


def test_s3_client_kwargs_region_and_endpoint():
    got = translate_storage_options("s3", {"client_kwargs": {"region_name": "us-east-1"}})
    assert got == {"region": "us-east-1"}


def test_s3_unknown_client_kwarg_is_rejected():
    with pytest.raises(UnsupportedStorageOption, match="verify"):
        translate_storage_options("s3", {"client_kwargs": {"verify": False}})


def test_unknown_scheme_is_rejected():
    with pytest.raises(UnsupportedStorageOption):
        translate_storage_options("ftp", {})


# --- backend selection --------------------------------------------------------


def test_fsspec_is_chosen_when_asked_for():
    assert resolve_remote_backend("gs://b/k.bgen", {"anon": True}, "fsspec") == "fsspec"


def test_invalid_backend_name_is_rejected():
    with pytest.raises(ValueError, match="remote_backend"):
        resolve_remote_backend("gs://b/k.bgen", None, "no-such-backend")
    assert "auto" in REMOTE_BACKENDS


def test_auto_falls_back_to_fsspec_for_an_untranslatable_option():
    # An option obstore cannot express must not be silently dropped: it decides
    # which bytes come back, so the read goes to the transport that honors it.
    assert resolve_remote_backend("gs://b/k.bgen", {"some_future_option": 1}, "auto") == "fsspec"


def test_explicit_obstore_raises_for_an_untranslatable_option():
    if not obstore_backend.is_available():
        with pytest.raises(ImportError, match="obstore"):
            resolve_remote_backend("gs://b/k.bgen", {"some_future_option": 1}, "obstore")
    else:
        with pytest.raises(UnsupportedStorageOption, match="some_future_option"):
            resolve_remote_backend("gs://b/k.bgen", {"some_future_option": 1}, "obstore")


def test_environment_variable_selects_the_backend(monkeypatch):
    monkeypatch.setenv("LAZYBGEN_REMOTE_BACKEND", "fsspec")
    assert resolve_remote_backend("gs://b/k.bgen", None, None) == "fsspec"


def test_explicit_argument_beats_the_environment_variable(monkeypatch):
    monkeypatch.setenv("LAZYBGEN_REMOTE_BACKEND", "obstore")
    assert resolve_remote_backend("gs://b/k.bgen", None, "fsspec") == "fsspec"


@requires_obstore
def test_auto_prefers_obstore_when_it_is_installed_and_options_translate():
    assert resolve_remote_backend("gs://b/k.bgen", {"anon": True}, "auto") == "obstore"


# --- filesystem surface -------------------------------------------------------


@requires_obstore
def test_filesystem_open_rejects_write_modes():
    fs = obstore_backend.ObstoreFileSystem()
    with pytest.raises(ValueError, match="'rb'"):
        fs.open("gs://bucket/key", "wb")


@requires_obstore
def test_cat_ranges_requires_matching_lengths():
    fs = obstore_backend.ObstoreFileSystem()
    with pytest.raises(ValueError, match="equal length"):
        fs.cat_ranges(["gs://b/k"], [0, 1], [1, 2])


@requires_obstore
def test_cat_ranges_of_nothing_is_nothing():
    assert obstore_backend.ObstoreFileSystem().cat_ranges([], [], []) == []


def test_thread_count_honors_the_environment(monkeypatch):
    monkeypatch.setenv("LAZYBGEN_OBSTORE_THREADS", "3")
    assert obstore_backend._thread_count() == 3
    # A value that is not a positive integer leaves the default in place.
    monkeypatch.setenv("LAZYBGEN_OBSTORE_THREADS", "not-a-number")
    assert obstore_backend._thread_count() == obstore_backend._DEFAULT_THREADS
    monkeypatch.setenv("LAZYBGEN_OBSTORE_THREADS", "0")
    assert obstore_backend._thread_count() == 1
    monkeypatch.delenv("LAZYBGEN_OBSTORE_THREADS")
    assert obstore_backend._thread_count() == obstore_backend._DEFAULT_THREADS


def test_module_is_importable_without_obstore(monkeypatch):
    # The package must import and select a transport on a machine that has never
    # heard of obstore, which is the default install.
    monkeypatch.setattr(obstore_backend, "is_available", lambda: False)
    assert resolve_remote_backend("gs://b/k.bgen", None, "auto") == "fsspec"
    with pytest.raises(ImportError, match="pip install obstore"):
        resolve_remote_backend("gs://b/k.bgen", None, "obstore")


def test_environment_is_not_leaked_between_tests():
    assert "LAZYBGEN_REMOTE_BACKEND" not in os.environ
