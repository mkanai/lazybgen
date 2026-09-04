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


def test_gcs_anon_is_refused_because_gcsfs_ignores_it():
    # gcsfs has no `anon` parameter: it lands in **kwargs and is discarded, so a
    # gs:// read with anon=True is AUTHENTICATED there. Honoring it here would
    # send different credentials than the fsspec transport does for the same
    # options, so it is refused and the read falls back.
    with pytest.raises(UnsupportedStorageOption, match="anon"):
        translate_storage_options("gs", {"anon": True})
    # A false flag asks for nothing and is not a divergence.
    assert translate_storage_options("gs", {"anon": False}) == {}


def test_gcs_anon_sends_the_read_to_fsspec():
    assert resolve_remote_backend("gs://b/k.bgen", {"anon": True}, "auto") == "fsspec"


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
    for token in (None, "google_default"):
        assert translate_storage_options("gs", {"token": token}) == {}


def test_gcs_token_cloud_and_default_are_refused():
    # gcsfs's "cloud" is the metadata server ONLY, while obstore's default chain
    # prefers an ADC file, so treating them as equivalent can read as a different
    # principal. "default" is not a gcsfs method at all: gcsfs signs with the
    # literal string and fails, so making it work here would hide a bad option.
    for token in ("cloud", "default"):
        with pytest.raises(UnsupportedStorageOption, match="token"):
            translate_storage_options("gs", {"token": token})
        assert resolve_remote_backend("gs://b/k.bgen", {"token": token}, "auto") == "fsspec"


def test_gcs_token_anon_maps_to_skip_signature():
    assert translate_storage_options("gs", {"token": "anon"}) == {"skip_signature": True}


def test_gcs_token_dict_becomes_a_service_account_key():
    key = {"type": "service_account", "project_id": "p", "private_key": "x"}
    got = translate_storage_options("gs", {"token": key})
    assert "service_account_key" in got
    assert '"project_id": "p"' in got["service_account_key"]


def test_gcs_token_path_becomes_a_service_account_file(tmp_path):
    key_file = tmp_path / "sa.json"
    key_file.write_text('{"type": "service_account", "private_key": "x"}')
    got = translate_storage_options("gs", {"token": str(key_file)})
    assert got == {"service_account": str(key_file)}


def test_gcs_token_rejects_credentials_obstore_cannot_load(tmp_path):
    # gcsfs also accepts a gcloud user ADC document; obstore's store cannot load
    # one, so it must be refused here rather than failing later at open.
    adc = tmp_path / "adc.json"
    adc.write_text('{"type": "authorized_user", "refresh_token": "x"}')
    with pytest.raises(UnsupportedStorageOption, match="service-account"):
        translate_storage_options("gs", {"token": str(adc)})
    with pytest.raises(UnsupportedStorageOption, match="service-account"):
        translate_storage_options("gs", {"token": {"type": "authorized_user"}})


def test_gcs_unknown_option_is_rejected_by_name():
    with pytest.raises(UnsupportedStorageOption, match="some_future_option"):
        translate_storage_options("gs", {"some_future_option": 1})


def test_s3_plain_http_endpoint_allows_http():
    # object_store refuses http:// unless told; s3fs allows it, and a plaintext
    # endpoint is how local S3 stand-ins (MinIO, localstack) are addressed.
    got = translate_storage_options("s3", {"endpoint_url": "http://localhost:9000"})
    assert got == {"endpoint": "http://localhost:9000", "client_options": {"allow_http": True}}
    https = translate_storage_options("s3", {"endpoint_url": "https://example.com"})
    assert "client_options" not in https


def test_s3_is_not_auto_selected_for_obstore():
    # obstore's S3 store does not resolve a bucket's region and skips the
    # file/profile credential chain s3fs handles, so "auto" leaves S3 on fsspec.
    assert resolve_remote_backend("s3://b/k.bgen", None, "auto") == "fsspec"
    assert resolve_remote_backend("s3://b/k.bgen", {"anon": True}, "auto") == "fsspec"


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
    assert resolve_remote_backend("gs://b/k.bgen", {"token": "anon"}, "auto") == "obstore"
    assert resolve_remote_backend("gs://b/k.bgen", None, "auto") == "obstore"


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


# --- anonymous fallback -------------------------------------------------------


class _CredentialError(Exception):
    pass


def _missing_credentials():
    return _CredentialError(
        "Generic GCS error: Error performing token request: Error performing GET "
        "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token"
    )


def test_a_credential_failure_retries_unsigned_once():
    """gcsfs's ladder ends at anonymous, so a public bucket reads with no creds."""
    fs = obstore_backend.ObstoreFileSystem()
    calls = []

    def operation():
        calls.append(dict(fs._store_kwargs.get("gs", {})))
        if len(calls) == 1:
            raise _missing_credentials()
        return "read"

    assert fs._call_with_anon_retry("gs://bucket/key", operation) == "read"
    assert len(calls) == 2
    assert fs._store_kwargs["gs"]["skip_signature"] is True


def test_the_anonymous_retry_happens_at_most_once():
    fs = obstore_backend.ObstoreFileSystem()

    def always_fails():
        raise _missing_credentials()

    with pytest.raises(_CredentialError):
        fs._call_with_anon_retry("gs://bucket/key", always_fails)
    # A second failure must propagate rather than retry again.
    with pytest.raises(_CredentialError):
        fs._call_with_anon_retry("gs://bucket/key", always_fails)


def test_a_requester_pays_read_is_never_retried_anonymously():
    # The read is billed to a project, so an unsigned retry cannot be what the
    # caller wanted.
    fs = obstore_backend.ObstoreFileSystem(requester_pays="billed-project")
    with pytest.raises(_CredentialError):
        fs._call_with_anon_retry("gs://bucket/key", lambda: (_ for _ in ()).throw(_missing_credentials()))
    assert "skip_signature" not in fs._store_kwargs.get("gs", {})


def test_an_explicit_credential_is_never_retried_anonymously(tmp_path):
    key_file = tmp_path / "sa.json"
    key_file.write_text('{"type": "service_account", "private_key": "x"}')
    fs = obstore_backend.ObstoreFileSystem(token=str(key_file))
    fs._store_kwargs["gs"] = translate_storage_options("gs", fs.storage_options)
    with pytest.raises(_CredentialError):
        fs._call_with_anon_retry("gs://bucket/key", lambda: (_ for _ in ()).throw(_missing_credentials()))


def test_an_unrelated_error_is_not_retried():
    fs = obstore_backend.ObstoreFileSystem()

    def boom():
        raise _CredentialError("404 Not Found")

    with pytest.raises(_CredentialError, match="404"):
        fs._call_with_anon_retry("gs://bucket/key", boom)
    assert "skip_signature" not in fs._store_kwargs.get("gs", {})


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
    # The package must still import and select a transport in an environment
    # where obstore is absent, even though it is a declared dependency.
    monkeypatch.setattr(obstore_backend, "is_available", lambda: False)
    assert resolve_remote_backend("gs://b/k.bgen", None, "auto") == "fsspec"
    with pytest.raises(ImportError, match="pip install obstore"):
        resolve_remote_backend("gs://b/k.bgen", None, "obstore")


@pytest.fixture
def _forked_after_use(monkeypatch):
    """Put the module in the state a fork leaves it in after a parent read."""
    monkeypatch.setattr(obstore_backend, "_did_io", True)
    obstore_backend._reset_after_fork()
    yield
    # Assigned directly, not through monkeypatch: a setattr made during teardown
    # would itself be undone, leaving the flag set for every later test.
    obstore_backend._broken_by_fork = False


def test_auto_falls_back_to_fsspec_in_a_forked_child(_forked_after_use):
    # obstore's runtime does not survive fork(); a read on it would hang rather
    # than fail, so the child must be sent to fsspec.
    assert obstore_backend.fork_broken() is True
    assert obstore_backend.is_available() is False
    assert resolve_remote_backend("gs://b/k.bgen", None, "auto") == "fsspec"


def test_explicit_obstore_in_a_forked_child_raises_rather_than_hanging(_forked_after_use):
    with pytest.raises(RuntimeError, match="fork"):
        resolve_remote_backend("gs://b/k.bgen", None, "obstore")


def test_the_fork_hook_drops_state_the_child_cannot_inherit(_forked_after_use):
    # The pool's threads do not exist in the child and a cached store holds the
    # parent's token, so both must be gone.
    assert obstore_backend._pool is None
    assert obstore_backend._store_cache == {}


def test_a_fork_before_any_read_leaves_obstore_usable(monkeypatch):
    monkeypatch.setattr(obstore_backend, "_did_io", False)
    monkeypatch.setattr(obstore_backend, "_broken_by_fork", False)
    obstore_backend._reset_after_fork()
    assert obstore_backend.fork_broken() is False


def test_environment_is_not_leaked_between_tests():
    assert "LAZYBGEN_REMOTE_BACKEND" not in os.environ
