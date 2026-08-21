"""The runtime provider key: pasting one, and falling back when there is none.

The point of the feature is that a reader can spend their OWN quota instead of
the deployment's, so what has to hold is narrow and specific:

  * the pasted key really becomes the one the backends read -- os.environ is
    the only thing app.agents.llm consults, so nothing short of writing there
    takes effect without a redeploy;
  * clearing returns the process to the key it BOOTED with, not to no key;
  * a key that does not authenticate never displaces a working one;
  * the key never travels back out.

Hermetic: no provider is contacted. `verify_key` is the only network path and
it is either monkeypatched or exercised against fabricated HTTP errors.
"""
from __future__ import annotations

import io
import os
import urllib.error
from pathlib import Path

import pytest
from fastapi import HTTPException

import app.api
import app.config
from app.agents import llm

REAL = "gsk_configured_by_the_deployment"
PASTED = "gsk_pasted_by_the_reader_abcd"


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    """Redirect the on-disk key store at the real DATA_DIR's expense.

    Autouse and not optional: without it every test that pastes a key would
    write one into the developer's actual data/db, and the suite's cleanliness
    would depend on a later test happening to clear it.
    """
    path = tmp_path / "provider_keys.json"
    monkeypatch.setattr(app.config, "KEY_STORE", path)
    monkeypatch.setattr(app.config, "PERSIST_API_KEY", True)
    return path


@pytest.fixture
def groq(monkeypatch):
    """A process that booted as groq with a configured key."""
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", REAL)
    monkeypatch.setitem(llm._BASELINE_KEYS, "GROQ_API_KEY", REAL)


@pytest.fixture
def keyless(monkeypatch):
    """A process that booted as groq with NO key -- the deploy-and-paste case."""
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setitem(llm._BASELINE_KEYS, "GROQ_API_KEY", None)


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://api.groq.com/openai/v1/models",
                                  code, "nope", {}, io.BytesIO(b""))


def _raiser(exc):
    def _f(*_a, **_k):
        raise exc
    return _f


# -- state reporting -------------------------------------------------------

def test_reports_the_configured_key_as_environment(groq):
    assert llm.key_state() == {"provider": "groq", "source": "environment",
                               "hint": "...ment"}


def test_reports_missing_when_the_host_has_no_key(keyless):
    assert llm.key_state()["source"] == "missing"
    assert llm.key_state()["hint"] == ""
    assert llm.available() is False


def test_a_provider_that_needs_no_key_reports_none(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "stub")
    assert llm.key_state()["source"] == "none"
    # and refuses to pretend otherwise rather than silently doing nothing
    with pytest.raises(llm.LLMError):
        llm.set_api_key("anything")


def test_hint_discloses_four_characters_at_most(groq):
    hint = llm.key_state()["hint"]
    assert hint == "..." + REAL[-4:]
    assert REAL[:-4] not in hint


# -- installing ------------------------------------------------------------

def test_set_puts_the_key_where_the_backends_read_it(groq):
    llm.set_api_key(PASTED)
    # not a parallel registry: the environment itself, which is what _groq()
    # reads on the next call
    assert os.environ["GROQ_API_KEY"] == PASTED
    assert llm.key_state()["source"] == "override"


def test_set_strips_whitespace(groq):
    """A key copied out of a console arrives with a newline on it."""
    llm.set_api_key(f"  {PASTED}\n")
    assert os.environ["GROQ_API_KEY"] == PASTED


def test_pasting_the_configured_key_is_not_an_override(groq):
    """Source is derived from the value, so re-pasting the deployment's own
    key reports honestly rather than claiming a personal one is in use."""
    llm.set_api_key(REAL)
    assert llm.key_state()["source"] == "environment"


# -- falling back ----------------------------------------------------------

def test_clear_restores_the_configured_key(groq):
    llm.set_api_key(PASTED)
    assert llm.clear_api_key()["source"] == "environment"
    assert os.environ["GROQ_API_KEY"] == REAL


def test_clear_without_a_baseline_unsets_rather_than_blanks(keyless):
    """An empty string is not the same as absent. `_groq` happens to test the
    value's truthiness, but anything reading os.environ.get(var, default)
    would take "" as a configured key and skip its own fallback."""
    llm.set_api_key(PASTED)
    assert llm.clear_api_key()["source"] == "missing"
    assert "GROQ_API_KEY" not in os.environ


def test_clear_is_idempotent(groq):
    llm.clear_api_key()
    llm.clear_api_key()
    assert os.environ["GROQ_API_KEY"] == REAL


# -- verification ----------------------------------------------------------

def test_rejected_key_never_displaces_a_working_one(groq, monkeypatch):
    """The failure this guards is a typo taking down a live deployment: with
    install-then-verify, every question fails until someone redeploys."""
    monkeypatch.setattr(llm, "verify_key",
                        _raiser(llm.LLMError("groq rejected that key")))
    with pytest.raises(HTTPException) as exc:
        app.api.set_api_key(app.api.ApiKeyIn(api_key=PASTED))
    assert exc.value.status_code == 400
    assert os.environ["GROQ_API_KEY"] == REAL


def test_a_rate_limited_account_still_has_a_valid_key(groq, monkeypatch):
    """429 means the key authenticated and the quota is spent -- which is the
    very situation someone is pasting a new key to escape. Refusing it here
    would lock them out of the fix."""
    monkeypatch.setattr(llm.urllib.request, "urlopen", _raiser(_http_error(429)))
    llm.verify_key(PASTED)          # must not raise


@pytest.mark.parametrize("code", [401, 403])
def test_unauthorised_is_reported_as_a_bad_key(groq, monkeypatch, code):
    monkeypatch.setattr(llm.urllib.request, "urlopen", _raiser(_http_error(code)))
    with pytest.raises(llm.LLMError, match="rejected that key"):
        llm.verify_key(PASTED)


def test_unreachable_provider_is_not_a_bad_key(groq, monkeypatch):
    monkeypatch.setattr(llm.urllib.request, "urlopen",
                        _raiser(urllib.error.URLError("dns")))
    with pytest.raises(llm.LLMError, match="cannot reach"):
        llm.verify_key(PASTED)


def test_verification_costs_no_tokens(groq, monkeypatch):
    """It lists models. A completion would spend part of the quota the key was
    pasted to recover, and would count against the per-minute limit."""
    seen = {}

    def fake_urlopen(req, *_a, **_k):
        seen["url"] = req.full_url
        seen["method"] = req.get_method()
        seen["auth"] = req.headers.get("Authorization")
        return io.BytesIO(b"{}")

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    llm.verify_key(PASTED)
    assert seen["method"] == "GET"
    assert seen["url"].endswith("/models")
    assert seen["auth"] == f"Bearer {PASTED}"


def test_verification_uses_the_candidate_not_the_installed_key(groq, monkeypatch):
    """Otherwise every paste 'succeeds' by re-checking the key already in
    place, and a bad key installs cleanly."""
    def fake_urlopen(req, *_a, **_k):
        if req.headers.get("Authorization") != f"Bearer {PASTED}":
            raise _http_error(401)
        return io.BytesIO(b"{}")

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    llm.verify_key(PASTED)


# -- the wire --------------------------------------------------------------

def test_the_response_cannot_carry_the_key(groq, monkeypatch):
    monkeypatch.setattr(llm, "verify_key", lambda *_a, **_k: None)
    out = app.api.set_api_key(app.api.ApiKeyIn(api_key=PASTED))
    assert PASTED not in out.model_dump_json()
    assert set(out.model_dump()) == {"provider", "source", "hint"}


def test_the_key_is_not_in_the_request_models_repr(groq):
    """A validation error or an unhandled traceback prints the model."""
    assert PASTED not in repr(app.api.ApiKeyIn(api_key=PASTED))


def test_get_reports_without_installing_anything(groq):
    assert app.api.get_api_key().source == "environment"
    assert os.environ["GROQ_API_KEY"] == REAL


def test_delete_falls_back(groq, monkeypatch):
    monkeypatch.setattr(llm, "verify_key", lambda *_a, **_k: None)
    app.api.set_api_key(app.api.ApiKeyIn(api_key=PASTED))
    assert app.api.clear_api_key().source == "environment"
    assert os.environ["GROQ_API_KEY"] == REAL


# -- persistence -----------------------------------------------------------
#
# A restart is simulated by calling restore_saved_keys() against a cleared
# environment: that is exactly what the lifespan does on boot, and it is the
# only step between "the file on disk" and "the key a backend reads".

def _restart(monkeypatch, baseline: str | None) -> None:
    """Put the process back to how it boots: baseline key, no override."""
    if baseline:
        monkeypatch.setenv("GROQ_API_KEY", baseline)
    else:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setitem(llm._BASELINE_KEYS, "GROQ_API_KEY", baseline)


def test_a_pasted_key_survives_a_restart(groq, monkeypatch):
    llm.set_api_key(PASTED)
    _restart(monkeypatch, REAL)
    assert llm.key_state()["source"] == "environment"    # the restart itself
    assert llm.restore_saved_keys() == ["groq"]
    assert os.environ["GROQ_API_KEY"] == PASTED
    assert llm.key_state()["source"] == "override"


def test_it_survives_on_a_host_that_booted_with_no_key(keyless, monkeypatch):
    """The deploy-keyless case: nothing to fall back to, so persistence is
    the only thing standing between a restart and a dead deployment."""
    llm.set_api_key(PASTED)
    _restart(monkeypatch, None)
    assert llm.restore_saved_keys() == ["groq"]
    assert os.environ["GROQ_API_KEY"] == PASTED


def test_clearing_does_not_come_back_after_a_restart(groq, monkeypatch, store):
    """Otherwise "use the server's key instead" is a lie with a one-boot
    shelf life."""
    llm.set_api_key(PASTED)
    llm.clear_api_key()
    assert not store.exists()
    _restart(monkeypatch, REAL)
    assert llm.restore_saved_keys() == []
    assert os.environ["GROQ_API_KEY"] == REAL


def test_rotating_the_configured_key_discards_a_stale_override(groq, monkeypatch, store):
    """The footgun this exists to prevent: the operator rotates the real key
    in the platform dashboard and redeploys, and a months-old pasted key
    silently keeps shadowing it."""
    llm.set_api_key(PASTED)
    _restart(monkeypatch, "gsk_the_operator_rotated_this")
    assert llm.restore_saved_keys() == []
    assert os.environ["GROQ_API_KEY"] == "gsk_the_operator_rotated_this"
    # and the dead record is swept, not left to be re-evaluated every boot
    assert not store.exists()


def test_the_store_does_not_hold_the_configured_key(groq, store):
    """It records a fingerprint of the baseline, never the baseline. The
    deployment's own key has no business being written to disk twice."""
    llm.set_api_key(PASTED)
    text = store.read_text(encoding="utf-8")
    assert REAL not in text
    assert PASTED in text          # this one is the whole point of the file


def test_persistence_can_be_switched_off(groq, monkeypatch, store):
    """A public deployment strangers can reach should not write their keys
    to its disk."""
    monkeypatch.setattr(app.config, "PERSIST_API_KEY", False)
    llm.set_api_key(PASTED)
    assert os.environ["GROQ_API_KEY"] == PASTED    # still live for this run
    assert not store.exists()
    _restart(monkeypatch, REAL)
    assert llm.restore_saved_keys() == []


def test_a_corrupt_store_is_not_fatal(groq, monkeypatch, store):
    """Falling back to the configured key is correct behaviour; refusing to
    start is not."""
    store.write_text("{not json", encoding="utf-8")
    _restart(monkeypatch, REAL)
    assert llm.restore_saved_keys() == []
    assert os.environ["GROQ_API_KEY"] == REAL


def test_a_store_entry_for_an_unknown_variable_is_ignored(groq, monkeypatch, store):
    store.write_text('{"AWS_SECRET_ACCESS_KEY": {"key": "x", "over": "y"}}',
                     encoding="utf-8")
    _restart(monkeypatch, REAL)
    assert llm.restore_saved_keys() == []
    assert "AWS_SECRET_ACCESS_KEY" not in os.environ


def test_an_unwritable_store_still_installs_the_key(groq, monkeypatch):
    """Persistence is a convenience. A read-only DATA_DIR must not turn a
    working paste into an error the reader cannot act on."""
    monkeypatch.setattr(app.config, "KEY_STORE",
                        Path("/nonexistent-dir-abc/provider_keys.json"))
    assert llm.set_api_key(PASTED)["source"] == "override"
    assert os.environ["GROQ_API_KEY"] == PASTED
