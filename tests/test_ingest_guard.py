"""The switch that lets a host serve a library without being able to build one.

Indexing is minutes of pegged CPU and the app's memory high-water mark, so a
small instance is better off refusing than accepting a job the platform will
kill half way through. What matters here is only *when* the refusal happens:
in the request, before a job row exists, so the client gets an error it can
act on rather than a job that never resolves.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

import app.api
import app.config


def test_allowed_by_default():
    """A laptop and any host with a disk must ingest without configuration."""
    assert app.config.ALLOW_RUNTIME_INGEST is True
    app.api._require_ingest_allowed()          # must not raise


def test_refuses_with_503_when_disabled(monkeypatch):
    monkeypatch.setattr(app.config, "ALLOW_RUNTIME_INGEST", False)
    with pytest.raises(HTTPException) as exc:
        app.api._require_ingest_allowed()
    # 503, not 403: the request is permitted, the host just cannot serve it.
    assert exc.value.status_code == 503
    assert "build_seed" in exc.value.detail


def test_flag_is_read_at_call_time(monkeypatch):
    """Two traps in one: a `from app.config import ...` would freeze the
    value at import, and a plain `import app.config` resolves to the FastAPI
    object, because app/api.py binds `app = api` for the ASGI convention."""
    monkeypatch.setattr(app.config, "ALLOW_RUNTIME_INGEST", False)
    with pytest.raises(HTTPException):
        app.api._require_ingest_allowed()
    monkeypatch.setattr(app.config, "ALLOW_RUNTIME_INGEST", True)
    app.api._require_ingest_allowed()          # must not raise
