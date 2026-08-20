"""EMBED_PROVIDER selection and the voyage backend.

The local backend is exercised everywhere else in the suite; what needs
covering here is the remote path, which no other test touches, and the
guards that stop a provider switch from silently corrupting an index.

No network: `requests.post` is monkeypatched throughout, so this stays as
hermetic as the rest of the suite.
"""
from __future__ import annotations

import numpy as np
import pytest

import app.config
from app.store import embedders


class FakeResponse:
    def __init__(self, payload, status_code=200, text="", headers=None):
        self._payload = payload
        self.status_code = status_code
        self.text = text or str(payload)
        self.headers = headers or {}

    def json(self):
        return self._payload


@pytest.fixture
def voyage(monkeypatch):
    """Select the voyage backend with a 4-dim model and a key present."""
    monkeypatch.setattr(app.config, "EMBED_PROVIDER", "voyage")
    monkeypatch.setattr(app.config, "EMBED_MODEL", "voyage-test")
    monkeypatch.setattr(app.config, "EMBED_DIM", 4)
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")


def _rows(vectors):
    return {"data": [{"embedding": v, "index": i}
                     for i, v in enumerate(vectors)]}


def test_local_is_the_default():
    assert embedders.provider_name() == "local"
    assert embedders.available() is True


def test_unknown_provider_names_the_valid_ones(monkeypatch):
    monkeypatch.setattr(app.config, "EMBED_PROVIDER", "nope")
    with pytest.raises(embedders.EmbedError, match="unknown EMBED_PROVIDER"):
        embedders.embed(["anything"])


def test_empty_input_never_calls_a_backend(monkeypatch):
    # An ingest that produced no chunks must fail in the pipeline with a
    # useful message, not here with a provider error.
    monkeypatch.setattr(app.config, "EMBED_PROVIDER", "nope")
    out = embedders.embed([])
    assert out.shape == (0, app.config.EMBED_DIM)


def test_voyage_missing_key_is_reported_plainly(monkeypatch):
    monkeypatch.setattr(app.config, "EMBED_PROVIDER", "voyage")
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    assert embedders.available() is False
    with pytest.raises(embedders.EmbedError, match="VOYAGE_API_KEY is not set"):
        embedders.embed(["text"])


def test_voyage_sends_model_dim_and_input_type(voyage, monkeypatch):
    seen = {}

    def fake_post(url, json, headers, timeout):
        seen.update(url=url, json=json, headers=headers)
        return FakeResponse(_rows([[1.0, 0.0, 0.0, 0.0]]))

    monkeypatch.setattr("requests.post", fake_post)
    embedders.embed(["a question"], input_type="query")

    assert seen["url"].endswith("/v1/embeddings")
    assert seen["headers"]["Authorization"] == "Bearer test-key"
    assert seen["json"]["model"] == "voyage-test"
    assert seen["json"]["output_dimension"] == 4
    # the asymmetric side: a query must not be embedded as a document
    assert seen["json"]["input_type"] == "query"


def test_voyage_reorders_by_index(voyage, monkeypatch):
    """A shuffled response must not pair the wrong vector with the wrong chunk.

    Nothing downstream could detect this -- retrieval would just quietly
    return the wrong passage -- so the ordering is enforced here.
    """
    shuffled = {"data": [
        {"embedding": [2.0, 0.0, 0.0, 0.0], "index": 1},
        {"embedding": [1.0, 0.0, 0.0, 0.0], "index": 0},
    ]}
    monkeypatch.setattr("requests.post",
                        lambda *a, **k: FakeResponse(shuffled))
    out = embedders.embed(["first", "second"])
    assert out[0][0] == 1.0
    assert out[1][0] == 2.0


def test_voyage_dimension_mismatch_says_reindex(voyage, monkeypatch):
    """The failure mode that matters: a model whose width is not EMBED_DIM.

    Left unchecked this writes a matrix that no stored index can be compared
    against, so the message has to name the fix.
    """
    monkeypatch.setattr("requests.post",
                        lambda *a, **k: FakeResponse(_rows([[1.0, 2.0]])))
    with pytest.raises(embedders.EmbedError, match="re-index"):
        embedders.embed(["text"])


def test_voyage_short_batch_is_an_error(voyage, monkeypatch):
    monkeypatch.setattr("requests.post",
                        lambda *a, **k: FakeResponse(_rows([[1.0, 0.0, 0.0, 0.0]])))
    with pytest.raises(embedders.EmbedError, match="1 vectors for 2 inputs"):
        embedders.embed(["one", "two"])


def test_voyage_4xx_is_not_retried(voyage, monkeypatch):
    calls = []

    def fake_post(*a, **k):
        calls.append(1)
        return FakeResponse({}, status_code=401, text="unauthorized")

    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr(embedders, "_VOYAGE_RETRIES", 2)
    monkeypatch.setattr(embedders.time, "sleep", lambda *_: None)
    with pytest.raises(embedders.EmbedError, match="401"):
        embedders.embed(["text"])
    # a bad key will still be bad on the third attempt
    assert len(calls) == 1


def test_voyage_batches_long_inputs(voyage, monkeypatch):
    monkeypatch.setattr(embedders, "_VOYAGE_BATCH", 2)
    sizes = []

    def fake_post(url, json, headers, timeout):
        sizes.append(len(json["input"]))
        return FakeResponse(_rows([[1.0, 0.0, 0.0, 0.0]] * len(json["input"])))

    monkeypatch.setattr("requests.post", fake_post)
    out = embedders.embed([f"chunk {i}" for i in range(5)])
    assert sizes == [2, 2, 1]
    assert out.shape == (5, 4)


def test_vectors_normalises_whatever_the_backend_returns(voyage, monkeypatch):
    """vectors.embed_texts owns normalisation, so cosine stays a dot product
    no matter which provider produced the raw vector."""
    from app.store import vectors

    monkeypatch.setattr("requests.post",
                        lambda *a, **k: FakeResponse(_rows([[3.0, 4.0, 0.0, 0.0]])))
    out = vectors.embed_texts(["text"])
    assert np.isclose(np.linalg.norm(out[0]), 1.0)


def test_voyage_429_is_waited_out_not_abandoned(voyage, monkeypatch):
    """Rate limits are the expected case on a free project, not an anomaly.

    A whole book is many minutes of token budget, so a 429 has to be slept
    through -- giving up would strand the ingest partway.
    """
    replies = [FakeResponse({}, status_code=429, text="rate limited"),
               FakeResponse(_rows([[1.0, 0.0, 0.0, 0.0]]))]
    monkeypatch.setattr("requests.post", lambda *a, **k: replies.pop(0))
    slept = []
    monkeypatch.setattr(embedders.time, "sleep", slept.append)

    out = embedders.embed(["text"])
    assert out.shape == (1, 4)
    assert slept == [embedders._VOYAGE_BACKOFF]


def test_voyage_429_honours_retry_after(voyage, monkeypatch):
    replies = [FakeResponse({}, status_code=429, headers={"Retry-After": "7"}),
               FakeResponse(_rows([[1.0, 0.0, 0.0, 0.0]]))]
    monkeypatch.setattr("requests.post", lambda *a, **k: replies.pop(0))
    slept = []
    monkeypatch.setattr(embedders.time, "sleep", slept.append)

    embedders.embed(["text"])
    # the server's own number wins over our default guess
    assert slept == [7.0]


def test_default_batch_fits_the_free_tier_token_budget():
    """A batch that exceeds the per-minute token cap can never succeed.

    Guards the default against being raised back to a value that 429s on
    every request regardless of how long it waits.
    """
    tokens_per_chunk = 315          # measured on this corpus
    assert embedders._VOYAGE_BATCH * tokens_per_chunk <= 10_000
