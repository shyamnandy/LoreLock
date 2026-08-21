"""The embedding backend and the guards around stored matrix width.

The local backend is exercised indirectly all over the suite; what needs
covering here is the contract `app.store.vectors` relies on -- shape, dtype,
the empty case, and the mismatch guard that stops a model change from
silently corrupting an index.

Hermetic: fastembed is never loaded. `_cached_model` is monkeypatched, so
these run in milliseconds and need no ONNX session.
"""
from __future__ import annotations

import numpy as np
import pytest

import app.config
from app.store import embedders, vectors


class FakeModel:
    """Stands in for fastembed's TextEmbedding."""

    def __init__(self, dim, *, short_by=0):
        self.dim = dim
        self.short_by = short_by
        self.calls = []

    def embed(self, texts, batch_size=None, parallel=None):
        texts = list(texts)
        self.calls.append({"n": len(texts), "batch_size": batch_size,
                           "parallel": parallel})
        for i in range(len(texts) - self.short_by):
            yield np.full(self.dim, float(i + 1), dtype=np.float32)


@pytest.fixture
def fake_model(monkeypatch):
    model = FakeModel(app.config.EMBED_DIM)
    monkeypatch.setattr(embedders, "_cached_model", model)
    return model


def test_model_name_is_the_configured_one():
    assert embedders.model_name() == app.config.EMBED_MODEL
    assert app.config.EMBED_DIM == 384


def test_embed_returns_the_declared_shape_and_dtype(fake_model):
    out = embedders.embed(["a", "b", "c"])
    assert out.shape == (3, app.config.EMBED_DIM)
    assert out.dtype == np.float32


def test_empty_input_never_loads_a_model(monkeypatch):
    """An ingest that produced no chunks must fail in the pipeline with a
    useful message, not here by loading 190 MB of ONNX to embed nothing."""
    monkeypatch.setattr(embedders, "_embedder",
                        lambda: pytest.fail("model was loaded for empty input"))
    monkeypatch.setattr(embedders, "_cached_model", None)
    out = embedders.embed([])
    assert out.shape == (0, app.config.EMBED_DIM)


def test_short_result_is_an_error_not_a_silent_gap(monkeypatch):
    """A backend returning fewer vectors than chunks would leave a row of
    uninitialised memory, which retrieval would happily score against."""
    monkeypatch.setattr(embedders, "_cached_model",
                        FakeModel(app.config.EMBED_DIM, short_by=1))
    with pytest.raises(embedders.EmbedError, match="returned 2 vectors for 3"):
        embedders.embed(["a", "b", "c"])


def test_memory_defaults_are_passed_through(fake_model):
    """CHALLENGES.md §6: batch drives ONNX's arena and `parallel` not being
    None spawns a second process holding its own copy of the model."""
    embedders.embed(["a"])
    assert fake_model.calls[0]["batch_size"] == embedders._BATCH
    assert fake_model.calls[0]["parallel"] is embedders._PARALLEL
    assert embedders._BATCH <= 8
    assert embedders._PARALLEL is None


def test_input_type_is_accepted_and_ignored(fake_model):
    """bge draws no query/document distinction; the kwarg exists because
    `vectors.embed_query` passes it."""
    doc = embedders.embed(["same text"], input_type="document")
    qry = embedders.embed(["same text"], input_type="query")
    assert np.array_equal(doc, qry)


def test_vectors_normalises_whatever_the_backend_returns(fake_model):
    """Cosine is a plain dot product downstream, which is only true if every
    stored row is unit length."""
    out = vectors.embed_texts(["a", "b"])
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_save_rejects_a_matrix_of_the_wrong_width(tmp_path, monkeypatch):
    """The guard that makes a model change fail loudly instead of writing an
    index that no query can ever be compared against."""
    monkeypatch.setattr(vectors, "VEC_DIR", tmp_path)
    wrong = np.zeros((4, app.config.EMBED_DIM + 1), dtype=np.float32)
    with pytest.raises(ValueError, match=rf"expected \(n,{app.config.EMBED_DIM}\)"):
        vectors.save("bookid", wrong)
