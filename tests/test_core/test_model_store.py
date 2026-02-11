"""Tests for core/model_store.py."""

import pytest
from sklearn.linear_model import LinearRegression

from pyanalytica.core.model_store import ModelArtifact, ModelStore


@pytest.fixture
def store():
    return ModelStore()


@pytest.fixture
def artifact():
    model = LinearRegression()
    return ModelArtifact(
        name="test_model",
        model_type="linear_regression",
        model=model,
        feature_names=["x1", "x2"],
        target_name="y",
    )


def test_save_and_get(store, artifact):
    store.save("m1", artifact)
    got = store.get("m1")
    assert got.name == "m1"
    assert got.model_type == "linear_regression"


def test_list_models(store, artifact):
    assert store.list_models() == []
    store.save("b_model", artifact)
    store.save("a_model", artifact)
    assert store.list_models() == ["a_model", "b_model"]


def test_remove(store, artifact):
    store.save("m1", artifact)
    store.remove("m1")
    assert "m1" not in store


def test_has(store, artifact):
    assert not store.has("m1")
    store.save("m1", artifact)
    assert store.has("m1")


def test_get_missing(store):
    with pytest.raises(KeyError, match="not found"):
        store.get("missing")


def test_len_and_contains(store, artifact):
    assert len(store) == 0
    store.save("m1", artifact)
    assert len(store) == 1
    assert "m1" in store
    assert "m2" not in store
