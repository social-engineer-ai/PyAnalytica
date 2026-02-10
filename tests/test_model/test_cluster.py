"""Tests for model/cluster.py."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from pyanalytica.model.cluster import hierarchical_cluster, kmeans_cluster


@pytest.fixture
def df():
    np.random.seed(42)
    return pd.DataFrame({
        "x": np.concatenate([np.random.normal(0, 1, 50), np.random.normal(5, 1, 50)]),
        "y": np.concatenate([np.random.normal(0, 1, 50), np.random.normal(5, 1, 50)]),
    })


def test_kmeans(df):
    result = kmeans_cluster(df, ["x", "y"], chosen_k=2)
    assert result.n_clusters == 2
    assert len(result.labels) == 100
    assert result.elbow_plot is not None


def test_kmeans_profiles(df):
    result = kmeans_cluster(df, ["x", "y"], chosen_k=2)
    assert len(result.cluster_profiles) == 2


def test_hierarchical(df):
    result = hierarchical_cluster(df, ["x", "y"], n_clusters=2)
    assert result.n_clusters == 2
    assert len(result.labels) == 100


def test_scatter_plot(df):
    result = kmeans_cluster(df, ["x", "y"], chosen_k=2)
    assert result.scatter_plot is not None
