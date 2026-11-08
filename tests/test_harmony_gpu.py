"""Harmony backend semantics and explicit cuGraph Leiden selection."""

import types

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

import cellscope as cs
from cellscope import _device
from cellscope.tools import _api, _harmony


@pytest.fixture
def data():
    adata = AnnData(sparse.csr_matrix(np.ones((40, 6), dtype=np.int64)),
                    obs=pd.DataFrame({"batch": ["a"] * 20 + ["b"] * 20},
                                     index=[f"c{i}" for i in range(40)]))
    adata.obsm["X_pca"] = np.random.default_rng(0).normal(size=(40, 4))
    adata.layers["raw_counts"] = adata.X.copy()
    return adata


def fake_gpu(monkeypatch, function):
    monkeypatch.setattr(_harmony, "select_gpu", lambda *args, **kwargs: True)
    monkeypatch.setattr(_harmony, "dependency", lambda *args:
                        types.SimpleNamespace(pp=types.SimpleNamespace(harmony_integrate=function)))


@pytest.mark.parametrize("flavor", [None, "harmony1", "harmony2"])
def test_gpu_harmony_flavor_and_parameter_translation(monkeypatch, data, flavor):
    original = data.obsm["X_pca"].copy()

    def integrate(working, key, basis, adjusted_basis, flavor, max_iter_harmony,
                  n_clusters, tol_harmony):
        assert working.n_vars == 0  # Expression assay never copied/transferred.
        assert key == ["batch"]
        assert max_iter_harmony == 17 and n_clusters == 3 and tol_harmony == 0.02
        working.obsm[adjusted_basis] = working.obsm[basis] + 1

    fake_gpu(monkeypatch, integrate)
    result = cs.tl.harmony(data, batch_key="batch", device="gpu", flavor=flavor,
                           nclust=3, epsilon_harmony=0.02, max_iter_harmony=17)
    assert isinstance(result, _harmony.GPUHarmonyResult)
    assert result.flavor == (flavor or "harmony1")
    np.testing.assert_array_equal(data.obsm["X_pca_harmony"], original + 1)
    np.testing.assert_array_equal(data.obsm["X_pca"], original)
    assert (data.X != data.layers["raw_counts"]).nnz == 0
    assert data.X.dtype == np.int64
    assert data.uns["cellscope"]["devices"]["tl.harmony"]["device"] == "gpu"


def test_cpu_keeps_native_fit_and_transposes_old_embedding(monkeypatch, data):
    monkeypatch.setattr(_harmony, "select_gpu", lambda *args, **kwargs: False)
    original = data.obsm["X_pca"].copy()
    expected = original + 2
    fit = types.SimpleNamespace(Z_corr=expected.T)

    def run(matrix, obs, keys, **kwargs):
        assert kwargs == {"nclust": 3}
        matrix[:] = 99
        return fit

    monkeypatch.setattr(_harmony, "dependency", lambda *args: types.SimpleNamespace(run_harmony=run))
    assert cs.tl.harmony(data, batch_key="batch", device="cpu", nclust=3) is fit
    np.testing.assert_array_equal(data.obsm["X_pca_harmony"], expected)
    np.testing.assert_array_equal(data.obsm["X_pca"], original)
    assert data.uns["cellscope"]["devices"]["tl.harmony"]["flavor"] == "harmonypy_native"
    with pytest.raises(RuntimeError, match="Explicit Harmony flavor"):
        cs.tl.harmony(data, batch_key="batch", device="cpu", flavor="harmony2")


@pytest.mark.parametrize("bad_result", ["nan", "shape", "exception"])
def test_gpu_harmony_failure_does_not_write_result(monkeypatch, data, bad_result):
    original = data.obsm["X_pca"].copy()

    def integrate(working, key, basis, adjusted_basis, flavor):
        if bad_result == "exception":
            raise ValueError("upstream failure")
        working.obsm[adjusted_basis] = (
            np.full((40, 4), np.nan) if bad_result == "nan" else np.zeros((40, 2)))

    fake_gpu(monkeypatch, integrate)
    with pytest.raises(ValueError):
        cs.tl.harmony(data, batch_key="batch", device="gpu")
    assert "X_pca_harmony" not in data.obsm
    np.testing.assert_array_equal(data.obsm["X_pca"], original)


def test_explicit_cugraph_flavor_routes_gpu(monkeypatch, data):
    calls = []
    monkeypatch.setattr(_api, "run_scanpy", lambda *args, **kwargs: calls.append(kwargs))
    cs.tl.leiden(data, flavor="cugraph", directed=False)
    assert calls[0]["device"] == "gpu"
    assert "flavor" not in calls[0]
    with pytest.raises(ValueError, match="requires"):
        cs.tl.leiden(data, flavor="cugraph", device="cpu")
    with pytest.raises(RuntimeError, match="selects CPU"):
        cs.tl.leiden(data, flavor="igraph", device="gpu")


def test_duplicate_harmony_parameter_names_rejected(data):
    with pytest.raises(ValueError, match="Duplicate"):
        cs.tl.harmony(data, batch_key="batch", nclust=3, n_clusters=3)


@pytest.mark.parametrize("lamb", [None, -1, [1, 2]])
def test_gpu_rejects_native_dynamic_or_vector_lambda(monkeypatch, data, lamb):
    def integrate(working, key, basis, adjusted_basis, flavor, ridge_lambda):
        pytest.fail("unsupported lambda must be rejected before GPU execution")

    monkeypatch.setattr(_device, "_probe", lambda kind: (True, "test GPU"))
    monkeypatch.setattr(_harmony, "dependency", lambda *args:
                        types.SimpleNamespace(pp=types.SimpleNamespace(harmony_integrate=integrate)))
    with pytest.raises(RuntimeError, match="positive scalar"):
        cs.tl.harmony(data, batch_key="batch", device="gpu", lamb=lamb)
