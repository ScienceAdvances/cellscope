"""Device precedence, optional dependencies, isolated GPU dispatch and CPU workflow."""

import types

import numpy as np
import pytest
from anndata import AnnData
from scipy import sparse

import cellscope as cs
from cellscope import _device


@pytest.fixture(autouse=True)
def restore_settings():
    original = cs.settings.device
    yield
    cs.settings.device = original


def test_settings_and_legacy_precedence(monkeypatch):
    cs.settings.device = "auto"
    assert _device.requested_device() == "auto"
    assert _device.requested_device("cpu") == "cpu"
    assert _device.requested_device(legacy=False) == "cpu"
    assert _device.requested_device(legacy=True) == "gpu"
    with pytest.raises(ValueError, match="conflicts"):
        _device.requested_device("cpu", True)
    with pytest.raises(ValueError, match="device"):
        cs.settings.device = "cuda-ish"
    monkeypatch.setattr(_device, "_probe", lambda kind: pytest.fail("CPU must not probe CUDA"))
    assert not _device.select_gpu("test", device="cpu")


def test_auto_fallback_and_strict_gpu(monkeypatch, capsys):
    monkeypatch.setattr(_device, "_probe", lambda kind: (False, "missing GPU package"))
    assert not _device.select_gpu("test", device="auto")
    assert "using CPU" in capsys.readouterr().err
    with pytest.raises(RuntimeError, match="missing GPU package"):
        _device.select_gpu("test", device="gpu")


def test_gpu_failure_probe_returns_reason(monkeypatch):
    def missing(name):
        raise ImportError("missing dependency")

    monkeypatch.setattr(_device, "import_module", missing)
    assert _device._probe.__wrapped__("rapids") == (False, "ImportError: missing dependency")


def fake_rapids(monkeypatch, gpu_function, *, name="neighbors"):
    rsc = types.SimpleNamespace(
        pp=types.SimpleNamespace(**{name: gpu_function}),
        get=types.SimpleNamespace(
            anndata_to_GPU=lambda *args, **kwargs: None,
            anndata_to_CPU=lambda *args, **kwargs: None,
        ),
    )
    monkeypatch.setattr(_device, "_probe", lambda kind: (True, "test GPU"))
    real_import = _device.import_module
    monkeypatch.setattr(_device, "import_module", lambda name:
                        rsc if name == "rapids_singlecell" else real_import(name))


def test_gpu_graph_outputs_on_host_and_original_counts_untouched(monkeypatch):
    adata = AnnData(sparse.csr_matrix(np.ones((25, 5), dtype=np.int64)))
    adata.layers["raw_counts"] = adata.X.copy()

    def gpu_neighbors(working, n_neighbors=15):
        assert working.X.dtype == np.float64
        assert working.layers["raw_counts"].dtype == np.int64
        working.obsp["distances"] = sparse.csr_array(np.eye(25))
        working.uns["neighbors"] = {"params": {"n_neighbors": n_neighbors}}
        working.X.data[:] = 99

    fake_rapids(monkeypatch, gpu_neighbors)
    cs.pp.neighbors(adata, n_neighbors=5, device="gpu")
    assert sparse.isspmatrix_csr(adata.obsp["distances"])
    assert adata.uns["neighbors"]["params"]["n_neighbors"] == 5
    assert adata.X.dtype == np.int64
    assert (adata.X != adata.layers["raw_counts"]).nnz == 0
    assert adata.X.max() == 1
    assert adata.uns["cellscope"]["devices"]["pp.neighbors"]["device"] == "gpu"


def test_gpu_error_discards_partial_results(monkeypatch):
    adata = AnnData(np.ones((25, 5)))

    def broken(working):
        working.obs["partial"] = True
        working.X[:] = 99
        raise ValueError("invalid graph input")

    fake_rapids(monkeypatch, broken)
    with pytest.raises(ValueError, match="invalid graph"):
        cs.pp.neighbors(adata, device="auto")
    assert "partial" not in adata.obs
    assert adata.X.max() == 1


def test_explicit_cpu_algorithm_fallback_without_probe(monkeypatch, capsys):
    import scanpy as sc

    adata = AnnData(np.ones((25, 5)))
    monkeypatch.setattr(_device, "_probe", lambda kind: pytest.fail("no GPU probe needed"))
    calls = []
    monkeypatch.setattr(sc.tl, "leiden", lambda data, **kwargs: calls.append(kwargs))
    cs.tl.leiden(adata, device="auto", flavor="igraph")
    assert calls[0]["flavor"] == "igraph"
    assert "using CPU" in capsys.readouterr().err
    with pytest.raises(RuntimeError, match="flavor"):
        cs.tl.leiden(adata, device="gpu", flavor="igraph")


def test_gpu_normalization_commits_only_corrected_assay(monkeypatch):
    adata = AnnData(sparse.csr_matrix(np.full((25, 5), 2, dtype=np.int64)))
    adata.layers["counts"] = adata.X.copy()

    def normalize_total(working, target_sum=1e4):
        working.X.data[:] = 7

    fake_rapids(monkeypatch, normalize_total, name="normalize_total")
    _device.run_scanpy(adata, "pp", "normalize_total", device="gpu")
    assert adata.X.max() == 7
    assert adata.layers["counts"].max() == 2
    assert adata.layers["counts"].dtype == np.int64


def test_cpu_full_workflow():
    adata = AnnData(sparse.csr_matrix(np.random.default_rng(3).poisson(3, (80, 50))))
    cs.tl.workflow(adata, n_top_genes=30, n_pcs=5, n_neighbors=5, device="cpu")
    assert adata.obsm["X_umap"].shape == (80, 2)
    assert np.isfinite(adata.obsm["X_umap"]).all()
    assert "leiden" in adata.obs
    assert adata.layers["counts"].dtype.kind in "iu"
    assert all(value["device"] == "cpu"
               for value in adata.uns["cellscope"]["devices"].values())


@pytest.mark.parametrize("gpu,expected_iterations", [(True, 100), (False, 2)])
def test_workflow_uses_backend_specific_leiden_iterations(monkeypatch, gpu, expected_iterations):
    from cellscope.tools import _api

    adata = AnnData(np.ones((30, 6)))
    adata.obsm["X_pca"] = np.ones((30, 3))
    for name in ("normalize", "highly_variable_genes", "neighbors"):
        monkeypatch.setattr(cs.pp, name, lambda *args, **kwargs: None)
    for name in ("pca", "umap"):
        monkeypatch.setattr(_api, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(_api, "select_gpu", lambda *args, **kwargs: gpu)
    calls = []
    monkeypatch.setattr(_api, "leiden", lambda *args, **kwargs: calls.append(kwargs))
    _api.workflow(adata, n_pcs=3)
    assert calls[0]["n_iterations"] == expected_iterations
    assert calls[0]["device"] == ("gpu" if gpu else "cpu")
