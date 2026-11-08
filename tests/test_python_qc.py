"""Protect library isolation, count provenance, and barcode/feature alignment."""

import types

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

import cellscope as cs
from cellscope.preprocessing import _python_qc


def test_doublets_are_library_local_and_only_annotations_change(monkeypatch):
    matrix = sparse.csr_matrix(np.arange(48 * 5).reshape(48, 5) % 7)
    data = AnnData(matrix.astype(float), obs=pd.DataFrame({
        "library_id": ["a", "b"] * 24,
    }, index=[f"cell{i}" for i in range(48)]))
    data.layers["counts"] = matrix.copy()
    original_x, original_counts = data.X.copy(), matrix.copy()
    calls = []

    def compute(working, **kwargs):
        assert working.obs.library_id.nunique() == 1
        assert kwargs["return_type"] == "adata"
        calls.append(working.obs_names.tolist())
        working.obs["scDblFinder_score"] = np.linspace(0, 1, working.n_obs)
        working.obs["scDblFinder_class"] = np.where(
            working.obs.scDblFinder_score > 0.5, "doublet", "singlet"
        )
        working.uns["scDblFinder_threshold"] = 0.5
        working.X[:] = 0  # backend mutations must not reach original matrices
        return working[::-1].copy()  # annotation alignment must follow cell IDs

    monkeypatch.setattr(_python_qc, "dependency", lambda *args: types.SimpleNamespace(
        compute_doublet_score=compute,
    ))
    cs.pp.scdblfinder(data, backend="python", batch_key="library_id")
    assert len(calls) == 2
    assert (data.X != original_x).nnz == 0
    assert (data.layers["counts"] != original_counts).nnz == 0
    np.testing.assert_allclose(data.obs.iloc[::2].scDblFinder_score, np.linspace(0, 1, 24))
    assert data.obs.predicted_doublet.sum() == 24
    assert data.uns["scDblFinder_thresholds"] == {"a": 0.5, "b": 0.5}


@pytest.mark.parametrize("use_gpu", [False, True])
@pytest.mark.parametrize("sparse_input", [False, True])
def test_doublet_gpu_count_dtype_preserves_input(monkeypatch, use_gpu, sparse_input):
    monkeypatch.setattr(_python_qc, "select_gpu", lambda *args, **kwargs: use_gpu)
    values = np.arange(40 * 5, dtype=np.int64).reshape(40, 5)
    # Above float32's consecutive-integer range: float64 must preserve this UMI.
    values[0, 0] = 2**24 + 1
    matrix = sparse.csr_matrix(values) if sparse_input else values.copy()
    data = AnnData(matrix, obs=pd.DataFrame(
        {"library": ["a"] * 20 + ["b"] * 20},
        index=[f"c{i}" for i in range(40)],
    ))
    data.layers["raw_counts"] = matrix.copy()
    calls = []

    def compute(working, **kwargs):
        expected = np.float64 if use_gpu else np.int64
        assert working.X.dtype == expected
        assert working.layers["counts"].dtype == expected
        assert kwargs["use_gpu"] == use_gpu
        actual = working.X.toarray() if sparse.issparse(working.X) else working.X
        offset = len(calls) * 20
        np.testing.assert_array_equal(actual, values[offset:offset + 20])
        # Match upstream's concat of real cells and integer artificial doublets.
        combined = sparse.vstack([sparse.csr_matrix(working.X),
                                  sparse.csr_matrix(np.ones((2, 5), dtype=np.int64))])
        assert combined.dtype == expected
        calls.append(working.n_obs)
        working.obs["scDblFinder_score"] = 0.1
        working.obs["scDblFinder_class"] = "singlet"
        working.uns["scDblFinder_threshold"] = 0.5
        return working

    monkeypatch.setattr(_python_qc, "dependency", lambda *args: types.SimpleNamespace(
        compute_doublet_score=compute,
    ))
    cs.pp.scDblFinder(data, batch_key="library", use_gpu=use_gpu)
    assert calls == [20, 20]
    assert data.X.dtype == np.int64
    assert data.layers["raw_counts"].dtype == np.int64
    actual = data.X.toarray() if sparse_input else data.X
    np.testing.assert_array_equal(actual, values)
    raw = data.layers["raw_counts"].toarray() if sparse_input else data.layers["raw_counts"]
    np.testing.assert_array_equal(raw, values)


def test_doublets_reject_implicit_pooled_libraries_and_normalized_input(monkeypatch):
    monkeypatch.setattr(_python_qc, "dependency", lambda *args: object())
    data = AnnData(sparse.csr_matrix(np.ones((40, 5))), obs=pd.DataFrame({
        "library_id": ["a"] * 20 + ["b"] * 20,
    }, index=[f"c{i}" for i in range(40)]))
    data.layers["counts"] = data.X.copy()
    with pytest.raises(ValueError, match="Multiple libraries"):
        cs.pp.scdblfinder_py(data)
    data.layers["counts"].data[:] = 0.1
    with pytest.raises(ValueError, match="integer counts"):
        cs.pp.scdblfinder_py(data, batch_key="library_id")


def test_cellbender_reader_aligns_barcodes_and_stable_gene_ids(monkeypatch, tmp_path):
    downstream = pytest.importorskip("cellbender.remove_background.downstream")
    import h5py
    import scanpy as sc

    raw = AnnData(sparse.csr_matrix([[10, 20, 30], [40, 50, 60], [1, 1, 1]]),
                  obs=pd.DataFrame(index=["a", "b", "empty"]),
                  var=pd.DataFrame({"gene_ids": ["id1", "id2", "id3"]},
                                   index=["one", "two", "three"]))
    corrected = AnnData(sparse.csr_matrix([[54, 36, 45], [27, 9, 18], [0, 0, 0]]),
                        obs=pd.DataFrame({"cell_probability": [0.9, 0.99, 0.1]},
                                         index=["b", "a", "empty"]),
                        var=pd.DataFrame({"gene_id": ["id3", "id1", "id2"]},
                                         index=["duplicate", "duplicate", "other"]))
    monkeypatch.setattr(downstream, "anndata_from_h5", lambda *a, **k: corrected.copy())
    monkeypatch.setattr(sc, "read_10x_h5", lambda *a, **k: raw.copy())
    output = tmp_path / "out.h5"
    with h5py.File(output, "w") as handle:
        handle.create_dataset("matrix/features/id", data=np.array(["id3", "id1", "id2"], dtype="S"))
    result = cs.io.read_cellbender(output, raw_file="raw.h5", library_id="lib")
    assert result.obs_names.tolist() == ["lib:b", "lib:a"]
    assert result.obs.barcode.tolist() == ["b", "a"]
    np.testing.assert_array_equal(result.layers["raw_counts"].toarray(), [[60, 40, 50], [30, 10, 20]])
    np.testing.assert_array_equal(result.layers["counts"].toarray(), [[54, 36, 45], [27, 9, 18]])
    assert result.var_names.is_unique
    result.write_h5ad(tmp_path / "corrected.h5ad")
