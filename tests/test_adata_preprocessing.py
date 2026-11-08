"""Merged-data contracts: batch isolation, X defaults, and non-destructive backup."""

import json
import types

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

import cellscope as cs
from cellscope.preprocessing import _adata_preprocessing as api
from cellscope.preprocessing import _python_qc


def _merged():
    obj = AnnData(sparse.csr_matrix([[10, 20], [30, 40], [50, 60], [70, 80]]),
                  obs=pd.DataFrame({"batch": ["a", "b", "a", "b"],
                                    "barcode": ["a0", "b0", "a1", "b1"]},
                                   index=["a:a0", "b:b0", "a:a1", "b:b1"]),
                  var=pd.DataFrame({"gene_ids": ["g1", "g2"]}, index=["G1", "G2"]))
    obj.layers["counts"] = sparse.csr_matrix(np.full((4, 2), 999))
    return obj


def test_cellbender_batches_align_and_backup_original_x(monkeypatch, tmp_path):
    raw_files = {b: tmp_path / f"{b}_raw.h5" for b in ["a", "b"]}
    for f in raw_files.values():
        f.write_text("raw fixture")
    calls = []

    def run(raw, output, **options):
        batch = raw.name[0]
        calls.append((batch, options))
        output.write_text("output fixture")
        output.with_suffix(".cellscope.json").write_text(json.dumps({
            "status": "complete", "input_file": str(raw),
        }))

    def read(output, **kwargs):
        batch = kwargs["raw_file"].name[0]
        original = _merged()
        block = original[original.obs.batch.eq(batch)].copy()[::-1, ::-1].copy()
        block.var["gene_id"] = block.var.gene_ids
        block.layers["raw_counts"] = block.X.copy()
        block.X = block.X // 2 if not sparse.issparse(block.X) else block.X.multiply(0.5).tocsr()
        block.obs["cell_probability"] = [0.8, 0.2]
        return block

    monkeypatch.setattr(api, "cellbender_file", run)
    monkeypatch.setattr(cs.io, "read_cellbender", read)
    data = _merged()
    before = data.X.copy()
    result = cs.pp.cellbender(
        data, batch_key="batch", raw_files=raw_files, output_dir=tmp_path / "results",
        expected_cells={"a": 2, "b": 3}, epochs=2,
    )
    assert result is data
    np.testing.assert_array_equal(data.layers["raw_counts"].toarray(), before.toarray())
    np.testing.assert_array_equal(data.X.toarray(), before.toarray() / 2)
    np.testing.assert_array_equal(data.layers["counts"].toarray(), before.toarray() / 2)
    assert data.obs_names.tolist() == _merged().obs_names.tolist()
    assert data.var_names.tolist() == ["G1", "G2"]
    assert [x[1]["expected_cells"] for x in calls] == [2, 3]
    np.testing.assert_allclose(data.obs.cellbender_cell_probability, [0.2, 0.2, 0.8, 0.8])
    # Cache reuse must neither retrain nor silently accept changed settings.
    cs.pp.cellbender(_merged(), batch_key="batch", raw_files=raw_files,
                     output_dir=tmp_path / "results", expected_cells={"a": 2, "b": 3},
                     epochs=2, reuse_outputs=True)
    assert len(calls) == 2
    with pytest.raises(ValueError, match="scientific settings"):
        cs.pp.cellbender(_merged(), batch_key="batch", raw_files=raw_files,
                         output_dir=tmp_path / "results", epochs=3, reuse_outputs=True)
    with pytest.raises(ValueError, match="Backup layer"):
        cs.pp.cellbender(data, batch_key="batch", raw_files=raw_files, output_dir=tmp_path)
    data.write_h5ad(tmp_path / "backed_up.h5ad")
    assert "raw_counts" in ad.read_h5ad(tmp_path / "backed_up.h5ad").layers

    def bad_read(output, **kwargs):
        result = read(output, **kwargs)
        if kwargs["raw_file"].name.startswith("b"):
            result.obs.iloc[0, result.obs.columns.get_loc("barcode")] = "missing"
        return result

    monkeypatch.setattr(cs.io, "read_cellbender", bad_read)
    failed = _merged()
    with pytest.raises(ValueError, match="output omits"):
        cs.pp.cellbender(failed, batch_key="batch", raw_files=raw_files,
                         output_dir=tmp_path / "bad_alignment")
    assert "raw_counts" not in failed.layers
    np.testing.assert_array_equal(failed.X.toarray(), before.toarray())


def test_basic_qc_uses_batch_gene_support_and_preserves_count_layers():
    matrix = sparse.csr_matrix([
        [2, 3, 0, 0], [2, 3, 0, 0], [2, 3, 0, 0], [0, 0, 0, 0],
        [2, 0, 3, 0], [2, 0, 3, 0], [2, 0, 3, 0], [0, 0, 0, 0],
    ])
    data = AnnData(matrix, obs=pd.DataFrame({"batch": ["a"] * 4 + ["b"] * 4},
                                          index=[f"c{i}" for i in range(8)]),
                   var=pd.DataFrame(index=["shared", "a_only", "b_only", "MT-X"]))
    data.layers["raw_counts"] = matrix.copy()
    result = cs.pp.qc(data, batch_key="batch", min_genes=2, min_cells=2, filter=True)
    assert result.shape == (6, 3)
    assert result.var_names.tolist() == ["shared", "a_only", "b_only"]
    assert result.obs.n_genes_by_counts.eq(2).all()
    assert (result.X != result.layers["raw_counts"]).nnz == 0
    np.testing.assert_array_equal(result.varm["qc_gene_pass_by_batch"], [[True, True], [True, False], [False, True]])
    intersection = cs.pp.qc(data, batch_key="batch", min_genes=1, min_cells=2,
                            filter=True, gene_join="intersection")
    assert intersection.shape == (6, 1)


def test_scdblfinder_defaults_to_x_and_optional_filter_retains_all_calls(monkeypatch):
    obj = AnnData(sparse.csr_matrix(np.ones((40, 5))),
                  obs=pd.DataFrame({"batch": ["a"] * 20 + ["b"] * 20},
                                   index=[f"c{i}" for i in range(40)]))
    obj.layers["counts"] = sparse.csr_matrix(np.full((40, 5), 99))

    def compute(working, **kwargs):
        assert np.all(working.X.data == 1)
        working.obs["scDblFinder_score"] = [1.] + [0.] * 19
        working.obs["scDblFinder_class"] = ["doublet"] + ["singlet"] * 19
        working.uns["scDblFinder_threshold"] = 0.5
        return working

    monkeypatch.setattr(_python_qc, "dependency", lambda *args: types.SimpleNamespace(compute_doublet_score=compute))
    result = cs.pp.scDblFinder(obj, batch_key="batch", remove_doublets=True)
    assert result.n_obs == 38
    assert obj.obs.predicted_doublet.sum() == 2
    assert result.uns["scDblFinder_all_calls"].shape[0] == 40
    assert np.all(obj.layers["counts"].data == 99)


def test_filter_qc_mad_is_batch_local_and_roundtrips_mixed_thresholds(tmp_path):
    values = [10, 11, 12, 13, 200, 1000, 1100, 1200, 1300, 20000]
    data = AnnData(sparse.csr_matrix([[v, 2 * v, 0] for v in values]),
                   obs=pd.DataFrame({"batch": ["a"] * 5 + ["b"] * 5},
                                    index=[f"c{i}" for i in range(10)]),
                   var=pd.DataFrame(index=["G1", "G2", "MT-X"]))
    data.layers["raw_counts"] = data.X.copy()
    result = cs.pp.filter_qc(data, batch_key="batch", method="both", nmads=3,
                             min_counts={"a": 20, "b": 2000}, max_pct_mito=20)
    assert result.n_obs == 8
    assert set(data.obs_names[data.obs.qc_fail_mad]) == {"c4", "c9"}
    assert "MT-X" in result.var_names  # mitochondrial-cell filtering does not delete genes
    result.write_h5ad(tmp_path / "qc.h5ad")
    restored = ad.read_h5ad(tmp_path / "qc.h5ad")
    assert restored.uns["filter_qc_all_cells"].shape[0] == 10
    assert restored.uns["filter_qc_thresholds"].shape[0] == 8


def test_fixed_limits_and_mito_gene_removal_are_separate():
    data = AnnData(sparse.csr_matrix([[100, 100, 0], [1000, 1000, 0], [1, 1, 100]]),
                   obs=pd.DataFrame({"batch": ["a"] * 3}, index=["normal", "high_umi", "high_mt"]),
                   var=pd.DataFrame(index=["G1", "G2", "MT-X"]))
    result = cs.pp.filter_qc(data, batch_key="batch", method="fixed", max_counts=500,
                             max_pct_mito=20, remove_mito_genes=True)
    assert result.obs_names.tolist() == ["normal"]
    assert result.var_names.tolist() == ["G1", "G2"]
    assert data.obs.loc["high_umi", "qc_fail_fixed"]
    assert data.obs.loc["high_mt", "qc_fail_mito"]
