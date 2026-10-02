"""Behavior tests with actual scverse and statistical backends."""

import numpy as np
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix

import cellscope as cs


@pytest.fixture
def rna():
    return cs.datasets.toy_rna(n_cells=80, n_genes=40)


def test_namespaces_and_legacy_entry_points():
    for name in ("io", "pp", "tl", "pl", "get", "datasets"):
        assert hasattr(cs, name)
    assert callable(cs.tl.deseq)
    assert callable(cs.pp.normalise)
    assert callable(cs.pl.dimplot)


def test_identifiers_idempotent_and_conflict(rna):
    cs.io.set_cell_ids(rna, library_id="library", sample_id="sample")
    ids = rna.obs_names.copy()
    cs.io.set_cell_ids(rna, library_id="library", sample_id="sample")
    assert ids.equals(rna.obs_names)
    assert rna.obs["barcode"].is_unique
    with pytest.raises(ValueError, match="conflicts"):
        cs.io.set_cell_ids(rna, library_id="other")


def test_concat_rejects_colliding_libraries(rna):
    with pytest.raises(ValueError, match="collide"):
        cs.io.concat({"a": rna, "b": rna.copy()})


def test_mudata_outer_inner_metadata_and_future_modality(rna):
    other = AnnData(csr_matrix(np.ones((10, 3))), obs=rna.obs.iloc[:10].copy())
    outer = cs.io.create_mudata({"gex": rna, "atac": other})
    assert outer.n_obs == 80
    assert outer.obs["has_atac"].sum() == 10
    assert "gex:cell_type" in outer.obs
    assert cs.io.create_mudata({"gex": rna, "atac": other}, join="inner").n_obs == 10
    other.obs["donor_id"] = "wrong"
    with pytest.raises(ValueError, match="Conflicting donor_id"):
        cs.io.create_mudata({"gex": rna, "atac": other})


def test_normalization_preserves_counts_and_is_repeatable(rna):
    expected = rna.layers["counts"].toarray().copy()
    cs.pp.normalize(rna)
    normalized = rna.X.toarray().copy()
    cs.pp.normalize(rna)
    np.testing.assert_array_equal(rna.layers["counts"].toarray(), expected)
    np.testing.assert_allclose(rna.X.toarray(), normalized)
    assert cs.get.result(rna, "normalize")["backend"] == "scanpy"


def test_qc_mask_and_cross_modality_filter(rna):
    other = AnnData(csr_matrix(np.ones((10, 2))), obs=rna.obs.iloc[:10].copy())
    data = cs.io.create_mudata({"gex": rna, "atac": other})
    cs.pp.qc(data, min_genes=1000)
    assert not data.mod["gex"].obs["qc_pass"].any()
    filtered = cs.pp.filter_cells(data)
    assert filtered.n_obs == 0
    assert filtered.mod["atac"].n_obs == 0
    assert data.n_obs == 80


def test_counts_validation_and_views(rna):
    rna.layers["counts"] = rna.layers["counts"].astype(float) / 3
    with pytest.raises(ValueError, match="integer"):
        cs.pp.normalize(rna)
    with pytest.raises(ValueError, match="views"):
        cs.pp.qc(rna[:10])


def test_composition_includes_zeros_and_correct_denominator(rna):
    table = cs.tl.cell_composition(rna, condition_col="condition", donor_col="donor_id")
    assert len(table) == 8
    assert table.groupby("sample_id")["fraction"].sum().eq(1).all()
    rna.obs.loc[rna.obs["sample_id"].eq("s0"), "cell_type"] = "T"
    table = cs.tl.cell_composition(rna)
    assert table.loc[table["sample_id"].eq("s0") & table["group"].eq("B"), "fraction"].item() == 0


def test_paired_composition_statistics(rna):
    table = cs.tl.cell_composition(rna, condition_col="condition", donor_col="donor_id")
    tested = cs.tl.composition_test(
        table, condition_col="condition", comparison="post", reference="pre", donor_col="donor_id"
    )
    assert tested["fdr"].between(0, 1).all()
    assert set(tested["method"]) == {"paired_wilcoxon"}


def test_composition_rejects_inconsistent_sample_covariates(rna):
    rna.obs.iloc[0, rna.obs.columns.get_loc("condition")] = "other"
    with pytest.raises(ValueError, match="constant"):
        cs.tl.cell_composition(rna, condition_col="condition")


def test_pseudobulk_matches_raw_sums(rna):
    pytest.importorskip("decoupler")
    cs.pp.normalize(rna)
    data = cs.tl.pseudobulk(rna, min_cells=1, metadata_cols=["condition", "donor_id"])
    for name, row in data.obs.iterrows():
        mask = rna.obs["sample_id"].eq(row["sample_id"]) & rna.obs["cell_type"].eq(row["cell_type"])
        np.testing.assert_array_equal(
            data[name].X.ravel(),
            np.asarray(rna.layers["counts"][mask.to_numpy()].sum(axis=0)).ravel(),
        )
    assert data.obs["condition"].notna().all()


def test_pseudobulk_rejects_covariate_conflicts(rna):
    pytest.importorskip("decoupler")
    rna.obs.iloc[0, rna.obs.columns.get_loc("donor_id")] = "other"
    with pytest.raises(ValueError, match="constant"):
        cs.tl.pseudobulk(rna, metadata_cols=["donor_id"])


def test_actual_paired_pydeseq2(rna):
    pytest.importorskip("pydeseq2")
    pytest.importorskip("decoupler")
    data = cs.tl.pseudobulk(rna, min_cells=1, metadata_cols=["condition", "donor_id"])
    table, models = cs.tl.differential_expression(
        data,
        design="~ donor_id + condition",
        contrast=["condition", "post", "pre"],
        groups_col="cell_type",
    )
    assert {"gene", "log2FoldChange", "pvalue", "padj", "group"}.issubset(table.columns)
    assert len(models) == 2
    assert table["pvalue"].dropna().between(0, 1).all()


def test_differential_rejects_cell_level_profiles(rna):
    pytest.importorskip("pydeseq2")
    with pytest.raises(ValueError, match="one profile"):
        cs.tl.differential_expression(
            rna, design="~ condition", contrast=["condition", "post", "pre"]
        )


def test_actual_aucell(rna):
    pytest.importorskip("decoupler")
    cs.pp.normalize(rna)
    values = cs.tl.aucell(rna, gene_list=["G1", "G2", "G3"], n_up=10)
    assert len(values) == rna.n_obs
    assert np.isfinite(values).all()


def test_scanpy_pipeline_markers_and_plots(rna):
    pytest.importorskip("igraph")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cs.tl.workflow(rna, n_top_genes=30, n_pcs=10)
    assert rna.obsm["X_pca"].shape == (80, 10)
    assert rna.obsm["X_umap"].shape == (80, 2)
    table = cs.tl.markers(rna, groupby="cell_type")
    assert {"names", "group", "pvals_adj"}.issubset(table)
    assert cs.pl.embedding(rna, color="cell_type") is not None
    assert cs.pl.cell_composition(cs.tl.cell_composition(rna)) is not None
    plt.close("all")


def test_persistence_and_get_copy(rna, tmp_path):
    cs.tl.cell_composition(rna)
    cs.io.write(rna, tmp_path / "rna.h5ad")
    restored = cs.io.read_h5ad(tmp_path / "rna.h5ad")
    assert restored.obs_names.equals(rna.obs_names)
    copy = cs.get.obs_df(restored)
    copy["cell_type"] = "changed"
    assert not restored.obs["cell_type"].eq("changed").all()
    with pytest.raises(ValueError, match="h5ad"):
        cs.io.write(rna, tmp_path / "wrong.h5mu")


def test_annotation_retains_unknown_cells(rna):
    rna.obs["leiden"] = np.where(np.arange(rna.n_obs) % 2, "1", "0")
    cs.tl.annotate(rna, {"0": "T"})
    assert rna.n_obs == 80
    assert rna.obs["cell_type"].isna().sum() == 40


def test_legacy_qc_ratio_and_inplace_subset(rna, tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    copied = cs.pp.fastqc(
        rna,
        qc_vars=[],
        sample="sample_id",
        outdir=tmp_path,
        min_genes=1,
        min_cells=1,
        inplace=False,
        formats=("png",),
    )
    assert "total_counts" in copied.obs
    assert "total_counts" not in rna.obs
    fractions = cs.pl.cell_ratio(rna, "sample_id", "cell_type")
    assert np.allclose(fractions.sum(axis=1), 1)
    expected = int(rna.obs["cell_type"].eq("T").sum())
    cs.tl.subset(rna, {"cell_type": ["T"]}, inplace=True)
    assert rna.n_obs == expected
    plt.close("all")
