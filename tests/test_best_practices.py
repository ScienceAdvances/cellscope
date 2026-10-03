"""Scientific data-contract checks against real optional Python and R backends."""

import shutil
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

import cellscope as cs


@pytest.fixture
def rna():
    return cs.datasets.toy_rna(n_cells=100, n_genes=60)


def _matrix(matrix):
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def _require_r(packages):
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript is not installed")
    try:
        cs.r.require_packages(packages)
    except ImportError as error:
        pytest.skip(str(error))


def test_public_interfaces_are_importable():
    for name in (
        "soupx",
        "scran",
        "sctransform",
        "pearson_residuals",
        "deviance_features",
        "scdblfinder",
    ):
        assert callable(getattr(cs.pp, name))
    for name in (
        "harmony",
        "celltypist",
        "bbknn",
        "sccoda",
        "milo",
        "mixscape",
        "communication",
        "nichenet",
        "slingshot",
        "tradeseq",
        "glmpca",
        "velocity",
    ):
        assert callable(getattr(cs.tl, name))
    for name in ("best_practices", "r"):
        assert hasattr(cs, name)


def test_method_catalog_resolves_public_interfaces():
    table = cs.best_practices.methods()
    assert table.interface.is_unique
    for interface in table.interface:
        namespace, name = interface.split(".")
        assert callable(getattr(getattr(cs, namespace), name))
    assert set(cs.best_practices.methods("Normalization").backend) >= {"R/scran", "scanpy"}


def test_rna_package_excludes_archived_assay_interfaces_and_dependencies():
    for name in ("atac", "spatial", "multimodal"):
        assert not hasattr(cs, name)
        assert find_spec(f"cellscope.{name}") is None
    for name in ("protein_clr", "protein_dsb", "atac_tfidf"):
        assert not hasattr(cs.pp, name)
    for name in ("music", "lineage_tree", "guide_assignment", "Chrom_size"):
        assert not hasattr(cs.tl, name)
    stages = set(cs.best_practices.methods().stage)
    assert stages.isdisjoint(
        {"Spatial", "ATAC", "Multimodal", "Bulk deconvolution", "Lineage tracing"}
    )
    config = pytest.importorskip("tomllib").loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text()
    )
    extras = config["project"]["optional-dependencies"]
    assert set(extras).isdisjoint({"atac", "spatial", "multimodal", "mofa", "glue", "lineage"})
    requirements = " ".join(
        config["project"]["dependencies"]
        + [dependency for group in extras.values() for dependency in group]
    )
    for dependency in (
        "muon",
        "snapatac2",
        "squidpy",
        "SpatialDE",
        "SpaGCN",
        "tangram",
        "cell2location",
        "mofapy2",
        "scglue",
        "cassiopeia",
    ):
        assert dependency.lower() not in requirements.lower()


def test_pearson_residuals_match_scanpy_and_preserve_counts(rna):
    import scanpy as sc

    before = rna.copy()
    expected = sc.experimental.pp.normalize_pearson_residuals(rna, layer="counts", inplace=False)
    assert cs.pp.pearson_residuals(rna) is rna
    np.testing.assert_allclose(rna.layers["pearson_residuals"], expected["X"])
    np.testing.assert_array_equal(_matrix(rna.X), _matrix(before.X))
    np.testing.assert_array_equal(_matrix(rna.layers["counts"]), _matrix(before.layers["counts"]))
    cs.pp.pearson_hvg(rna, n_top_genes=12)
    assert rna.var.highly_variable.sum() == 12


def test_python_soupx_aligns_raw_genes_and_preserves_input(rna):
    pytest.importorskip("pysoupx")
    rng = np.random.default_rng(3)
    raw = AnnData(
        sparse.vstack([rna.layers["counts"], sparse.csr_matrix(rng.poisson(0.3, (90, 60)))]),
        obs=pd.DataFrame(index=[*rna.obs_names, *[f"empty{i}" for i in range(90)]]),
        var=rna.var.copy(),
    )
    raw = raw[:, ::-1].copy()
    before = _matrix(rna.X).copy()
    cs.pp.soupx(rna, raw, cluster_key="cell_type", contamination=0.1)
    corrected = _matrix(rna.layers["soupx_counts"])
    assert corrected.shape == rna.shape and (corrected >= 0).all()
    np.testing.assert_array_equal(corrected, np.rint(corrected))
    assert corrected.sum() <= before.sum()
    np.testing.assert_array_equal(_matrix(rna.X), before)
    np.testing.assert_allclose(rna.obs.soupx_contamination, 0.1)


def test_soupx_rejects_missing_raw_identifiers(rna):
    with pytest.raises(ValueError, match="identifiers"):
        cs.pp.soupx(rna, rna[:50].copy(), cluster_key="cell_type", contamination=0.1)


def test_python_sctransform_named_output(rna):
    pytest.importorskip("sctransform")
    before = _matrix(rna.layers["counts"]).copy()
    cs.pp.sctransform(rna, vst_flavor="v2", n_genes=40, verbosity=0)
    assert rna.layers["sct_residuals"].shape == rna.shape
    assert rna.var.sct_modeled.any()
    assert np.isfinite(rna.layers["sct_residuals"][:, rna.var.sct_modeled]).all()
    np.testing.assert_array_equal(_matrix(rna.layers["counts"]), before)


def test_harmony_keeps_expression_assays(rna):
    pytest.importorskip("harmonypy")
    cs.pp.normalize(rna)
    cs.tl.pca(rna, n_comps=5)
    before = _matrix(rna.X).copy()
    cs.tl.harmony(rna, batch_key="sample_id", max_iter_harmony=3, verbose=False)
    assert rna.obsm["X_pca_harmony"].shape == (100, 5)
    np.testing.assert_array_equal(_matrix(rna.X), before)


def test_nichenet_python_ranks_plausible_ligand():
    pytest.importorskip("nichenetpy")
    genes = [f"g{i}" for i in range(100)]
    signal = np.r_[np.linspace(1, 0.8, 20), np.linspace(0.2, 0, 80)]
    matrix = pd.DataFrame({"active": signal, "inactive": signal[::-1]}, index=genes)
    result = cs.tl.nichenet(genes[:20], genes, ["active", "inactive"], matrix).set_index("ligand")
    assert result.loc["active", "auroc"] > result.loc["inactive", "auroc"]


def test_r_transport_sparse_metadata_and_native_objects():
    _require_r(["Matrix", "jsonlite"])
    matrix = sparse.csr_matrix([[0, 2, 5], [1, 0, 3]])
    metadata = pd.DataFrame(
        {"condition": ["post", "pre"], "number": [3.5, 2.5], "flag": [True, False]},
        index=["b", "a"],
    )
    result = cs.r._call("function(x, obs) list(matrix=x*2, obs=obs)", x=matrix, obs=metadata)
    assert sparse.issparse(result["matrix"])
    np.testing.assert_array_equal(result["matrix"].toarray(), matrix.toarray() * 2)
    assert result["obs"].index.equals(metadata.index)
    assert list(result["obs"].condition) == ["post", "pre"]
    np.testing.assert_allclose(result["obs"].number, metadata.number)
    native = cs.r._call("function() new.env()")
    assert isinstance(native, cs.r.RObject)
    result = cs.r._call("function(x) is.environment(x)", x=native)
    assert result[0]


def test_r_missing_package_and_error_propagation():
    _require_r(["Matrix", "jsonlite"])
    with pytest.raises(ImportError, match="CELLSCOPE_MISSING_PACKAGES"):
        cs.r.require_packages(["cellscope_nonexistent_package"])
    with pytest.raises(RuntimeError, match="intentional failure"):
        cs.r._call('function() stop("intentional failure")')


def test_sce_roundtrip_keeps_sparse_counts_and_identifiers():
    _require_r(["SingleCellExperiment"])
    x = AnnData(
        sparse.csr_matrix([[0, 2], [1, 4]]),
        obs=pd.DataFrame(index=["b", "a"]),
        var=pd.DataFrame(index=["z", "y"]),
    )
    x.layers["counts"] = x.X.copy()
    restored = cs.r.from_sce(cs.r.to_sce(x))
    assert sparse.issparse(restored.X)
    assert restored.obs_names.equals(x.obs_names) and restored.var_names.equals(x.var_names)
    np.testing.assert_array_equal(restored.layers["counts"].toarray(), x.X.toarray())


def test_actual_scran_scry_and_persistence(rna, tmp_path):
    _require_r(["scran", "scry", "SingleCellExperiment"])
    before = _matrix(rna.layers["counts"]).copy()
    cs.pp.scran(rna, cluster_key="cell_type")
    factors = rna.obs.size_factors.to_numpy()
    np.testing.assert_allclose(
        _matrix(rna.layers["scran_normalization"]), np.log1p(before / factors[:, None])
    )
    cs.pp.deviance_features(rna, n_top_genes=10)
    assert rna.var.highly_deviant.sum() == 10
    assert rna.uns["cellscope"]["scran_normalization"]["r_packages"]["scran"]
    rna.write_h5ad(tmp_path / "results.h5ad")
    np.testing.assert_array_equal(_matrix(rna.layers["counts"]), before)


def test_sccoda_rejects_sample_covariate_conflicts(rna):
    pytest.importorskip("pertpy")
    rna.obs.iloc[0, rna.obs.columns.get_loc("condition")] = "conflicting"
    with pytest.raises(ValueError, match="constant"):
        cs.tl.sccoda(
            rna,
            sample_key="sample_id",
            cell_type_key="cell_type",
            covariate_keys=["condition"],
            formula="condition",
        )


def test_python_slingshot_handles_nonconsecutive_numeric_cluster_ids():
    pytest.importorskip("pyslingshot")
    rng = np.random.default_rng(1)
    t = np.linspace(0, 10, 120)
    adata = AnnData(np.ones((120, 2)), obs=pd.DataFrame({"cluster": np.repeat([7, 12, 30], 40)}))
    adata.obsm["X_pca"] = np.column_stack([t, rng.normal(0, 0.4, 120)])
    cs.tl.slingshot(adata, cluster_key="cluster", start_cluster=7, fit_kwargs={"num_epochs": 2})
    assert adata.obsm["slingshot_pseudotime"].shape == (120, 1)
    assert adata.obsm["slingshot_weights"].shape == (120, 1)
    assert np.isfinite(adata.obsm["slingshot_pseudotime"]).all()
    assert (
        adata.obs.slingshot_pseudotime.iloc[-20:].mean()
        > adata.obs.slingshot_pseudotime.iloc[:20].mean()
    )


def test_actual_glmpca_python_and_r(rna):
    pytest.importorskip("glmpca")
    cs.tl.glmpca(rna, n_comps=2, ctl={"maxIter": 20, "eps": 0.01})
    assert rna.obsm["X_glmpca"].shape == (100, 2)
    assert np.isfinite(rna.obsm["X_glmpca"]).all()
    _require_r(["glmpca"])
    cs.tl.glmpca(
        rna,
        n_comps=2,
        backend="r",
        key_added="X_glmpca_r",
        ctl={"maxIter": 20, "minIter": 5, "tol": 0.01},
    )
    assert rna.obsm["X_glmpca_r"].shape == (100, 2)
    assert np.isfinite(rna.obsm["X_glmpca_r"]).all()


def test_r_slingshot_and_tradeseq_alignment():
    _require_r(["slingshot", "tradeSeq", "DelayedMatrixStats"])
    rng = np.random.default_rng(2)
    t = np.linspace(0, 10, 100)
    x = AnnData(
        sparse.csr_matrix(rng.poisson(np.exp(np.outer(t / 10, np.linspace(0.1, 1, 20)) + 1))),
        obs=pd.DataFrame(
            {"cluster": np.repeat(["root", "middle", "end", "terminal"], 25)},
            index=[f"cell{i}" for i in range(100)],
        ),
        var=pd.DataFrame(index=[f"gene{i}" for i in range(20)]),
    )
    x.layers["counts"] = x.X.copy()
    x.obsm["X_pca"] = np.column_stack([t, rng.normal(0, 0.3, 100)])
    cs.tl.slingshot(x, cluster_key="cluster", start_cluster="root", backend="r")
    assert x.obsm["slingshot_pseudotime"].shape[0] == 100
    table, fit = cs.tl.tradeseq(x, n_knots=3)
    assert set(table.gene) == set(x.var_names)
    assert isinstance(fit, cs.r.RObject)
    assert table.pvalue.dropna().between(0, 1).all()


def test_r_sctransform_and_soupx_keep_raw_assays(rna):
    _require_r(["sctransform", "SoupX"])
    before = _matrix(rna.layers["counts"]).copy()
    cs.pp.sctransform(rna, backend="r", n_genes=40, verbosity=0)
    assert np.isfinite(rna.layers["sct_residuals"][:, rna.var.sct_modeled]).all()
    rng = np.random.default_rng(8)
    raw = AnnData(
        sparse.vstack([rna.layers["counts"], sparse.csr_matrix(rng.poisson(0.3, (80, 60)))]),
        obs=pd.DataFrame(index=[*rna.obs_names, *[f"empty{i}" for i in range(80)]]),
        var=rna.var.copy(),
    )
    cs.pp.soupx(rna, raw, cluster_key="cell_type", contamination=0.1, backend="r")
    assert _matrix(rna.layers["soupx_counts"]).sum() <= before.sum()
    np.testing.assert_array_equal(_matrix(rna.layers["counts"]), before)


def test_liana_explicit_resource_with_named_groups():
    pytest.importorskip("liana")
    adata = AnnData(
        np.tile(np.array([1.0, 2.0, 3.0, 4.0]), (40, 1)),
        obs=pd.DataFrame({"cell_type": pd.Categorical(["sender"] * 20 + ["receiver"] * 20)}),
        var=pd.DataFrame(index=["L1", "R1", "L2", "R2"]),
    )
    resource = pd.DataFrame({"ligand": ["L1", "L2"], "receptor": ["R1", "R2"]})
    table = cs.tl.communication(
        adata, groupby="cell_type", method="geometric_mean", resource=resource
    )
    assert set(table.source) == {"sender", "receiver"}
    assert set(table.target) == {"sender", "receiver"}
    assert set(table.ligand_complex) == {"L1", "L2"}


def test_milo_uses_python_solver_and_sample_level_counts(rna):
    pytest.importorskip("pertpy")
    cs.pp.normalize(rna)
    cs.tl.pca(rna, n_comps=5)
    result, model = cs.tl.milo(
        rna,
        sample_key="sample_id",
        design="~condition",
        prop=0.15,
        neighbors_kwargs={"n_neighbors": 15, "use_rep": "X_pca"},
    )
    assert result.mod["milo"].n_obs == rna.obs.sample_id.nunique()
    assert "SpatialFDR" in result.mod["milo"].var
    assert model.__class__.__name__ == "Milo"


def test_sccoda_native_sample_level_posterior(rna):
    pytest.importorskip("numpyro")
    pytest.importorskip("pertpy")
    result, model = cs.tl.sccoda(
        rna,
        sample_key="sample_id",
        cell_type_key="cell_type",
        formula="condition",
        covariate_keys=["condition"],
        reference_cell_type="T",
        sampling_kwargs={"num_samples": 10, "num_warmup": 10},
    )
    assert result.mod["coda"].n_obs == rna.obs.sample_id.nunique()
    assert "intercept_df" in result.mod["coda"].varm
    assert model.__class__.__name__ == "Sccoda"


def test_native_regulon_auc_has_named_cell_alignment(rna):
    pytest.importorskip("pyscenic")
    genesig = pytest.importorskip("ctxcore.genesig")
    before = _matrix(rna.layers["counts"]).copy()
    cs.pp.normalize(rna)
    signature = genesig.GeneSignature("TF1", {"G1": 1.0, "G2": 1.0, "G3": 1.0, "G4": 1.0})
    result = cs.tl.regulon_activity(rna, [signature], num_workers=1, auc_threshold=0.5)
    assert result.index.equals(rna.obs_names)
    assert list(result.columns) == ["TF1"]
    assert result.TF1.between(0, 1).all()
    np.testing.assert_array_equal(_matrix(rna.layers["counts"]), before)
