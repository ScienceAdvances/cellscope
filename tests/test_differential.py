"""Actual backend tests for cell markers and paired sample-level inference."""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

import cellscope as cs
from cellscope.tools._differential import _design


@pytest.fixture
def expression():
    rng = np.random.default_rng(10)
    raw = rng.poisson(2, (90, 20))
    raw[:30, :3] += 15
    raw[30:60, 3:6] += 15
    raw[:, -1] = 0
    obj = AnnData(
        sparse.csr_matrix(raw),
        obs=pd.DataFrame(
            {"identity": np.repeat(["a", "b", "c"], 30)}, index=[f"cell{i}" for i in range(90)]
        ),
        var=pd.DataFrame(index=[f"gene{i}" for i in range(20)]),
    )
    obj.layers["counts"] = obj.X.copy()
    cs.pp.normalize(obj)
    return obj


@pytest.fixture
def profiles():
    rng = np.random.default_rng(13)
    matrix = rng.negative_binomial(20, 0.45, size=(12, 60))
    matrix[1::2, :6] *= 5
    obs = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(12)],
            "donor_id": np.repeat([f"d{i}" for i in range(6)], 2),
            "condition": ["control", "treated"] * 6,
            "cell_type": "T",
        },
        index=[f"profile{i}" for i in range(12)],
    )
    return AnnData(
        sparse.csr_matrix(matrix), obs=obs, var=pd.DataFrame(index=[f"gene{i}" for i in range(60)])
    )


@pytest.mark.parametrize("method", ["wilcoxon", "t-test", "t-test_overestim_var", "logreg"])
def test_actual_marker_methods(expression, method):
    matrix = expression.X.copy()
    result = cs.tl.find_markers(
        expression, groupby="identity", ident_1="a", ident_2="b", method=method
    )
    assert {"gene", "group", "log2FoldChange", "pvalue", "padj", "pct_nz_group"}.issubset(result)
    assert set(result["method"]) == {method}
    assert set(result["gene"]).isdisjoint({"gene19"})
    indexed = result.set_index("gene")
    assert indexed.loc["gene0", "log2FoldChange"] > 1
    assert indexed.loc["gene3", "log2FoldChange"] < -1
    np.testing.assert_array_equal(matrix.toarray(), expression.X.toarray())
    if method == "logreg":
        assert result["pvalue"].isna().all()
        assert indexed.loc["gene0", "scores"] > 0
        assert indexed.loc["gene3", "scores"] < 0
    else:
        assert result["pvalue"].between(0, 1).all()
        assert result["padj"].between(0, 1).all()


def test_find_all_filters_and_legacy_layout(expression):
    table = cs.tl.find_all_markers(expression, groupby="identity", only_pos=True, logfc_threshold=1)
    assert {"a", "b"}.issubset(set(table["group"]))
    assert set(table["group"]).issubset({"a", "b", "c"})
    assert table["log2FoldChange"].ge(1).all()
    extracted = cs.get.markers_df(expression, key="find_all_markers", group="a")
    assert set(extracted["group"]) == {"a"}
    extracted["gene"] = "changed"
    assert not cs.get.de_df(expression, key="find_all_markers")["gene"].eq("changed").any()
    legacy = cs.tl.find_all_markers(expression, groupby="identity", legacy=True)
    assert legacy.index.name == "Feature"
    assert {"Identy", "LogFC", "Padj"}.issubset(legacy)


def test_pooled_identities_raw_layer_and_empty_results(expression):
    expression.raw = expression.copy()
    expression.layers["lognorm"] = expression.X.copy()
    kwargs = {
        "groupby": "identity",
        "ident_1": ["a", "c"],
        "ident_2": "b",
        "features": ["gene0", "gene3"],
    }
    raw = cs.tl.find_markers(expression, use_raw=True, **kwargs)
    layer = cs.tl.find_markers(expression, layer="lognorm", **kwargs)
    pd.testing.assert_frame_equal(raw, layer)
    assert set(raw["gene"]) == {"gene0", "gene3"}
    empty = cs.tl.find_markers(expression, groupby="identity", ident_1="a", min_diff_pct=1)
    assert empty.empty and "padj" in empty


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"ident_1": "absent"}, "absent"),
        ({"ident_1": "a", "ident_2": "a"}, "overlap"),
        ({"ident_1": "a", "method": "pydeseq2"}, "sample|pseudobulk|count-model"),
        ({"ident_1": "a", "min_pct": 2}, "fractions"),
        ({"ident_1": "a", "layer": "counts"}, "Normalize"),
    ],
)
def test_marker_validation(expression, kwargs, message):
    with pytest.raises(ValueError, match=message):
        cs.tl.find_markers(expression, groupby="identity", **kwargs)


def test_marker_effect_is_mean_linear_not_mean_log(expression):
    result = cs.tl.find_markers(expression, groupby="identity", ident_1="a", ident_2="b")
    linear = np.expm1(expression.X.toarray().astype(float))
    expected = np.log2((linear[:30].mean(axis=0) + 1e-9) / (linear[30:60].mean(axis=0) + 1e-9))
    ids = expression.var_names.get_indexer(result["gene"])
    np.testing.assert_allclose(result["log2FoldChange"], expected[ids])


@pytest.mark.parametrize(
    "method, module",
    [
        ("pylimma", "pylimma"),
        ("edgepython_ql", "edgepython"),
        ("edgepython_lrt", "edgepython"),
        ("pydeseq2", "pydeseq2"),
    ],
)
def test_actual_paired_backends_and_gene_alignment(profiles, method, module):
    pytest.importorskip(module)
    if method == "pylimma":
        pytest.importorskip("edgepython")
    before = profiles.X.toarray().copy()
    obs = profiles.obs.copy()
    result, models = cs.tl.differential_expression(
        profiles,
        design="~ donor_id + condition",
        contrast=("condition", "treated", "control"),
        method=method,
        groups_col="cell_type",
    )
    assert len(result) == 60 and len(models) == 1
    assert set(result["gene"]) == set(profiles.var_names)
    assert set(result["unit"]) == {"sample"}
    assert result["pvalue"].dropna().between(0, 1).all()
    assert result["padj"].dropna().between(0, 1).all()
    assert (
        result.set_index("gene").loc[[f"gene{i}" for i in range(6)], "log2FoldChange"].gt(1).all()
    )
    np.testing.assert_array_equal(before, profiles.X.toarray())
    pd.testing.assert_frame_equal(obs, profiles.obs)
    entry = cs.get.result(profiles, "differential_expression")
    assert entry["params"]["method"] == method
    assert entry["designs"]["T"].shape == (12, 7)
    assert entry["contrasts"]["T"]["weight"].iloc[-1] == 1
    if method == "pylimma":
        import pylimma as lm

        native = lm.top_table(models["T"][-1], coef=0, number=60, sort_by="none")
        np.testing.assert_allclose(result["log2FoldChange"], native["log_fc"])
    elif method.startswith("edgepython"):
        native = models["T"][-1]["table"].copy()
        native["gene"] = models["T"][-1]["genes"]["gene"].to_numpy()
        np.testing.assert_allclose(
            result.set_index("gene")["log2FoldChange"].sort_index(),
            native.set_index("gene")["logFC"].sort_index(),
        )
    else:
        np.testing.assert_allclose(
            result["log2FoldChange"], models["T"][1].results_df["log2FoldChange"]
        )


@pytest.mark.parametrize("method", ["pylimma", "edgepython_ql", "edgepython_lrt", "pydeseq2"])
def test_contrast_reversal(profiles, method):
    pytest.importorskip(
        "pylimma" if method == "pylimma" else ("pydeseq2" if method == "pydeseq2" else "edgepython")
    )
    result, _ = cs.tl.differential_expression(
        profiles,
        design="~ donor_id + condition",
        contrast=("condition", "control", "treated"),
        method=method,
    )
    assert (
        result.set_index("gene").loc[[f"gene{i}" for i in range(6)], "log2FoldChange"].lt(-1).all()
    )


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("duplicate", "one profile"),
        ("zero", "zero-depth"),
        ("fractional", "integer"),
        ("confounded", "rank deficient"),
        ("missing", "Invalid design"),
        ("insufficient", "needs 2"),
    ],
)
def test_sample_level_guards(profiles, mutation, message):
    if mutation == "duplicate":
        profiles.obs["sample_id"] = "same"
        # Keep condition constant so the duplicate-profile check is reached.
        profiles.obs["sample_id"] = profiles.obs["condition"]
    elif mutation == "zero":
        dense = profiles.X.toarray()
        dense[0] = 0
        profiles.X = sparse.csr_matrix(dense)
    elif mutation == "fractional":
        profiles.X = profiles.X.astype(float) / 3
    elif mutation == "confounded":
        profiles.obs["donor_id"] = profiles.obs["condition"]
    elif mutation == "missing":
        profiles.obs.iloc[0, profiles.obs.columns.get_loc("donor_id")] = None
    elif mutation == "insufficient":
        profiles = profiles[:3].copy()
    with pytest.raises(ValueError, match=message):
        cs.tl.differential_expression(
            profiles, design="~ donor_id + condition", contrast=("condition", "treated", "control")
        )


def test_design_with_unusual_labels_and_interactions(profiles):
    profiles.obs["condition"] = profiles.obs["condition"].map(
        {"control": "before treatment", "treated": "after/treatment"}
    )
    profiles.obs["age"] = np.repeat(np.arange(6) + 20, 2)
    matrix, contrast = _design(
        profiles.obs, "~ age * condition", ("condition", "after/treatment", "before treatment")
    )
    assert matrix.shape == (12, 4)
    assert contrast[-1] == profiles.obs["age"].mean()
    with pytest.raises(ValueError, match="no effect"):
        _design(profiles.obs, "~ age", ("condition", "after/treatment", "before treatment"))


def test_one_step_pseudobulk_de_and_multimodal(expression):
    pytest.importorskip("edgepython")
    pytest.importorskip("decoupler")
    data = cs.datasets.toy_rna(n_cells=160, n_genes=60)
    joint = cs.io.create_mudata({"gex": data})
    table, pdata, models = cs.tl.pseudobulk_de(
        joint,
        design="~ donor_id + condition",
        contrast=("condition", "post", "pre"),
        method="edgepython_ql",
        metadata_cols=["donor_id"],
        min_cells=1,
    )
    assert set(table["group"]) == {"T", "B"}
    assert set(pdata.obs["condition"]) == {"pre", "post"}
    assert len(models) == 2


def test_results_h5_roundtrip_and_method_inventory(expression, profiles, tmp_path):
    pytest.importorskip("edgepython")
    cs.tl.find_all_markers(expression, groupby="identity")
    cs.tl.differential_expression(
        profiles,
        method="edgepython_lrt",
        design="~ condition",
        contrast=("condition", "treated", "control"),
    )
    for name, obj, key in [
        ("markers", expression, "find_all_markers"),
        ("bulk", profiles, "differential_expression"),
    ]:
        path = tmp_path / f"{name}.h5ad"
        cs.io.write(obj, path)
        restored = cs.io.read_h5ad(path)
        pd.testing.assert_frame_equal(
            cs.get.de_df(obj, key=key).reset_index(drop=True),
            cs.get.de_df(restored, key=key).reset_index(drop=True),
            check_dtype=False,
            check_categorical=False,
        )
    inventory = cs.tl.de_methods()
    assert inventory["method"].is_unique and set(inventory["unit"]) == {"cell", "sample"}


def test_pylimma_import_collision(profiles, monkeypatch):
    import cellscope.tools._differential as de

    original = de.version
    monkeypatch.setattr(
        de, "version", lambda name: "0.1.0" if name == "python-limma" else original(name)
    )
    with pytest.raises(ImportError, match="share an import name"):
        cs.tl.differential_expression(
            profiles,
            method="pylimma",
            design="~ condition",
            contrast=("condition", "treated", "control"),
        )


def test_mature_expression_filter_keeps_gene_annotations(profiles):
    pytest.importorskip("edgepython")
    dense = profiles.X.toarray()
    dense[:, -1] = 1
    profiles.X = sparse.csr_matrix(dense)
    result, _ = cs.tl.differential_expression(
        profiles,
        method="edgepython_ql",
        design="~ condition",
        contrast=("condition", "treated", "control"),
        filter_method="filter_by_expr",
    )
    assert "gene59" not in set(result["gene"])
    assert {f"gene{i}" for i in range(6)}.issubset(set(result["gene"]))
    assert (
        cs.get.result(profiles, "differential_expression")["params"]["filter_method"]
        == "filter_by_expr"
    )


def test_unused_categories_and_nullable_numeric_covariates(profiles):
    profiles.obs["donor_id"] = pd.Categorical(
        profiles.obs["donor_id"],
        categories=[*[f"d{i}" for i in range(6)], "absent"],
    )
    matrix, effect = _design(
        profiles.obs, "~ donor_id + condition", ("condition", "treated", "control")
    )
    assert matrix.shape == (12, 7) and effect[-1] == 1
    profiles.obs["age"] = pd.array(np.repeat(np.arange(6) + 20, 2), dtype="Int64")
    matrix, _ = _design(profiles.obs, "~ age + condition", ("condition", "treated", "control"))
    assert matrix.shape == (12, 3)


def test_native_pts_option_and_scalar_numeric_identities(expression):
    expression.obs["numeric"] = (
        expression.obs["identity"].map({"a": 1.0, "b": 2.0, "c": 3.0}).astype("float32")
    )
    result = cs.tl.find_all_markers(expression, groupby="numeric", backend_kwargs={"pts": True})
    assert set(result["group"]) == {"1.0", "2.0", "3.0"}
    assert result["pct_nz_reference"].between(0, 1).all()


def test_pseudobulk_without_cell_group_and_custom_sample_key():
    pytest.importorskip("edgepython")
    pytest.importorskip("decoupler")
    data = cs.datasets.toy_rna(n_cells=160, n_genes=60)
    data.obs = data.obs.rename(columns={"sample_id": "biological_sample"})
    result, profiles, models = cs.tl.pseudobulk_de(
        data,
        sample_col="biological_sample",
        groups_col=None,
        min_cells=1,
        method="edgepython_lrt",
        metadata_cols=["donor_id", "biological_sample"],
        design="~ donor_id + condition",
        contrast=("condition", "post", "pre"),
    )
    assert len(profiles) == 4 and set(models) == {"all"}
    assert set(result["group"]) == {"all"}
    np.testing.assert_array_equal(
        np.asarray(data.layers["counts"].sum(axis=0)).ravel(),
        np.asarray(profiles.X.sum(axis=0)).ravel(),
    )
