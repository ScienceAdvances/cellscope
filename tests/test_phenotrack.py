"""Protect PhenoTrack normalization, donor separation, flow geometry and persistence."""

import json

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cellscope as cs
from cellscope.tools._phenotrack import resolve_phenotrack


@pytest.fixture
def data():
    obs = pd.DataFrame({
        "donor_id": ["P1"] * 7,
        "condition": ["Pre"] * 4 + ["Post"] * 3,
        "sample_id": ["P1_Pre"] * 4 + ["P1_Post"] * 3,
        "clone_id": ["A", "A", "Lost", "Other", "A", "A", "New"],
        "cell_type": ["Eff", "Mem", "Eff", "Eff", "Mem", "Mem", "Eff"],
    }, index=[f"c{i}" for i in range(7)])
    return ad.AnnData(sparse.csr_matrix(np.arange(21).reshape(7, 3)), obs=obs)


def run(data, **kwargs):
    return cs.tl.phenotrack(data, time_key="condition", time_order=["Pre", "Post"], **kwargs)


def test_counts_flows_flux_and_subset_denominator(data):
    before = data.X.copy()
    result = run(data, clones=["A", "Lost", "New"])
    assert result.denominators.denominator.tolist() == [4, 3]
    nodes = result.nodes.set_index(["time", "phenotype"]).value
    assert nodes["Pre", "Eff"] == 0.5
    assert nodes["Pre", "Mem"] == 0.25
    assert nodes["Post", "Eff"] == 1 / 3
    assert nodes["Post", "Mem"] == 2 / 3
    assert len(result.flows) == 2
    np.testing.assert_allclose(result.flows.outflow, [0.25, 0.25])
    np.testing.assert_allclose(result.flows.inflow, [1 / 3, 1 / 3])
    assert result.flux.phenotypic_flux.tolist() == [1]
    assert dict(zip(result.events.clone_id, result.events.event, strict=True)) == {
        "A": "shared", "Lost": "disappeared", "New": "appeared"}
    assert (data.X != before).nnz == 0
    assert "leiden" not in data.obs
    assert isinstance(result, cs.tl.PhenoTrackResult)
    assert result.parameters["denominator_before_clone_selection"]


def test_probabilities_and_raw_count_flow_mode(data):
    probabilities = np.array([[.75, .25], [.25, .75], [1, 0], [1, 0],
                              [.2, .8], [.2, .8], [1, 0]])
    data.obsm["phenotype_probs"] = pd.DataFrame(probabilities, columns=["Eff", "Mem"], index=data.obs_names)
    result = run(data, probabilities_key="phenotype_probs", clones=["A"], normalize=False)
    expected = np.outer([.5, .5], [.2, .8]) * 2
    flow = result.flows.pivot(index="from_phenotype", columns="to_phenotype", values="outflow")
    np.testing.assert_allclose(flow.loc[["Eff", "Mem"], ["Eff", "Mem"]], expected)
    assert result.flux.phenotypic_flux.iloc[0] == pytest.approx(.6)
    assert not result.parameters["normalize"]


def test_invalid_probabilities_are_not_silently_renormalized(data):
    data.obsm["probs"] = np.full((7, 2), .6)
    with pytest.raises(ValueError, match="sum to 1"):
        run(data, probabilities_key="probs", phenotype_order=["Eff", "Mem"])
    data.obsm["probs"] = np.ones((7, 2)) / 2
    with pytest.raises(ValueError, match="phenotype_order"):
        run(data, probabilities_key="probs")


def test_missing_labels_count_in_default_denominator(data):
    extra = data[0:1].copy()
    extra.obs_names = ["missing"]
    extra.obs["clone_id"] = None
    extra.obs["cell_type"] = None
    merged = ad.concat([data, extra])
    result = run(merged, clones=["A"])
    assert result.denominators.denominator.tolist() == [5, 3]
    alternative = run(merged, clones=["A"], denominator="annotated_cells")
    assert alternative.denominators.denominator.tolist() == [4, 3]
    assert result.nodes.query("time == 'Pre'").value.sum() == pytest.approx(2 / 5)


def test_equal_donor_mean_does_not_pool_cells_or_clone_ids(data):
    other = data[data.obs.clone_id.eq("A")].copy()
    other.obs_names = [f"other_{x}" for x in other.obs_names]
    other.obs["donor_id"] = "P2"
    other.obs["sample_id"] = other.obs["sample_id"].str.replace("P1", "P2")
    other.obs["cell_type"] = "Eff"
    joint = ad.concat([data, other])
    result = run(joint, clones={"P1": ["A"], "P2": ["A"]})
    assert set(result.flows.donor_id) == {"P1", "P2"}
    mean = cs.tl.phenotrack_aggregate(result)
    nodes = mean.nodes.set_index(["time", "phenotype"]).value
    assert nodes["Pre", "Eff"] == pytest.approx((.25 + 1) / 2)
    assert nodes["Post", "Mem"] == pytest.approx((2 / 3 + 0) / 2)
    assert mean.parameters["aggregation"] == "equal_donor_mean"
    with pytest.raises(ValueError, match="already aggregated"):
        cs.tl.phenotrack_aggregate(mean)


def test_missing_donor_time_not_treated_as_disappearance(data):
    other = data[data.obs.condition.eq("Post")].copy()
    other.obs_names = [f"other_{x}" for x in other.obs_names]
    other.obs["donor_id"] = "P2"
    other.obs["sample_id"] = "P2_Post"
    joint = ad.concat([data, other])
    result = run(joint)
    assert not result.nodes.query("donor_id == 'P2' and time == 'Pre'").sampled.any()
    assert "P2" not in set(result.events.donor_id)
    with pytest.raises(ValueError, match="every donor/time"):
        cs.tl.phenotrack_aggregate(result)
    mean = cs.tl.phenotrack_aggregate(result, missing="available")
    assert mean.parameters["n_donors_by_time"] == {"Pre": 1, "Post": 2}
    assert mean.flows.outflow.sum() == pytest.approx(.5)
    assert mean.flows.inflow.sum() == pytest.approx(1 / 3)


def test_strict_flux_threshold_and_empty_shared_clone_plot(data):
    result = run(data, min_cell_count=2)
    assert result.flux.empty
    unshared = run(data, clones=["Lost", "New"])
    assert unshared.flows.empty
    ax = cs.pl.phenotrack(unshared)
    assert len(ax.patches) == 2
    plt.close(ax.figure)


def test_native_storage_roundtrip_and_export(data, tmp_path):
    result = run(data)
    cs.io.write(data, tmp_path / "rna.h5ad")
    restored = ad.read_h5ad(tmp_path / "rna.h5ad")
    loaded = resolve_phenotrack(restored)
    pd.testing.assert_frame_equal(loaded.flows, result.flows)
    loaded.write(tmp_path / "tables")
    assert json.loads((tmp_path / "tables/parameters.json").read_text())["time_order"] == ["Pre", "Post"]
    with pytest.raises(FileExistsError):
        loaded.write(tmp_path / "tables")
    loaded.write(tmp_path / "tables", overwrite=True)
    ax = cs.pl.phenotrack(restored)
    plt.close(ax.figure)


def test_plot_export_and_nonbalanced_band_endpoints(data, tmp_path):
    result = run(data, clones=["A", "Lost", "New"])
    ax = cs.pl.phenotrack(result, outdir=tmp_path / "nothing")
    assert not (tmp_path / "nothing").exists()
    assert len(ax.collections) == 2
    path = ax.collections[0].get_paths()[0].vertices
    left, right = path[path[:, 0] == path[:, 0].min(), 1], path[path[:, 0] == path[:, 0].max(), 1]
    assert np.ptp(left) == pytest.approx(.25)
    assert np.ptp(right) == pytest.approx(1 / 3)
    ax.set_title("Editable")
    ax.figure.savefig(tmp_path / "manual.png")
    plt.close(ax.figure)
    ax = cs.pl.phenotrack(result, filename="flow", outdir=tmp_path, formats=("pdf", "png"), dpi=80)
    assert (tmp_path / "flow.pdf").exists() and (tmp_path / "flow.png").exists()
    plt.close(ax.figure)


def test_validation_of_time_order_and_library_scope(data):
    with pytest.raises(ValueError, match="chronological"):
        cs.tl.phenotrack(data, time_key="condition")
    with pytest.raises(ValueError, match="covering"):
        run(data, phenotype_order=["Eff"])
    data.obs["sample_id"] = "same_capture"
    with pytest.raises(ValueError, match="exactly one"):
        run(data)


def test_ternary_projection_and_state_selection(data, tmp_path):
    result = run(data, phenotype_order=["Eff", "Mem", "Prolif"])
    ax = cs.pl.phenotrack_ternary(result, from_time="Pre", to_time="Post",
                                 filename="ternary", outdir=tmp_path)
    coordinates = ax.cellscope_table.set_index(["clone_id", "time"])
    assert coordinates.loc[("A", "Pre"), "x"] == pytest.approx(.25)
    assert coordinates.loc[("A", "Pre"), "y"] == pytest.approx(np.sqrt(3) / 4)
    assert coordinates.loc[("A", "Post"), "x"] == 0
    assert "New" not in coordinates.index.get_level_values("clone_id")
    assert (tmp_path / "ternary.pdf").exists() and (tmp_path / "ternary.png").exists()
    plt.close(ax.figure)
    ax = cs.pl.phenotrack_ternary(result, from_time="Pre", to_time="Post", include_new=True)
    assert "New" in set(ax.cellscope_table.clone_id)
    plt.close(ax.figure)
    with pytest.raises(ValueError, match="exactly three"):
        cs.pl.phenotrack_ternary(result, from_time="Pre", phenotypes=["Eff", "Mem"])


def test_categorical_state_order_and_dates(data):
    data.obs["cell_type"] = pd.Categorical(data.obs.cell_type, categories=["Mem", "Eff"])
    data.obs["date"] = pd.to_datetime(data.obs.condition.map({"Pre": "2020-01-01", "Post": "2020-02-01"}))
    dates = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")]
    result = cs.tl.phenotrack(data, time_key="date", time_order=dates)
    assert result.parameters["phenotype_order"] == ["Mem", "Eff"]
    assert result.denominators.denominator.tolist() == [4, 3]
    assert not result.flows.empty


def test_missing_sampling_plot_and_empty_clone_selection(data):
    result = cs.tl.phenotrack(data, time_key="condition", time_order=["Pre", "Missing", "Post"])
    assert result.flows.empty
    ax = cs.pl.phenotrack(result)
    assert any(text.get_text() == "not sampled" for text in ax.texts)
    plt.close(ax.figure)
    mean = cs.tl.phenotrack_aggregate(result, missing="available")
    absent = mean.nodes[mean.nodes.time.eq("Missing")]
    assert absent.value.isna().all() and not absent.sampled.any()
    assert mean.parameters["n_donors_by_time"]["Missing"] == 0
    ax = cs.pl.phenotrack(mean)
    assert any(text.get_text() == "not sampled" for text in ax.texts)
    plt.close(ax.figure)
    empty = run(data, clones=[])
    assert empty.flows.empty and empty.composition.empty
    mean = cs.tl.phenotrack_aggregate(empty)
    assert mean.nodes.value.sum() == 0
    ax = cs.pl.phenotrack(mean)
    plt.close(ax.figure)
