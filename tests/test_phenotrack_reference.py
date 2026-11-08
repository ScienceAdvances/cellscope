"""Optional comparisons against external, unmodified PhenoTrack author sources."""

import hashlib
import importlib
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
import scipy.stats
from anndata import AnnData

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cellscope as cs
from cellscope.tools._phenotrack import REFERENCE_SHA256


@contextmanager
def author_modules():
    root = os.environ.get("PHENOTRACK_REFERENCE_DIR")
    support = os.environ.get("CLONETRACK_REFERENCE_DIR")
    if not root or not support:
        pytest.skip("External author sources not supplied")
    for file in ("pheno_def.py", "plotting_phenos.py"):
        if not (Path(root) / file).is_file():
            pytest.skip(f"Missing author source: {file}")
        assert hashlib.sha256((Path(root) / file).read_bytes()).hexdigest() == REFERENCE_SHA256[file]
    names = ["pheno_def", "plotting_phenos", "tcr_utils", "tcr_errors", "tcr_seq_and_clone_definitions"]
    saved = {name: sys.modules.pop(name, None) for name in names}
    path = list(sys.path)
    had_binom = hasattr(scipy.stats, "binom_test")
    if not had_binom:
        # Compatibility alias for an imported, unused old SciPy API name.
        scipy.stats.binom_test = lambda x, n=None, p=.5, alternative="two-sided": (
            scipy.stats.binomtest(x, n=n, p=p, alternative=alternative).pvalue)
    sys.path[:0] = [root, support]
    try:
        yield importlib.import_module("pheno_def"), importlib.import_module("plotting_phenos")
    finally:
        sys.path[:] = path
        for name in names:
            sys.modules.pop(name, None)
            if saved[name] is not None:
                sys.modules[name] = saved[name]
        if not had_binom:
            del scipy.stats.binom_test


class Repertoire:
    """Minimal author repertoire protocol; test data are supplied directly."""

    def __init__(self, counts, norm):
        self.clones_and_phenos, self.norm = counts, norm
        self.clones = list(counts)
        self.name = "test"
        self.phenotypes = ["Eff", "Mem"]

    def __getitem__(self, clone):
        return self.clones_and_phenos.get(clone, dict.fromkeys(self.phenotypes, 0))

    def get_phenotype_counts(self, clone, phenotypes):
        return np.array([self[clone].get(state, 0) for state in phenotypes], dtype=float)

    def clone_union(self, other):
        return sorted(set(self.clones) | set(other.clones))


@pytest.mark.parametrize("normalize", [True, False])
def test_author_static_nodes_inflow_outflow_and_l1(monkeypatch, normalize):
    obs = pd.DataFrame({
        "donor_id": ["P1"] * 7, "sample_id": ["pre"] * 4 + ["post"] * 3,
        "time": ["Pre"] * 4 + ["Post"] * 3,
        "clone_id": ["A", "A", "Lost", "Other", "A", "A", "New"],
        "state": ["Eff", "Mem", "Eff", "Eff", "Mem", "Mem", "Eff"],
    }, index=[f"c{i}" for i in range(7)])
    adata = AnnData(np.zeros((7, 1)), obs=obs)
    clones = ["A", "Lost", "New"]
    result = cs.tl.phenotrack(adata, time_key="time", time_order=["Pre", "Post"],
                             phenotype_key="state", clones=clones, normalize=normalize)
    origin = Repertoire({"A": {"Eff": 1, "Mem": 1}, "Lost": {"Eff": 1, "Mem": 0},
                         "Other": {"Eff": 1, "Mem": 0}}, 4)
    destination = Repertoire({"A": {"Eff": 0, "Mem": 2}, "New": {"Eff": 1, "Mem": 0}}, 3)
    with author_modules() as (definitions, plotting):
        endpoints = []
        monkeypatch.setattr(plotting.SankeyNode, "plot_node_connection",
                            lambda a, b, ax, **kwargs: endpoints.append((a.dy, b.dy)))
        fig, ax = plotting.plot_pheno_sankey(["Eff", "Mem"], [origin, destination],
                                            clones=clones, normalize=normalize, times=[0, 1], return_axes=True)
        np.testing.assert_allclose([p.get_height() for p in ax.patches], result.nodes.value, atol=1e-15)
        flow = result.flows.groupby(["from_phenotype", "to_phenotype"])[["outflow", "inflow"]].sum()
        full = flow.reindex(pd.MultiIndex.from_product([["Eff", "Mem"], ["Eff", "Mem"]]), fill_value=0)
        np.testing.assert_allclose(np.array(endpoints), full.to_numpy(), atol=1e-15)
        original_flux = definitions.TcellPhenotypeRepertoire.get_phenotypic_flux(origin, destination)
        assert original_flux["A"] == result.flux.phenotypic_flux.iloc[0]
        plt.close(fig)


def test_ternary_projection_matches_author():
    with author_modules() as (_, plotting):
        for p in [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]), np.array([.2, .3, .5])]:
            expected = np.array([1 - p[1] - p[0] / 2, np.sqrt(3) * p[0] / 2])
            np.testing.assert_allclose(plotting.ternary_plot_projection(p), expected, atol=1e-15)


def test_equal_donor_mean_matches_author_complete_cohort(monkeypatch):
    first_pre = Repertoire({"A": {"Eff": 1, "Mem": 1}, "Other": {"Eff": 2, "Mem": 0}}, 4)
    first_post = Repertoire({"A": {"Eff": 0, "Mem": 2}, "Other": {"Eff": 1, "Mem": 0}}, 3)
    second_pre = Repertoire({"A": {"Eff": 8, "Mem": 0}, "Other": {"Eff": 0, "Mem": 2}}, 10)
    second_post = Repertoire({"A": {"Eff": 1, "Mem": 1}, "Other": {"Eff": 2, "Mem": 0}}, 4)
    records = []
    for donor, reps in {"P1": [first_pre, first_post], "P2": [second_pre, second_post]}.items():
        for time, rep in zip(["Pre", "Post"], reps, strict=True):
            for clone, counts in rep.clones_and_phenos.items():
                for state, count in counts.items():
                    for _ in range(count):
                        records.append({"donor_id": donor, "sample_id": f"{donor}_{time}",
                                        "time": time, "clone_id": clone, "state": state})
    frame = pd.DataFrame(records, index=[f"c{i}" for i in range(len(records))])
    data = AnnData(np.zeros((len(frame), 1)), obs=frame)
    result = cs.tl.phenotrack(data, time_key="time", time_order=["Pre", "Post"],
                             phenotype_key="state", clones=["A"])
    mean = cs.tl.phenotrack_aggregate(result)
    with author_modules() as (_, plotting):
        endpoints = []
        monkeypatch.setattr(plotting.SankeyNode, "plot_node_connection",
                            lambda a, b, ax, **kwargs: endpoints.append((a.dy, b.dy)))
        fig, ax = plotting.plot_agg_pheno_sankey_w_dict(
            ["Eff", "Mem"], [{"P1": first_pre, "P2": second_pre}, {"P1": first_post, "P2": second_post}],
            clones_by_pt={"P1": ["A"], "P2": ["A"]}, times=[0, 1], return_axes=True)
        np.testing.assert_allclose([patch.get_height() for patch in ax.patches], mean.nodes.value, atol=1e-15)
        flow = mean.flows.set_index(["from_phenotype", "to_phenotype"])[["outflow", "inflow"]]
        full = flow.reindex(pd.MultiIndex.from_product([["Eff", "Mem"], ["Eff", "Mem"]]), fill_value=0)
        np.testing.assert_allclose(endpoints, full.to_numpy(), atol=1e-15)
        plt.close(fig)
