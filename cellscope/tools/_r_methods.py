"""Explicit adapters for tutorial methods without an equivalent Python algorithm."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import r
from .._core import counts, dependency, modality, record, sync_obs
from ..preprocessing._best_practices import _labels


def glmpca(
    data, *, mod="gex", layer="counts", n_comps=30, key_added="X_glmpca", backend="python", **kwargs
):
    """Fit GLM-PCA with the original author's Python port, or explicitly choose R.

    The Python backend requires a dense genes-by-cells count matrix.
    """
    adata = modality(data, mod)
    if not 0 < n_comps < min(adata.shape):
        raise ValueError("n_comps must be positive and smaller than both matrix dimensions")
    if backend == "python":
        gp = dependency("glmpca.glmpca", "glmpca")
        matrix = counts(adata, layer).T
        fit = gp.glmpca(
            matrix.toarray() if hasattr(matrix, "toarray") else matrix, n_comps, **kwargs
        )
    elif backend == "r":
        fit = r._call(
            """function(x, k, options) {
        fit <- do.call(glmpca::glmpca, c(list(Y=x, L=k), options))
        list(factors=fit$factors, loadings=fit$loadings)
        }""",
            packages=["glmpca"],
            x=counts(adata, layer).T,
            k=n_comps,
            options=kwargs,
        )
    else:
        raise ValueError("backend must be 'python' or 'r'")
    adata.obsm[key_added] = np.asarray(fit["factors"])
    adata.varm[f"{key_added}_loadings"] = np.asarray(fit["loadings"])
    params = {"layer": layer, "n_comps": n_comps}
    if backend == "r":
        r.record_r(adata, key_added, params, ["glmpca"])
    else:
        record(adata, key_added, params, backend="glmpca")
    return data


def slingshot(
    data,
    *,
    cluster_key,
    use_rep="X_pca",
    mod="gex",
    start_cluster=None,
    key_added="slingshot",
    backend="python",
    fit_kwargs=None,
    **kwargs,
):
    """Fit Slingshot with pyslingshot or R; store aligned lineage times and weights."""
    adata = modality(data, mod)
    labels = _labels(adata, cluster_key)
    if backend == "python":
        module = dependency("pyslingshot", "slingshot")
        levels = sorted(set(labels))
        if len(levels) < 2:
            raise ValueError("Slingshot needs at least two clusters")
        if start_cluster is None or str(start_cluster) not in levels:
            raise ValueError("Choose an observed start_cluster for Python Slingshot")
        codes = np.asarray([levels.index(label) for label in labels])
        onehot = np.eye(len(levels))[codes]
        fit = module.Slingshot(
            np.asarray(adata.obsm[use_rep]),
            cluster_labels_onehot=onehot,
            start_node=levels.index(str(start_cluster)),
            **kwargs,
        )
        fit.fit(**(fit_kwargs or {}))
        adata.obsm[f"{key_added}_pseudotime"] = np.column_stack(
            [curve.pseudotimes_interp for curve in fit.curves]
        )
        adata.obsm[f"{key_added}_weights"] = np.asarray(fit.cell_weights)
        adata.obs[f"{key_added}_pseudotime"] = fit.unified_pseudotime
        record(
            adata,
            key_added,
            {"cluster_key": cluster_key, "use_rep": use_rep, "start_cluster": start_cluster},
            backend="pyslingshot",
        )
        sync_obs(data, mod)
        return fit
    if backend != "r":
        raise ValueError("backend must be 'python' or 'r'")
    result = r._call(
        """function(x, clusters, start, options) {
        fit <- do.call(slingshot::slingshot,
            c(list(data=x, clusterLabels=clusters, start.clus=start), options))
        list(pseudotime=slingshot::slingPseudotime(fit),
             weights=slingshot::slingCurveWeights(fit), fit=fit)
    }""",
        packages=["slingshot"],
        x=np.asarray(adata.obsm[use_rep]),
        clusters=labels,
        start=start_cluster,
        options=kwargs,
    )
    for name in ("pseudotime", "weights"):
        adata.obsm[f"{key_added}_{name}"] = np.asarray(result[name])
    r.record_r(
        adata,
        key_added,
        {"cluster_key": cluster_key, "use_rep": use_rep, "start_cluster": start_cluster},
        ["slingshot"],
    )
    return result["fit"]


def tradeseq(
    data,
    *,
    mod="gex",
    layer="counts",
    trajectory_key="slingshot",
    n_knots=6,
    test="associationTest",
    random_state=0,
    **kwargs,
):
    """Fit R tradeSeq GAMs to Slingshot outputs; return a gene table and native fit."""
    if test not in {"associationTest", "startVsEndTest", "diffEndTest", "patternTest"}:
        raise ValueError("Unsupported tradeSeq test")
    adata = modality(data, mod)
    times = np.asarray(adata.obsm[f"{trajectory_key}_pseudotime"])
    weights = np.asarray(adata.obsm[f"{trajectory_key}_weights"])
    if times.shape != weights.shape or times.shape[0] != adata.n_obs:
        raise ValueError(
            "Pseudotime and lineage weights must be aligned cells-by-lineages matrices"
        )
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Lineage weights must be finite and nonnegative")
    if np.any(~np.isfinite(times) & (weights > 0)):
        raise ValueError("Positive lineage weights require finite pseudotimes")
    result = r._call(
        """function(x, genes, times, weights, knots, test, seed, options) {
        set.seed(seed); rownames(x) <- genes
        fit <- do.call(tradeSeq::fitGAM, c(list(counts=x, pseudotime=times,
                       cellWeights=weights, nknots=knots, verbose=FALSE), options))
        list(table=as.data.frame(getExportedValue("tradeSeq", test)(fit)), fit=fit)
    }""",
        packages=["tradeSeq"],
        x=counts(adata, layer).T,
        genes=adata.var_names,
        times=np.nan_to_num(times),
        weights=weights,
        knots=n_knots,
        test=test,
        seed=random_state,
        options=kwargs,
    )
    table = result["table"].rename_axis("gene").reset_index()
    r.record_r(
        adata,
        "tradeseq",
        {"layer": layer, "trajectory_key": trajectory_key, "test": test, "n_knots": n_knots},
        ["tradeSeq"],
    )
    adata.uns["cellscope"]["tradeseq"]["table"] = table
    return table, result["fit"]


def nichenet(
    geneset, background, potential_ligands, ligand_target_matrix, *, backend="python", **kwargs
):
    """Predict NicheNet ligand activities with an explicit named gene-by-ligand prior.

    The caller derives the receiver gene set from sample-level differential
    expression and selects expressed sender ligands/receiver receptors.
    """
    if not isinstance(ligand_target_matrix, pd.DataFrame):
        raise TypeError("ligand_target_matrix must be a genes-by-ligands DataFrame")
    geneset, background, ligands = list(geneset), list(background), list(potential_ligands)
    if not geneset or not set(geneset).issubset(background):
        raise ValueError("geneset must be a nonempty subset of the expressed background")
    if not ligands or not set(ligands).issubset(ligand_target_matrix.columns):
        raise ValueError("All potential ligands must occur in the ligand-target prior")
    if not set(background).issubset(ligand_target_matrix.index):
        raise ValueError("All background genes must occur in the ligand-target prior")
    if backend == "python":
        if kwargs:
            raise ValueError("nichenetpy does not accept additional activity options")
        module = dependency("nichenetpy.prediction", "communication")
        predictor = module.LigandActivityPredictor(
            ligand_target_matrix.to_numpy(),
            list(ligand_target_matrix.index),
            list(ligand_target_matrix.columns),
        )
        activities = predictor.predict_ligand_activities(geneset, background, ligands)
        return (
            pd.DataFrame.from_dict(activities, orient="index").rename_axis("ligand").reset_index()
        )
    if backend != "r":
        raise ValueError("backend must be 'python' or 'r'")
    return r._call(
        """function(genes, background, ligands, prior, options) {
        as.data.frame(do.call(nichenetr::predict_ligand_activities,
            c(list(geneset=genes, background_expressed_genes=background,
                   ligand_target_matrix=as.matrix(prior), potential_ligands=ligands), options)))
    }""",
        packages=["nichenetr"],
        genes=geneset,
        background=background,
        ligands=ligands,
        prior=ligand_target_matrix,
        options=kwargs,
    )
