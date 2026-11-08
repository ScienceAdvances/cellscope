"""Adapters for the author's pre-publication GeneVector 0.1.0 implementation.

No PCA/Harmony fallback. The optional accelerated MI backend reproduces the
upstream co-expressed-cell, ten-bin histogram definition (including scaling).
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .. import _logging as feedback
from .._core import counts, dependency, modality, record


def _mi_numba_exact(matrix):
    """Compute upstream's conditional ten-bin MI; no zero/zero observations."""
    from numba import njit, prange

    @njit(parallel=True, cache=False)
    def compute(x):
        genes, cells = x.shape
        result = np.zeros((genes, genes), dtype=np.float64)
        for i in prange(genes):
            for j in range(i + 1, genes):
                lo_x, hi_x, lo_y, hi_y, n = np.inf, -np.inf, np.inf, -np.inf, 0
                for c in range(cells):
                    a, b = x[i, c], x[j, c]
                    if a > 0 and b > 0:
                        lo_x, hi_x = min(lo_x, a), max(hi_x, a)
                        lo_y, hi_y = min(lo_y, b), max(hi_y, b)
                        n += 1
                if n == 0:
                    continue
                if hi_x == lo_x:
                    lo_x, hi_x = lo_x - 0.5, hi_x + 0.5
                if hi_y == lo_y:
                    lo_y, hi_y = lo_y - 0.5, hi_y + 0.5
                joint = np.zeros((10, 10), dtype=np.float64)
                for c in range(cells):
                    a, b = x[i, c], x[j, c]
                    if a > 0 and b > 0:
                        ix = min(9, int((a - lo_x) * 10 / (hi_x - lo_x)))
                        iy = min(9, int((b - lo_y) * 10 / (hi_y - lo_y)))
                        joint[ix, iy] += 1
                joint /= n
                px, py = joint.sum(axis=1), joint.sum(axis=0)
                value = 0.0
                for u in range(10):
                    for v in range(10):
                        p = joint[u, v]
                        if p > 0:
                            value += p * np.log2(p / (px[u] * py[v]))
                result[i, j] = value
                result[j, i] = value
        return result

    dense = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    return compute(np.ascontiguousarray(dense.T, dtype=np.float64))


def genevector_fit(
    data,
    *,
    output_dir,
    mod="gex",
    layer="counts",
    mi_backend="original",
    emb_dimension=100,
    epochs=1000,
    threshold=1e-5,
    batch_size=100000,
    c=1.0,
    random_state=0,
    n_threads=8,
):
    """Train author's GeneVector 0.1.0 on explicitly selected raw-count genes.

    Select genes before calling; no hidden filtering/normalization. Epochs,
    markers and selected genes are caller choices, not alleged paper settings.
    ``numba_exact`` only accelerates the upstream MI formula, tested against
    original histogram results. Model/Adadelta training remain upstream code.
    Returns runtime objects; serializable provenance is in data.uns.
    """
    if mi_backend not in {"original", "numba_exact"}:
        raise ValueError("mi_backend must be original or numba_exact")
    backend = dependency("genevector", "genevector")
    if version("genevector") != "0.1.0":
        raise ImportError("This adapter requires the recorded author GeneVector 0.1.0 sources")
    import torch
    from genevector.data import GeneVectorDataset
    from genevector.embedding import GeneEmbedding
    from genevector.model import GeneVector

    adata = modality(data, mod)
    matrix = counts(adata, layer)
    if adata.n_obs < 2 or adata.n_vars < 2:
        raise ValueError("GeneVector needs at least two cells and genes")
    if not np.all(adata.var_names == adata.var_names.str.upper()):
        raise ValueError("Provide unique uppercase symbols, as required by upstream GeneVector")
    directory = Path(output_dir)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(directory)
    directory.mkdir(parents=True, exist_ok=True)
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.set_num_threads(n_threads)
    from anndata import AnnData

    working = AnnData(matrix.copy(), obs=adata.obs.copy(), var=adata.var.copy())
    feedback.progress("GeneVector | constructing author's dataset")
    dataset = GeneVectorDataset(working, device="cpu")
    if mi_backend == "numba_exact":
        from numba import set_num_threads

        set_num_threads(n_threads)
        feedback.progress("GeneVector | computing exact histogram mutual information")
        mi = _mi_numba_exact(matrix)
        # Context excludes genes expressed in fewer than 10 cells from MI.
        valid = set(dataset.data.data)
        scores = defaultdict(lambda: defaultdict(float))
        names = working.var_names.tolist()
        for i, a in enumerate(names):
            if a not in valid:
                continue
            for j in range(i + 1, len(names)):
                b = names[j]
                if b in valid:
                    scores[a][b] = float(mi[i, j] * c)
                    scores[b][a] = float(mi[i, j] * c)
        dataset.mi_scores = scores
        np.save(directory / "mi_unscaled.npy", mi)
    filename = directory / "genes.vec"
    model = GeneVector(
        dataset,
        str(filename),
        emb_dimension=emb_dimension,
        batch_size=batch_size,
        c=c,
        device="cpu",
    )
    feedback.progress("GeneVector | author's Adadelta training")
    model.train(epochs, threshold=threshold)
    # Upstream 0.1.0 returns before saving if early stopping fires. Always save
    # the actual final weights, without additional optimization steps.
    model.model.save_embedding(dataset.data.id2gene, str(filename), 0)
    model.model.save_embedding(dataset.data.id2gene, str(directory / "genes2.vec"), 1)
    embedding = GeneEmbedding(str(filename), dataset, vector="average")
    params = {
        "source_version": version("genevector"),
        "mi_backend": mi_backend,
        "mi_definition": "10 equal-width bins over jointly positive cells; log2 MI",
        "c": float(c),
        "upstream_target_scale": float(c * c),
        "emb_dimension": emb_dimension,
        "max_epochs": epochs,
        "epochs_executed": len(model.mean_loss_values),
        "threshold": threshold,
        "early_stopped": len(model.mean_loss_values) < epochs,
        "last_mean_loss": float(model.mean_loss_values[-1]),
        "batch_size": batch_size,
        "random_state": random_state,
        "n_cells": adata.n_obs,
        "n_genes": adata.n_vars,
        "selected_genes_sha256": hashlib.sha256("\n".join(adata.var_names).encode()).hexdigest(),
        "paper_hyperparameters_fully_reported": False,
        "source_sha256": {
            name: hashlib.sha256((Path(backend.__file__).parent / name).read_bytes()).hexdigest()
            for name in ("data.py", "model.py", "embedding.py")
        },
    }
    pd.DataFrame({"mean_loss": model.mean_loss_values}).to_csv(
        directory / "training_loss.csv", index=False
    )
    record(adata, "genevector_fit", params, backend="genevector")
    return {
        "dataset": dataset,
        "model": model,
        "gene_embedding": embedding,
        "parameters": params,
        "output_dir": directory,
    }


def genevector_embedding(
    data, fit, *, batch_key, reference_batch, mod="gex", key_added="X_genevector"
):
    """Author expression-weighted cell vectors and vector-shift batch correction."""
    from genevector.embedding import CellEmbedding

    adata = modality(data, mod)
    if batch_key not in adata.obs or adata.obs[batch_key].isna().any():
        raise ValueError("Provide nonmissing sample batch labels")
    fit["dataset"].data.adata.obs[batch_key] = adata.obs[batch_key].copy()
    fit["dataset"].data.metadata[batch_key] = adata.obs[batch_key].copy()
    cell_embedding = CellEmbedding(fit["dataset"], fit["gene_embedding"])
    uncorrected = np.asarray(cell_embedding.matrix).copy()
    ids = list(cell_embedding.data)
    cell_embedding.batch_correct(column=batch_key, reference=reference_batch)
    corrected = np.asarray(cell_embedding.matrix)
    aligned = pd.DataFrame(corrected, index=ids).reindex(adata.obs_names)
    adata.obsm[key_added] = aligned.to_numpy()
    adata.obsm[key_added + "_uncorrected"] = (
        pd.DataFrame(uncorrected, index=ids).reindex(adata.obs_names).to_numpy()
    )
    fit["cell_embedding"] = cell_embedding
    record(
        adata,
        "genevector_embedding",
        {
            "batch_key": batch_key,
            "reference_batch": reference_batch,
            "key_added": key_added,
            "missing_embedding_cells": int(aligned.isna().any(axis=1).sum()),
        },
        backend="genevector",
    )
    return data


def genevector_phenotypes(
    data, fit, markers, *, mod="gex", key_added="genevector_state", undefined_policy="author"
):
    """Call upstream softmax marker classifier with an explicit undefined policy.

    Explicit marker dictionaries are mandatory. No invented paper marker set.
    Default author retains upstream labels. Optional flag masks unsupported/zero-marker labels, without changing the computed probabilities.
    """
    if undefined_policy not in {"author", "flag"}:
        raise ValueError("undefined_policy must be author or flag")
    adata = modality(data, mod)
    if not markers or any(not values for values in markers.values()):
        raise ValueError("Provide nonempty explicit marker sets")
    requested = {name: list(values) for name, values in markers.items()}
    missing = {
        name: [g for g in values if g not in adata.var_names] for name, values in requested.items()
    }
    if any(len(missing[name]) == len(values) for name, values in requested.items()):
        raise ValueError("Every phenotype needs at least one present marker")
    classified = fit["cell_embedding"].phenotype_probability(
        adata.copy(),
        requested,
        {},
        target_col=key_added,
    )
    columns = [name + " Pseudo-probability" for name in requested]
    probabilities = classified.obs[columns].reindex(adata.obs_names)
    finite = np.isfinite(probabilities.to_numpy()).all(axis=1)
    present = sorted(
        {g for values in requested.values() for g in values if g in adata.var_names}
    )
    marker_matrix = adata.layers.get("counts", adata.X)
    marker_signal = np.asarray(
        marker_matrix[:, adata.var_names.get_indexer(present)].sum(axis=1)
    ).ravel()
    valid = finite & (marker_signal > 0)
    adata.obsm[key_added + "_probabilities"] = probabilities.to_numpy()
    labels = classified.obs[key_added].reindex(adata.obs_names).astype("string")
    if undefined_policy == "flag":
        labels.loc[~valid] = pd.NA
    adata.obs[key_added] = labels.astype("category")
    adata.obs[key_added + "_defined"] = valid
    record(
        adata,
        key_added,
        {
            "markers": requested,
            "missing_markers": missing,
            "probability_columns": list(requested),
            "undefined_cells": int((~valid).sum()),
            "undefined_policy": undefined_policy,
            "zero_marker_cells": int((marker_signal == 0).sum()),
            "exact_paper_marker_set_available": False,
        },
        backend="genevector",
    )
    return probabilities
