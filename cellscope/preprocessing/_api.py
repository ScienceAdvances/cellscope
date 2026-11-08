"""Expression preprocessing without automatic file or figure creation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import _logging as feedback
from .._core import counts, modality, record, sync_obs
from .._device import execution_backend, run_scanpy


def qc(
    data,
    *,
    mod="gex",
    layer=None,
    mito_pattern=r"(?i)^mt-",
    min_genes=0,
    min_counts=0,
    max_pct_mito=100.0,
    key_added="qc_pass",
    batch_key=None,
    min_cells=None,
    filter=False,
    gene_join="union",
):
    """QC defaults to X; optional batch-wise cell/gene filtering via filter=True.

    Legacy calls without batch_key/min_cells only annotate cells by default.
    Batch-wise QC also records gene support per library; gene_join controls the
    shared feature space. Always assign the return value when filtering.
    """
    if batch_key is not None or min_cells is not None or filter:
        from ._adata_preprocessing import basic_qc

        return basic_qc(
            data, mod=mod, layer=layer, batch_key=batch_key, min_genes=min_genes,
            min_cells=0 if min_cells is None else min_cells, min_counts=min_counts,
            max_pct_mito=max_pct_mito, mito_pattern=mito_pattern, key_added=key_added,
            filter=filter, gene_join=gene_join,
        )
    import scanpy as sc

    adata = modality(data, mod)
    counts(adata, layer)
    if min_genes < 0 or min_counts < 0 or not 0 <= max_pct_mito <= 100:
        raise ValueError("QC thresholds must be nonnegative; mitochondrial percentage is 0..100")
    adata.var["mt"] = adata.var_names.str.contains(mito_pattern, regex=True)
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], layer=layer, percent_top=None, log1p=False, inplace=True
    )
    adata.obs[key_added] = (
        adata.obs["n_genes_by_counts"].ge(min_genes)
        & adata.obs["total_counts"].ge(min_counts)
        & adata.obs["pct_counts_mt"].le(max_pct_mito)
    )
    record(
        adata,
        key_added,
        {
            "layer": layer,
            "min_genes": min_genes,
            "min_counts": min_counts,
            "max_pct_mito": max_pct_mito,
        },
        backend="scanpy",
    )
    sync_obs(data, mod)
    return data


def filter_cells(data, *, key="qc_pass", mod="gex", copy=True):
    """Return a subset across all modalities, using a selected modality's mask."""
    adata = modality(data, mod)
    keep = adata.obs[key]
    if keep.isna().any() or not pd.api.types.is_bool_dtype(keep):
        raise ValueError("Filter key must contain nonmissing boolean values")
    result = data[adata.obs_names[keep]]
    return result.copy() if copy else result


def normalize(data, *, mod="gex", layer=None, target_sum=1e4, counts_layer="counts", device=None):
    """Preserve raw counts once and compute log1p normalized expression in X."""
    adata = modality(data, mod)
    if target_sum <= 0:
        raise ValueError("target_sum must be positive")
    if layer is None and counts_layer in adata.layers:
        layer = counts_layer
    matrix = counts(adata, layer)
    if counts_layer not in adata.layers:
        adata.layers[counts_layer] = matrix.copy()
    adata.X = matrix.copy()
    adata.uns.pop("log1p", None)
    feedback.progress(f"Normalization | library-size scaling to {target_sum:g}")
    run_scanpy(adata, "pp", "normalize_total", device=device, target_sum=target_sum)
    feedback.progress("Normalization | log1p transformation")
    run_scanpy(adata, "pp", "log1p", device=device)
    record(
        adata,
        "normalize",
        {"layer": layer, "target_sum": target_sum, "counts_layer": counts_layer},
        backend=execution_backend(adata, "pp", "log1p"),
    )
    return data


def highly_variable_genes(data, *, mod="gex", n_top_genes=2000, batch_key=None, device=None, **kwargs):
    adata = modality(data, mod)
    run_scanpy(
        adata, "pp", "highly_variable_genes", device=device,
        n_top_genes=min(n_top_genes, adata.n_vars),
        batch_key=batch_key,
        subset=False,
        **kwargs,
    )
    if kwargs.get("inplace", True) and "highly_variable" in adata.var:
        feedback.info(f"HVG selection | {int(adata.var['highly_variable'].sum()):,} genes selected")
    record(
        adata,
        "highly_variable_genes",
        {"n_top_genes": n_top_genes, "batch_key": batch_key},
        backend=execution_backend(adata, "pp", "highly_variable_genes"),
    )
    return data


def neighbors(data, *, mod="gex", device=None, **kwargs):
    run_scanpy(modality(data, mod), "pp", "neighbors", device=device, **kwargs)
    return data


def doublets(data, *, mod="gex", layer="counts", batch_key=None, **kwargs):
    """Run Scrublet on raw counts, copying only its observation annotations back."""
    import scanpy as sc
    from anndata import AnnData

    adata = modality(data, mod)
    working = AnnData(counts(adata, layer).copy(), obs=adata.obs.copy(), var=adata.var.copy())
    sc.pp.scrublet(working, batch_key=batch_key, **kwargs)
    for key in ("doublet_score", "predicted_doublet"):
        adata.obs[key] = working.obs[key]
    adata.uns["scrublet"] = working.uns["scrublet"]
    record(adata, "doublets", {"layer": layer, "batch_key": batch_key}, backend="scanpy")
    sync_obs(data, mod)
    return data


def mad_outliers(data, *, metrics, mod="gex", batch_key=None, nmads=3, key_added="qc_outlier"):
    """Mark robust QC outliers within batches; do not filter observations."""
    from scipy.stats import median_abs_deviation

    adata = modality(data, mod)
    if nmads <= 0 or not metrics:
        raise ValueError("Supply metrics and positive nmads")
    flags = pd.Series(False, index=adata.obs_names)
    groups = (
        adata.obs.groupby(batch_key, observed=True).groups
        if batch_key
        else {"all": adata.obs_names}
    )
    for idx in groups.values():
        for metric in metrics:
            values = adata.obs.loc[idx, metric]
            threshold = nmads * median_abs_deviation(values.dropna())
            flags.loc[idx] |= values.isna() | np.abs(values - values.median()).gt(threshold)
    adata.obs[key_added] = flags
    sync_obs(data, mod)
    return data
