"""Expression preprocessing without automatic file or figure creation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._core import counts, modality, record, sync_obs


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
):
    """Calculate QC and mark passing cells, retaining all modalities and cells."""
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


def normalize(data, *, mod="gex", layer=None, target_sum=1e4, counts_layer="counts"):
    """Preserve raw counts once and compute log1p normalized expression in X."""
    import scanpy as sc

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
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    record(
        adata,
        "normalize",
        {"layer": layer, "target_sum": target_sum, "counts_layer": counts_layer},
        backend="scanpy",
    )
    return data


def highly_variable_genes(data, *, mod="gex", n_top_genes=2000, batch_key=None, **kwargs):
    import scanpy as sc

    adata = modality(data, mod)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=min(n_top_genes, adata.n_vars),
        batch_key=batch_key,
        subset=False,
        **kwargs,
    )
    record(
        adata,
        "highly_variable_genes",
        {"n_top_genes": n_top_genes, "batch_key": batch_key},
        backend="scanpy",
    )
    return data


def neighbors(data, *, mod="gex", **kwargs):
    import scanpy as sc

    sc.pp.neighbors(modality(data, mod), **kwargs)
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
