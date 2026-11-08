"""Count-preserving preprocessing methods used in Single-cell Best Practices."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

from .. import _logging as feedback
from .. import r
from .._core import counts, dependency, modality, record, sync_obs
from ._adata_preprocessing import _groups


def _labels(adata, key):
    if key not in adata.obs or adata.obs[key].isna().any():
        raise ValueError(f"Nonmissing observation metadata required: {key}")
    return adata.obs[key].astype(str).to_numpy()


def _output(adata, matrix, key):
    if matrix.shape != adata.shape:
        raise ValueError("Backend output does not match the input cell/feature dimensions")
    adata.layers[key] = sparse.csr_matrix(matrix) if sparse.issparse(matrix) else matrix


def pearson_residuals(
    data,
    *,
    mod="gex",
    layer="counts",
    key_added="pearson_residuals",
    theta=100,
    clip=None,
    **kwargs,
):
    """Store Scanpy analytic Pearson residuals in a layer; leave X and counts intact.

    These signed residuals are for dimension reduction, not count-model inference.
    The backend allocates a dense cells-by-genes residual matrix.
    """
    import scanpy as sc

    adata = modality(data, mod)
    counts(adata, layer)
    result = sc.experimental.pp.normalize_pearson_residuals(
        adata, layer=layer, theta=theta, clip=clip, inplace=False, **kwargs
    )
    _output(adata, result["X"], key_added)
    record(adata, key_added, {"layer": layer, "theta": theta, "clip": clip}, backend="scanpy")
    return data


def pearson_hvg(data, *, mod="gex", layer="counts", n_top_genes=2000, batch_key=None, **kwargs):
    """Select genes with Scanpy's analytic Pearson-residual variance model."""
    import scanpy as sc

    adata = modality(data, mod)
    counts(adata, layer)
    sc.experimental.pp.highly_variable_genes(
        adata,
        layer=layer,
        n_top_genes=min(n_top_genes, adata.n_vars),
        batch_key=batch_key,
        subset=False,
        **kwargs,
    )
    record(
        adata,
        "pearson_hvg",
        {"layer": layer, "batch_key": batch_key, "n_top_genes": n_top_genes},
        backend="scanpy",
    )
    return data


def scran(
    data,
    *,
    cluster_key,
    mod="gex",
    layer="counts",
    min_mean=0.1,
    key_added="scran_normalization",
    **kwargs,
):
    """Use R scran pooled deconvolution factors, then store log1p(counts/factor).

    scranpy library-size/median normalization is a different method. This adapter
    uses scran::computeSumFactors, with serial R execution by default.
    """
    adata = modality(data, mod)
    matrix = counts(adata, layer)
    factors = r._call(
        """function(x, clusters, min_mean, options) {
        sce <- SingleCellExperiment::SingleCellExperiment(list(counts=x))
        fit <- do.call(scran::computeSumFactors, c(list(x=sce, clusters=clusters,
            min.mean=min_mean, BPPARAM=BiocParallel::SerialParam()), options))
        SingleCellExperiment::sizeFactors(fit)
    }""",
        packages=["scran", "SingleCellExperiment", "BiocParallel"],
        x=matrix.T,
        clusters=_labels(adata, cluster_key),
        min_mean=min_mean,
        options=kwargs,
    )
    factors = np.asarray(factors).ravel()
    if factors.shape != (adata.n_obs,) or not np.isfinite(factors).all() or (factors <= 0).any():
        raise ValueError("scran returned nonpositive or invalid size factors")
    normalized = (
        sparse.diags(1 / factors) @ matrix
        if sparse.issparse(matrix)
        else np.asarray(matrix) / factors[:, None]
    )
    if sparse.issparse(normalized):
        normalized = normalized.tocsr()
        normalized.data = np.log1p(normalized.data)
    else:
        normalized = np.log1p(normalized)
    _output(adata, normalized, key_added)
    adata.obs["size_factors"] = factors
    r.record_r(
        adata,
        key_added,
        {"layer": layer, "cluster_key": cluster_key, "min_mean": min_mean},
        ["scran"],
    )
    sync_obs(data, mod)
    return data


def soupx(
    data, raw=None, *, batch_key=None, raw_files=None, cluster_key=None,
    mod="gex", layer="auto", raw_layer=None, backend="python", contamination=None,
    key_added="soupx_counts", random_state=0, estimate_kwargs=None, adjust_kwargs=None,
    backup_layer="raw_counts", counts_layer="counts",
):
    """Remove ambient RNA with NiRuff's soupx-python.

    For a merged object, pass batch_key and raw_files (capture-library labels
    mapped to unfiltered 10x files). Counts default to X. Temporary clusters
    are computed separately per library unless cluster_key is supplied.
    After all libraries succeed, the input counts are copied to backup_layer
    and corrected counts replace X and counts_layer. Cells and genes keep
    their order. contamination can be a scalar or a per-library mapping.

    The single-library form soupx(data, raw, cluster_key=...) defaults to the
    counts layer and writes only key_added, preserving X.
    """
    options = dict(
        mod=mod, backend=backend, key_added=key_added, random_state=random_state,
        estimate_kwargs=estimate_kwargs, adjust_kwargs=adjust_kwargs,
    )
    if batch_key is None and raw_files is None:
        if raw is None or cluster_key is None:
            raise ValueError("Supply raw and cluster_key for single-library SoupX")
        return _soupx_library(
            data, raw, cluster_key=cluster_key, layer="counts" if layer == "auto" else layer,
            raw_layer=raw_layer, contamination=contamination, **options,
        )
    if raw is not None or raw_layer is not None or batch_key is None or not isinstance(raw_files, Mapping):
        raise ValueError("For merged SoupX, supply batch_key and a raw_files mapping of unfiltered 10x inputs")
    return _soupx_batches(
        data, raw_files, batch_key=batch_key, cluster_key=cluster_key,
        layer=None if layer == "auto" else layer, contamination=contamination,
        backup_layer=backup_layer, counts_layer=counts_layer, **options,
    )


def _soupx_clusters(matrix, random_state):
    import scanpy as sc

    working = AnnData(matrix.copy())
    if working.n_obs < 3 or working.n_vars < 2:
        raise ValueError("Automatic SoupX clustering needs at least three cells and two genes; supply cluster_key")
    sc.pp.normalize_total(working, target_sum=1e4)
    sc.pp.log1p(working)
    sc.pp.highly_variable_genes(working, n_top_genes=min(2000, working.n_vars))
    n_pcs = min(30, working.n_obs - 1, int(working.var["highly_variable"].sum()) - 1)
    if n_pcs < 1:
        raise ValueError("Too few variable genes for SoupX clustering; supply cluster_key")
    sc.pp.pca(working, n_comps=n_pcs, mask_var="highly_variable", random_state=random_state)
    sc.pp.neighbors(
        working, n_neighbors=min(15, working.n_obs - 1), use_rep="X_pca",
        transformer="sklearn", random_state=random_state,
    )
    sc.tl.leiden(
        working, flavor="igraph", directed=False, n_iterations=2, random_state=random_state,
    )
    return working.obs["leiden"].to_numpy()


def _soupx_batches(
    data, raw_files, *, batch_key, cluster_key, layer, contamination,
    backup_layer, counts_layer, **options,
):
    from .. import io

    mod = options["mod"]
    adata = modality(data, mod)
    groups = _groups(adata, batch_key)
    matrix = counts(adata, layer)
    if backup_layer in adata.layers or backup_layer == counts_layer:
        raise ValueError("SoupX backup layer already exists or conflicts with counts_layer")
    paths = {str(batch): Path(path) for batch, path in raw_files.items()}
    if len(paths) != len(raw_files):
        raise ValueError("raw_files library labels must be unique after conversion to strings")
    missing = [batch for batch in groups if batch not in paths or not paths[batch].exists()]
    if missing:
        raise ValueError(f"Missing unfiltered 10x input for SoupX libraries: {missing}")
    gene_ids = adata.var.get("gene_ids", adata.var.get("gene_id"))
    if gene_ids is None or gene_ids.isna().any() or not gene_ids.astype(str).is_unique:
        raise ValueError("Merged SoupX requires unique, nonmissing gene_ids or gene_id")

    blocks, order = [], []
    rho = np.empty(adata.n_obs)
    contamination_by_sample = {}
    labels = _labels(adata, cluster_key) if cluster_key is not None else None
    for index, (batch, positions) in enumerate(groups.items(), 1):
        with feedback.batch("SoupX", batch, index, len(groups), cells=len(positions)):
            library = AnnData(matrix[positions].copy(), obs=adata.obs.iloc[positions].copy())
            barcodes = library.obs.get("barcode", library.obs_names)
            barcodes = pd.Index(barcodes, dtype="string")
            if barcodes.isna().any() or not barcodes.is_unique or (barcodes == "").any():
                raise ValueError(f"{batch}: SoupX requires unique, nonempty barcodes within a library")
            library.obs_names = barcodes.astype(str)
            library.var_names = pd.Index(gene_ids.astype(str))
            library.obs["_soupx_cluster"] = (
                _soupx_clusters(library.X, options["random_state"]) if labels is None else labels[positions]
            )

            raw = io.read_10x(paths[batch], library_id="soupx")
            raw.obs_names = pd.Index(raw.obs["barcode"].astype(str))
            raw_ids = raw.var.get("gene_ids", raw.var.get("gene_id"))
            if raw_ids is None or raw_ids.isna().any() or not raw_ids.astype(str).is_unique:
                raise ValueError(f"{batch}: unfiltered SoupX input requires unique, nonmissing gene IDs")
            raw.var_names = pd.Index(raw_ids.astype(str))
            fraction = contamination.get(batch) if isinstance(contamination, Mapping) else contamination
            _soupx_library(
                library, raw, cluster_key="_soupx_cluster", layer=None, contamination=fraction, **options,
            )
            blocks.append(library.layers[options["key_added"]])
            order.extend(positions)
            rho[positions] = library.obs["soupx_contamination"].to_numpy()
            contamination_by_sample[batch] = float(rho[positions[0]])
            feedback.info(f"SoupX | {batch} | estimated ambient fraction {contamination_by_sample[batch]:.1%}")
            del raw, library

    corrected = sparse.vstack(blocks, format="csr")[np.argsort(order)]
    adata.layers[backup_layer] = matrix.copy()
    adata.X = corrected
    if counts_layer is not None:
        adata.layers[counts_layer] = corrected.copy()
    adata.obs["soupx_contamination"] = rho
    adata.uns["ambient_correction"] = {
        "method": "SoupX",
        "backend": "soupx-python",
        "version": version("soupx-python"),
        "batch_key": batch_key,
        "contamination_by_sample": contamination_by_sample,
    }
    sync_obs(data, mod)
    return data


def _soupx_library(
    data,
    raw,
    *,
    cluster_key,
    mod="gex",
    layer="counts",
    raw_layer=None,
    backend="python",
    contamination=None,
    key_added="soupx_counts",
    random_state=0,
    estimate_kwargs=None,
    adjust_kwargs=None,
):
    """Correct one library with NiRuff's soupx-python (imported as soupx).

    Supply unfiltered droplets and provisional cluster labels. Features are
    aligned by name; raw must contain every filtered cell and gene. Corrected
    counts go into a layer. Call separately for each capture library.
    """
    if backend != "python":
        raise ValueError("SoupX uses soupx-python; remove backend='r' from this call")
    adata = modality(data, mod)
    raw = modality(raw, mod)
    matrix = counts(adata, layer)
    raw_matrix = counts(raw, raw_layer)
    if (
        not adata.var_names.isin(raw.var_names).all()
        or not adata.obs_names.isin(raw.obs_names).all()
    ):
        raise ValueError("Raw droplets must contain all filtered cell and feature identifiers")
    if contamination is not None and not 0 <= contamination <= 1:
        raise ValueError("contamination must be between zero and one")
    groups = _labels(adata, cluster_key)
    raw_matrix = raw_matrix[:, raw.var_names.get_indexer(adata.var_names)]
    estimate_kwargs, adjust_kwargs = dict(estimate_kwargs or {}), dict(adjust_kwargs or {})
    if "sc" in estimate_kwargs or "sc" in adjust_kwargs:
        raise ValueError("SoupX options cannot override the input channel 'sc'")
    if "round_to_int" in adjust_kwargs:
        if "roundToInt" in adjust_kwargs:
            raise ValueError("Specify only one of roundToInt and round_to_int")
        adjust_kwargs["roundToInt"] = adjust_kwargs.pop("round_to_int")
    soup = dependency("soupx", "ambient")
    channel = soup.SoupChannel(
        sparse.csr_matrix(raw_matrix.T),
        sparse.csr_matrix(matrix.T),
        gene_names=adata.var_names.tolist(),
    )
    if not channel.soupProfile["counts"].sum():
        raise ValueError("SoupX found no background droplets in its default 0 < UMI < 100 range")
    channel.setClusters(groups)
    if contamination is None:
        soup.autoEstCont(channel, **{"verbose": False, **estimate_kwargs})
    else:
        channel.set_contamination_fraction(contamination)
    corrected = soup.adjustCounts(
        channel, **{"roundToInt": True, "seed": random_state, "verbose": 0, **adjust_kwargs}
    ).T
    rho = np.asarray(channel.metaData["rho"], dtype=float)
    _output(adata, corrected, key_added)
    adata.obs["soupx_contamination"] = rho
    record(
        adata, key_added,
        {"layer": layer, "cluster_key": cluster_key, "random_state": random_state},
        backend="soupx-python",
    )
    sync_obs(data, mod)
    return data


def sctransform(
    data, *, mod="gex", layer="counts", backend="python", key_added="sct_residuals", **kwargs
):
    """Store SCTransform residuals and feature statistics using Python or R vst.

    Genes excluded by vst are represented by NaN in the layer and flagged by
    var['sct_modeled']; subset to modeled genes before PCA. Signed residuals
    are not normalized expression for marker testing or count models.
    """
    if backend not in {"python", "r"}:
        raise ValueError("backend must be 'python' or 'r'")
    adata = modality(data, mod)
    matrix = counts(adata, layer)
    if backend == "python":
        sct = dependency("sctransform", "normalization")
        fit = sct.vst(matrix.T, gene_names=list(adata.var_names), **kwargs)
        residuals = fit["y"]
        stats = fit["gene_attr"]
        record(adata, key_added, {"layer": layer, **kwargs}, backend="sctransform")
    else:
        fit = r._call(
            """function(x, genes, cells, options) {
            dimnames(x) <- list(genes, cells)
            fit <- do.call(sctransform::vst, c(list(umi=x), options))
            list(residuals=as.data.frame(fit$y), gene_attr=fit$gene_attr)
        }""",
            packages=["sctransform"],
            x=matrix.T,
            genes=adata.var_names,
            cells=adata.obs_names,
            options=kwargs,
        )
        residuals, stats = fit["residuals"], fit["gene_attr"]
        r.record_r(adata, key_added, {"layer": layer, **kwargs}, ["sctransform"])
    if not isinstance(residuals, pd.DataFrame):
        raise TypeError("Expected named genes-by-cells residuals from sctransform")
    # Python vst may use integer column labels; vst retains the input cell order.
    if residuals.shape[1] != adata.n_obs:
        raise ValueError("SCTransform changed the number of cells")
    residuals.columns = adata.obs_names
    adata.var["sct_modeled"] = adata.var_names.isin(residuals.index)
    _output(adata, residuals.reindex(adata.var_names).to_numpy().T, key_added)
    if isinstance(stats, pd.DataFrame):
        for col in stats:
            adata.var[f"sct_{col}"] = stats[col].reindex(adata.var_names)
    return data


def deviance_features(
    data, *, mod="gex", layer="counts", n_top_genes=4000, key_added="highly_deviant", **kwargs
):
    """Select genes using R scry binomial deviance on raw counts."""
    if n_top_genes < 1:
        raise ValueError("n_top_genes must be positive")
    adata = modality(data, mod)
    values = r._call(
        """function(x, options) {
        sce <- SingleCellExperiment::SingleCellExperiment(list(counts=x))
        sce <- do.call(scry::devianceFeatureSelection, c(list(object=sce, assay="counts"), options))
        SummarizedExperiment::rowData(sce)$binomial_deviance
    }""",
        packages=["scry", "SingleCellExperiment"],
        x=counts(adata, layer).T,
        options=kwargs,
    )
    values = np.asarray(values).ravel()
    selected = np.argsort(np.where(np.isfinite(values), values, -np.inf))[
        -min(n_top_genes, adata.n_vars) :
    ]
    mask = np.zeros(adata.n_vars, dtype=bool)
    mask[selected] = np.isfinite(values[selected])
    adata.var["binomial_deviance"], adata.var[key_added] = values, mask
    r.record_r(adata, key_added, {"layer": layer, "n_top_genes": n_top_genes}, ["scry"])
    return data


def scdblfinder(
    data, *, mod="gex", layer="counts", batch_key=None, random_state=0, backend="r", **kwargs
):
    """Detect doublets per capture; legacy R default or explicit Python backend.

    ``backend='python'`` calls scDblFinderPy directly in process. Neither branch
    filters cells or overwrites count/expression matrices.
    """
    if backend == "python":
        from ._python_qc import scdblfinder_py

        return scdblfinder_py(
            data, mod=mod, layer=layer, batch_key=batch_key, random_state=random_state, **kwargs
        )
    if backend != "r":
        raise ValueError("backend must be 'python' or 'r'")
    adata = modality(data, mod)
    samples = None if batch_key is None else _labels(adata, batch_key)
    result = r._call(
        """function(x, samples, seed, options) {
        set.seed(seed)
        sce <- SingleCellExperiment::SingleCellExperiment(list(counts=x))
        sce <- do.call(scDblFinder::scDblFinder, c(list(sce=sce, samples=samples,
                        BPPARAM=BiocParallel::SerialParam()), options))
        list(score=sce$scDblFinder.score, class=as.character(sce$scDblFinder.class))
    }""",
        packages=["scDblFinder", "SingleCellExperiment", "BiocParallel"],
        x=counts(adata, layer).T,
        samples=samples,
        seed=random_state,
        options=kwargs,
    )
    adata.obs["scDblFinder_score"] = np.asarray(result["score"]).ravel()
    adata.obs["scDblFinder_class"] = pd.Categorical(result["class"])
    r.record_r(
        adata,
        "scdblfinder",
        {"layer": layer, "batch_key": batch_key, "random_state": random_state},
        ["scDblFinder"],
    )
    sync_obs(data, mod)
    return data
