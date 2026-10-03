"""Count-preserving preprocessing methods used in Single-cell Best Practices."""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

from .. import r
from .._core import counts, dependency, modality, record, sync_obs


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
    """Correct one library with SoupX, using pysoupx by default or R SoupX explicitly.

    raw contains unfiltered droplets. Features are aligned by name; raw must
    contain every filtered cell and gene. Call separately for each library.
    """
    if backend not in {"python", "r"}:
        raise ValueError("backend must be 'python' or 'r'")
    adata = modality(data, mod)
    raw = modality(raw, mod)
    matrix = counts(adata, layer)
    counts(raw, raw_layer)
    if (
        not adata.var_names.isin(raw.var_names).all()
        or not adata.obs_names.isin(raw.obs_names).all()
    ):
        raise ValueError("Raw droplets must contain all filtered cell and feature identifiers")
    if contamination is not None and not 0 <= contamination <= 1:
        raise ValueError("contamination must be between zero and one")
    groups = _labels(adata, cluster_key)
    raw_matrix = counts(raw, raw_layer)[:, raw.var_names.get_indexer(adata.var_names)]
    estimate_kwargs, adjust_kwargs = dict(estimate_kwargs or {}), dict(adjust_kwargs or {})
    if "sc" in estimate_kwargs or "sc" in adjust_kwargs:
        raise ValueError("SoupX options cannot override the input channel 'sc'")
    if backend == "python":
        soup = dependency("pysoupx", "ambient")
        working = AnnData(matrix.copy(), obs=adata.obs.copy(), var=adata.var.copy())
        droplets = AnnData(raw_matrix.copy(), obs=raw.obs.copy(), var=adata.var.copy())
        channel = soup.SoupChannel.from_anndata(working, raw=droplets, cluster_key=cluster_key)
        if contamination is None:
            channel = soup.auto_est_cont(channel, **estimate_kwargs)
        else:
            channel = soup.set_contamination_fraction(channel, contamination)
        corrected = soup.adjust_counts(
            channel, **{"round_to_int": True, "seed": random_state, **adjust_kwargs}
        ).T
        rho = np.asarray(channel.meta_data["rho"], dtype=float)
        record(
            adata,
            key_added,
            {"layer": layer, "cluster_key": cluster_key, "random_state": random_state},
            backend="pysoupx",
        )
    else:
        result = r._call(
            """function(x, raw, genes, cells, droplets, clusters, rho,
                                     seed, estimate_options, adjust_options) {
            set.seed(seed)
            dimnames(x) <- list(genes, cells); dimnames(raw) <- list(genes, droplets)
            channel <- SoupX::SoupChannel(raw, x, calcSoupProfile=TRUE)
            channel <- SoupX::setClusters(channel, setNames(clusters, cells))
            if (is.null(rho)) channel <- do.call(SoupX::autoEstCont,
                c(list(sc=channel), utils::modifyList(list(doPlot=FALSE), estimate_options,
                                                     keep.null=TRUE)))
            else channel <- SoupX::setContaminationFraction(channel, rho)
            out <- do.call(SoupX::adjustCounts,
                c(list(sc=channel), utils::modifyList(list(roundToInt=TRUE), adjust_options,
                                                     keep.null=TRUE)))
            list(counts=out, rho=channel$metaData$rho)
        }""",
            packages=["SoupX"],
            x=matrix.T,
            raw=raw_matrix.T,
            genes=adata.var_names,
            cells=adata.obs_names,
            droplets=raw.obs_names,
            clusters=groups,
            rho=contamination,
            seed=random_state,
            estimate_options=estimate_kwargs,
            adjust_options=adjust_kwargs,
        )
        corrected, rho = result["counts"].T, np.asarray(result["rho"]).ravel()
        r.record_r(
            adata,
            key_added,
            {"layer": layer, "cluster_key": cluster_key, "random_state": random_state},
            ["SoupX"],
        )
    _output(adata, corrected, key_added)
    adata.obs["soupx_contamination"] = rho
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


def scdblfinder(data, *, mod="gex", layer="counts", batch_key=None, random_state=0, **kwargs):
    """Run R scDblFinder by capture batch; retain every cell and the raw counts."""
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
