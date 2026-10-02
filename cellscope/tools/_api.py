"""Scanpy workflows, native model adapters, and sample-level inference."""

from __future__ import annotations

import pandas as pd

from .._core import counts, dependency, modality, record, sync_obs
from ._differential import differential_expression  # noqa: F401


def pca(data, *, mod="gex", n_comps=50, **kwargs):
    import scanpy as sc

    adata = modality(data, mod)
    mask = kwargs.get("mask_var")
    n_vars = int(adata.var[mask].sum()) if isinstance(mask, str) else adata.n_vars
    n_comps = min(n_comps, adata.n_obs - 1, n_vars - 1)
    if n_comps < 1:
        raise ValueError("PCA requires at least two cells and two selected genes")
    sc.pp.pca(adata, n_comps=n_comps, **kwargs)
    return data


def umap(data, *, mod="gex", **kwargs):
    import scanpy as sc

    sc.tl.umap(modality(data, mod), **kwargs)
    return data


def leiden(data, *, mod="gex", resolution=1.0, key_added="leiden", **kwargs):
    import scanpy as sc

    sc.tl.leiden(modality(data, mod), resolution=resolution, key_added=key_added, **kwargs)
    sync_obs(data, mod)
    return data


def workflow(
    data,
    *,
    mod="gex",
    layer=None,
    n_top_genes=2000,
    n_pcs=30,
    n_neighbors=15,
    resolution=1.0,
    batch_key=None,
    random_state=0,
):
    """Run normalization, HVG, PCA, neighbors, UMAP, and Leiden on one modality."""
    from .. import pp

    pp.normalize(data, mod=mod, layer=layer)
    pp.highly_variable_genes(data, mod=mod, n_top_genes=n_top_genes, batch_key=batch_key)
    pca(data, mod=mod, n_comps=n_pcs, mask_var="highly_variable", random_state=random_state)
    adata = modality(data, mod)
    pp.neighbors(
        data,
        mod=mod,
        n_neighbors=min(n_neighbors, adata.n_obs - 1),
        n_pcs=adata.obsm["X_pca"].shape[1],
        random_state=random_state,
    )
    umap(data, mod=mod, random_state=random_state)
    leiden(
        data,
        mod=mod,
        resolution=resolution,
        flavor="igraph",
        n_iterations=2,
        random_state=random_state,
    )
    record(
        adata,
        "workflow",
        {
            "random_state": random_state,
            "resolution": resolution,
            "n_pcs": n_pcs,
            "n_neighbors": n_neighbors,
        },
        backend="scanpy",
    )
    return data


def markers(data, *, groupby, mod="gex", key_added="rank_genes_groups", **kwargs):
    import scanpy as sc

    adata = modality(data, mod)
    kwargs.setdefault("use_raw", False)
    kwargs.setdefault("method", "wilcoxon")
    sc.tl.rank_genes_groups(adata, groupby=groupby, key_added=key_added, **kwargs)
    record(adata, key_added, {"groupby": groupby, **kwargs}, backend="scanpy")
    return sc.get.rank_genes_groups_df(adata, group=None, key=key_added)


def annotate(data, mapping, *, reference_key="leiden", key_added="cell_type", mod="gex"):
    """Map clusters to labels; retain unannotated cells as missing labels."""
    adata = modality(data, mod)
    adata.obs[key_added] = adata.obs[reference_key].map(mapping).astype("category")
    sync_obs(data, mod)
    return data


def score_genes(data, gene_sets, *, mod="gex", **kwargs):
    import scanpy as sc

    adata = modality(data, mod)
    kwargs.setdefault("use_raw", False)
    for name, genes in gene_sets.items():
        sc.tl.score_genes(adata, gene_list=list(genes), score_name=name, **kwargs)
    sync_obs(data, mod)
    return data


def aucell(data, gene_list=None, *, network=None, mod="gex", use_raw=False, **kwargs):
    """Use decoupler 2.x AUCell; preserve native scores in obsm['score_aucell']."""
    dc = dependency("decoupler", "functional")
    adata = modality(data, mod)
    if network is None:
        if not gene_list:
            raise ValueError("Supply gene_list or a source/target network")
        network = pd.DataFrame({"source": "gene_set", "target": list(gene_list)})
    kwargs.setdefault("tmin", 1)
    dc.mt.aucell(adata, network, raw=use_raw, **kwargs)
    record(adata, "aucell", {"use_raw": use_raw}, backend="decoupler")
    scores = adata.obsm["score_aucell"]
    return scores.iloc[:, 0].tolist() if gene_list is not None else scores


def scvi(
    data,
    *,
    mod="gex",
    layer="counts",
    batch_key=None,
    key_added="X_scVI",
    model_kwargs=None,
    train_kwargs=None,
):
    backend = dependency("scvi", "integration")
    adata = modality(data, mod)
    counts(adata, layer)
    backend.model.SCVI.setup_anndata(adata, layer=layer, batch_key=batch_key)
    model = backend.model.SCVI(adata, **(model_kwargs or {}))
    model.train(**(train_kwargs or {}))
    adata.obsm[key_added] = model.get_latent_representation()
    record(adata, key_added, {"layer": layer, "batch_key": batch_key}, backend="scvi-tools")
    return model


def scanvi(
    data,
    *,
    labels_key="cell_type",
    unlabeled_category="Unknown",
    mod="gex",
    layer="counts",
    batch_key=None,
    model_kwargs=None,
    train_kwargs=None,
    key_added="predicted_cell_type",
):
    """Train semi-supervised scANVI; retain native model and label probabilities."""
    backend = dependency("scvi", "integration")
    adata = modality(data, mod)
    counts(adata, layer)
    if adata.obs[labels_key].isna().any():
        raise ValueError("Represent unlabeled cells with unlabeled_category, not missing values")
    backend.model.SCANVI.setup_anndata(
        adata, layer=layer, batch_key=batch_key, labels_key=labels_key
    )
    model = backend.model.SCANVI(
        adata, unlabeled_category=unlabeled_category, **(model_kwargs or {})
    )
    model.train(**(train_kwargs or {}))
    adata.obs[key_added] = model.predict()
    adata.obsm[f"{key_added}_probabilities"] = model.predict(soft=True)
    adata.obsm["X_scANVI"] = model.get_latent_representation()
    sync_obs(data, mod)
    record(
        adata,
        key_added,
        {"labels_key": labels_key, "unlabeled_category": unlabeled_category},
        backend="scvi-tools",
    )
    return model


def reference_mapping(
    data, reference_model, *, mod="gex", train_kwargs=None, key_added="X_reference"
):
    """Adapt a native scvi-tools model to query cells using its public scArches API."""
    adata = modality(data, mod)
    cls = type(reference_model)
    cls.prepare_query_anndata(adata, reference_model)
    model = cls.load_query_data(adata, reference_model)
    model.train(**(train_kwargs or {}))
    adata.obsm[key_added] = model.get_latent_representation()
    record(adata, key_added, {}, backend="scvi-tools")
    return model


def activity(data, network, *, method="ulm", mod="gex", use_raw=False, **kwargs):
    """Infer pathway or TF activities using an explicit decoupler network."""
    if method not in {"ulm", "mlm", "aucell", "ora", "gsva", "gsea", "viper"}:
        raise ValueError("Unsupported decoupler activity method")
    dc = dependency("decoupler", "functional")
    adata = modality(data, mod)
    getattr(dc.mt, method)(adata, network, raw=use_raw, **kwargs)
    record(adata, f"activity_{method}", {"method": method, "use_raw": use_raw}, backend="decoupler")
    return adata.obsm[f"score_{method}"]


def pseudobulk(
    data,
    *,
    sample_col="sample_id",
    groups_col="cell_type",
    mod="gex",
    layer="counts",
    min_cells=10,
    metadata_cols=(),
    **kwargs,
):
    """Aggregate raw counts with decoupler; attach validated sample covariates."""
    dc = dependency("decoupler", "differential")
    adata = modality(data, mod)
    counts(adata, layer)
    grouping_cols = [sample_col, *([groups_col] if groups_col is not None else [])]
    metadata_cols = list(dict.fromkeys(col for col in metadata_cols if col != sample_col))
    for col in (*grouping_cols, *metadata_cols):
        if col not in adata.obs or adata.obs[col].isna().any():
            raise ValueError(f"Nonmissing observation metadata required: {col}")
    metadata = adata.obs[[sample_col, *metadata_cols]].drop_duplicates()
    if metadata[sample_col].duplicated().any():
        raise ValueError("Sample covariates must be constant within each sample")
    from anndata import AnnData

    # Grouping comparisons must yield NumPy masks for decoupler sparse
    # indexing, including with Pandas 3/Arrow-backed string columns.
    working = AnnData(X=counts(adata, layer), obs=adata.obs.copy(), var=adata.var.copy())
    for col in grouping_cols:
        working.obs[col] = pd.Categorical(working.obs[col])
    result = dc.pp.pseudobulk(
        working, sample_col=sample_col, groups_col=groups_col, layer=None, mode="sum", **kwargs
    )
    if "psbulk_cells" in result.obs:
        result = result[result.obs["psbulk_cells"].ge(min_cells)].copy()
    indexed = metadata.set_index(sample_col)
    for col in metadata_cols:
        result.obs[col] = result.obs[sample_col].map(indexed[col])
    record(
        result,
        "pseudobulk",
        {
            "sample_col": sample_col,
            "groups_col": groups_col,
            "layer": layer,
            "min_cells": min_cells,
        },
        backend="decoupler",
    )
    return result


def trajectory(data, *, mod="gex", groupby="leiden", root=None, key_added="dpt_pseudotime"):
    """Compute PAGA; optionally run diffusion pseudotime from a named root cell."""
    import scanpy as sc

    adata = modality(data, mod)
    sc.tl.paga(adata, groups=groupby)
    if root is not None:
        if root not in adata.obs_names:
            raise KeyError(root)
        adata.uns["iroot"] = int(adata.obs_names.get_loc(root))
        sc.tl.diffmap(adata)
        sc.tl.dpt(adata)
        adata.obs[key_added] = adata.obs["dpt_pseudotime"]
    record(adata, "trajectory", {"groupby": groupby, "root": root}, backend="scanpy")
    sync_obs(data, mod)
    return data
