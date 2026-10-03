"""Python-first adapters for annotation, integration, trajectories and mechanisms."""

from __future__ import annotations

import numpy as np

from .._core import counts, dependency, modality, record, sync_obs
from ..preprocessing._best_practices import _labels


def celltypist(data, model, *, mod="gex", layer=None, key_added="celltypist", **kwargs):
    """Annotate log1p(10,000-normalized) RNA with an explicit local CellTypist model."""
    ct = dependency("celltypist", "annotation")
    adata = modality(data, mod)
    working = adata.copy()
    if layer is not None:
        working.X = working.layers[layer].copy()
    prediction = ct.annotate(working, model=model, **kwargs)
    for column in prediction.predicted_labels:
        adata.obs[f"{key_added}_{column}"] = prediction.predicted_labels[column].reindex(
            adata.obs_names
        )
    adata.obsm[f"{key_added}_probabilities"] = prediction.probability_matrix.reindex(
        adata.obs_names
    )
    record(adata, key_added, {"layer": layer}, backend="celltypist")
    sync_obs(data, mod)
    return prediction


def harmony(data, *, batch_key, mod="gex", use_rep="X_pca", key_added="X_pca_harmony", **kwargs):
    """Correct an embedding using harmonypy; expression and count assays stay intact."""
    hm = dependency("harmonypy", "batch")
    adata = modality(data, mod)
    keys = [batch_key] if isinstance(batch_key, str) else list(batch_key)
    for key in keys:
        _labels(adata, key)
    embedding = np.asarray(adata.obsm[use_rep])
    fit = hm.run_harmony(embedding, adata.obs, keys, **kwargs)
    corrected = np.asarray(fit.Z_corr)
    # harmonypy <2 returns dimensions-by-cells; >=2 returns cells-by-dimensions.
    if corrected.shape != embedding.shape and corrected.T.shape == embedding.shape:
        corrected = corrected.T
    if corrected.shape != embedding.shape:
        raise ValueError("Harmony returned an incompatible embedding shape")
    adata.obsm[key_added] = corrected
    record(adata, key_added, {"batch_key": keys, "use_rep": use_rep}, backend="harmonypy")
    return fit


def bbknn(data, *, batch_key, mod="gex", **kwargs):
    """Construct the Python BBKNN batch-balanced graph in the selected modality."""
    module = dependency("bbknn", "batch")
    adata = modality(data, mod)
    _labels(adata, batch_key)
    module.bbknn(adata, batch_key=batch_key, **kwargs)
    record(adata, "bbknn", {"batch_key": batch_key}, backend="bbknn")
    return data


def tsne(data, *, mod="gex", **kwargs):
    import scanpy as sc

    sc.tl.tsne(modality(data, mod), **kwargs)
    return data


def velocity(
    data,
    *,
    mod="gex",
    mode="stochastic",
    preprocess=True,
    preprocessing_kwargs=None,
    moments_kwargs=None,
    dynamics_kwargs=None,
    velocity_kwargs=None,
    graph_kwargs=None,
):
    """Run scVelo on a copy while preserving the input assays.

    Requires spliced and unspliced layers. Returns the working AnnData, including
    normalized splicing layers and any feature filtering performed by scVelo.
    """
    scv = dependency("scvelo", "trajectory")
    adata = modality(data, mod)
    if mode not in {"deterministic", "stochastic", "dynamical"}:
        raise ValueError("Unsupported RNA velocity mode")
    for layer in ("spliced", "unspliced"):
        counts(adata, layer)
    working = adata.copy()
    if preprocess:
        scv.pp.filter_and_normalize(working, **(preprocessing_kwargs or {}))
    scv.pp.moments(working, **(moments_kwargs or {}))
    if mode == "dynamical":
        scv.tl.recover_dynamics(working, **(dynamics_kwargs or {}))
    scv.tl.velocity(working, mode=mode, **(velocity_kwargs or {}))
    scv.tl.velocity_graph(working, **(graph_kwargs or {}))
    record(working, "velocity", {"mode": mode, "preprocess": preprocess}, backend="scvelo")
    return working


def palantir(
    data,
    *,
    root,
    mod="gex",
    use_rep="X_pca",
    diffusion_kwargs=None,
    multiscale_kwargs=None,
    **kwargs,
):
    """Infer Palantir pseudotime, entropy and branch probabilities from a named root cell."""
    module = dependency("palantir", "trajectory")
    adata = modality(data, mod)
    if root not in adata.obs_names:
        raise ValueError("root must name an observed cell")
    module.utils.run_diffusion_maps(adata, pca_key=use_rep, **(diffusion_kwargs or {}))
    module.utils.determine_multiscale_space(adata, **(multiscale_kwargs or {}))
    result = module.core.run_palantir(adata, root, **kwargs)
    record(adata, "palantir", {"root": root, "use_rep": use_rep}, backend="palantir")
    sync_obs(data, mod)
    return result


def milo(
    data,
    *,
    sample_key,
    design,
    mod="gex",
    prop=0.1,
    neighbors_kwargs=None,
    da_kwargs=None,
    graph_kwargs=None,
):
    """Run Pertpy Milo using Python PyDESeq2 by default; return MuData and native model."""
    import scanpy as sc

    pt = dependency("pertpy", "composition")
    adata = modality(data, mod)
    _labels(adata, sample_key)
    model = pt.tl.Milo()
    working = model.load(adata.copy())
    sc.pp.neighbors(working.mod["rna"], **(neighbors_kwargs or {}))
    neighbor_key = (neighbors_kwargs or {}).get("key_added")
    model.make_nhoods(working.mod["rna"], prop=prop, neighbors_key=neighbor_key)
    working = model.count_nhoods(working, sample_col=sample_key)
    model.da_nhoods(working, design=design, **{"solver": "pydeseq2", **(da_kwargs or {})})
    basis = (
        "X_umap"
        if "X_umap" in working.mod["rna"].obsm
        else working.mod["rna"].uns[neighbor_key or "neighbors"]["params"].get("use_rep", "X_pca")
    )
    model.build_nhood_graph(working, **{"basis": basis, **(graph_kwargs or {})})
    record(
        working,
        "milo",
        {"sample_key": sample_key, "design": design, "prop": prop},
        backend="pertpy",
    )
    return working, model


def sccoda(
    data,
    *,
    sample_key,
    cell_type_key,
    formula,
    covariate_keys,
    reference_cell_type="automatic",
    mod="gex",
    tree_levels=None,
    prepare_kwargs=None,
    sampling_kwargs=None,
    random_state=0,
):
    """Fit scCODA or tree-aware tascCODA using Pertpy's Python/JAX implementation.

    Each sample covariate must be constant within that biological sample.
    Returns sample-level MuData and the native model for posterior diagnostics.
    """
    pt = dependency("pertpy", "composition")
    adata = modality(data, mod)
    for key in (sample_key, cell_type_key, *covariate_keys):
        _labels(adata, key)
    for key in covariate_keys:
        if adata.obs.groupby(sample_key, observed=True)[key].nunique().gt(1).any():
            raise ValueError(f"Covariate {key!r} must be constant within each sample")
    if adata.obs[sample_key].nunique() < 3:
        raise ValueError("Composition inference requires biological sample replication")
    model = pt.tl.Sccoda() if tree_levels is None else pt.tl.Tasccoda()
    options = (
        {} if tree_levels is None else {"levels_orig": list(tree_levels), "add_level_name": True}
    )
    working = model.load(
        adata.copy(),
        type="cell_level",
        cell_type_identifier=cell_type_key,
        sample_identifier=sample_key,
        covariate_obs=list(covariate_keys),
        **options,
    )
    prepared = model.prepare(
        working,
        modality_key="coda",
        formula=formula,
        reference_cell_type=reference_cell_type,
        **(prepare_kwargs or {}),
    )
    if prepared is not None:
        working = prepared
    model.run_nuts(working, modality_key="coda", rng_key=random_state, **(sampling_kwargs or {}))
    record(
        working,
        "sccoda",
        {
            "sample_key": sample_key,
            "formula": formula,
            "tree": tree_levels is not None,
            "random_state": random_state,
        },
        backend="pertpy",
    )
    return working, model


def mixscape(
    data,
    *,
    perturbation_key,
    target_key,
    control,
    mod="gex",
    split_by=None,
    signature_kwargs=None,
    model_kwargs=None,
):
    """Infer perturbation signatures and classify successful knockouts with Pertpy."""
    pt = dependency("pertpy", "perturbation")
    adata = modality(data, mod)
    _labels(adata, perturbation_key)
    _labels(adata, target_key)
    model = pt.tl.Mixscape()
    model.perturbation_signature(
        adata,
        pert_key=perturbation_key,
        control=control,
        split_by=split_by,
        **(signature_kwargs or {}),
    )
    model.mixscape(
        adata,
        pert_key=target_key,
        control=control,
        layer="X_pert",
        split_by=split_by,
        **(model_kwargs or {}),
    )
    record(
        adata,
        "mixscape",
        {
            "perturbation_key": perturbation_key,
            "target_key": target_key,
            "control": control,
            "split_by": split_by,
        },
        backend="pertpy",
    )
    sync_obs(data, mod)
    return model


def perturbation_distance(
    data,
    *,
    groupby,
    control,
    mod="gex",
    use_rep="X_pca",
    metric="edistance",
    n_perms=None,
    **kwargs,
):
    """Compare perturbations to controls with Pertpy distances or permutation tests."""
    pt = dependency("pertpy", "perturbation")
    adata = modality(data, mod)
    _labels(adata, groupby)
    if n_perms is None:
        model = pt.tl.Distance(metric, obsm_key=use_rep)
        table = model.onesided_distances(adata, groupby=groupby, selected_group=control, **kwargs)
    else:
        model = pt.tl.DistanceTest(metric, n_perms=n_perms, obsm_key=use_rep)
        table = model(adata, groupby=groupby, contrast=control, **kwargs)
    return table, model


def communication(
    data,
    *,
    groupby,
    mod="gex",
    method="rank_aggregate",
    layer=None,
    key_added="liana_res",
    **kwargs,
):
    """Infer ligand-receptor communication using LIANA's Python implementations."""
    li = dependency("liana", "communication")
    if method not in {
        "rank_aggregate",
        "cellphonedb",
        "connectome",
        "natmi",
        "logfc",
        "singlecellsignalr",
        "cellchat",
        "geometric_mean",
    }:
        raise ValueError("Unsupported LIANA method")
    adata = modality(data, mod)
    _labels(adata, groupby)
    getattr(li.mt, method)(
        adata,
        groupby=groupby,
        layer=layer,
        use_raw=False,
        key_added=key_added,
        inplace=True,
        **kwargs,
    )
    record(
        adata, key_added, {"groupby": groupby, "layer": layer, "method": method}, backend="liana"
    )
    return adata.uns[key_added]


def grn(data, *, tf_names, mod="gex", layer=None, method="grnboost2", **kwargs):
    """Infer expression-based TF edges with Arboreto; return its native edge table.

    Use scenic_regulons for motif pruning, then regulon_activity or
    cellscope.tl.aucell/activity for network scores. Arboreto uses a dense matrix.
    """
    module = dependency("arboreto.algo", "grn")
    adata = modality(data, mod)
    if method not in {"grnboost2", "genie3"}:
        raise ValueError("method must be 'grnboost2' or 'genie3'")
    matrix = adata.X if layer is None else adata.layers[layer]
    matrix = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    table = getattr(module, method)(
        expression_data=matrix, gene_names=list(adata.var_names), tf_names=list(tf_names), **kwargs
    )
    record(adata, "grn", {"layer": layer, "method": method}, backend="arboreto")
    return table


def scenic_regulons(
    data,
    adjacencies,
    *,
    ranking_databases,
    motif_annotations,
    mod="gex",
    layer=None,
    module_kwargs=None,
    prune_kwargs=None,
):
    """Prune expression edges with pySCENIC motifs; return motif table and regulons.

    ranking_databases maps database labels to local cisTarget Feather paths.
    The gene identifiers and genome build must match the expression and motifs.
    Uses dense expression; no reference database is downloaded automatically.
    """
    from pathlib import Path

    import pandas as pd

    try:
        utils = dependency("pyscenic.utils", "grn")
        prune = dependency("pyscenic.prune", "grn")
    except AttributeError as error:
        if "numpy" not in str(error) or "object" not in str(error):
            raise
        raise ImportError(
            "Installed pySCENIC uses removed NumPy aliases. Install its corrected "
            "official source revision; see docs/best_practices.md."
        ) from error
    rnkdb = dependency("ctxcore.rnkdb", "grn")
    adata = modality(data, mod)
    if not {"TF", "target", "importance"}.issubset(adjacencies.columns):
        raise ValueError("adjacencies requires Arboreto TF, target and importance columns")
    if not ranking_databases:
        raise ValueError("Provide at least one local cisTarget ranking database")
    paths = [*ranking_databases.values(), motif_annotations]
    for path in paths:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    dbs = [
        rnkdb.FeatherRankingDatabase(fname=str(path), name=name)
        for name, path in ranking_databases.items()
    ]
    matrix = adata.X if layer is None else adata.layers[layer]
    matrix = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    expression = pd.DataFrame(matrix, index=adata.obs_names, columns=adata.var_names)
    modules = list(utils.modules_from_adjacencies(adjacencies, expression, **(module_kwargs or {})))
    motifs = prune.prune2df(dbs, modules, str(motif_annotations), **(prune_kwargs or {}))
    regulons = prune.df2regulons(motifs)
    record(
        adata,
        "scenic_regulons",
        {"layer": layer, "databases": list(ranking_databases)},
        backend="pyscenic",
    )
    return motifs, regulons


def regulon_activity(data, regulons, *, mod="gex", layer=None, key_added="X_regulon_auc", **kwargs):
    """Score native pySCENIC regulons with its AUCell implementation."""
    import pandas as pd

    backend = dependency("pyscenic.aucell", "grn")
    adata = modality(data, mod)
    matrix = adata.X if layer is None else adata.layers[layer]
    matrix = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    expression = pd.DataFrame(matrix, index=adata.obs_names, columns=adata.var_names)
    table = backend.aucell(expression, regulons, **kwargs).reindex(adata.obs_names)
    adata.obsm[key_added] = table
    record(adata, key_added, {"layer": layer, "regulons": list(table.columns)}, backend="pyscenic")
    return table
