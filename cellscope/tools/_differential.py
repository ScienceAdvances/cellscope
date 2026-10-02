"""Cell-level marker discovery and sample-level differential expression adapters."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pandas as pd
from scipy import sparse

from .._core import counts, dependency, modality, record

_CELL_METHODS = {
    "wilcox": "wilcoxon",
    "wilcoxon": "wilcoxon",
    "t": "t-test",
    "t-test": "t-test",
    "t-test_overestim_var": "t-test_overestim_var",
    "logreg": "logreg",
}
_BULK_METHODS = {
    "pydeseq2": "pydeseq2",
    "deseq2": "pydeseq2",
    "pylimma": "pylimma",
    "limma": "pylimma",
    "limma_voom": "pylimma",
    "edgepython": "edgepython_ql",
    "edgepython_ql": "edgepython_ql",
    "edger": "edgepython_ql",
    "edgepython_lrt": "edgepython_lrt",
}


def de_methods():
    """Describe supported methods, their statistical units, and installation extras."""
    return pd.DataFrame(
        [
            ("wilcoxon", "cell", "scanpy", "base", "rank-sum test"),
            ("t-test", "cell", "scanpy", "base", "Welch t-test"),
            ("t-test_overestim_var", "cell", "scanpy", "base", "conservative t-test"),
            ("logreg", "cell", "scikit-learn", "base", "ranking only; no p-values"),
            ("pydeseq2", "sample", "pydeseq2", "differential", "negative-binomial Wald"),
            ("pylimma", "sample", "pylimma", "limma", "TMM + voom + moderated t"),
            ("edgepython_ql", "sample", "edgepython", "edger", "TMM + QL F-test"),
            ("edgepython_lrt", "sample", "edgepython", "edger", "TMM + GLM LRT"),
        ],
        columns=["method", "unit", "backend", "extra", "test"],
    )


def _labels(value):
    return [value] if pd.api.types.is_scalar(value) else list(value)


def _cell_method(method):
    if method not in _CELL_METHODS:
        raise ValueError(
            f"Unknown cell-level method {method!r}. Use {list(_CELL_METHODS)}; "
            "count-model methods belong in differential_expression on pseudobulk profiles."
        )
    return _CELL_METHODS[method]


def _marker_comparison(
    adata,
    *,
    groupby,
    ident_1,
    ident_2,
    method,
    layer,
    use_raw,
    features,
    min_pct,
    min_diff_pct,
    logfc_threshold,
    only_pos,
    min_cells,
    corr_method,
    backend_kwargs,
):
    import scanpy as sc
    from anndata import AnnData

    if groupby not in adata.obs:
        raise KeyError(groupby)
    if adata.obs[groupby].isna().any():
        raise ValueError("Marker grouping annotations must be nonmissing")
    labels = adata.obs[groupby]
    first = _labels(ident_1)
    second = None if ident_2 is None else _labels(ident_2)
    if not first or (second is not None and not second):
        raise ValueError("Identity selections must be nonempty")
    available = set(labels.unique())
    if not set(first).issubset(available) or (
        second is not None and not set(second).issubset(available)
    ):
        raise ValueError("Requested identity is absent from the grouping annotations")
    if second is not None and set(first).intersection(second):
        raise ValueError("Comparison and reference identities must not overlap")
    mask1 = labels.isin(first).to_numpy()
    mask2 = (~mask1) if second is None else labels.isin(second).to_numpy()
    if min(mask1.sum(), mask2.sum()) < min_cells:
        raise ValueError(f"Each marker comparison needs at least {min_cells} cells per side")
    if use_raw and layer is not None:
        raise ValueError("Choose either use_raw=True or layer, not both")
    source = adata.raw if use_raw else adata
    if source is None:
        raise ValueError("use_raw=True requires adata.raw")
    matrix = source.X if layer is None else source.layers[layer]
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Marker analysis expects finite nonnegative log1p expression")
    if layer == "counts" or (
        "log1p" not in adata.uns and np.any(values > 0) and np.allclose(values, np.rint(values))
    ):
        raise ValueError(
            "Normalize/log1p expression before marker discovery; use counts for pseudobulk"
        )
    pct1 = np.asarray((matrix[mask1] > 0).mean(axis=0)).ravel()
    pct2 = np.asarray((matrix[mask2] > 0).mean(axis=0)).ravel()
    keep = (np.maximum(pct1, pct2) >= min_pct) & (np.abs(pct1 - pct2) >= min_diff_pct)
    if features is not None:
        requested = pd.Index(list(features))
        if not requested.isin(source.var_names).all():
            raise ValueError("Some requested features are absent from the expression matrix")
        keep &= source.var_names.isin(requested)
    columns = [
        "gene",
        "group",
        "comparison",
        "reference",
        "log2FoldChange",
        "pvalue",
        "padj",
        "scores",
        "pct_nz_group",
        "pct_nz_reference",
        "method",
        "unit",
    ]
    if not keep.any():
        return pd.DataFrame(columns=columns)
    selected = mask1 | mask2
    working = AnnData(
        X=matrix[selected][:, keep].copy(),
        obs=adata.obs.loc[selected].copy(),
        var=source.var.loc[keep].copy(),
    )
    working.obs["_cellscope_comparison"] = pd.Categorical(
        np.where(mask1[selected], "comparison", "reference"),
        categories=["reference", "comparison"],
    )
    if "log1p" in adata.uns:
        working.uns["log1p"] = adata.uns["log1p"].copy()
    if method == "logreg":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(**{"max_iter": 1000, **backend_kwargs})
        model.fit(working.X, mask1[selected].astype(int))
        table = pd.DataFrame({"names": working.var_names, "scores": model.coef_[0]})
        table["pvals"] = np.nan
        table["pvals_adj"] = np.nan
    else:
        reserved = {"groupby", "groups", "reference", "method", "use_raw", "n_genes", "corr_method"}
        if reserved.intersection(backend_kwargs):
            raise ValueError("backend_kwargs cannot override marker comparison settings")
        sc.tl.rank_genes_groups(
            working,
            groupby="_cellscope_comparison",
            groups=["comparison"],
            reference="reference",
            method=method,
            use_raw=False,
            n_genes=working.n_vars,
            corr_method=corr_method,
            **backend_kwargs,
        )
        table = sc.get.rank_genes_groups_df(working, group="comparison")
    # Effect size uses log2(mean(expm1(expression)) + epsilon), rather than
    # Scanpy's exp(mean(log-expression)) approximation. Input must be log1p.
    base = adata.uns.get("log1p", {}).get("base")
    multiplier = 1.0 if base is None else np.log(base)
    linear = working.X.astype(float).copy()
    if sparse.issparse(linear):
        linear.data = np.expm1(linear.data * multiplier)
    else:
        linear = np.expm1(linear * multiplier)
    mean1 = np.asarray(linear[mask1[selected]].mean(axis=0)).ravel()
    mean2 = np.asarray(linear[mask2[selected]].mean(axis=0)).ravel()
    lfc = np.log2((mean1 + 1e-9) / (mean2 + 1e-9))
    metrics = pd.DataFrame(
        {"log2FoldChange": lfc, "pct_nz_group": pct1[keep], "pct_nz_reference": pct2[keep]},
        index=working.var_names,
    )
    table = table.rename(columns={"names": "gene", "pvals": "pvalue", "pvals_adj": "padj"})
    table = table.drop(columns=["pct_nz_group", "pct_nz_reference"], errors="ignore")
    table = table.join(metrics, on="gene")
    table["group"] = "+".join(map(str, first))
    table["comparison"] = table["group"]
    table["reference"] = "rest" if second is None else "+".join(map(str, second))
    table["method"] = method
    table["unit"] = "cell"
    keep_rows = (
        table["log2FoldChange"].ge(logfc_threshold)
        if only_pos
        else (table["log2FoldChange"].abs().ge(logfc_threshold))
    )
    return (
        table.loc[keep_rows, columns].sort_values("scores", ascending=False).reset_index(drop=True)
    )


def find_markers(
    data,
    *,
    groupby,
    ident_1,
    ident_2=None,
    method="wilcoxon",
    mod="gex",
    layer=None,
    use_raw=False,
    features=None,
    min_pct=0.01,
    min_diff_pct=0.0,
    logfc_threshold=0.0,
    only_pos=False,
    min_cells=3,
    corr_method="benjamini-hochberg",
    key_added="find_markers",
    backend_kwargs=None,
):
    """Find markers for one identity (or pooled identities) versus another/rest.

    Input is nonnegative log1p-normalized expression, not scaled/integrated
    values or raw counts. Detection filters precede testing; fold-change
    filtering follows adjustment over the tested genes in each comparison.
    Cell-level p-values are exploratory, not donor-level condition inference.
    Logistic regression supplies signed ranking coefficients, not p-values.
    Returns a canonical long table stored in uns['cellscope'][key_added].
    """
    if not 0 <= min_pct <= 1 or not 0 <= min_diff_pct <= 1:
        raise ValueError("Detection fractions must be between zero and one")
    if logfc_threshold < 0 or min_cells < 2:
        raise ValueError("Use a nonnegative fold threshold and at least two cells per side")
    if corr_method not in {"benjamini-hochberg", "bonferroni"}:
        raise ValueError("corr_method must be benjamini-hochberg or bonferroni")
    adata = modality(data, mod)
    selected_method = _cell_method(method)
    table = _marker_comparison(
        adata,
        groupby=groupby,
        ident_1=ident_1,
        ident_2=ident_2,
        method=selected_method,
        layer=layer,
        use_raw=use_raw,
        features=features,
        min_pct=min_pct,
        min_diff_pct=min_diff_pct,
        logfc_threshold=logfc_threshold,
        only_pos=only_pos,
        min_cells=min_cells,
        corr_method=corr_method,
        backend_kwargs=backend_kwargs or {},
    )
    record(
        adata,
        key_added,
        {
            "groupby": groupby,
            "ident_1": list(map(str, _labels(ident_1))),
            "ident_2": "rest" if ident_2 is None else list(map(str, _labels(ident_2))),
            "method": selected_method,
            "layer": layer,
            "use_raw": use_raw,
            "min_pct": min_pct,
            "min_diff_pct": min_diff_pct,
            "logfc_threshold": logfc_threshold,
            "only_pos": only_pos,
            "min_cells": min_cells,
            "features": None if features is None else list(map(str, features)),
            "backend_kwargs": backend_kwargs or {},
            "unit": "cell",
            "corr_method": corr_method,
        },
        backend="scikit-learn" if selected_method == "logreg" else "scanpy",
    )
    adata.uns["cellscope"][key_added]["table"] = table.copy()
    return table


def find_all_markers(
    data, groupby="Cluster", use_raw=False, *, key_added="find_all_markers", legacy=False, **kwargs
):
    """Run one-versus-rest marker discovery for every observed identity.

    Uses the same method/filter parameters and output schema as find_markers.
    Use legacy=True for the previous column layout (also available from
    gene_level_analysis); default outputs use the canonical long-table schema.
    """
    adata = modality(data, kwargs.get("mod", "gex"))
    if groupby not in adata.obs or adata.obs[groupby].isna().any():
        raise ValueError("Nonmissing marker grouping annotations are required")
    groups = adata.obs[groupby].unique()
    if len(groups) < 2:
        raise ValueError("FindAllMarkers requires at least two observed identities")
    tables = []
    for group in groups:
        tables.append(
            find_markers(
                data,
                groupby=groupby,
                ident_1=group,
                use_raw=use_raw,
                key_added=key_added,
                **kwargs,
            )
        )
    table = pd.concat(tables, ignore_index=True)
    entry = adata.uns["cellscope"][key_added]
    entry["params"].pop("ident_1")
    entry["params"]["comparisons"] = "each_observed_identity_vs_rest"
    entry["table"] = table.copy()
    if legacy:
        return table.rename(
            columns={
                "gene": "Feature",
                "group": "Identy",
                "scores": "Score",
                "pvalue": "Pvalue",
                "padj": "Padj",
                "log2FoldChange": "LogFC",
                "pct_nz_group": "PTS",
                "pct_nz_reference": "PTS_Rest",
            }
        ).set_index("Feature")
    return table


def _design(metadata, formula, contrast):
    patsy = dependency("patsy", "differential")
    if not isinstance(formula, str) or not formula.lstrip().startswith("~"):
        raise ValueError(
            "design must be an explicit right-hand-side formula, e.g. '~ donor_id + condition'"
        )
    factor, comparison, reference = contrast
    clean = metadata.copy()
    for col in clean:
        if isinstance(clean[col].dtype, pd.CategoricalDtype):
            clean[col] = clean[col].cat.remove_unused_categories()
        elif isinstance(clean[col].dtype, pd.StringDtype):
            clean[col] = clean[col].astype(object)
        elif isinstance(clean[col].dtype, pd.api.extensions.ExtensionDtype) and (
            pd.api.types.is_numeric_dtype(clean[col]) and not pd.api.types.is_bool_dtype(clean[col])
        ):
            clean[col] = clean[col].astype(float)
    levels = [reference, *[value for value in clean[factor].unique() if value != reference]]
    clean[factor] = pd.Categorical(clean[factor], categories=levels)
    try:
        matrix = patsy.dmatrix(formula, clean, return_type="dataframe", NA_action="raise")
        vectors = []
        for level in (comparison, reference):
            counterfactual = clean.copy()
            counterfactual[factor] = pd.Categorical([level] * len(clean), categories=levels)
            vectors.append(
                np.asarray(
                    patsy.build_design_matrices(
                        [matrix.design_info],
                        counterfactual,
                        NA_action="raise",
                    )[0]
                ).mean(axis=0)
            )
    except (patsy.PatsyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid design/metadata: {error}") from error
    effect = vectors[0] - vectors[1]
    if not np.isfinite(matrix.to_numpy()).all():
        raise ValueError("Design covariates must be finite")
    if np.linalg.matrix_rank(matrix) != matrix.shape[1]:
        raise ValueError("Design matrix is rank deficient; check confounded covariates")
    if len(matrix) <= matrix.shape[1]:
        raise ValueError("Design needs residual degrees of freedom for inference")
    if np.allclose(effect, 0):
        raise ValueError("Contrast factor has no effect in the design")
    return matrix, effect


def _backend_options(options, allowed):
    unknown = set(options).difference(allowed)
    if unknown:
        raise ValueError(
            f"Unknown backend_kwargs sections: {sorted(unknown)}; use {sorted(allowed)}"
        )
    if any(not isinstance(value, dict) for value in options.values()):
        raise ValueError("backend_kwargs sections must be dictionaries")


def _fit_pydeseq2(frame, metadata, design, contrast, n_cpus, options):
    _backend_options(options, {"dds", "stats"})
    dds_module = dependency("pydeseq2.dds", "differential")
    stats_module = dependency("pydeseq2.ds", "differential")
    dds_kwargs = {"refit_cooks": True, "n_cpus": n_cpus, "quiet": True, **options.get("dds", {})}
    stats_kwargs = {"n_cpus": n_cpus, "quiet": True, **options.get("stats", {})}
    dds = dds_module.DeseqDataSet(
        counts=frame.astype(int), metadata=metadata, design=design, **dds_kwargs
    )
    dds.deseq2()
    statistics = stats_module.DeseqStats(dds, contrast=contrast, **stats_kwargs)
    statistics.summary()
    return statistics.results_df.rename_axis("gene").reset_index(), (dds, statistics)


def _dge(frame, lib_sizes, options):
    ep = dependency("edgepython", "edger")
    dge = ep.make_dgelist(
        counts=frame.to_numpy().T,
        lib_size=lib_sizes,
        genes=pd.DataFrame({"gene": frame.columns}),
    )
    dge = ep.calc_norm_factors(dge, **options.get("normalization", {}))
    return ep, dge


def _fit_edgepython(frame, lib_sizes, design, contrast, method, options):
    _backend_options(options, {"normalization", "dispersion", "fit", "test"})
    ep, dge = _dge(frame, lib_sizes, options)
    dge = ep.estimate_disp(dge, design=design.to_numpy(), **options.get("dispersion", {}))
    if method == "edgepython_ql":
        fit = ep.glm_ql_fit(dge, design=design.to_numpy(), **options.get("fit", {}))
        test = ep.glm_ql_ftest(fit, contrast=contrast, **options.get("test", {}))
    else:
        fit = ep.glm_fit(dge, design=design.to_numpy(), **options.get("fit", {}))
        test = ep.glm_lrt(fit, contrast=contrast, **options.get("test", {}))
    table = ep.top_tags(test, n=frame.shape[1])["table"].reset_index(drop=True)
    table = table.rename(columns={"logFC": "log2FoldChange", "PValue": "pvalue", "FDR": "padj"})
    return table, (dge, fit, test)


def _fit_pylimma(frame, lib_sizes, design, contrast, options):
    _backend_options(options, {"normalization", "voom", "fit", "ebayes"})
    try:
        version("python-limma")
    except PackageNotFoundError:
        pass
    else:
        raise ImportError(
            "python-limma and pylimma share an import name. Use a clean environment "
            "with John Mulvey's PyPI pylimma distribution (cellscope[limma])."
        )
    lm = dependency("pylimma", "limma")
    _, dge = _dge(frame, lib_sizes, options)
    effective_sizes = np.asarray(dge["samples"]["lib.size"] * dge["samples"]["norm.factors"])
    transformed = lm.voom(
        frame.T,
        design=design,
        lib_size=effective_sizes,
        **options.get("voom", {}),
    )
    fit = lm.lm_fit(transformed, design=design, **options.get("fit", {}))
    fit = lm.contrasts_fit(fit, contrasts=pd.DataFrame({"effect": contrast}, index=design.columns))
    fit = lm.e_bayes(fit, **options.get("ebayes", {}))
    table = lm.top_table(
        fit,
        coef=0,
        number=frame.shape[1],
        sort_by="none",
        genelist=pd.DataFrame({"gene": frame.columns}),
    ).reset_index(drop=True)
    table = table.rename(
        columns={"log_fc": "log2FoldChange", "p_value": "pvalue", "adj_p_value": "padj"}
    )
    return table, (dge, transformed, fit)


def differential_expression(
    pdata,
    *,
    design,
    contrast,
    method="pydeseq2",
    groups_col=None,
    sample_col="sample_id",
    mod="gex",
    layer=None,
    min_total_count=10,
    filter_method="total_count",
    filter_kwargs=None,
    min_replicates=2,
    n_cpus=1,
    key_added="differential_expression",
    backend_kwargs=None,
):
    """Fit sample-level DE with PyDESeq2, TMM/voom limma, or edgePython QL/LRT.

    Input is one raw-count profile per biological sample per cell group.
    contrast=(factor, comparison, reference) has positive effects in comparison.
    A common Patsy design is used across backends. Interaction contrasts are
    averaged comparison-minus-reference predictions over observed covariates.
    Returns (canonical long table, group-to-native-models mapping); models are
    not serialized. Adjustment is performed within each group's tested genes.
    """
    if method not in _BULK_METHODS:
        raise ValueError(f"Unknown sample-level method {method!r}; use {list(_BULK_METHODS)}")
    selected_method = _BULK_METHODS[method]
    if filter_method not in {"total_count", "filter_by_expr"}:
        raise ValueError("filter_method must be total_count or filter_by_expr")
    if filter_kwargs and filter_method != "filter_by_expr":
        raise ValueError("filter_kwargs require filter_method='filter_by_expr'")
    if len(contrast) != 3 or contrast[1] == contrast[2]:
        raise ValueError("contrast must be (factor, comparison, reference) with distinct levels")
    if min_replicates < 2 or min_total_count < 0 or n_cpus < 1:
        raise ValueError(
            "Require at least two samples per condition, nonnegative count threshold, and positive n_cpus"
        )
    obj = modality(pdata, mod)
    counts(obj, layer)
    factor, comparison, reference = contrast
    for col in (sample_col, factor, *([groups_col] if groups_col else [])):
        if col not in obj.obs or obj.obs[col].isna().any():
            raise ValueError(f"Nonmissing profile metadata required: {col}")
    if obj.obs.groupby(sample_col, observed=True)[factor].nunique().gt(1).any():
        raise ValueError("Condition metadata must be constant within each sample")
    groups = obj.obs[groups_col].unique() if groups_col else ["all"]
    if not len(groups):
        raise ValueError("No sample-level profiles are available")
    results, models, designs, effects = [], {}, {}, {}
    for group in groups:
        subset = obj[obj.obs[groups_col].eq(group)].copy() if groups_col else obj.copy()
        if subset.obs[sample_col].duplicated().any():
            raise ValueError("Each cell group must contain one profile per sample")
        for level in (comparison, reference):
            if subset.obs[factor].eq(level).sum() < min_replicates:
                raise ValueError(f"Group {group!r} needs {min_replicates} samples in {level!r}")
        matrix = counts(subset, layer)
        lib_sizes = np.asarray(matrix.sum(axis=1)).ravel()
        if (lib_sizes <= 0).any():
            raise ValueError(f"Group {group!r} has zero-depth sample profiles")
        keep = np.asarray(matrix.sum(axis=0)).ravel() >= min_total_count
        keep &= np.asarray((matrix > 0).sum(axis=0)).ravel() > 0
        design_matrix, effect = _design(subset.obs, design, contrast)
        if filter_method == "filter_by_expr":
            ep = dependency("edgepython", "edger")
            dense = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
            keep &= np.asarray(
                ep.filter_by_expr(
                    dense.T,
                    design=design_matrix.to_numpy(),
                    lib_size=lib_sizes,
                    **(filter_kwargs or {}),
                )
            )
        if not keep.any():
            raise ValueError(f"No genes pass filtering in group {group!r}")
        filtered = matrix[:, keep]
        frame = pd.DataFrame(
            np.rint(filtered.toarray() if sparse.issparse(filtered) else np.asarray(filtered)),
            index=subset.obs_names,
            columns=subset.var_names[keep],
        )
        options = backend_kwargs or {}
        if selected_method == "pydeseq2":
            table, native = _fit_pydeseq2(
                frame, subset.obs.copy(), design_matrix, effect, n_cpus, options
            )
        elif selected_method == "pylimma":
            table, native = _fit_pylimma(frame, lib_sizes, design_matrix, effect, options)
        else:
            table, native = _fit_edgepython(
                frame, lib_sizes, design_matrix, effect, selected_method, options
            )
        if not {"gene", "log2FoldChange", "pvalue", "padj"}.issubset(table.columns):
            raise RuntimeError(f"Unexpected {selected_method} result schema")
        table["group"] = str(group)
        table["comparison"] = str(comparison)
        table["reference"] = str(reference)
        table["method"] = selected_method
        table["unit"] = "sample"
        results.append(table)
        models[str(group)] = native
        designs[str(group)] = design_matrix.copy()
        effects[str(group)] = pd.DataFrame({"coefficient": design_matrix.columns, "weight": effect})
    result = pd.concat(results, ignore_index=True)
    backend = "edgepython" if selected_method.startswith("edgepython") else selected_method
    record(
        obj,
        key_added,
        {
            "method": selected_method,
            "design": design,
            "contrast": list(map(str, contrast)),
            "sample_col": sample_col,
            "groups_col": groups_col,
            "layer": layer,
            "min_total_count": min_total_count,
            "filter_method": filter_method,
            "filter_kwargs": filter_kwargs or {},
            "min_replicates": min_replicates,
            "unit": "sample",
            "adjustment_scope": "tested_genes_within_group",
            "contrast_rule": "average_counterfactual_design_difference",
            "backend_kwargs": backend_kwargs or {},
        },
        backend=backend,
    )
    obj.uns["cellscope"][key_added].update(
        {"table": result.copy(), "designs": designs, "contrasts": effects}
    )
    if selected_method == "pylimma":
        obj.uns["cellscope"][key_added]["normalization_backend"] = "edgepython"
        try:
            obj.uns["cellscope"][key_added]["normalization_backend_version"] = version("edgepython")
        except PackageNotFoundError:
            obj.uns["cellscope"][key_added]["normalization_backend_version"] = "source"
    return result, models


def pseudobulk_de(
    data,
    *,
    design,
    contrast,
    method="pydeseq2",
    sample_col="sample_id",
    groups_col="cell_type",
    mod="gex",
    layer="counts",
    min_cells=10,
    metadata_cols=(),
    pseudobulk_kwargs=None,
    de_kwargs=None,
):
    """Aggregate raw counts, then fit sample-level DE; return table/profiles/models.

    Supply every design covariate in metadata_cols. The contrast factor is
    automatically included. See differential_expression for backend options.
    """
    from ._api import pseudobulk

    metadata_cols = list(dict.fromkeys([contrast[0], *metadata_cols]))
    pdata = pseudobulk(
        data,
        sample_col=sample_col,
        groups_col=groups_col,
        mod=mod,
        layer=layer,
        min_cells=min_cells,
        metadata_cols=metadata_cols,
        **(pseudobulk_kwargs or {}),
    )
    table, models = differential_expression(
        pdata,
        design=design,
        contrast=contrast,
        method=method,
        groups_col=groups_col,
        sample_col=sample_col,
        **(de_kwargs or {}),
    )
    return table, pdata, models
