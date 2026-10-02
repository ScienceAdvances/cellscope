"""Plots return native axes and never save files implicitly."""

from .._core import modality


def embedding(data, *, mod="gex", basis="umap", color=None, ax=None, **kwargs):
    import scanpy as sc

    kwargs.setdefault("show", False)
    return sc.pl.embedding(modality(data, mod), basis=basis, color=color, ax=ax, **kwargs)


def dotplot(data, var_names, *, groupby, mod="gex", **kwargs):
    import scanpy as sc

    kwargs.setdefault("show", False)
    kwargs.setdefault("use_raw", False)
    return sc.pl.dotplot(modality(data, mod), var_names, groupby=groupby, **kwargs)


def heatmap(data, var_names, *, groupby, mod="gex", **kwargs):
    import scanpy as sc

    kwargs.setdefault("show", False)
    kwargs.setdefault("use_raw", False)
    return sc.pl.heatmap(modality(data, mod), var_names, groupby=groupby, **kwargs)


def qc(
    data,
    *,
    keys=("total_counts", "n_genes_by_counts", "pct_counts_mt"),
    mod="gex",
    groupby=None,
    **kwargs,
):
    import scanpy as sc

    kwargs.setdefault("show", False)
    return sc.pl.violin(modality(data, mod), list(keys), groupby=groupby, **kwargs)


def cell_composition(table, *, value="fraction", ax=None, **kwargs):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    matrix = table.pivot(index="sample_id", columns="group", values=value).fillna(0)
    matrix.plot.bar(stacked=True, ax=ax, **kwargs)
    ax.set_ylabel(value)
    return ax


def volcano(results, *, effect="log2FoldChange", pvalue="padj", ax=None, **kwargs):
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(
        results[effect], -np.log10(results[pvalue].clip(lower=np.finfo(float).tiny)), **kwargs
    )
    ax.set(xlabel=effect, ylabel=f"-log10({pvalue})")
    return ax
