"""Plots return native axes and never save files implicitly."""

import numpy as np

from .._core import modality
from ._output import finish_plot, reject_save


def embedding(data, *, mod="gex", basis="umap", color=None, ax=None, filename=None, outdir=".", formats=("pdf", "png"), dpi=300, show=False, **kwargs):
    import scanpy as sc

    reject_save(kwargs)
    kwargs["show"] = False
    kwargs["save"] = False
    result = sc.pl.embedding(modality(data, mod), basis=basis, color=color, ax=ax, **kwargs)
    return finish_plot(result, filename=filename, outdir=outdir, formats=formats, dpi=dpi, show=show)


def dotplot(data, var_names, *, groupby, mod="gex", filename=None, outdir=".", formats=("pdf", "png"), dpi=300, show=False, **kwargs):
    import scanpy as sc

    reject_save(kwargs)
    kwargs["show"] = False
    kwargs["save"] = False
    kwargs.setdefault("use_raw", False)
    kwargs.setdefault("return_fig", True)
    result = sc.pl.dotplot(modality(data, mod), var_names, groupby=groupby, **kwargs)
    return finish_plot(result, filename=filename, outdir=outdir, formats=formats, dpi=dpi, show=show)


def heatmap(data, var_names, *, groupby, mod="gex", filename=None, outdir=".", formats=("pdf", "png"), dpi=300, show=False, **kwargs):
    import scanpy as sc

    reject_save(kwargs)
    kwargs["show"] = False
    kwargs["save"] = False
    kwargs.setdefault("use_raw", False)
    result = sc.pl.heatmap(modality(data, mod), var_names, groupby=groupby, **kwargs)
    return finish_plot(result, filename=filename, outdir=outdir, formats=formats, dpi=dpi, show=show)


def qc(
    data,
    *,
    keys=("total_counts", "n_genes_by_counts", "pct_counts_mt"),
    mod="gex",
    groupby=None,
    ax=None, figsize=None, xlabel=None, ylabel=None, title=None, ylim=None,
    filename=None, outdir=".", formats=("pdf", "png"), dpi=300, show=False,
    **kwargs,
):
    import scanpy as sc

    reject_save(kwargs)
    kwargs["show"] = False
    kwargs["save"] = False
    import matplotlib.pyplot as plt

    with plt.rc_context({"figure.figsize": figsize} if figsize is not None else {}):
        result = sc.pl.violin(modality(data, mod), list(keys), groupby=groupby, ax=ax, **kwargs)
    panel_source = result.axes if not hasattr(result, "set") and hasattr(result, "axes") else result
    panels = np.atleast_1d(panel_source).ravel()
    for panel in panels:
        if hasattr(panel, "set"):
            options = {k: v for k, v in {"xlabel": xlabel, "ylabel": ylabel, "title": title, "ylim": ylim}.items() if v is not None}
            panel.set(**options)
    return finish_plot(result, filename=filename, outdir=outdir, formats=formats, dpi=dpi, show=show)


def cell_composition(table, *, value="fraction", ax=None, filename=None, outdir=".", formats=("pdf", "png"), dpi=300, show=False, **kwargs):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    reject_save(kwargs)
    matrix = table.pivot(index="sample_id", columns="group", values=value).fillna(0)
    matrix.plot.bar(stacked=True, ax=ax, **kwargs)
    ax.set_ylabel(value)
    return finish_plot(ax, filename=filename, outdir=outdir, formats=formats, dpi=dpi, show=show)


def volcano(results, *, effect="log2FoldChange", pvalue="padj", ax=None, filename=None, outdir=".", formats=("pdf", "png"), dpi=300, show=False, **kwargs):
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        _, ax = plt.subplots()
    reject_save(kwargs)
    ax.scatter(
        results[effect], -np.log10(results[pvalue].clip(lower=np.finfo(float).tiny)), **kwargs
    )
    ax.set(xlabel=effect, ylabel=f"-log10({pvalue})")
    return finish_plot(ax, filename=filename, outdir=outdir, formats=formats, dpi=dpi, show=show)
