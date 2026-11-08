"""Compatibility cell-ratio plot with an explicit within-sample denominator."""

import matplotlib.pyplot as plt
import pandas as pd

from ._output import finish_plot


def cell_ratio(adata, x, y, *, palette=None, normalize=True, legend=True, figsize=(6, 3), filename=None, outdir=".", formats=("pdf", "png"), dpi=300, show=False):
    """Return plot Axes; the computed composition is in ax.cellscope_table."""
    table = pd.crosstab(adata.obs[x], adata.obs[y], dropna=False)
    if normalize:
        table = table.div(table.sum(axis=1), axis=0)
    if palette is None and f"{y}_colors" in adata.uns:
        categories = (
            adata.obs[y].cat.categories
            if isinstance(adata.obs[y].dtype, pd.CategoricalDtype)
            else table.columns
        )
        palette = dict(zip(categories, adata.uns[f"{y}_colors"]))
    _, ax = plt.subplots(figsize=figsize)
    colors = [palette.get(item, "gray") for item in table.columns] if palette else None
    table.plot.bar(stacked=True, ax=ax, color=colors, legend=legend)
    ax.set_ylabel("Fraction" if normalize else "Cell count")
    if normalize:
        ax.set_ylim(0, 1)
    if legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
    ax.cellscope_table = table
    return finish_plot(ax, filename=filename, outdir=outdir, formats=formats, dpi=dpi, show=show)
