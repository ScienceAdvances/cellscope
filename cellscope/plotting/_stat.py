"""Compatibility cell-ratio plot with an explicit within-sample denominator."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def cell_ratio(adata, x, y, *, palette=None, normalize=True, od=None, legend=True, figsize=(6, 3)):
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
    if od is not None:
        destination = Path(od)
        destination.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(destination / f"{x}_{y}_cell_ratio.pdf", dpi=300, bbox_inches="tight")
    return table
