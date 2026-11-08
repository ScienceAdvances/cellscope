"""Numeric-axis PCA scree plots without per-component text annotations."""

from numbers import Integral

import numpy as np

from .._core import modality
from ._output import finish_plot


def pca_variance_ratio(
    data,
    *,
    mod="gex",
    n_pcs=50,
    log=True,
    tick_step=10,
    color="#286D91",
    figsize=(7, 4.2),
    title="PCA explained variance",
    ax=None,
    filename=None,
    outdir=".",
    formats=("pdf", "png"),
    dpi=300,
    show=False,
):
    """Plot explained variance with numeric PC ticks and return Matplotlib Axes.

    Read existing ``uns['pca']['variance_ratio']``; do not recompute PCA or
    change the input object. Percentages refer to the original PCA variance,
    not a renormalization over the displayed PCs. PC data start at 1; the
    tick at 0 is only an axis reference. ``n_pcs=None`` displays all computed
    components, and requests beyond the available components are capped.
    ``filename`` with ``outdir`` and ``formats`` saves multiple formats, as
    in ``dimplot``. No file is saved without filename.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullFormatter, StrMethodFormatter

    adata = modality(data, mod)
    if "pca" not in adata.uns or "variance_ratio" not in adata.uns["pca"]:
        raise KeyError("Run cs.tl.pca before plotting PCA explained variance")
    ratio = np.asarray(adata.uns["pca"]["variance_ratio"], dtype=float)
    if ratio.ndim != 1 or not ratio.size or not np.isfinite(ratio).all():
        raise ValueError("PCA variance_ratio must be a nonempty finite 1D array")
    if (ratio < 0).any() or (ratio > 1 + 1e-6).any():
        raise ValueError("PCA variance_ratio must contain proportions between 0 and 1")
    if n_pcs is not None and (
        isinstance(n_pcs, bool) or not isinstance(n_pcs, Integral) or n_pcs < 1
    ):
        raise ValueError("n_pcs must be a positive integer or None")
    if isinstance(tick_step, bool) or not isinstance(tick_step, Integral) or tick_step < 1:
        raise ValueError("tick_step must be a positive integer")
    n = len(ratio) if n_pcs is None else min(int(n_pcs), len(ratio))
    y = ratio[:n] * 100
    if log and (y <= 0).any():
        raise ValueError("Zero explained variance cannot use a log axis; set log=False")
    x = np.arange(1, n + 1)
    ticks = np.arange(0, n + 1, tick_step)
    if ticks[-1] != n:
        ticks = np.append(ticks, n)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.plot(
        x, y, color=color, linewidth=2, marker="o", markersize=3.5,
        markerfacecolor="white", markeredgewidth=1, zorder=3,
    )
    ax.set_facecolor("white")
    ax.set_yscale("log" if log else "linear")
    if log:
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.set_ylim(bottom=0)
    ax.set_xticks(ticks)
    ax.set_xlim(-0.5, n + 0.75)
    ax.set_xlabel("Principal component", fontsize=11, labelpad=8)
    ax.set_ylabel("Explained variance (%)", fontsize=11, labelpad=8)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=14)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color("#B8C2CC")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(axis="both", labelsize=9, colors="#3D4A57", length=3)
    ax.grid(False)
    ax.grid(axis="y", which="major", color="#E5EAF0", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    return finish_plot(ax, filename=filename, outdir=outdir, formats=formats, dpi=dpi, show=show)
