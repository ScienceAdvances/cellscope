import functools
import pathlib
from collections.abc import Sequence

import matplotlib.pyplot as plt
import scanpy as sc

from ._output import finish_plot, reject_save


def dimplot(
    adata: sc.AnnData,
    reduction: str,
    filename: str | None = None,
    dim1label: str = "UMAP1",
    dim2label: str = "UMAP2",
    color: str | list[str] | tuple[str] | Sequence[str] = "CellType",
    formats: tuple[str] = ("pdf", "png"),
    frameon: bool = False,
    outdir: pathlib.PosixPath | str = ".",
    width=7,
    height=6,
    dpi: int = 300,
    axis_length: float = 0.12,
    axis_origin: tuple[float, float] = (0.06, 0.06),
    show: bool = False,
    **kwds,
):
    """Plot/save an embedding with optional small lower-left axes.

    With frameon=False (default), axis_origin and axis_length use fractions
    of each panel, independent of embedding coordinates. Larger axis_length
    gives longer arrows. frameon=True keeps the normal coordinate axes.
    Returns the native Matplotlib axis or list of axes.
    """
    origin_x, origin_y = axis_origin
    if not 0 < axis_length < 1 or not (
        0 <= origin_x < 1 - axis_length and 0 <= origin_y < 1 - axis_length
    ):
        raise ValueError("axis_origin and axis_length must keep the arrows inside each panel")
    reject_save(kwds)
    kwds["show"] = False
    kwds["save"] = False
    return_figure = kwds.pop("return_fig", False)
    with plt.rc_context({"figure.figsize": (width, height)}):
        ax = sc.pl.embedding(
            adata, basis=reduction, color=color, frameon=frameon, **kwds,
        )
        axes = ax if isinstance(ax, list) else [ax]
        if not frameon:
            for panel in axes:
                # Axes-relative coordinates never expand the UMAP data limits.
                origin = (origin_x, origin_y)
                style = {"arrowstyle": "->", "color": "black", "lw": 0.8,
                         "shrinkA": 0, "shrinkB": 0, "mutation_scale": 8}
                panel.annotate(
                    "", xy=(origin_x + axis_length, origin_y), xytext=origin,
                    xycoords="axes fraction", textcoords="axes fraction", arrowprops=style,
                )
                panel.annotate(
                    "", xy=(origin_x, origin_y + axis_length), xytext=origin,
                    xycoords="axes fraction", textcoords="axes fraction", arrowprops=style,
                )
                panel.annotate(
                    dim1label, xy=(origin_x + axis_length / 2, origin_y),
                    xycoords="axes fraction", xytext=(0, -5), textcoords="offset points",
                    ha="center", va="top", fontsize=8,
                )
                panel.annotate(
                    dim2label, xy=(origin_x, origin_y + axis_length / 2),
                    xycoords="axes fraction", xytext=(-5, 0), textcoords="offset points",
                    ha="right", va="center", rotation=90, fontsize=8,
                )
                panel.set_axis_off()
        # Multi-panel figures retain Scanpy's panel layout rather than being
        # squeezed into a single width x height canvas.
        if len(axes) == 1:
            ax.figure.set_size_inches(width, height)
    result = axes[0].figure if return_figure else ax
    return finish_plot(result, filename=filename, outdir=outdir, formats=formats, dpi=dpi, show=show)


def plot_batch_effect(
    adata: sc.AnnData,
    *,
    cluster_key: str = "Cluster",
    batch_key: str = "Sample",
    n_jobs: int = 8,
    use_rep: str = "X_pca",  # X_harmony
    neighbors_key="X_pca",  # harmony_neighbors
    resolution: float = 1.0,
    outdir: pathlib.PosixPath | str = ".",
    legend_loc: str = "right margin",  # right margin, on data,
    legend_fontsize: str = "small",  # [‘xx-small’, ‘x-small’, ‘small’, ‘medium’, ‘large’, ‘x-large’, ‘xx-large’]
    n_pcs: int = 20,
    mask_var: str | None = "highly_variable",
    n_neighbors: int = 15,
    dpi: int = 300,
    formats: tuple[str] = ("pdf", "png"),
    filename=None,
    show=False,
):
    plots = {}
    if use_rep not in adata.obsm:
        if use_rep != "X_pca":
            raise KeyError(f"Embedding {use_rep!r} must be computed before plotting")
        selected = int(adata.var[mask_var].sum()) if mask_var in adata.var else adata.n_vars
        n_comps = min(n_pcs, adata.n_obs - 1, selected - 1)
        sc.pp.pca(
            adata,
            svd_solver="arpack",
            n_comps=n_comps,
            mask_var=mask_var if mask_var in adata.var else None,
        )
        from ._pca import pca_variance_ratio

        plots["pca_variance"] = pca_variance_ratio(adata, n_pcs=n_comps)
    if neighbors_key not in adata.uns:
        sc.pp.neighbors(
            adata,
            n_neighbors=n_neighbors,
            n_pcs=min(n_pcs, adata.obsm[use_rep].shape[1]),
            use_rep=use_rep,
            method="umap",
            key_added=neighbors_key,  #  .uns[key_added] .obsp[key_added+'_distances'] .obsp[key_added+'_connectivities']
            random_state=0,
        )
    sc.tl.umap(
        adata,
        method="umap",
        min_dist=0.5,
        spread=1.0,
        init_pos="spectral",
        neighbors_key=neighbors_key,
        key_added=f"{neighbors_key}_umap",
        random_state=0,
    )
    # sc.tl.tsne(adata,n_jobs=n_jobs)
    sc.tl.leiden(
        adata, key_added=cluster_key, resolution=resolution, neighbors_key=neighbors_key
    )  # , flavor='igraph',n_iterations=2

    _dimplot = functools.partial(
        dimplot,
        adata=adata,
        legend_fontsize=legend_fontsize,
        outdir=outdir,
        legend_loc=legend_loc, formats=formats, dpi=dpi,
    )
    plots["batch"] = _dimplot(
        reduction=f"{neighbors_key}_umap",
        color=batch_key,
        filename=None,
    )
    plots["cluster"] = _dimplot(
        reduction=f"{neighbors_key}_umap",
        color=cluster_key,
        filename=None,
    )
    # _dimplot(reduction='tsne',color=batch_key,filename=f"{batch_key}_TSNE",dim1label='TSNE1',dim2label='TSNE2')
    # _dimplot(reduction='tsne',color=cluster_key,filename=f"{cluster_key}_TSNE",dim1label='TSNE1',dim2label='TSNE2')
    # batch_key='Sample';legend_loc='right margin';reduction='umap';outdir=f"{OUTDIR}/batch_effect_before_integratation"
    # legend_fontsize='small';cluster_key='Cluster_before_integratation'

    for name, plot in plots.items():
        finish_plot(plot, filename=f"{filename}_{name}" if filename is not None else None, outdir=outdir, formats=formats, dpi=dpi)
    if show:
        plt.show()
    return plots
