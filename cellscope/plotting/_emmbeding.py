import functools
import pathlib
from collections.abc import Sequence

import matplotlib.pyplot as plt
import scanpy as sc

from .._legacy_save import save_images


def dimplot(
    adata: sc.AnnData,
    reduction: str,
    filename: str,
    dim1label: str = "UMAP1",
    dim2label: str = "UMAP2",
    color: str | list[str] | tuple[str] | Sequence[str] = "CellType",
    formats: tuple[str] = ("pdf", "png"),
    frameon: bool = False,
    outdir: pathlib.PosixPath | str = ".",
    width=7,
    height=6,
    dpi: int = 300,
    **kwds,
):
    _saveimg = save_images(formats=formats, outdir=outdir, dpi=dpi)
    plt.rcParams["figure.figsize"] = (width, height)
    ax = sc.pl.embedding(adata, basis=reduction, color=color, show=False, frameon=frameon, **kwds)
    axes = ax if isinstance(ax, list) else [ax]
    for x in axes:
        x.arrow(
            -7,
            -12,
            5 * height / width,
            0,
            head_width=0.5,
            head_length=0.5,
            width=0.1,
            color="black",
        )
        x.arrow(
            -7,
            -12,
            0,
            5,
            head_width=0.5,
            head_length=0.5,
            width=0.1 * height / width,
            color="black",
        )
        x.text(-6.5, -13.5, dim1label, fontdict={"weight": "bold", "color": "black"})
        x.text(
            -8,
            -11.3,
            dim2label,
            fontdict={"fontweight": "bold", "rotation": "vertical"},
        )
    _saveimg(filename=filename, figsize=(width, height))


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
) -> sc.AnnData | None:
    _saveimg = save_images(formats=formats, outdir=outdir, dpi=dpi)
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
        sc.pl.pca_variance_ratio(adata, n_pcs=n_comps, show=False)
        _saveimg("PCA_variance_ratio")
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
        legend_loc=legend_loc,
    )
    _dimplot(
        reduction=f"{neighbors_key}_umap",
        color=batch_key,
        filename=f"{batch_key}_UMAP",
    )
    _dimplot(
        reduction=f"{neighbors_key}_umap",
        color=cluster_key,
        filename=f"{cluster_key}_UMAP",
    )
    # _dimplot(reduction='tsne',color=batch_key,filename=f"{batch_key}_TSNE",dim1label='TSNE1',dim2label='TSNE2')
    # _dimplot(reduction='tsne',color=cluster_key,filename=f"{cluster_key}_TSNE",dim1label='TSNE1',dim2label='TSNE2')
    # batch_key='Sample';legend_loc='right margin';reduction='umap';outdir=f"{OUTDIR}/batch_effect_before_integratation"
    # legend_fontsize='small';cluster_key='Cluster_before_integratation'
