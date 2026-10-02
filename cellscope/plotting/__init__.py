"""Plotting namespace with optional legacy file-saving entry points."""

from ._api import cell_composition, dotplot, embedding, heatmap, qc, volcano

__all__ = [
    "cell_composition",
    "cell_ratio",
    "dimplot",
    "dotplot",
    "embedding",
    "heatmap",
    "plot_batch_effect",
    "qc",
    "volcano",
]


def __getattr__(name):
    if name in {"dimplot", "plot_batch_effect"}:
        from . import _emmbeding

        return getattr(_emmbeding, name)
    if name == "cell_ratio":
        from ._stat import cell_ratio

        return cell_ratio
    raise AttributeError(name)
