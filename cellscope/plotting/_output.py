"""Shared explicit figure export, independent of pyplot's current figure."""

from pathlib import Path


def figures(result):
    import numpy as np
    from matplotlib.figure import Figure

    found = []
    def visit(obj):
        if isinstance(obj, Figure):
            if obj not in found:
                found.append(obj)
        elif isinstance(obj, dict):
            for value in obj.values():
                visit(value)
        elif isinstance(obj, np.ndarray):
            for value in obj.flat:
                visit(value)
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                visit(value)
        elif hasattr(obj, "get_axes"):
            visit(obj.get_axes())
        elif hasattr(obj, "figure"):
            visit(obj.figure)
        elif hasattr(obj, "fig"):
            visit(obj.fig)
    visit(result)
    if not found:
        raise TypeError("The plotting backend did not return a usable figure object")
    return found


def finish_plot(result, *, filename=None, outdir=".", formats=("pdf", "png"), dpi=300, show=False):
    import matplotlib.pyplot as plt

    targets = figures(result)
    if filename is not None:
        if not str(filename).strip():
            raise ValueError("filename must be a nonempty base name")
        requested = (formats,) if isinstance(formats, str) else tuple(formats)
        if not requested or any(not isinstance(fmt, str) or not fmt.strip(". ") for fmt in requested):
            raise ValueError("formats must contain nonempty format names")
        extensions = list(dict.fromkeys(fmt.strip().lstrip(".").lower() for fmt in requested))
        for fig in targets:
            invalid = set(extensions) - set(fig.canvas.get_supported_filetypes())
            if invalid:
                raise ValueError(f"Unsupported figure formats: {sorted(invalid)}")
        folder = Path(outdir)
        folder.mkdir(parents=True, exist_ok=True)
        for i, fig in enumerate(targets, 1):
            stem = str(filename) if len(targets) == 1 else f"{filename}_{i}"
            for extension in extensions:
                fig.savefig(folder / f"{stem}.{extension}", dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return result


def reject_save(kwargs):
    if "save" in kwargs:
        raise TypeError("save is not supported; use filename, outdir and formats")
