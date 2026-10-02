"""File saving for existing APIs, implemented with Matplotlib and pathlib."""

from pathlib import Path


def save_images(*, outdir, formats=("pdf", "png"), dpi=300):
    destination = Path(outdir)
    destination.mkdir(parents=True, exist_ok=True)
    formats = (formats,) if isinstance(formats, str) else formats

    def save(filename, figsize=None):
        import matplotlib.pyplot as plt

        if figsize is not None:
            plt.gcf().set_size_inches(*figsize)
        for extension in formats:
            plt.savefig(destination / f"{filename}.{extension}", dpi=dpi, bbox_inches="tight")

    return save
