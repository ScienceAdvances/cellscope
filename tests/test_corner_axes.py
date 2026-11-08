"""Embedding corner axes stay within panels and never alter UMAP coordinates."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

import cellscope as cs


@pytest.mark.parametrize("shift,scale", [(0, 1), (1000, 100), (-10000, 0.001)])
def test_dimplot_corner_axes_ignore_coordinate_range(tmp_path, shift, scale):
    data = cs.datasets.toy_rna(n_cells=60, n_genes=20)
    coordinates = np.random.default_rng(0).normal(size=(60, 2)) * scale + shift
    data.obsm["X_umap"] = coordinates.copy()
    ordinary = cs.pl.embedding(data, color="sample_id", frameon=False)
    expected_limits = ordinary.get_xlim(), ordinary.get_ylim()
    plt.close(ordinary.figure)
    axis = cs.pl.dimplot(
        data, reduction="umap", filename="corner", color="sample_id",
        formats=("png",), outdir=tmp_path, axis_length=0.1,
    )
    assert axis.axison is False
    arrows = [text for text in axis.texts if getattr(text, "arrow_patch", None) is not None]
    assert len(arrows) == 2
    assert all(text.xycoords == "axes fraction" for text in arrows)
    np.testing.assert_allclose((axis.get_xlim(), axis.get_ylim()), expected_limits)
    np.testing.assert_array_equal(data.obsm["X_umap"], coordinates)
    assert (tmp_path / "corner.png").stat().st_size > 0
    plt.close(axis.figure)


def test_multi_panel_corner_axes_and_normal_frame(tmp_path):
    data = cs.datasets.toy_rna(n_cells=60, n_genes=20)
    data.obsm["X_umap"] = np.random.default_rng(0).normal(size=(60, 2))
    axes = cs.pl.dimplot(data, "umap", "panels", color=["sample_id", "cell_type"],
                         formats=("png",), outdir=tmp_path)
    assert len(axes) == 2
    assert all(not axis.axison for axis in axes)
    assert axes[0].figure.get_size_inches()[0] > 7
    plt.close(axes[0].figure)
    axis = cs.pl.dimplot(data, "umap", "frame", color="sample_id", frameon=True,
                         formats=("png",), outdir=tmp_path)
    assert axis.axison
    assert not any(getattr(text, "arrow_patch", None) is not None for text in axis.texts)
    plt.close(axis.figure)
