"""Scree plot semantics: true variance, 1-based data and numeric interval ticks."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from anndata import AnnData

import cellscope as cs


def pca_data(n=50):
    data = AnnData(np.zeros((3, 2)))
    data.uns['pca'] = {'variance_ratio': np.geomspace(.05, .0005, n)}
    return data


def test_plot_has_numeric_interval_ticks_and_preserves_pca(tmp_path):
    data = pca_data()
    before = data.uns['pca']['variance_ratio'].copy()
    axis = cs.pl.pca_variance_ratio(data, n_pcs=50, log=True, tick_step=10,
                                    filename="scree", outdir=tmp_path, formats=("png",))
    np.testing.assert_array_equal(axis.get_xticks(), [0, 10, 20, 30, 40, 50])
    assert [tick.get_text() for tick in axis.get_xticklabels()] == ['0','10','20','30','40','50']
    assert not axis.texts  # no PC1...PC50 per-point annotations
    assert axis.get_yscale() == 'log'
    np.testing.assert_array_equal(axis.lines[0].get_xdata(), np.arange(1, 51))
    np.testing.assert_allclose(axis.lines[0].get_ydata(), before * 100)
    np.testing.assert_array_equal(data.uns['pca']['variance_ratio'], before)
    assert (tmp_path/'scree.png').stat().st_size > 0
    plt.close(axis.figure)


def test_short_pca_caps_plot_and_accepts_supplied_axis():
    data = pca_data(7)
    fig, axis = plt.subplots()
    assert cs.pl.pca_variance_ratio(data, n_pcs=50, log=False, ax=axis) is axis
    np.testing.assert_array_equal(axis.lines[0].get_xdata(), np.arange(1, 8))
    np.testing.assert_array_equal(axis.get_xticks(), [0, 7])
    plt.close(fig)


def test_zero_variance_is_only_valid_on_linear_axis():
    data = pca_data(3)
    data.uns['pca']['variance_ratio'][-1] = 0
    with pytest.raises(ValueError, match='log=False'):
        cs.pl.pca_variance_ratio(data)
    axis = cs.pl.pca_variance_ratio(data, log=False, n_pcs=None)
    assert axis.lines[0].get_ydata()[-1] == 0
    plt.close(axis.figure)


@pytest.mark.parametrize('options', [{'tick_step':0}, {'n_pcs':0}, {'n_pcs':2.5}])
def test_invalid_plot_options(options):
    with pytest.raises(ValueError):
        cs.pl.pca_variance_ratio(pca_data(), **options)



def test_multiple_formats_export_the_same_plot(tmp_path):
    data = pca_data()
    axis = cs.pl.pca_variance_ratio(
        data, filename="pca_variance", outdir=tmp_path, formats=("pdf", "png"),
    )
    assert (tmp_path / "pca_variance.pdf").read_bytes().startswith(b"%PDF-")
    assert (tmp_path / "pca_variance.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    np.testing.assert_array_equal(axis.get_xticks(), [0, 10, 20, 30, 40, 50])
    plt.close(axis.figure)


def test_formats_accepts_single_string(tmp_path):
    axis = cs.pl.pca_variance_ratio(
        pca_data(), filename="single", outdir=tmp_path, formats="png",
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == ["single.png"]
    plt.close(axis.figure)


def test_save_keyword_is_removed(tmp_path):
    import inspect

    assert "save" not in inspect.signature(cs.pl.pca_variance_ratio).parameters
    with pytest.raises(TypeError, match="save"):
        cs.pl.pca_variance_ratio(pca_data(), save=tmp_path / "single.png")
    assert not list(tmp_path.iterdir())
