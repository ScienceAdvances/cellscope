"""Real-backend regression checks for PCA, R transport and SoupX overrides."""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

import cellscope as cs


def _require_r(packages):
    try:
        cs.r.require_packages(packages)
    except ImportError as error:
        pytest.skip(str(error))


@pytest.mark.parametrize("selection", ["implicit", "named", "array", "disabled", "legacy"])
def test_pca_caps_components_for_actual_feature_selection(selection):
    data = cs.datasets.toy_rna(n_cells=100, n_genes=60)
    cs.pp.normalize(data)
    mask = np.arange(data.n_vars) < 10
    data.var["highly_variable"] = mask
    options = {
        "implicit": {},
        "named": {"mask_var": "highly_variable"},
        "array": {"mask_var": mask},
        "disabled": {"mask_var": None},
        "legacy": {"use_highly_variable": False},
    }[selection]
    cs.tl.pca(data, **options)
    assert data.obsm["X_pca"].shape == (100, 50 if selection in {"disabled", "legacy"} else 9)


def test_sce_roundtrip_keeps_nullable_boolean_values():
    _require_r(["SingleCellExperiment"])
    data = AnnData(
        sparse.csr_matrix([[1, 2], [3, 4], [5, 6]]), obs=pd.DataFrame(index=["c3", "c1", "c2"])
    )
    data.layers["counts"] = data.X.copy()
    data.obs["flag"] = pd.array([True, False, None], dtype="boolean")
    data.obs["all_missing"] = pd.array([None] * 3, dtype="boolean")
    data.obs["number"] = pd.array([1.5, None, 3.5], dtype="Float64")
    restored = cs.r.from_sce(cs.r.to_sce(data))
    pd.testing.assert_series_equal(restored.obs.flag, data.obs.flag)
    pd.testing.assert_series_equal(restored.obs.all_missing, data.obs.all_missing)
    np.testing.assert_allclose(
        restored.obs.number.to_numpy(dtype=float), [1.5, np.nan, 3.5], equal_nan=True
    )
    np.testing.assert_array_equal(restored.layers["counts"].toarray(), data.X.toarray())


def test_r_named_vectors_keep_values_order_and_duplicate_names():
    _require_r(["jsonlite"])
    result = cs.r._call("function() c(b=2, a=1, b=NA_real_)")
    assert isinstance(result, pd.Series)
    assert result.index.tolist() == ["b", "a", "b"]
    np.testing.assert_allclose(result.to_numpy(), [2, 1, np.nan], equal_nan=True)
    strings = cs.r._call('function() c(second="x", first="y")')
    assert strings.tolist() == ["x", "y"]
    assert strings.index.tolist() == ["second", "first"]


def test_soupx_r_allows_rounding_override_without_changing_counts():
    _require_r(["SoupX"])
    data = cs.datasets.toy_rna(n_cells=100, n_genes=60)
    before = data.layers["counts"].toarray().copy()
    cs.pp.soupx(
        data,
        data.copy(),
        cluster_key="cell_type",
        backend="r",
        contamination=0.1,
        adjust_kwargs={"roundToInt": False},
    )
    corrected = data.layers["soupx_counts"].toarray()
    assert np.isfinite(corrected).all() and (corrected >= 0).all()
    assert not np.allclose(corrected, np.rint(corrected))
    np.testing.assert_array_equal(data.layers["counts"].toarray(), before)


def test_soupx_rejects_input_channel_override():
    data = cs.datasets.toy_rna(n_cells=100, n_genes=60)
    with pytest.raises(ValueError, match="input channel"):
        cs.pp.soupx(
            data,
            data.copy(),
            cluster_key="cell_type",
            backend="r",
            contamination=0.1,
            adjust_kwargs={"sc": "other"},
        )
