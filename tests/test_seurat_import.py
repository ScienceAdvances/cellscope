"""Seurat v5 RDS interoperability checks using real R objects."""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import cellscope as cs


@pytest.fixture(scope="module")
def seurat_files(tmp_path_factory):
    try:
        cs.r.require_packages(["Seurat", "SeuratObject"])
    except ImportError as error:
        pytest.skip(str(error))
    root = tmp_path_factory.mktemp("seurat-rds")
    cs.r._call(
        r"""function(root) {
        options(Seurat.object.assay.version="v5")
        x <- Matrix::Matrix(matrix(c(1,0,3,0,5,6,7,8,0,10,11,12), nrow=3), sparse=TRUE)
        dimnames(x) <- list(c("g3", "g1", "g2"), c("c3", "c1", "c4", "c2"))
        obj <- SeuratObject::CreateSeuratObject(x)
        obj$group <- factor(c("B", "A", NA, "B"), levels=c("B", "A", "unused"), ordered=TRUE)
        obj$flag <- c(TRUE, FALSE, NA, TRUE)
        obj$score <- c(0.5, NA, 1.5, 2.5)
        SeuratObject::Idents(obj) <- factor(c("T", "B", "T", "B"), levels=c("T", "B"))
        obj[["RNA"]]$data <- log1p(x)
        obj[["RNA"]]$scale.data <- as.matrix(x[c("g2", "g3"), ]) - 2
        obj[["RNA"]]$counts <- x
        obj[["RNA"]][["symbol"]] <- setNames(c("third", "first", "second"), rownames(x))
        SeuratObject::VariableFeatures(obj[["RNA"]]) <- c("g1", "g2")
        embedding <- matrix(seq_len(8), ncol=2,
            dimnames=list(colnames(x), c("PC_1", "PC_2")))
        obj[["pca"]] <- SeuratObject::CreateDimReducObject(embedding, key="PC_", assay="RNA")
        saveRDS(obj, file.path(root, "standard.rds"))
        split <- obj
        split[["RNA"]] <- split(split[["RNA"]], f=c("a", "b", "a", "b"), layers=c("counts", "data"))
        saveRDS(split, file.path(root, "split.rds"))
        overlap <- obj
        overlap[["RNA"]]$counts.duplicate <- x
        overlap[["RNA"]]$counts.other <- x
        overlap[["RNA"]]$counts <- NULL
        saveRDS(overlap, file.path(root, "overlap.rds"))
        slim <- SeuratObject::CreateSeuratObject(SeuratObject::CreateAssay5Object(data=log1p(x)))
        saveRDS(slim, file.path(root, "slim.rds"))
        legacy <- obj
        legacy[["RNA"]] <- as(legacy[["RNA"]], "Assay")
        saveRDS(legacy, file.path(root, "legacy.rds"))
        subset <- obj
        subset[["RNA2"]] <- SeuratObject::CreateAssay5Object(counts=x[, c("c2", "c3")])
        saveRDS(subset, file.path(root, "subset.rds"))
        unequal <- SeuratObject::CreateSeuratObject(list(
            a=x[c("g3", "g1"), c("c3", "c1")], b=x[c("g1", "g2"), c("c4", "c2")]))
        saveRDS(unequal, file.path(root, "unequal.rds"))
        single <- SeuratObject::CreateSeuratObject(x[1, 1, drop=FALSE])
        saveRDS(single, file.path(root, "single.rds"))
        partial <- obj
        partial[["pca"]] <- SeuratObject::CreateDimReducObject(
            embedding[c("c2", "c3"), ], key="PC_", assay="RNA")
        saveRDS(partial, file.path(root, "partial.rds"))
        saveRDS(list(x=1), file.path(root, "invalid.rds"))
        NULL
    }""",
        packages=["Seurat"],
        root=str(root),
    )
    return root


def _counts():
    return np.array([[1, 0, 3], [0, 5, 6], [7, 8, 0], [10, 11, 12]])


@pytest.mark.parametrize("name", ["standard", "split", "legacy"])
def test_import_preserves_expression_metadata_and_embeddings(seurat_files, tmp_path, name):
    data = cs.io.read_seurat_rds(seurat_files / f"{name}.rds")
    assert data.obs_names.tolist() == ["c3", "c1", "c4", "c2"]
    assert data.var_names.tolist() == ["g3", "g1", "g2"]
    assert sparse.issparse(data.X)
    np.testing.assert_array_equal(data.X.toarray(), _counts())
    np.testing.assert_allclose(data.layers["data"].toarray(), np.log1p(_counts()))
    assert set(data.layers) == {"counts", "data"}
    assert data.obs.group.cat.categories.tolist() == ["B", "A", "unused"]
    assert data.obs.group.cat.ordered
    assert pd.isna(data.obs.group.iloc[2])
    assert data.obs.seurat_ident.tolist() == ["T", "B", "T", "B"]
    assert data.obs.flag.dtype == "boolean"
    assert pd.isna(data.obs.flag.iloc[2])
    assert data.var.symbol.tolist() == ["third", "first", "second"]
    assert data.var.highly_variable.tolist() == [False, True, True]
    np.testing.assert_array_equal(data.obsm["X_pca"], np.arange(1, 9).reshape(2, 4).T)
    destination = tmp_path / "converted.h5ad"
    data.write_h5ad(destination)
    restored = ad.read_h5ad(destination)
    pd.testing.assert_frame_equal(restored.obs, data.obs)
    np.testing.assert_array_equal(restored.X.toarray(), _counts())


def test_partial_scaled_layer_is_nan_padded_and_aligned(seurat_files):
    with pytest.warns(UserWarning, match="padding with NaN"):
        data = cs.io.read_seurat_rds(seurat_files / "standard.rds", layers=("scale.data",))
    assert np.isnan(data.layers["scale.data"][:, 1]).all()
    np.testing.assert_array_equal(data.layers["scale.data"][:, [2, 0]], _counts()[:, [2, 0]] - 2)


def test_data_only_object_and_missing_x(seurat_files):
    path = seurat_files / "slim.rds"
    with pytest.raises(ValueError, match="Layer 'counts' not found"):
        cs.io.read_seurat_rds(path)
    data = cs.io.read_seurat_rds(path, x_layer="data")
    np.testing.assert_allclose(data.X.toarray(), np.log1p(_counts()))
    assert "counts" not in data.layers
    assert data.uns["seurat"]["missing_layers"] == ["counts"]


def test_selected_assay_uses_own_cells_and_excludes_other_reductions(seurat_files):
    data = cs.io.read_seurat_rds(seurat_files / "subset.rds", assay="RNA2")
    assert data.obs_names.tolist() == ["c3", "c2"]
    assert not data.obsm
    np.testing.assert_array_equal(data.X.toarray(), _counts()[[0, 3]])


def test_exact_split_selection_and_disable_reductions(seurat_files):
    data = cs.io.read_seurat_rds(
        seurat_files / "split.rds", x_layer="counts.b", layers=(), reductions=False
    )
    assert data.obs_names.tolist() == ["c1", "c2"]
    assert set(data.layers) == {"counts.b"}
    assert not data.obsm
    np.testing.assert_array_equal(data.X.toarray(), _counts()[[1, 3]])


def test_unequal_split_features_record_coverage(seurat_files):
    with pytest.warns(UserWarning, match="different feature sets"):
        data = cs.io.read_seurat_rds(seurat_files / "unequal.rds")
    expected = _counts()
    expected[:2, 2] = 0
    expected[2:, 0] = 0
    np.testing.assert_array_equal(data.X.toarray(), expected)
    assert set(data.uns["seurat"]["split_coverage"]["counts"]) == {"counts.a", "counts.b"}


def test_overlapping_split_cells_rejected(seurat_files):
    with pytest.raises(ValueError, match="overlapping cells"):
        cs.io.read_seurat_rds(seurat_files / "overlap.rds")


def test_single_cell_and_feature_keep_two_dimensions(seurat_files):
    data = cs.io.read_seurat_rds(seurat_files / "single.rds")
    assert data.shape == (1, 1)
    assert data.obs_names.tolist() == ["c3"]
    assert data.var_names.tolist() == ["g3"]
    assert data.X[0, 0] == 1


def test_partial_embeddings_align_by_cell_name(seurat_files):
    data = cs.io.read_seurat_rds(seurat_files / "partial.rds")
    assert np.isnan(data.obsm["X_pca"][[1, 2]]).all()
    np.testing.assert_array_equal(data.obsm["X_pca"][[0, 3]], [[1, 5], [4, 8]])


def test_invalid_rds_and_assay(seurat_files):
    with pytest.raises(RuntimeError, match="RDS must contain a Seurat"):
        cs.io.read_seurat_rds(seurat_files / "invalid.rds")
    with pytest.raises(RuntimeError, match="Assay not found"):
        cs.io.read_seurat_rds(seurat_files / "standard.rds", assay="missing")


def test_invalid_path_and_arguments(seurat_files):
    with pytest.raises(FileNotFoundError):
        cs.io.read_seurat_rds(Path("/missing/cellscope-test-seurat.rds"))
    with pytest.raises(TypeError, match="sequence"):
        cs.io.read_seurat_rds(seurat_files / "standard.rds", layers="counts")
