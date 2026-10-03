# Import Seurat RDS objects

`cellscope.io.read_seurat_rds` reads a local Seurat RDS and returns an in-memory
AnnData. The reader uses Seurat's public layer accessors through the existing
isolated Rscript bridge. No intermediate h5Seurat file, rpy2 installation, or
manual export in an R session is needed.

The implementation is tested with Seurat **5.5.1** and SeuratObject **5.4.0**,
including `Assay5` and legacy `Assay` objects stored by the current version.
The [official release](https://github.com/satijalab/seurat/releases/tag/v5.5.1)
and [v5 object guide](https://satijalab.org/seurat/articles/seurat5_essential_commands)
describe the release and split-layer representation used by these tests.
This does not guarantee compatibility with every historical RDS class version.

## Requirements

Install R and these packages in the library used by Rscript:

```r
install.packages(c("Seurat", "SeuratObject", "Matrix", "jsonlite"))
```

Use Seurat and SeuratObject >= 5. Set `CELLSCOPE_RSCRIPT` to an alternate
Rscript executable and `R_LIBS_USER` to a private R library if needed.
Importing cellscope does not start R or install anything.

## Basic use

```python
import cellscope as cs

adata = cs.io.read_seurat_rds("sample.rds")
adata.write_h5ad("sample.h5ad")

# X and layers['counts'] contain RNA counts by default.
cs.pp.qc(adata, min_genes=200)
cs.pp.normalize(adata)
```

Select normalized expression for X explicitly when you want to continue from
Seurat's normalized values, or when the object contains no counts:

```python
adata = cs.io.read_seurat_rds("sample.rds", assay="RNA", x_layer="data")
# Do not normalize these values a second time.
```

The reader never fabricates missing counts. A missing X layer is an error;
missing optional layers are listed in `adata.uns['seurat']['missing_layers']`.
The default assay is `RNA`, even if Seurat's active assay is `SCT` or
`integrated`. Select another transcriptome assay explicitly when appropriate.
SCT corrected counts are imported as stored and are not original RNA counts.

## Data mapping

| Seurat content | AnnData destination |
|---|---|
| Selected `x_layer` | `X` and `layers[x_layer]` |
| `counts`, `data` when available | `layers['counts']`, `layers['data']` |
| Cell names and metadata | `obs_names`, `obs` |
| Feature names and metadata | `var_names`, `var` |
| Active identities | Categorical `obs['seurat_ident']` |
| Variable features | Boolean `var['highly_variable']` |
| Embeddings associated with the selected assay | `obsm['X_pca']`, `obsm['X_umap']`, etc. |
| Versions, source layers and split coverage | `uns['seurat']` |

Factor levels, their ordering and missing values are preserved. A pre-existing
`seurat_ident` metadata column is rejected to avoid overwriting it. List-valued
metadata is rejected with an error. This reader does not populate `raw`,
translate Seurat PCA model/loadings into Scanpy models, or transfer graphs,
commands, images, model fits or other assays. Embeddings can be plotted after
import; recompute neighbors in Python before downstream graph analyses.

X determines the cell and feature sets. Cells are ordered by the original
object metadata and features by the selected expression layer. Other layers
are aligned by identifiers and restricted to X's axes; extra cells/features
outside X are not retained. Embeddings with missing cells are NaN-padded.

## Split layers and matrix size

For an exact match, such as `counts` or `counts.batchA`, the reader imports
that layer. If no exact match exists, it combines matching names such as
`counts.batchA` and `counts.batchB`. Cells in these split layers must be
disjoint; overlaps raise an error instead of silently summing expression.
Sparse expression stays sparse. If batches have different feature sets, the
union is zero-filled with a warning; inspect the original per-layer features
and cells in `uns['seurat']['split_coverage']` before comparing such batches.

```python
# Import a single split layer, with no additional layers or embeddings.
adata = cs.io.read_seurat_rds(
    "merged.rds", x_layer="counts.batchA", layers=(), reductions=False
)

# Explicitly import scaled expression, which is often dense and HVG-only.
adata = cs.io.read_seurat_rds(
    "sample.rds", layers=("counts", "data", "scale.data"), timeout=600
)
```

`scale.data` is excluded by default to avoid loading a large dense matrix.
An additional layer missing some X cells/features is padded with **NaN**, not
zero, and a warning explains the dense allocation. Alignments requiring more
than 512 MiB for that output array are rejected; omit the incomplete layer or
select it as X. This is a guard for padding only, not a total memory limit.

The result always resides in memory. Disk-backed assays require the original
external files, valid paths, and their R packages (such as BPCells); an RDS
alone may not contain the expression matrix. The reader attempts sparse
materialization of these matrices but disk-backed formats are not covered by
the current integration tests. Unsupported backends fail with the R error.

## Validation

Real R-generated fixtures cover standard v5, split layers, legacy Assay,
normalized-only objects, partial scaled layers, assay-specific cells,
metadata types, feature alignment, embeddings and h5ad persistence. Run:

```bash
python -m pytest tests/test_seurat_import.py -q
```

The R integration tests skip when their dependencies are unavailable.
