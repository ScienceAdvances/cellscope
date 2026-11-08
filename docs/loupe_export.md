# Export to 10x Loupe Browser

`cs.io.write_cloupe` exports single-cell GEX from AnnData or the GEX modality
of MuData. Python writes the intermediate HDF5 and invokes the official 10x
`louper` converter. The Loupe Browser viewer opens the resulting `.cloupe`.

```python
import cellscope as cs

adata = cs.io.read_h5ad("rna_scanpy.h5ad")
output = cs.io.write_cloupe(
    adata,
    "rna_scanpy.cloupe",
    layer="counts",
    cluster_keys=["leiden", "sample_id"],
    projection_keys=["X_umap"],
    executable_path="/path/to/louper-linux-x64",
)
```

Provide a 10x Louper CLI binary via `executable_path`, the
`CELLSCOPE_LOUPER` environment variable, or a `louper` executable on PATH.
If none is found, the function checks its local cache and supports downloading
10x's Linux x86-64 v1.1.5 binary. After reviewing and accepting the
[10x software license](https://10xgen.com/EULA), omit `executable_path` and pass
`accept_eula=True` to authorize the download. The binary is verified against a
pinned checksum and cached under `~/.cache/cellscope/louper/v1.1.5` by default.
`cache_dir` selects the parent of the versioned directory, and
`auto_download=False` requires an available executable.

Choose a layer with nonnegative integer UMI counts. Use
`layer="raw_counts"` for observed counts or `layer=None` when the integer
matrix is in X. Cell IDs default to `obs_names`. The usual
`sample:ACGTACGTACGTACGT-1` format preserves sample identity in merged data.
Exported labels come from `obs` and the projections from two-column arrays in
`obsm`. Feature IDs default to `var["gene_ids"]` or `var["gene_id"]` when
available. The export preserves the original row and column order.

The return value is the output path. Use `overwrite=True` to replace an
existing file after a successful conversion. The function accepts AnnData
views after materialization with `.copy()`.

[10x LoupeR converter and compatibility notes](https://github.com/10XGenomics/loupeR)
