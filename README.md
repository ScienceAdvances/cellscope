# cellscope

Single-cell transcriptome and cell-state analysis on AnnData and MuData.
The package contains RNA algorithms; other assay implementations are archived
outside this repository for separate packages.
Use cellscope for RNA workflows and immune for VDJ, clones, and joint
clone/state analysis. Both use native scverse data objects.

```python
import cellscope as cs

adata = cs.datasets.toy_rna()
cs.pp.qc(adata, min_genes=5)
cs.tl.workflow(adata, n_top_genes=30, n_pcs=10)
cs.pl.embedding(adata, color="cell_type")
composition = cs.tl.cell_composition(adata, groupby="cell_type")
```

Install `cellscope[clustering]` for Leiden, `cellscope[functional]` for
decoupler scores, `cellscope[differential]` for sample-level count inference,
and `cellscope[integration]` for optional scvi-tools models.

The public namespaces are `io`, `pp`, `tl`, `pl`, `get`, and `datasets`.

Clone phenotype tracking is available through `cs.tl.phenotrack`,
`cs.tl.phenotrack_aggregate`, `cs.pl.phenotrack`, and
`cs.pl.phenotrack_ternary`. These accept existing Scanpy state annotations
or state probabilities and supplied clone identities; no GeneVector is required.
See [PhenoTrack](docs/phenotrack.md) for normalization, unbalanced flows,
equal-donor means, plotting and export.
Runtime feedback defaults to concise INFO: an ASCII startup logo, current
library and key results. DEBUG adds parameters, nested steps, backend epochs
and long-task heartbeats. WARNING shows only warnings; QUIET disables messages. Configure it with
`cs.settings.verbosity = "quiet"` or `cs.logging.configure(log_file="analysis.log")`.
Set `CELLSCOPE_BANNER=0` before import to hide only the logo. See the
[runtime feedback guide](docs/user_feedback.md) for levels, batch messages and logging.

Core RNA workflows support lazy CPU/GPU selection with
`cs.settings.device = "auto"` (default), `"cpu"`, or strict `"gpu"`.
Supported functions accept a per-call `device=` override. Auto mode checks the
required GPU stack per operation and reports CPU fallbacks; successful calls
record the backend in `adata.uns["cellscope"]["devices"]`.
See [device selection](docs/device.md) for supported functions, parameter
compatibility, legacy flags, and algorithms that retain their CPU interface.

See [DESIGN.md](DESIGN.md) for purpose, ownership, data conventions,
interfaces, statistical assumptions, dependencies, and extension plans.

The [Single-cell Best Practices guide](docs/best_practices.md) maps tutorial
chapters to Python-first interfaces, focused optional dependencies and explicit
R backends. The RNA extension namespaces are `r` and `best_practices`. Inspect available routes with `cs.best_practices.methods()`.

Chinese version: [DESIGN.zh-CN.md](DESIGN.zh-CN.md).

Import Seurat v5 RDS objects directly into AnnData with
`cs.io.read_seurat_rds("sample.rds")`. The reader preserves RNA expression,
metadata, identities and embeddings, including split v5 layers. It requires
R with Seurat installed; see the [Seurat import guide](docs/seurat_import.md).

Use `cs.io.write_cloupe()` to export AnnData or MuData for 10x Loupe Browser.
See the [Loupe export guide](docs/loupe_export.md).

Cell-level `tl.find_markers` / `tl.find_all_markers` and sample-level
`tl.differential_expression` / `tl.pseudobulk_de` support explicit method
selection. Sample-level backends include PyDESeq2, pylimma (TMM/voom), and
edgePython (QL/LRT). Install `cellscope[de]` for all DE backends or select
`differential`, `limma`, or `edger`. See the
[differential analysis guide](docs/differential-expression.md) and
[executable example](examples/differential_expression.py).

Embedding plots with small lower-left axes: see [dimplot guide](docs/dimplot.md).

Numeric-axis PCA scree plots: see [PCA variance guide](docs/pca_variance_ratio.md).

All plotting functions share optional filename/outdir/formats/dpi export and native plot returns: [plot output guide](docs/plotting_output.md).
