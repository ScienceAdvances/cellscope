# cellscope

Single-cell gene expression and cell-state analysis on AnnData and MuData.
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
See [DESIGN.md](DESIGN.md) for purpose, ownership, data conventions,
interfaces, statistical assumptions, dependencies, and extension plans.

Chinese version: [DESIGN.zh-CN.md](DESIGN.zh-CN.md).

Cell-level `tl.find_markers` / `tl.find_all_markers` and sample-level
`tl.differential_expression` / `tl.pseudobulk_de` support explicit method
selection. Sample-level backends include PyDESeq2, pylimma (TMM/voom), and
edgePython (QL/LRT). Install `cellscope[de]` for all DE backends or select
`differential`, `limma`, or `edger`. See the
[differential analysis guide](docs/differential-expression.md) and
[executable example](examples/differential_expression.py).
