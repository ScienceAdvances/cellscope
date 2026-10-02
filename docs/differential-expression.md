# Differential expression and marker discovery

cellscope separates two analysis units. `find_markers` and `find_all_markers`
identify cell-level markers. `differential_expression` and `pseudobulk_de`
test sample-level count profiles. These interfaces take AnnData or MuData
(RNA modality `gex` by default) and call public backend APIs; they do not
launch R or command-line tools.

## Method selection

| Analysis | `method` | Backend and interpretation |
| --- | --- | --- |
| Cell markers | `wilcoxon` / `wilcox` | Scanpy Wilcoxon rank-sum |
| Cell markers | `t-test` / `t` | Scanpy Welch t-test |
| Cell markers | `t-test_overestim_var` | Scanpy conservative variance estimate |
| Cell markers | `logreg` | scikit-learn coefficient ranking; no inferential p-values |
| Sample DE | `pydeseq2` / `deseq2` | PyDESeq2 negative-binomial Wald test |
| Sample DE | `pylimma` / `limma` / `limma_voom` | TMM normalization, voom precision weights, moderated t-test |
| Sample DE | `edgepython` / `edgepython_ql` / `edger` | TMM normalization, negative-binomial GLM, QL F-test |
| Sample DE | `edgepython_lrt` | TMM normalization, negative-binomial GLM likelihood-ratio test |

`cs.tl.de_methods()` returns this inventory as a DataFrame. No method is
silently substituted when a backend is unavailable or a model fails.
The Python ports are not claimed to be identical to every R release.
cellscope delegates the statistics and does not independently establish
whole-package parity with Seurat, limma, edgeR, or DESeq2.

Install optional backends with `cellscope[differential]` (PyDESeq2),
`cellscope[limma]`, `cellscope[edger]`, or `cellscope[de]` (all DE backends).
The limma extra includes edgePython for TMM normalization.

The selected limma distribution is **`pylimma` from John Mulvey**, not
`python-limma` from omicverse. Both use the `pylimma` import namespace;
do not install them in the same environment. cellscope rejects that
collision with an actionable error. These Python limma/edgeR ports are
GPL-licensed optional dependencies; no source is vendored here. Review
upstream licenses when redistributing software or environments.

## Cell-level markers: FindMarkers and FindAllMarkers

Use nonnegative **log1p-normalized expression**, not raw counts,
scaled expression, a corrected embedding, or an integration latent space.
Normalize first, or select a layer containing that representation.
`use_raw=True` means AnnData's `.raw`, which must contain log-normalized
expression; `.raw` is not automatically synonymous with raw counts.

```python
import cellscope as cs

adata = cs.datasets.toy_rna()
cs.pp.normalize(adata)

pair = cs.tl.find_markers(
    adata,
    groupby="cell_type",
    ident_1="T",
    ident_2="B",
    method="wilcoxon",
    min_pct=0.1,
    logfc_threshold=0.25,
)

all_markers = cs.tl.find_all_markers(
    adata,
    groupby="cell_type",
    method="t-test",
    only_pos=True,
    min_pct=0.1,
    logfc_threshold=0.25,
)
```

`ident_1` and `ident_2` accept a scalar or list of identities; lists pool
cells, and omitting `ident_2` compares against the rest. Missing grouping
labels, overlapping identities, and undersized groups are rejected.

`min_pct`, `min_diff_pct`, and `features` filter the tested gene universe.
Correction defaults to BH within each comparison; `corr_method='bonferroni'`
is also supported. `logfc_threshold` and `only_pos` filter after testing
and correction. This is inspired by Seurat's interface, not an exact copy
of its filtering order or adjustment defaults.

Positive `log2FoldChange` means higher expression in `ident_1`. It is
`log2((mean(expm1(log-expression)) + 1e-9) / (reference mean + 1e-9))`,
accounting for the recorded log base. `pct_nz_group` and
`pct_nz_reference` are detected-cell fractions. Logistic regression is
binary for each comparison: a positive coefficient supports `ident_1`.
Its `pvalue` and `padj` are missing, not zero. It is not Seurat's inferential
`LR` likelihood-ratio test.

`cs.tl.markers` remains the low-level Scanpy-native interface, with Scanpy
columns and native `rank_genes_groups` results. The new interfaces return
canonical tables. For the old `Feature` index / `Identy`, `LogFC`, `Padj`
column layout, use `cs.tl.find_all_markers(..., legacy=True)`; the default
now uses the canonical schema and `use_raw=False`.

Cell-level p-values are exploratory marker statistics. They do not account
for cells nested within donors and should not establish condition effects
by treating every cell as an independent biological replicate.

## Sample-level DE: pseudobulk

Sum **raw integer counts** per biological sample and cell type/state.
Specify all design covariates in `metadata_cols`. Technical replicate
libraries must be combined into biological samples before inference.
Use `groups_col=None` to aggregate all included cells per sample rather
than splitting by cell type; interpret that comparison in light of changes
in cell composition.

```python
pdata = cs.tl.pseudobulk(
    adata,
    sample_col="sample_id",
    groups_col="cell_type",
    layer="counts",
    min_cells=1,
    metadata_cols=["condition", "donor_id"],
)

results, models = cs.tl.differential_expression(
    pdata,
    method="pylimma",
    design="~ donor_id + condition",
    contrast=("condition", "post", "pre"),
    groups_col="cell_type",
)
```

Switch `method` to `edgepython_ql`, `edgepython_lrt`, or `pydeseq2` without
changing the contrast direction or canonical result columns. Input is
raw counts in `pdata.X` by default; normalized/scaled matrices are rejected
when they violate count requirements. Each cell group must have one
profile per sample and at least two samples per contrasted condition.
This checks sample counts, not that the supplied sample IDs truly encode
independent biological replicates; experimental design remains essential.

Zero-depth profiles, inconsistent sample conditions, missing design
covariates, rank-deficient/confounded designs, and designs without
residual degrees of freedom are rejected. Small replicated datasets can
still be statistically underpowered even when these checks pass.

All backends use the same Patsy design matrix and contrast vector.
For an additive paired design, positive effects mean `post` versus `pre`.
For interactions, the contrast is the average difference between design
rows with the factor set to each level, using the observed covariate
distribution. It is not automatically an interaction-only coefficient.
Design matrices and contrast vectors are retained for inspection.

The default gene filter is `min_total_count=10`, excluding all-zero genes.
For edgeR-style expression filtering, set `filter_method='filter_by_expr'`
and optionally `filter_kwargs={'min_count': 10, 'min_prop': 0.7}`.
This uses edgePython's public filter, requires the `edger`/`de` extra,
and can be applied with any DE backend for a common gene universe.
Original library sizes are preserved for TMM/voom after gene filtering.

Backend-specific options are explicit nested dictionaries, for example:

```python
results, models = cs.tl.differential_expression(
    pdata,
    method="pydeseq2",
    design="~ donor_id + condition",
    contrast=("condition", "post", "pre"),
    groups_col="cell_type",
    backend_kwargs={"dds": {"size_factors_fit_type": "poscounts"}},
)
```

Accepted sections are `dds/stats` for PyDESeq2;
`normalization/voom/fit/ebayes` for pylimma;
`normalization/dispersion/fit/test` for edgePython. Unknown sections are
rejected. `n_cpus` controls PyDESeq2; the other adapters use their backend's
execution behavior. There is no automatic count-model fallback.

## One-step pseudobulk and DE

```python
results, pdata, models = cs.tl.pseudobulk_de(
    adata,
    method="edgepython_ql",
    sample_col="sample_id",
    groups_col="cell_type",
    layer="counts",
    min_cells=1,
    metadata_cols=["donor_id"],
    design="~ donor_id + condition",
    contrast=("condition", "post", "pre"),
)
```

The contrast factor is automatically carried into pseudobulk metadata;
other design covariates must be specified. `min_cells=1` is for the tiny
toy example, not a recommended universal threshold. Use `pseudobulk_kwargs`
and `de_kwargs` to configure the two stages separately.

## Result contract and persistence

Canonical columns include `gene`, `group`, `comparison`, `reference`,
`log2FoldChange`, `pvalue`, `padj`, `method`, and `unit`. Backend-specific
columns (such as `F`, `LR`, `t`, `baseMean`, and `lfcSE`) remain available.
Adjustment is within each group's tested genes, not across every cell type
and every contrast in the study. PyDESeq2's independent filtering/Cook's
rules can produce missing p-values or adjusted p-values; preserve them.

Results and provenance are in `uns['cellscope'][key_added]`, including
tables, method, backend version, formula, contrast, filters, and design
matrices. limma uses edgePython normalization in addition to pylimma.
Native models are returned, not serialized into `uns`:

- PyDESeq2: `(DeseqDataSet, DeseqStats)` per group.
- pylimma: `(DGEList, voom output, moderated fit)` per group.
- edgePython: `(DGEList, GLM fit, test result)` per group.

`cs.get.de_df(pdata)` extracts a copied differential table.
`cs.get.markers_df(adata, key='find_all_markers')` extracts a copied marker
table. Results can be persisted with `.h5ad`/`.h5mu`; retain or save native
models separately using supported backend formats.

## immune integration

`immune.tl.clone_expression(..., method='pylimma')` forwards method choice
to cellscope; `de_kwargs` passes additional DE settings. It compares
conditions within receptor-defined groups, not expanded versus unexpanded
cells directly. Install the selected cellscope extra in the same environment.

## References

- [Seurat FindMarkers](https://satijalab.org/seurat/reference/findmarkers)
  and [FindAllMarkers](https://satijalab.org/seurat/reference/findallmarkers).
- [Scanpy rank_genes_groups](https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.rank_genes_groups.html).
- [John Mulvey pylimma](https://github.com/john-mulvey/pylimma).
- [pachterlab edgePython](https://github.com/pachterlab/edgepython).
- [PyDESeq2](https://github.com/owkin/PyDESeq2).
