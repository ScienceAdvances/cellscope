# Single-cell Best Practices interfaces

cellscope 1.3 exposes Python interfaces for the single-cell transcriptome workflows in
[Single-cell Best Practices](https://www.sc-best-practices.org/). The tutorial
is a guide to study design and analysis, rather than one pipeline that should
be run on every dataset. Select the methods and parameters appropriate to the
assay and biological question. cellscope owns RNA analysis; immune owns V(D)J.
Spatial, ATAC, ADT, cross-modality models, bulk deconvolution and barcode
lineage implementations have been archived outside both packages for future reuse.
AnnData/MuData interchange is retained to share RNA objects and receptor annotations.

## Backend selection

Use an available Python implementation by default. For algorithms with a
Python port, `backend="r"` is an explicit option where implemented. R-only
adapters invoke the original R package through an isolated `Rscript` process.
No R process, download, package installation or model training happens on
import. A failed R call raises a Python error instead of terminating Python.

- SoupX: `pysoupx` by default; optional R `SoupX`.
- SCTransform: Python `sctransform.vst`; optional R `sctransform::vst`.
- GLM-PCA: the original author's Python `glmpca`; optional R `glmpca`.
- Slingshot: `pyslingshot`; optional R `slingshot`.
- NicheNet: the official `nichenetpy` ligand-activity implementation;
  optional R `nichenetr`.
- Pooled size factors: R `scran::computeSumFactors`. scranpy normalization
  functions do not implement this pooled-deconvolution algorithm.
- Binomial deviance: R `scry::devianceFeatureSelection`.
- scDblFinder: R `scDblFinder`; `pp.doublets` separately exposes Python Scrublet.
  These are different doublet algorithms.
- Trajectory-associated expression: R `tradeSeq`. The available Python
  tradeSeq project has not established a complete stable equivalent API.

Ports are implementations of the same named method, not a promise of identical
results or full feature parity. Python SCTransform exposes residuals and gene
statistics here; it does not provide Seurat's entire SCT assay workflow.
LIANA's Python CellChat scoring method does not expose the full CellChat R
object and its downstream plotting ecosystem. Statistical backends and their
parameter names remain explicit; a failed Python call never silently switches
to a different algorithm.

## Installation

Python 3.12 is the recommended environment for the current optional backends.
Install from the local checkout while developing these unreleased versions:

```bash
cd /Users/tim/Documents/Repo/cellscope
python -m pip install -e '.[best-practices,clustering,de,integration]'
```

The `best-practices` extra installs common RNA, composition, perturbation,
and RNA communication adapters. It deliberately does not bundle every large
model, regulatory reference resource or R package.
Focused extras are listed by `cs.best_practices.methods()`:

| Analysis | Extra |
|---|---|
| SoupX / SCTransform | `ambient` / `normalization` |
| CellTypist / Harmony, BBKNN | `annotation` / `batch` |
| scVelo, Palantir / GLM-PCA | `trajectory` / `glmpca` |
| scCODA, tascCODA, Milo | `composition` |
| RNA Mixscape and perturbation distances | `perturbation` |
| LIANA, NicheNet | `communication` |
| RNA regulatory networks: Arboreto, pySCENIC | `grn` |
| scVI and scANVI RNA models | `integration` |

PySlingshot 0.2 requires NumPy <2. Current Pertpy 1.4 imports functions
introduced in Scanpy 1.12, whose execution paths use NumPy 2 features. The
`slingshot` extra therefore uses Scanpy <1.12, while `composition`,
`perturbation` and `best-practices` use Scanpy >=1.12. Use separate environments
for these extras, or explicitly use `backend="r"` for Slingshot in a Pertpy
environment. The installer rejects the conflicting combination instead of
producing an environment that fails during analysis.

The tested common-environment versions are in
`requirements/best-practices-tested.txt`. Pass that file with pip's `-c`
option to constrain only the dependencies you install.

pySCENIC 0.12.1 on PyPI imports the removed `np.object` alias during motif
pruning. Its official source fixes that import. The `grn` extra pins
Setuptools <81 for ctxcore's older `pkg_resources` import. For motif pruning
with current NumPy, install the inspected official revision explicitly:

```bash
python -m pip install -e '.[grn]'
python -m pip install --no-deps --upgrade 'pyscenic @ git+https://github.com/aertslab/pySCENIC.git@06bafba412792f6efa5a552a23bb221cc3bdea1b'
```

The adapter reports an actionable error for the older NumPy-incompatible
release; it does not modify NumPy's global API.

```bash
# Dedicated Python Slingshot environment
uv venv --python 3.12 .venv-slingshot
uv pip install --python .venv-slingshot/bin/python -e '.[slingshot]'
```

For R adapters, install R and the needed packages yourself. `cellscope[r]`
is a marker extra; it requires no rpy2 installation. A helper installs the
common R packages only when explicitly run:

```bash
Rscript scripts/install_r_backends.R
# Optional SoupX/SCT/GLM-PCA and GitHub NicheNet packages:
Rscript scripts/install_r_backends.R --extended
```

`CELLSCOPE_RSCRIPT` can point to a specific executable. `R_LIBS_USER` selects
a private library. Both are inherited by analysis subprocesses.

```python
import cellscope as cs
print(cs.r.backend_status())
print(cs.best_practices.methods("Normalization"))
```

## Tutorial coverage and public interfaces

The complete machine-readable routing table is `cs.best_practices.methods()`.
The table below groups the interfaces by analysis task, including the existing
interfaces reused by this extension.

| Tutorial task | Interfaces |
|---|---|
| Native data loading and interoperability | `io.read_10x`, `io.read_h5ad`, `io.read_seurat_rds`, `io.read_h5mu`, `io.create_mudata`, `r.to_sce`, `r.from_sce` |
| RNA quality control and ambient RNA | `pp.qc`, `pp.filter_cells`, `pp.doublets`, `pp.scdblfinder`, `pp.soupx` |
| Normalization and feature selection | `pp.normalize`, `pp.scran`, `pp.sctransform`, `pp.pearson_residuals`, `pp.pearson_hvg`, `pp.deviance_features`, `pp.highly_variable_genes` |
| Embeddings and clustering | `tl.pca`, `tl.glmpca`, `tl.tsne`, `pp.neighbors`, `tl.leiden`, `tl.umap` |
| Cell annotation | `tl.celltypist`, `tl.scanvi`, `tl.reference_mapping`, `tl.annotate` |
| Batch integration | `tl.harmony`, `tl.bbknn`, `tl.scvi` |
| Trajectories and velocity | `tl.trajectory` (PAGA/DPT), `tl.slingshot`, `tl.palantir`, `tl.velocity`, `tl.tradeseq` |
| Markers and replicated differential expression | `tl.find_markers`, `tl.find_all_markers`, `tl.pseudobulk`, `tl.pseudobulk_de`, `tl.differential_expression` |
| Differential composition | `tl.sccoda` (also tascCODA via `tree_levels`), `tl.milo` |
| Pathway and TF activity | `tl.activity`, `tl.aucell`, `tl.score_genes` |
| Expression regulatory networks | `tl.grn` (Arboreto), `tl.scenic_regulons` (motif pruning), `tl.regulon_activity` (pySCENIC AUCell) |
| Cell communication and ligand activity | `tl.communication` (LIANA methods), `tl.nichenet` |
| RNA perturbation responses | `tl.mixscape`, `tl.perturbation_distance` |
| Immune receptor and joint RNA/VDJ analysis | `immune`; see its best-practices guide |

FASTQ alignment/quantification remains an upstream Cell Ranger/STARsolo/
kallisto workflow. This release does not wrap command-line quantifiers, the
book's data downloads, GPU-specific rapids-singlecell operations, or every R
plotting API. The adapters expose the principal
downstream analysis tasks and native outputs; they do not recreate all notebook
code or automatically choose reference genomes, models or biological labels.

## Data contracts

For RNA adapters, `data` is AnnData or MuData and `mod="gex"` selects the RNA
modality. Override `mod` for RNA objects using `rna` or another convention.
MuData is a shared container; the RNA adapters operate only on the selected RNA modality.

- Keep raw integer counts in `layers["counts"]`. Use `layer=None` only when
  `X` actually contains counts. Negative, nonfinite and fractional count input
  is rejected by count-based adapters.
- Cell and feature identifiers must be unique. Matrix exchange with R flips
  cells-by-genes to genes-by-cells explicitly and retains identifier order.
  Sparse inputs remain sparse across the bridge where the R algorithm permits.
- Nullable boolean metadata retains logical values and missingness across
  the SCE bridge. Named R vectors decode as pandas Series with their names and
  value order; unnamed vectors remain arrays.
- Ambient correction writes `soupx_counts`; scran writes
  `scran_normalization`; SCT and Pearson write residual layers. These functions
  leave `X` and the original count layer intact. `pp.normalize` explicitly
  changes `X`, while preserving the original counts.
- SCT excluded genes have NaN residuals and `var["sct_modeled"] == False`.
  Subset to modeled genes before PCA. Residuals are unsuitable input for
  integer-count differential-expression models.
- New annotations are published from the selected modality to MuData `obs`
  using the modality prefix. Each recorded result includes its backend version
  in `uns["cellscope"]`; R results include the R package versions.
- Statistical composition and differential expression use biological samples
  as replicates. Covariates must be constant within a sample. A cell is not an
  independent donor replicate.
- Methods that train models return native fitted models for diagnostics.
  Velocity returns a processed copy. R S4 fits are returned as `r.RObject` containing
  RDS bytes; they are not Python model objects or H5AD-compatible `uns` values.
- Optional native `**kwargs` remain backend-specific: Python SoupX uses
  `round_to_int`, R SoupX uses `roundToInt`; Python SCT uses `vst_flavor`,
  R SCT uses `vst.flavor`. Pass the latter with `**{"vst.flavor": "v2"}`.
  SoupX option dictionaries override defaults such as `roundToInt` and
  `doPlot`; the input channel `sc` is reserved.

For SCENIC, infer edges with `tl.grn`, then pass local cisTarget databases
and motif annotations to `tl.scenic_regulons`. It returns a motif-enrichment
table and native regulons; pass those regulons to `tl.regulon_activity`.
Database filenames and gene identifiers must match the genome build.

## RNA workflow

```python
import cellscope as cs

adata = cs.io.read_h5ad("filtered.h5ad")
raw = cs.io.read_h5ad("unfiltered.h5ad")  # same capture; includes empty droplets
# Existing cell_type or provisional clusters are needed for SoupX/scran.
cs.pp.soupx(adata, raw, cluster_key="provisional_cluster")
cs.pp.scran(adata, cluster_key="provisional_cluster", layer="soupx_counts")
cs.pp.deviance_features(adata, layer="soupx_counts", n_top_genes=4000)
adata.var["highly_variable"] = adata.var["highly_deviant"]
adata.X = adata.layers["scran_normalization"].copy()
cs.tl.pca(adata, n_comps=30)
cs.tl.harmony(adata, batch_key="sample_id")
cs.pp.neighbors(adata, use_rep="X_pca_harmony")
cs.tl.leiden(adata)
cs.tl.umap(adata)
# A CellTypist model requires log1p(10,000-normalized) expression separately.
```

Run the synthetic, download-free example with
`python examples/best_practices.py`; add `--with-r` for scran, scry and
R Slingshot/tradeSeq. This example avoids relying on external models or priors.

## Trajectory, composition and communication

```python
fit = cs.tl.slingshot(
    adata, cluster_key="leiden", start_cluster="0", backend="r"
)
gene_tests, gam_fit = cs.tl.tradeseq(adata, layer="counts", n_knots=6)

coda_data, coda = cs.tl.sccoda(
    adata, sample_key="sample_id", cell_type_key="cell_type",
    formula="condition", covariate_keys=["condition"],
    reference_cell_type="reference_type",
)
nhood_data, milo = cs.tl.milo(
    adata, sample_key="sample_id", design="~condition",
    neighbors_kwargs={"use_rep": "X_pca_harmony"},
)
# log-normalized expression, explicit ligand/receptor resource if needed
interactions = cs.tl.communication(adata, groupby="cell_type")
activities = cs.tl.nichenet(
    receiver_de_genes, expressed_receiver_genes, expressed_sender_ligands,
    ligand_target_prior,  # DataFrame: genes x ligands
)
```

NicheNet input gene sets, sender ligands and prior matrices are explicit; the
adapter does not infer them from unrelated marker tables.

## Verification and limits

Tests run real installed backends with synthetic data and check named
alignment, sparse orientation, unchanged count assays, backend errors and
native result structure. They include Python SoupX/SCT/Slingshot/GLM-PCA,
R scran/scry/SoupX/SCT/Slingshot/tradeSeq/GLM-PCA, Harmony, NicheNet, LIANA
communication, Pertpy Milo/scCODA, and pySCENIC AUCell. Optional tests skip
explicitly when a backend is absent. Scope checks ensure archived interfaces,
modules and dependency extras are absent from the active packages.

These checks do not establish scientific validity on a real cohort or numerical
parity between Python/R ports. Arboreto and cisTarget motif pruning,
scDblFinder and R NicheNet have not been executed end to end in the development
environment. Their adapters require validation on the intended cohort. GPU
training is not tested here.

Primary method references: [Python SoupX](https://github.com/omicverse/py-soupx),
[Python SCT](https://pypi.org/project/sctransform/),
[scranpy API](https://libscran.github.io/scranpy/api/scranpy.html),
[PySlingshot](https://github.com/mossjacob/pyslingshot/),
[Python tradeSeq project](https://github.com/WeilerP/tradeSeq-py),
[official Python NicheNet](https://github.com/saeyslab/nichenetpy).
