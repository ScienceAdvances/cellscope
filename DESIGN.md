# cellscope: purpose, architecture, and API

Status: implemented RNA/VDJ ecosystem foundation, October 2026.

## Purpose and package boundaries

cellscope analyzes single-cell gene expression and cell states using native
scverse data objects. It provides consistent workflows around Scanpy,
decoupler, PyDESeq2, and optional scvi-tools models. It owns RNA quality
control, expression preprocessing, embeddings, clustering, annotation,
functional scoring, pseudobulk inference, and cell composition.

immune owns immune receptor chains, clonotypes, repertoire statistics,
longitudinal clone tracking, clone/state relationships, and bulk-to-cell
matching. cellscope has no dependency on immune. immune only requires
cellscope for its optional clone-aware expression workflows.

Future ATAC and spatial packages can adopt this data contract. Their analysis
algorithms are outside this release. A common container does not imply that
every modality can use RNA normalization or share the same observation axis.

## Data contract

- Single-modality functions accept `AnnData`; multimodal functions accept
  `MuData` and select the `gex` modality by default using `mod="gex"`.
- `gex.X` stores the expression representation currently used for analysis.
  `gex.layers["counts"]` stores original nonnegative integer counts.
- `gex.obs` owns RNA QC, cluster labels, cell types, states, and RNA scores.
  Embeddings belong in `gex.obsm`; graphs in `gex.obsp`; backend outputs
  remain in native `gex.uns` locations.
- Cell identities use `library_id:barcode`. `barcode` preserves the original
  identifier. `sample_id` denotes a biological sample; `donor_id` denotes
  its biological donor. A donor can contribute several samples/libraries.
- Modality-specific columns are explicitly synchronized to global columns
  named `gex:<column>`. Public sample/donor/library/time/condition metadata
  is shared only when values agree for corresponding observations.
- Outer joins preserve observations without all modalities and provide
  `has_<modality>` flags. Inner joins are explicit. QC first annotates a
  mask; filtering a MuData object subsets every modality.
- Object-changing analysis rejects views: callers must materialize `.copy()`.
- Native `.h5ad` and `.h5mu` are persistence formats. No custom data class
  or Muon analysis dependency is required.

`uns["cellscope"][key]` records parameters, backend and backend version,
and schema version. Table-producing composition tools also store a table.
Large native models are returned, not serialized into `uns`; save them using
the model's own supported methods.

## Public namespaces

```python
import cellscope as cs

cs.io       # Native readers, identifiers, multimodal assembly, persistence
cs.pp       # Preprocessing and QC
cs.tl       # Expression/state tools and sample-level statistics
cs.pl       # Plotting
cs.get      # Tabular result access
cs.datasets # Deterministic example datasets
```

### IO

| Interface | Behavior |
| --- | --- |
| `io.read_10x(path, library_id=..., sample_id=..., donor_id=...)` | Scanpy 10x HDF5/directory reader with stable cell IDs |
| `io.set_cell_ids(adata, library_id=..., ...)` | Assign IDs before concatenation or modality assembly |
| `io.concat(samples, sample_key="sample_id", join="outer")` | Concatenate RNA AnnData; reject cell ID collisions |
| `io.create_mudata(modalities, join="outer")` | Assemble arbitrary modalities, validate shared metadata |
| `io.read_h5ad`, `io.read_h5mu`, `io.write` | Native persistence; extension checked on writing |

### Preprocessing

| Interface | Behavior and outputs |
| --- | --- |
| `pp.qc(data, mod="gex", layer=None, min_genes=..., min_counts=..., max_pct_mito=...)` | Scanpy QC metrics and `obs["qc_pass"]`; does not discard cells |
| `pp.filter_cells(data, key="qc_pass", mod="gex")` | Return an explicitly filtered native object |
| `pp.mad_outliers(data, metrics=..., batch_key=..., nmads=3)` | Batch-specific robust outlier annotations |
| `pp.normalize(data, layer=None, counts_layer="counts")` | Preserve counts once; normalize/log1p into X; repeated calls reuse counts |
| `pp.highly_variable_genes(data, batch_key=..., ...)` | Scanpy HVG selection without dropping other genes |
| `pp.neighbors(data, ...)` | Native Scanpy neighbor graph |
| `pp.doublets(data, layer="counts", batch_key=...)` | Scrublet on raw counts; attach annotations without changing RNA X |

### Tools

| Interface | Behavior and return |
| --- | --- |
| `tl.workflow(data, ...)` | Normalize, HVG, PCA, neighbors, UMAP, Leiden; return modified object |
| `tl.pca`, `tl.umap`, `tl.leiden` | Individually callable Scanpy stages |
| `tl.markers(data, groupby=..., method=...)` | Low-level Scanpy ranking plus native-column DataFrame; default `use_raw=False` |
| `tl.find_markers(data, groupby=..., ident_1=..., ident_2=None, method=...)` | Cell-level comparison versus selected identities/rest; canonical table, detection/effect filters |
| `tl.find_all_markers(data, groupby=..., method=...)` | Each observed identity versus rest; same filters/schema as find_markers |
| `tl.annotate(data, mapping, reference_key=..., key_added="cell_type")` | Cluster-to-label mapping; retain unannotated cells |
| `tl.score_genes(data, gene_sets)` | Scanpy gene-set scores in obs |
| `tl.aucell(data, gene_list=..., network=...)` | decoupler 2.x AUCell; native `obsm["score_aucell"]` |
| `tl.activity(data, network, method="ulm")` | Explicit pathway/TF network, native decoupler scores |
| `tl.scvi(data, layer="counts", batch_key=...)` | SCVI model; latent representation in obsm |
| `tl.scanvi(data, labels_key=..., unlabeled_category=...)` | SCANVI model, predictions, probabilities, latent representation |
| `tl.reference_mapping(data, reference_model, ...)` | scvi-tools query adaptation; return native model |
| `tl.pseudobulk(data, sample_col=..., groups_col=..., metadata_cols=...)` | decoupler sum aggregation from raw counts; return AnnData |
| `tl.differential_expression(pdata, method=..., design=..., contrast=..., groups_col=...)` | Per-group PyDESeq2, TMM/voom pylimma, or edgePython QL/LRT; canonical table and native models |
| `tl.pseudobulk_de(data, method=..., design=..., contrast=..., metadata_cols=...)` | Raw-count aggregation plus sample-level DE; return table, profiles, models |
| `tl.de_methods()` | Method inventory with statistical units, backends, installation extras |
| `tl.cell_composition(data, groupby=..., condition_col=..., donor_col=...)` | Complete sample/category counts and fractions, including zeros |
| `tl.composition_test(table, condition_col=..., comparison=..., reference=..., donor_col=...)` | Sample Mann-Whitney or paired-donor Wilcoxon, BH across categories |
| `tl.trajectory(data, groupby=..., root=...)` | PAGA, optional DPT with an explicit root cell |

### Plotting, extraction, examples

`pl.embedding`, `pl.dotplot`, `pl.heatmap`, and `pl.qc` delegate to Scanpy;
`pl.cell_composition` and `pl.volcano` use Matplotlib/Pandas. They return
native axes/plot objects and do not save files implicitly. A caller can pass
axes and explicitly save the resulting figure.

`get.obs_df`, `get.markers_df`, and `get.result` return extracted copies.
`datasets.toy_rna()` creates deterministic raw-count data.

`get.de_df` extracts copied canonical differential tables. Cell-marker
tables are available through `get.markers_df(..., key='find_all_markers')`.

## Statistical contract

Cell-level marker methods include Scanpy Wilcoxon, Welch t-test, its
conservative variance variant, and logistic coefficient ranking (no
p-values). Marker inputs are nonnegative log1p expression. Detection
filters precede testing; effect filters follow within-comparison correction.
These methods do not model cells nested in donors and do not replace
sample-level condition inference. This is a Seurat-inspired interface,
not a claim of numerical parity with Seurat or its filtering defaults.

Pseudobulk and DE require nonnegative integer counts. Sample covariates must
be constant within sample. At least two samples per contrasted condition
are required by default. Paired DE uses an explicit design such as
`~ donor_id + condition`. Samples must represent biological replicates;
technical replicate libraries should be merged before inference.

Sample-level backends use the same explicit Patsy design and contrast;
positive effects mean comparison versus reference. Interaction contrasts
average counterfactual design differences over observed covariates.
Rank-deficient designs, missing covariates, zero-depth profiles, and designs
without residual degrees of freedom are rejected. Gene filtering defaults
to total counts; optional edgePython `filter_by_expr` supports a common
gene universe across backends. Correction is within each cell group's
tested genes. Native model objects are returned, not stored in `uns`.

See [differential analysis](docs/differential-expression.md) for API examples,
backend options, method limitations, result schemas, and persistence.

Composition fractions use annotated cells per sample as the denominator.
Zero categories are retained. `composition_test` is an exploratory marginal
fraction test, not a complete compositional regression. Supply `donor_col`
for paired data; independent tests assume independently sampled subjects.
Low cell counts, imbalanced sampling, and multiple tissues require a
study-specific design rather than treating cells as independent donors.

## Dependencies and compatibility

Base: AnnData, MuData, NumPy, Pandas, SciPy, Scanpy, Matplotlib.
Extras: `clustering`, `functional`, `differential`, `limma`, `edger`, `de`,
`integration`, `dev`. `de` installs all differential backends. The limma
extra selects John Mulvey's PyPI `pylimma`, not the distinct `python-limma`
project sharing its import name; mixed installations are rejected.
Heavy models are imported only when requested. Importing cellscope no
longer requires the unrelated `atopos` utility package.

Existing `pp.normalise`, `pp.fastqc`, `pp.mad_filter`, `tl.add_label`,
`tl.find_all_markers`, `tl.deseq`, `pl.dimplot`, `pl.plot_batch_effect`, and
`pl.cell_ratio` remain accessible. New workflows should use the APIs above.
`tl.deseq` now requires an explicit raw-count layer and current PyDESeq2
metadata/design interfaces. Versions in package metadata and `__version__`
are consistent. Publishing/version bumps are separate release decisions.

`tl.find_all_markers` now defaults to canonical columns and `use_raw=False`.
Use `legacy=True` for the previous column layout, or the compatibility
entry point in `tools.gene_level_analysis`.

## References and reuse

Backend algorithms are called through public APIs; no third-party source
files are vendored. Source and design references:

- [Scanpy](https://github.com/scverse/scanpy): `pp/tl/pl/get`, native graph and ranking results.
- [AnnData](https://github.com/scverse/anndata): matrix/annotation storage and slicing.
- [MuData](https://github.com/scverse/mudata): modality containers and explicit metadata synchronization.
- [decoupler](https://github.com/saezlab/decoupler-py): activities and pseudobulk.
- [PyDESeq2](https://github.com/owkin/PyDESeq2): count-based differential expression.
- [pylimma](https://github.com/john-mulvey/pylimma): voom and empirical-Bayes moderation.
- [edgePython](https://github.com/pachterlab/edgepython): TMM, QL/LRT, expression filtering.
- [Seurat](https://satijalab.org/seurat/reference/findallmarkers): marker API inspiration.
- [scvi-tools](https://github.com/scverse/scvi-tools): latent models and query adaptation.

If future work copies source, inspect the specific repository license and
retain its required notices. Backend citations should also be included in
scientific publications using those algorithms.

## Roadmap and current limits

The foundation and interfaces above are implemented. RNA/VDJ joint examples
live in immune. Optional deep-model adapters require installed backends;
their presence does not guarantee suitability or convergence on a study.
Future work includes more reference atlases, model benchmarking, joint
compositional regression, neighborhood differential abundance, and larger
data/backed-array optimization. Spatial spot mapping and ATAC pipelines
remain separate package responsibilities.
