# GeneVector adapters for the CloneTrack 2023 single-cell method

Rojas et al. (Nature 2023, doi:10.1038/s41586-023-06063-y) describe Scirpy receptor matching, unnormalized merged counts, GeneVector sample batch correction, CD4/CD8 marker classification, four CD8 states and Scanpy UMAP.

`tl.genevector_fit`, `tl.genevector_embedding` and `tl.genevector_phenotypes` expose the author's GeneVector implementation; there is no PCA/Harmony substitute. The adapter explicitly supports the pre-publication PyPI 0.1.0 source release (2023-01-10). This is a recorded implementation choice: the paper does not identify its exact GeneVector revision, full marker dictionaries or training hyperparameters, so exact historical numerical reproduction cannot be asserted.

The old release is a Python egg. Repackaging its unchanged `.py` files as a wheel is an installation compatibility change, not a model rewrite. Record source hashes and packaging details in the analysis environment manifest. CPU PyTorch, fast-histogram and GeneVector must be installed separately; the historical package's legacy metadata may not resolve with modern pip.

```python
# Select raw-count genes explicitly; selected genes are recorded, never hidden.
fit = cs.tl.genevector_fit(adata, output_dir='new_genevector_run',
                          layer='counts', mi_backend='original', c=1.0)
cs.tl.genevector_embedding(adata, fit, batch_key='sample_id', reference_batch='pre')
cs.tl.genevector_phenotypes(adata, fit, markers, key_added='gv_state')
cs.pp.neighbors(adata, use_rep='X_genevector', n_neighbors=50)
cs.tl.umap(adata, min_dist=0.3, random_state=0)
```

`mi_backend='original'` is the default. Optional `numba_exact` accelerates only the same conditional ten-bin histogram MI, verified against the author's implementation. Both preserve 0.1.0's target scaling (`c*c`), original Adadelta, expression-weighted cell embeddings and batch-vector shifts. The adapter always saves final weights because the author's early-stop branch otherwise returns without writing embeddings. Default undefined_policy="author" preserves upstream labels. Optional undefined_policy="flag" records zero-marker/nonfinite cells with a missing label, preserving the original probabilities; it is an explicit robustness option, not an alleged original-paper filter.

Marker sets must be supplied explicitly. They are saved with missing marker lists; no built-in marker collection is represented as the original paper's unpublished dictionary. Feature selection, seeds, epochs, stopping criterion, actual epochs, final loss and source hashes are recorded. Failure to early-stop is not labelled convergence.
