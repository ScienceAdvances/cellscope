"""Harmony integration with explicit native CPU and RAPIDS GPU semantics."""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import numpy as np
from anndata import AnnData

from .. import _logging as feedback
from .._core import dependency, modality, record
from .._device import _host, _record_backend, select_gpu
from ..preprocessing._best_practices import _labels


@dataclass(frozen=True)
class GPUHarmonyResult:
    """GPU embedding result, not a harmonypy model or its convergence history."""

    Z_corr: np.ndarray
    flavor: str
    backend: str = "rapids_singlecell"


_ALIASES = {
    "nclust": "n_clusters", "lamb": "ridge_lambda", "block_size": "block_proportion",
    "max_iter_kmeans": "max_iter_clustering", "epsilon_cluster": "tol_clustering",
    "epsilon_harmony": "tol_harmony", "batch_prop_cutoff": "batch_prune_threshold",
}


def harmony(data, *, batch_key, mod="gex", use_rep="X_pca", key_added="X_pca_harmony",
            device=None, flavor=None, **kwargs):
    """Correct an embedding; leave expression, counts and source PCs unchanged.

    CPU returns the native harmonypy fit and preserves its installed defaults.
    GPU defaults to Harmony1 and returns GPUHarmonyResult with cells-by-PCs
    Z_corr. Explicit flavor='harmony1'/'harmony2' requires the GPU backend;
    it cannot silently fall back to a different native CPU algorithm.
    Common harmonypy parameter names are translated to RAPIDS names.
    """
    if flavor not in {None, "harmony1", "harmony2"}:
        raise ValueError("flavor must be None, 'harmony1', or 'harmony2'")
    adata = modality(data, mod)
    keys = [batch_key] if isinstance(batch_key, str) else list(batch_key)
    if not keys:
        raise ValueError("batch_key must contain at least one batch column")
    for key in keys:
        _labels(adata, key)
    embedding = np.asarray(adata.obsm[use_rep])
    if embedding.ndim != 2 or embedding.shape[0] != adata.n_obs or not np.isfinite(embedding).all():
        raise ValueError("Harmony requires a finite cells-by-components embedding")
    gpu_options = {}
    for key, value in kwargs.items():
        target = _ALIASES.get(key, key)
        if target in gpu_options:
            raise ValueError(f"Duplicate Harmony parameter after translation: {target}")
        gpu_options[target] = value
    use_gpu = select_gpu("tl.harmony", device=device, announce=False)
    gpu_function = None
    if use_gpu:
        rsc = dependency("rapids_singlecell", "gpu")
        gpu_function = getattr(rsc.pp, "harmony_integrate", None)
        reason = None
        if gpu_function is None or "flavor" not in inspect.signature(gpu_function).parameters:
            reason = "installed RAPIDS Harmony has no explicit Harmony1/Harmony2 selection"
        else:
            reserved = {"adata", "key", "basis", "adjusted_basis", "flavor"}
            unsupported = set(gpu_options) - (set(inspect.signature(gpu_function).parameters) - reserved)
            if unsupported:
                reason = "RAPIDS Harmony does not support options: " + ", ".join(sorted(unsupported))
            if "lamb" in kwargs and (
                kwargs["lamb"] is None or not np.isscalar(kwargs["lamb"]) or kwargs["lamb"] <= 0
            ):
                reason = "dynamic/vector lamb needs native harmonypy; GPU ridge_lambda must be positive scalar"
        if reason:
            use_gpu = select_gpu("tl.harmony", device=device, reason=reason, announce=False)
    if use_gpu:
        actual_flavor = flavor or "harmony1"
        feedback.info(f"tl.harmony | GPU / RAPIDS | {actual_flavor}")
        # Harmony operates on PCs only. Do not copy/transfer the expression assay.
        working = AnnData(obs=adata.obs[keys].copy())
        working.obsm[use_rep] = embedding.copy()
        gpu_function(working, key=keys, basis=use_rep, adjusted_basis=key_added,
                     flavor=actual_flavor, **gpu_options)
        corrected = np.asarray(_host(working.obsm[key_added]))
        fit = GPUHarmonyResult(corrected, actual_flavor)
        backend = "rapids_singlecell"
    else:
        if flavor is not None:
            raise RuntimeError("Explicit Harmony flavor requires GPU; omit flavor for native CPU Harmony")
        feedback.info("tl.harmony | CPU / harmonypy | installed native algorithm")
        hm = dependency("harmonypy", "batch")
        fit = hm.run_harmony(embedding.copy(), adata.obs[keys].copy(), keys, **kwargs)
        corrected = np.asarray(fit.Z_corr)
        # Old harmonypy returns PCs-by-cells; current versions use cells-by-PCs.
        if corrected.shape != embedding.shape and corrected.T.shape == embedding.shape:
            corrected = corrected.T
        backend, actual_flavor = "harmonypy", "harmonypy_native"
    if corrected.shape != embedding.shape or not np.isfinite(corrected).all():
        raise ValueError("Harmony returned a nonfinite or incompatible embedding")
    adata.obsm[key_added] = corrected
    record(adata, key_added, {
        "batch_key": keys, "use_rep": use_rep, "flavor": actual_flavor,
        "options": gpu_options if use_gpu else kwargs,
        "numerical_equivalence_claimed": False,
    }, backend=backend)
    if use_gpu:
        adata.uns["cellscope"][key_added]["backend_version"] = str(getattr(rsc, "__version__", "unknown"))
    _record_backend(adata, "tl", "harmony", "gpu" if use_gpu else "cpu", backend)
    adata.uns["cellscope"]["devices"]["tl.harmony"]["flavor"] = actual_flavor
    return fit
