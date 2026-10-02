"""Small data contract shared by all public cellscope namespaces."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

import numpy as np
from anndata import AnnData
from scipy import sparse


def modality(data, mod="gex") -> AnnData:
    """Resolve a modality without copying its expression matrix."""
    adata = data.mod[mod] if hasattr(data, "mod") else data
    if not isinstance(adata, AnnData):
        raise TypeError("Expected AnnData or MuData")
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError("Observation and feature identifiers must be unique")
    if adata.is_view:
        raise ValueError("Materialize AnnData views with .copy() before analysis")
    return adata


def dependency(name, extra):
    try:
        return import_module(name)
    except ImportError as error:
        raise ImportError(f"Install cellscope[{extra}] to use {name}") from error


def counts(adata, layer="counts", *, integer=True):
    matrix = adata.X if layer is None else adata.layers[layer]
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Counts must be finite and nonnegative")
    if integer and not np.allclose(values, np.rint(values)):
        raise ValueError("Raw integer counts are required; choose the counts layer")
    return matrix


def record(data, key, params, *, backend="cellscope"):
    try:
        backend_version = version(backend)
    except PackageNotFoundError:
        backend_version = "source"
    data.uns.setdefault("cellscope", {})[key] = {
        "params": params,
        "backend": backend,
        "backend_version": backend_version,
        "schema_version": "1.0",
    }


def sync_obs(data, mod="gex"):
    """Explicitly publish modality annotations, including on MuData >=0.4."""
    if hasattr(data, "mod"):
        adata = data.mod[mod]
        for col in adata.obs:
            data.obs[f"{mod}:{col}"] = adata.obs[col].reindex(data.obs_names)
