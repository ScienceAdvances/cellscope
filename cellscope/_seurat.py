"""Seurat RDS import through the isolated R bridge."""

from __future__ import annotations

import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from . import r


def _names(values, label):
    names = pd.Index(values, dtype=str)
    if names.empty or not names.is_unique or names.isna().any() or (names == "").any():
        raise ValueError(f"Seurat {label} must be nonempty and unique")
    return names


def _metadata(payload):
    frame = payload["frame"]
    for name, spec in payload["factors"].items():
        frame[name] = pd.Categorical(
            frame[name], categories=spec["levels"], ordered=bool(spec["ordered"][0])
        )
    return frame


def _merge_layer(matrices, key):
    members = [key] if key in matrices else [k for k in matrices if k.startswith(key + ".")]
    if not members:
        return None
    features = pd.Index([], dtype=str)
    cells = pd.Index([], dtype=str)
    parts = []
    for name in members:
        payload = matrices[name]
        genes = _names(payload["features"], f"features in {name}")
        barcodes = _names(payload["cells"], f"cells in {name}")
        if len(cells.intersection(barcodes)):
            raise ValueError(f"Split layer {key!r} has overlapping cells; resolve layers in Seurat")
        features = features.union(genes, sort=False)
        cells = cells.append(barcodes)
        parts.append((payload["matrix"].T, genes, barcodes))
    coverage = {}
    if len(parts) == 1:
        return parts[0][0], cells, features, members, coverage
    rows, columns, values = [], [], []
    for name, (matrix, genes, barcodes) in zip(members, parts):
        coo = sparse.coo_matrix(matrix)
        rows.append(cells.get_indexer(barcodes)[coo.row])
        columns.append(features.get_indexer(genes)[coo.col])
        values.append(coo.data)
        coverage[name] = {"cells": barcodes.to_numpy(), "features": genes.to_numpy()}
    if any(len(genes) != len(features) for _, genes, _ in parts):
        warnings.warn(
            f"Split layer {key!r} has different feature sets; absent features are zero-filled. "
            "Original feature/cell coverage is in uns['seurat']['split_coverage'].",
            UserWarning,
            stacklevel=3,
        )
    matrix = sparse.csr_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(len(cells), len(features)),
    )
    return matrix, cells, features, members, coverage


def _align_layer(matrix, cells, features, obs_names, var_names, key):
    row_index = cells.get_indexer(obs_names)
    col_index = features.get_indexer(var_names)
    if (row_index >= 0).all() and (col_index >= 0).all():
        if sparse.issparse(matrix):
            return matrix.tocsr()[row_index][:, col_index]
        return matrix[np.ix_(row_index, col_index)]
    # Missing normalized/scaled values are unknown, not measured zeroes.
    size = len(obs_names) * len(var_names) * np.dtype(float).itemsize
    if size > 512 * 1024**2:
        raise ValueError(
            f"Aligning incomplete layer {key!r} needs {size / 1024**2:.0f} MiB of dense memory. "
            "Exclude it with layers=(), select fewer layers, or import it as x_layer."
        )
    warnings.warn(
        f"Layer {key!r} lacks some X cells/features; padding with NaN in a dense layer.",
        UserWarning,
        stacklevel=3,
    )
    result = np.full((len(obs_names), len(var_names)), np.nan)
    present_rows, present_cols = row_index >= 0, col_index >= 0
    if sparse.issparse(matrix):
        subset = matrix.tocsr()[row_index[present_rows]][:, col_index[present_cols]].toarray()
    else:
        subset = matrix[np.ix_(row_index[present_rows], col_index[present_cols])]
    result[np.ix_(present_rows, present_cols)] = subset
    return result


def read_seurat_rds(
    path,
    *,
    assay="RNA",
    x_layer="counts",
    layers=("counts", "data"),
    reductions=True,
    timeout=None,
):
    """Read a Seurat RDS into an in-memory AnnData using R/Seurat >= 5.

    Parameters
    ----------
    path : str or pathlib.Path
        Local RDS containing a Seurat object. Disk-backed assays also require
        their external matrix files and their R backend to remain accessible.
    assay : str
        Transcriptome assay to import (default: ``RNA``).
    x_layer : str
        Layer used for X and its cell/feature axes (default: ``counts``).
        Use ``data`` for normalized-only objects. Exact layer names take
        precedence; otherwise disjoint ``<name>.<batch>`` layers are merged.
    layers : sequence of str
        Additional layers (default: counts and data). Missing optional layers
        are recorded in ``uns['seurat']['missing_layers']``. X is also retained
        as a layer. Pass ``()`` to import X alone. Add ``scale.data`` explicitly
        to import scaled values; missing values are padded with NaN, requiring
        dense storage (limited to 512 MiB for incomplete layers).
    reductions : bool
        Import embeddings associated with this assay into ``obsm['X_<name>']``.
        Missing cells receive NaN. Other assays' reductions are not imported.
    timeout : float or None
        R subprocess timeout in seconds.

    Notes
    -----
    Preserves cell/feature names, metadata, factors, active identities in
    ``obs['seurat_ident']``, and variable features in ``var['highly_variable']``.
    X's cells follow the original object metadata order. Sparse expression
    stays sparse. Split layers with different features use a zero-filled union
    and record original coverage with a warning. Extra layer axes outside X
    are discarded. No normalization or count reconstruction is performed.
    Graphs, loadings, images, commands, models and non-transcriptome assays are
    not converted. ``raw`` is unset. See ``docs/seurat_import.md``.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if not isinstance(assay, str) or not assay or not isinstance(x_layer, str) or not x_layer:
        raise ValueError("assay and x_layer must be nonempty strings")
    if isinstance(layers, str) or layers is None:
        raise TypeError("layers must be a sequence of layer names; use () for X alone")
    requested = list(dict.fromkeys([x_layer, *layers]))
    if any(not isinstance(key, str) or not key for key in requested):
        raise ValueError("Layer names must be nonempty strings")
    payload = r._call(
        Path(__file__).with_suffix(".R").read_text(),
        packages=["Seurat", "SeuratObject"],
        timeout=timeout,
        path=str(path),
        assay=assay,
        requested=requested,
        reductions=bool(reductions),
    )
    merged = {key: _merge_layer(payload["matrices"], key) for key in requested}
    if merged[x_layer] is None:
        raise ValueError(
            f"Layer {x_layer!r} not found in assay {assay!r}; "
            f"available layers: {list(payload['available_layers'])}"
        )
    matrix, cells, features, _, _ = merged[x_layer]
    obs, var = _metadata(payload["obs"]), _metadata(payload["var"])
    _names(obs.index, "metadata cells")
    _names(var.index, "metadata features")
    if not cells.isin(obs.index).all() or not features.isin(var.index).all():
        raise ValueError("Seurat layer names are missing from assay/object metadata")
    obs = obs.loc[obs.index.isin(cells)].copy()
    matrix = _align_layer(matrix, cells, features, obs.index, features, x_layer)
    result = ad.AnnData(matrix, obs=obs, var=var.loc[features].copy())
    result.var["highly_variable"] = features.isin(payload["variable_features"])
    sources, coverage = {}, {}
    for key, value in merged.items():
        if value is None:
            continue
        matrix, cells, genes, members, covered = value
        result.layers[key] = _align_layer(
            matrix, cells, genes, result.obs_names, result.var_names, key
        )
        sources[key] = members
        if covered:
            coverage[key] = covered
    for key, value in payload["reductions"].items():
        cells = _names(value["cells"], f"reduction {key} cells")
        index = cells.get_indexer(result.obs_names)
        embedding = np.full((result.n_obs, value["matrix"].shape[1]), np.nan)
        embedding[index >= 0] = value["matrix"][index[index >= 0]]
        result.obsm[f"X_{key}"] = embedding
    result.uns["seurat"] = {
        "assay": assay,
        "x_layer": x_layer,
        "source_layers": sources,
        "split_coverage": coverage,
        "missing_layers": [key for key, value in merged.items() if value is None],
        "available_layers": payload["available_layers"],
        "object_version": str(payload["object_version"][0]),
        "seurat_version": str(payload["seurat_version"][0]),
        "seuratobject_version": str(payload["seuratobject_version"][0]),
    }
    return result
