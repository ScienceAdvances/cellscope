"""Native AnnData/MuData readers, identity handling, and persistence."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import anndata as ad
import mudata as md
import pandas as pd

from ._core import modality, record
from ._seurat import read_seurat_rds

__all__ = [
    "concat",
    "create_mudata",
    "read_10x",
    "read_h5ad",
    "read_h5mu",
    "read_seurat_rds",
    "set_cell_ids",
    "write",
]


def set_cell_ids(adata, *, library_id, sample_id=None, donor_id=None, barcode_key="barcode"):
    """Assign stable library:barcode identifiers; preserve original barcodes."""
    modality(adata)
    if not library_id or ":" in str(library_id):
        raise ValueError("library_id must be nonempty and must not contain ':'")
    if barcode_key not in adata.obs:
        adata.obs[barcode_key] = adata.obs_names.astype(str)
    barcodes = adata.obs[barcode_key].astype("string")
    if barcodes.isna().any() or barcodes.eq("").any() or barcodes.duplicated().any():
        raise ValueError("Barcodes must be nonempty and unique within a library")
    if "library_id" in adata.obs and not adata.obs["library_id"].eq(library_id).all():
        raise ValueError("Existing library_id conflicts with the supplied identifier")
    adata.obs["library_id"] = str(library_id)
    for key, value in (("sample_id", sample_id), ("donor_id", donor_id)):
        if value is not None:
            adata.obs[key] = str(value)
    adata.obs_names = pd.Index(str(library_id) + ":" + barcodes.astype(str))
    return adata


def read_10x(path, *, library_id, sample_id=None, donor_id=None, **kwargs):
    """Read a 10x expression HDF5 file or matrix directory using Scanpy."""
    import scanpy as sc

    path = Path(path)
    adata = sc.read_10x_mtx(path, **kwargs) if path.is_dir() else sc.read_10x_h5(path, **kwargs)
    adata.var_names_make_unique()
    return set_cell_ids(adata, library_id=library_id, sample_id=sample_id, donor_id=donor_id)


def read_h5ad(path, **kwargs):
    return ad.read_h5ad(path, **kwargs)


def read_h5mu(path, **kwargs):
    return md.read_h5mu(path, **kwargs)


def create_mudata(modalities: Mapping, *, join="outer"):
    """Combine independent modalities, checking cell identity metadata conflicts.

    Outer joins retain observations missing a modality. Inner joins explicitly
    restrict all modalities to their common observations. Spatial spots must
    be connected by a mapping table rather than assigned invented cell IDs.
    """
    if not modalities or join not in {"outer", "inner"}:
        raise ValueError("Supply modalities and join='outer' or 'inner'")
    objects = dict(modalities)
    for obj in objects.values():
        modality(obj)
    for key in ("sample_id", "donor_id", "library_id", "barcode", "condition", "timepoint"):
        values = [obj.obs[key] for obj in objects.values() if key in obj.obs]
        if values:
            combined = pd.concat(values, axis=1)
            if combined.nunique(axis=1, dropna=True).gt(1).any():
                raise ValueError(f"Conflicting {key} for shared observations")
    if join == "inner":
        shared = next(iter(objects.values())).obs_names
        for obj in objects.values():
            shared = shared.intersection(obj.obs_names, sort=False)
        objects = {key: obj[shared].copy() for key, obj in objects.items()}
    data = md.MuData(objects)
    data.pull_obs()
    for key, obj in objects.items():
        data.obs[f"has_{key}"] = data.obs_names.isin(obj.obs_names)
        for column in obj.obs:
            data.obs[f"{key}:{column}"] = obj.obs[column].reindex(data.obs_names)
    for key in ("sample_id", "donor_id", "library_id", "barcode", "condition", "timepoint"):
        columns = [
            obj.obs[key].reindex(data.obs_names) for obj in objects.values() if key in obj.obs
        ]
        if columns:
            data.obs[key] = pd.concat(columns, axis=1).bfill(axis=1).iloc[:, 0]
    record(data, "create_mudata", {"join": join, "modalities": list(objects)})
    return data


def concat(samples: Mapping, *, sample_key="sample_id", join="outer"):
    """Concatenate libraries whose cell identifiers have already been assigned."""
    if not samples:
        raise ValueError("At least one sample is required")
    names = pd.Index([name for obj in samples.values() for name in obj.obs_names])
    if not names.is_unique:
        raise ValueError("Cell IDs collide; call set_cell_ids for every library first")
    return ad.concat(samples, label=sample_key, join=join, merge="same")


def write(data, path, **kwargs):
    """Persist native objects; reject extension/object mismatches."""
    path = Path(path)
    expected = ".h5mu" if hasattr(data, "mod") else ".h5ad"
    if path.suffix != expected:
        raise ValueError(f"Use {expected} for this data object")
    data.write(path, **kwargs)
