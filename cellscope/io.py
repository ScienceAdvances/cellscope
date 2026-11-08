"""Native AnnData/MuData readers, identity handling, and persistence."""

from __future__ import annotations

import os
import tempfile
import warnings
from collections.abc import Mapping
from pathlib import Path

import anndata as ad
import mudata as md
import pandas as pd

from ._core import modality, record
from ._loupe import write_cloupe
from ._seurat import read_seurat_rds

__all__ = [
    "concat",
    "create_mudata",
    "read_10x",
    "read_cellbender",
    "read_h5ad",
    "read_h5mu",
    "read_seurat_rds",
    "set_cell_ids",
    "write",
    "write_cloupe",
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
    adata.obs_names = pd.Index((str(library_id) + ":" + barcodes.astype(str)).to_numpy(), name=None)
    return adata


def _disambiguate_gene_names(adata):
    symbols = adata.var_names
    duplicates = symbols.duplicated(keep=False)
    if not duplicates.any():
        return
    if "gene_symbols" not in adata.var:
        adata.var["gene_symbols"] = symbols.to_numpy(copy=True)
    gene_ids = adata.var.get("gene_ids", adata.var.get("gene_id"))
    names = symbols.tolist()
    used = set(names)
    next_suffix = {}
    for position, duplicate in enumerate(duplicates):
        if not duplicate:
            continue
        symbol = names[position]
        gene_id = gene_ids.iloc[position] if gene_ids is not None else None
        gene_id = str(gene_id).strip() if pd.notna(gene_id) else ""
        base = f"{symbol}_{gene_id}" if gene_id else symbol
        name = base
        while name in used:
            next_suffix[base] = next_suffix.get(base, 0) + 1
            name = f"{base}_{next_suffix[base]}"
        names[position] = name
        used.add(name)
    adata.var_names = pd.Index(names, name=symbols.name)


def read_10x(path, *, library_id, sample_id=None, donor_id=None, **kwargs):
    """Read 10x counts, suffixing duplicate symbols with gene IDs when available."""
    import scanpy as sc

    path = Path(path)
    if path.is_dir():
        kwargs.setdefault("make_unique", False)
    # Scanpy constructs AnnData before we can uniquify 10x gene symbols.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"^Variable names are not unique\. To make them unique, call `\.var_names_make_unique`\.$",
            category=UserWarning,
            module=r"^anndata(\.|$)",
        )
        adata = sc.read_10x_mtx(path, **kwargs) if path.is_dir() else sc.read_10x_h5(path, **kwargs)
    _disambiguate_gene_names(adata)
    return set_cell_ids(adata, library_id=library_id, sample_id=sample_id, donor_id=donor_id)


def read_h5ad(path, **kwargs):
    return ad.read_h5ad(path, **kwargs)


def read_cellbender(
    output_file, *, raw_file, library_id, sample_id=None, donor_id=None,
    min_cell_probability=0.5,
):
    """Read native CellBender metadata and align raw counts by barcode/gene ID.

    ``X`` and ``layers['counts']`` contain corrected integer counts;
    ``layers['raw_counts']`` retains the observed counts. Keep analyzed droplets
    with cell_probability > min_cell_probability (not all analyzed droplets).
    Set min_cell_probability=None to retain all analyzed droplets for explicit
    matching to an existing AnnData, without implicit cell deletion.
    The source H5 files are never changed. Features are aligned by stable gene
    IDs, not potentially duplicated symbols or an assumed identical ordering.
    """
    import h5py
    import numpy as np
    import scanpy as sc

    from ._core import counts, dependency

    dependency("cellbender", "cellbender")
    from cellbender.remove_background.downstream import anndata_from_h5

    if min_cell_probability is not None and not 0 <= min_cell_probability <= 1:
        raise ValueError("min_cell_probability must be in [0, 1]")
    corrected = anndata_from_h5(str(output_file), analyzed_barcodes_only=True)
    # The upstream metadata heuristic confuses gene/cell fields when n_obs ==
    # n_vars. Read authoritative feature metadata directly from native v3 H5.
    with h5py.File(output_file) as handle:
        if "matrix/features/id" in handle:
            for source_key, target_key in (
                ("id", "gene_id"), ("feature_type", "feature_type"), ("genome", "genome")
            ):
                path = f"matrix/features/{source_key}"
                if path in handle:
                    values = handle[path][:].astype(str)
                    if len(values) != corrected.n_vars:
                        raise ValueError("Output feature metadata does not match the count matrix")
                    corrected.var[target_key] = values
                    if target_key in corrected.obs:
                        del corrected.obs[target_key]
            if "global_latents/ambient_expression" in handle:
                ambient = handle["global_latents/ambient_expression"][:]
                if ambient.shape == (corrected.n_vars,):
                    corrected.var["ambient_expression"] = ambient
                    if "ambient_expression" in corrected.obs:
                        del corrected.obs["ambient_expression"]
    if "cell_probability" not in corrected.obs:
        raise ValueError("CellBender cell_probability is required; use its native output H5")
    probability = pd.to_numeric(corrected.obs["cell_probability"])
    if not np.isfinite(probability).all() or not probability.between(0, 1).all():
        raise ValueError("Invalid CellBender cell probabilities")
    if min_cell_probability is not None:
        corrected = corrected[probability.gt(min_cell_probability)].copy()
    if "feature_type" in corrected.var:
        corrected = corrected[:, corrected.var["feature_type"].eq("Gene Expression")].copy()
    raw = sc.read_10x_h5(raw_file, gex_only=True)
    if "gene_id" not in corrected.var or "gene_ids" not in raw.var:
        raise ValueError("Stable gene IDs required to align raw and corrected features")
    gene_ids = pd.Index(corrected.var["gene_id"].astype(str))
    raw_ids = pd.Index(raw.var["gene_ids"].astype(str))
    if not gene_ids.is_unique or not raw_ids.is_unique:
        raise ValueError("Gene IDs must be unique")
    gene_positions = raw_ids.get_indexer(gene_ids)
    cell_positions = raw.obs_names.get_indexer(corrected.obs_names)
    if (gene_positions < 0).any() or (cell_positions < 0).any():
        raise ValueError("Raw input does not contain every output gene and barcode")
    observed = raw.X[cell_positions][:, gene_positions].tocsr()
    matrix = corrected.X.tocsr()
    counts(corrected, None)
    if ((matrix - observed) > 0).nnz:
        raise ValueError("Corrected counts exceed raw counts; check input/output correspondence")
    _disambiguate_gene_names(corrected)
    corrected.layers["raw_counts"] = observed
    corrected.layers["counts"] = matrix.copy()
    corrected.X = matrix
    set_cell_ids(
        corrected, library_id=library_id, sample_id=sample_id, donor_id=donor_id,
    )
    record(corrected, "read_cellbender", {
        "raw_file": str(Path(raw_file).resolve()), "output_file": str(Path(output_file).resolve()),
        "min_cell_probability": min_cell_probability,
        "raw_layer": "raw_counts", "corrected_layer": "counts", "alignment": "gene_id+barcode",
    }, backend="cellbender")
    return corrected


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


def write(data, path, *, atomic=True, **kwargs):
    """Persist native objects, replacing the destination only after success.

    Atomic writes require an in-memory object and a temporary file on the same
    filesystem. Exceptions remove that temporary file and retain an existing
    destination. Backed objects require to_memory() or explicit atomic=False.
    Native AnnData/MuData serialization and marker schemas remain unchanged.
    """
    path = Path(path)
    expected = ".h5mu" if hasattr(data, "mod") else ".h5ad"
    if path.suffix != expected:
        raise ValueError(f"Use {expected} for this data object")
    if not atomic:
        data.write(path, **kwargs)
        return
    if getattr(data, "isbacked", False):
        raise ValueError("Atomic write requires an in-memory object; use to_memory() or atomic=False")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=expected, dir=path.parent)
    os.close(descriptor)
    try:
        data.write(temporary, **kwargs)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
