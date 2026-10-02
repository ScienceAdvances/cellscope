"""Compatibility wrapper for annotation tables."""

from pathlib import Path

import pandas as pd


def add_label(adata, annotation, reference_key, cell_type_key="CellType"):
    """Accept a DataFrame or TSV; retain the historical labeled-cell return value."""
    table = (
        pd.read_csv(annotation, sep="\t", dtype=str)
        if isinstance(annotation, (str, Path))
        else annotation.copy()
    )
    if reference_key not in table and "Cluster" in table:
        table = table.rename(columns={"Cluster": reference_key})
    table[reference_key] = table[reference_key].astype(str).str.split(",")
    table = table.explode(reference_key)
    if table[reference_key].duplicated().any():
        raise ValueError("Each cluster must have exactly one annotation")
    mapping = table.set_index(reference_key)[cell_type_key].to_dict()
    reference = adata.obs[reference_key].astype(str)
    adata.obs[cell_type_key] = reference.map(mapping).astype("category")
    return adata[adata.obs[cell_type_key].notna()].copy()
