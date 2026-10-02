"""Extract tabular results without depending on a plotting backend."""

from ._core import modality


def obs_df(data, keys=None, *, mod="gex"):
    obs = modality(data, mod).obs
    return (obs if keys is None else obs.loc[:, list(keys)]).copy()


def markers_df(data, *, mod="gex", key="rank_genes_groups", group=None, **kwargs):
    import scanpy as sc

    adata = modality(data, mod)
    entry = adata.uns.get("cellscope", {}).get(key, {})
    if "table" in entry:
        if kwargs:
            raise ValueError("Filter canonical marker tables explicitly after extraction")
        table = entry["table"]
        return (table if group is None else table[table["group"].eq(str(group))]).copy()
    return sc.get.rank_genes_groups_df(adata, group=group, key=key, **kwargs)


def de_df(data, *, mod="gex", key="differential_expression", group=None):
    """Extract a copied canonical marker or pseudobulk differential result table."""
    table = modality(data, mod).uns["cellscope"][key]["table"]
    return (table if group is None else table[table["group"].eq(str(group))]).copy()


def result(data, key, *, mod="gex"):
    """Return a stored result/provenance entry."""
    import copy

    return copy.deepcopy(modality(data, mod).uns["cellscope"][key])
