from importlib import resources

import anndata

from .. import _data


def subset(adata: anndata.AnnData, subsets: dict, inplace: bool = False) -> anndata.AnnData | None:
    """
    filter/subset a AnnData according to subsets conditions
    """
    import numpy as np

    keep = np.ones(adata.n_obs, dtype=bool)
    for key, condition in subsets.items():
        series = adata.obs[key]
        if callable(condition):
            mask = condition(series)
        elif isinstance(condition, (list, tuple, set)):
            mask = series.isin(condition)
        elif isinstance(condition, str) and "x" in condition:
            # Preserve trusted, vectorized legacy predicates such as 'x > 3'.
            mask = series.to_frame("x").eval(condition)
        else:
            mask = series.eq(condition)
        keep &= np.asarray(mask, dtype=bool)
    if inplace:
        adata._inplace_subset_obs(keep)
        return None
    return adata[keep].copy()


def read_json(filename: str, encoding="utf-8"):
    import json

    with resources.open_text(_data, filename, encoding=encoding) as _:
        return json.load(_)
