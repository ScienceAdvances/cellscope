from importlib import resources
from typing import ClassVar

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


class Chrom_size:
    hg38: ClassVar[dict] = {
        "chr1": 248956422,
        "chr2": 242193529,
        "chr3": 198295559,
        "chr4": 190214555,
        "chr5": 181538259,
        "chr6": 170805979,
        "chr7": 159345973,
        "chr8": 145138636,
        "chr9": 138394717,
        "chr10": 133797422,
        "chr11": 135086622,
        "chr12": 133275309,
        "chr13": 114364328,
        "chr14": 107043718,
        "chr15": 101991189,
        "chr16": 90338345,
        "chr17": 83257441,
        "chr18": 80373285,
        "chr19": 58617616,
        "chr20": 64444167,
        "chr21": 46709983,
        "chr22": 50818468,
        "chrX": 156040895,
        "chrY": 57227415,
    }
    mm10: ClassVar[dict] = {
        "chr1": 195471971,
        "chr2": 182113224,
        "chr3": 160039680,
        "chr4": 156508116,
        "chr5": 151834684,
        "chr6": 149736546,
        "chr7": 145441459,
        "chr8": 129401213,
        "chr9": 124595110,
        "chr10": 130694993,
        "chr11": 122082543,
        "chr12": 120129022,
        "chr13": 120421639,
        "chr14": 124902244,
        "chr15": 104043685,
        "chr16": 98207768,
        "chr17": 94987271,
        "chr18": 90702639,
        "chr19": 61431566,
        "chrX": 171031299,
        "chrY": 91744698,
    }


def read_json(filename: str, encoding="utf-8"):
    import json

    with resources.open_text(_data, filename, encoding=encoding) as _:
        return json.load(_)
