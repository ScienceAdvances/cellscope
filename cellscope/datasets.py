"""Small deterministic datasets for examples and integration tests."""

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix


def toy_rna(*, n_cells=80, n_genes=40, seed=0):
    """Return raw RNA counts for four samples from two paired donors."""
    if n_cells < 8 or n_genes < 4:
        raise ValueError("Use at least eight cells and four genes")
    rng = np.random.default_rng(seed)
    state = np.arange(n_cells) % 2
    rates = np.full((n_cells, n_genes), 2.0)
    rates[state == 0, 1:4] = 8.0
    obs = pd.DataFrame(
        {
            "sample_id": [f"s{i % 4}" for i in range(n_cells)],
            "donor_id": [f"d{(i % 4) // 2}" for i in range(n_cells)],
            "condition": ["pre" if i % 2 == 0 else "post" for i in range(n_cells)],
            "cell_type": ["T" if i % 3 else "B" for i in range(n_cells)],
            "cell_state": np.where(state == 0, "naive", "effector"),
        },
        index=[f"lib{i % 4}:cell{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=["MT-CO1", *[f"G{i}" for i in range(1, n_genes)]])
    adata = AnnData(csr_matrix(rng.poisson(rates)), obs=obs, var=var)
    adata.layers["counts"] = adata.X.copy()
    return adata
