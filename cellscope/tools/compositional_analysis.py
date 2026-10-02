"""Cell-type proportions with samples, rather than cells, as replicates."""

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, mannwhitneyu, wilcoxon

from .._core import modality, record


def cell_composition(
    data,
    *,
    sample_col="sample_id",
    groupby="cell_type",
    mod="gex",
    condition_col=None,
    donor_col=None,
    key_added="cell_composition",
):
    """Count all sample/category combinations, including zero-cell categories."""
    adata = modality(data, mod)
    obs = adata.obs
    cols = [sample_col, groupby, *[x for x in (condition_col, donor_col) if x]]
    if obs[cols].isna().any().any():
        raise ValueError("Composition metadata must not be missing")
    table = pd.crosstab(obs[sample_col], obs[groupby], dropna=False)
    result = (
        table.rename_axis(index="sample_id", columns="group")
        .stack()
        .rename("n_cells")
        .reset_index()
    )
    result["sample_n_cells"] = result.groupby("sample_id", observed=True)["n_cells"].transform(
        "sum"
    )
    result["fraction"] = result["n_cells"] / result["sample_n_cells"]
    for col in (condition_col, donor_col):
        if col:
            metadata = obs[[sample_col, col]].drop_duplicates()
            if metadata[sample_col].duplicated().any():
                raise ValueError(f"{col} must be constant within sample")
            result[col] = result["sample_id"].map(metadata.set_index(sample_col)[col])
    entry = {
        "table": result,
        "params": {
            "sample_col": sample_col,
            "groupby": groupby,
            "denominator": "annotated_cells_per_sample",
        },
    }
    record(adata, key_added, entry["params"])
    adata.uns["cellscope"][key_added]["table"] = result
    return result


def composition_test(table, *, condition_col, comparison, reference, donor_col=None):
    """Test sample fractions with Mann-Whitney, or donor-paired Wilcoxon.

    This is a marginal, exploratory test of each fraction, not a joint
    compositional regression. FDR covers all categories in the returned table.
    """
    rows = []
    for group, subset in table.groupby("group", observed=True):
        subset = subset[subset[condition_col].isin([comparison, reference])]
        if subset["sample_id"].duplicated().any():
            raise ValueError("Expected one fraction per sample and category")
        if donor_col:
            if subset.duplicated([donor_col, condition_col]).any():
                raise ValueError("Supply one sample per donor and condition")
            paired = subset.pivot(index=donor_col, columns=condition_col, values="fraction")
            if comparison not in paired or reference not in paired:
                raise ValueError("Both conditions are required")
            paired = paired[[comparison, reference]].dropna()
            x, y = paired[comparison].to_numpy(), paired[reference].to_numpy()
            if len(x) < 2:
                raise ValueError("At least two complete donor pairs are required")
            pvalue = 1.0 if np.all(x == y) else float(wilcoxon(x, y).pvalue)
            method = "paired_wilcoxon"
        else:
            x = subset.loc[subset[condition_col].eq(comparison), "fraction"].to_numpy()
            y = subset.loc[subset[condition_col].eq(reference), "fraction"].to_numpy()
            if min(len(x), len(y)) < 2:
                raise ValueError("At least two independent samples per condition are required")
            pvalue = float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
            method = "mannwhitneyu"
        rows.append(
            {
                "group": group,
                "mean_difference": float(x.mean() - y.mean()),
                "n_comparison": len(x),
                "n_reference": len(y),
                "p_value": pvalue,
                "method": method,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No cell groups available for testing")
    result["fdr"] = false_discovery_control(result["p_value"].to_numpy(), method="bh")
    return result
