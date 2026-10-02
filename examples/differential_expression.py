"""Executable cell-marker and multi-backend pseudobulk examples."""

import cellscope as cs


def main(method="pydeseq2"):
    data = cs.datasets.toy_rna(n_cells=160, n_genes=60)
    cs.pp.normalize(data)
    markers = cs.tl.find_all_markers(data, groupby="cell_type", method="wilcoxon")
    results, profiles, models = cs.tl.pseudobulk_de(
        data,
        method=method,
        design="~ donor_id + condition",
        contrast=("condition", "post", "pre"),
        metadata_cols=["donor_id"],
        min_cells=1,
    )
    print(f"{len(markers)} marker rows; {len(results)} DE rows using {method}")
    return markers, results, profiles, models


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="pydeseq2")
    main(parser.parse_args().method)
