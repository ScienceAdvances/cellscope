"""Download-free RNA example; optional isolated R scran/scry/Slingshot/tradeSeq."""

import argparse

import cellscope as cs


def main(with_r=False):
    data = cs.datasets.toy_rna(n_cells=160, n_genes=80)
    cs.pp.qc(data, layer="counts", min_genes=5)
    data = cs.pp.filter_cells(data)
    cs.pp.normalize(data)
    cs.pp.pearson_residuals(data)
    cs.pp.pearson_hvg(data, n_top_genes=30)
    cs.tl.pca(data, n_comps=10, layer="pearson_residuals", mask_var="highly_variable")
    cs.pp.neighbors(data, use_rep="X_pca", n_neighbors=10)
    cs.tl.leiden(data, resolution=0.5)
    print(f"{data.n_obs} cells, {data.n_vars} genes; {data.obs.leiden.nunique()} clusters")
    if with_r:
        cs.pp.scran(data, cluster_key="cell_type")
        cs.pp.deviance_features(data, n_top_genes=30)
        cs.tl.slingshot(data, cluster_key="cell_type", start_cluster="T", backend="r")
        tests, fit = cs.tl.tradeseq(data, n_knots=3)
        print(f"tradeSeq: {len(tests)} gene tests; {fit}")
    print(cs.best_practices.methods("Normalization").to_string(index=False))
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-r", action="store_true")
    main(parser.parse_args().with_r)
