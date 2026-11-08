import contextlib
import io
import tempfile
import unittest

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

import cellscope as cs
from cellscope.tools._genevector import _mi_numba_exact


class GeneVectorAdapterTests(unittest.TestCase):
    def test_numba_exact_matches_author_histograms(self):
        from genevector.data import GeneVectorDataset

        for seed in range(3):
            rng = np.random.default_rng(seed)
            x = rng.poisson(2, (60, 6)).astype(float)
            x[:, 0] = 2
            adata = AnnData(
                sparse.csr_matrix(x), var=pd.DataFrame(index=[f"G{i}" for i in range(6)])
            )
            with contextlib.redirect_stdout(io.StringIO()):
                dataset = GeneVectorDataset(adata)
                dataset.generate_mi_scores(constant=1)
            expected = np.array(
                [[dataset.mi_scores[a][b] for b in adata.var_names] for a in adata.var_names]
            )
            np.testing.assert_allclose(_mi_numba_exact(adata.X), expected, atol=1e-12, rtol=1e-12)

    def test_backend_training_embedding_and_probability(self):
        rng = np.random.default_rng(4)
        x = rng.poisson(3, (40, 6)).astype(float)
        adata = AnnData(
            sparse.csr_matrix(x),
            obs=pd.DataFrame(
                {"sample_id": ["a"] * 20 + ["b"] * 20}, index=[f"cell{i}" for i in range(40)]
            ),
            var=pd.DataFrame(index=["CD4", "CD8A", "CCR7", "GZMB", "PDCD1", "X"]),
        )
        adata.X[0, :2] = 0
        x[0, :2] = 0
        adata.layers["counts"] = adata.X.copy()
        with tempfile.TemporaryDirectory() as d:
            with contextlib.redirect_stdout(io.StringIO()):
                fit = cs.tl.genevector_fit(
                    adata,
                    output_dir=d,
                    mi_backend="numba_exact",
                    epochs=3,
                    threshold=0,
                    emb_dimension=4,
                    n_threads=2,
                )
                cs.tl.genevector_embedding(adata, fit, batch_key="sample_id", reference_batch="a")
                probs = cs.tl.genevector_phenotypes(
                    adata, fit, {"CD4": ["CD4"], "CD8": ["CD8A"]}, key_added="lineage", undefined_policy="flag"
                )
            self.assertTrue(pd.isna(adata.obs.loc["cell0", "lineage"]))
            self.assertEqual(adata.obsm["X_genevector"].shape, (40, 4))
            self.assertTrue(np.isfinite(adata.obsm["X_genevector"]).all())
            np.testing.assert_allclose(probs.sum(axis=1), 1)
            self.assertEqual(fit["parameters"]["epochs_executed"], 3)
            np.testing.assert_array_equal(adata.X.toarray(), x)
            before = adata.obsm["X_genevector_uncorrected"][:20].mean(axis=0)
            after = adata.obsm["X_genevector"][20:].mean(axis=0)
            np.testing.assert_allclose(before, after, atol=1e-10)

    def test_unknown_backend_is_rejected(self):
        with self.assertRaises(ValueError):
            cs.tl.genevector_fit(None, output_dir="unused", mi_backend="harmony")
