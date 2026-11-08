import tempfile
from pathlib import Path
import anndata as ad
import numpy as np
import cellscope as cs


def test_prefixed_ids_do_not_inherit_raw_barcode_index_name():
    a = ad.AnnData(np.ones((2, 2)))
    a.obs_names = ["AAAC-1", "CCCG-1"]
    a.obs_names.name = "barcode"
    cs.io.set_cell_ids(a, library_id="lib", sample_id="sample")
    assert a.obs_names.name != "barcode"
    assert a.obs["barcode"].tolist() == ["AAAC-1", "CCCG-1"]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "rna.h5ad"
        cs.io.write(a, path)
        b = ad.read_h5ad(path)
        assert b.obs_names.tolist() == ["lib:AAAC-1", "lib:CCCG-1"]
        assert b.obs["barcode"].tolist() == ["AAAC-1", "CCCG-1"]
