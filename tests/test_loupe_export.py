"""Regression coverage for Python's Louper interchange and output handling."""

import sys

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

from cellscope import io


@pytest.fixture
def expression():
    data = AnnData(
        sparse.csr_matrix([[0, 1, 0], [2, 0, 5], [0, 0, 3]]),
        obs=pd.DataFrame(
            {"leiden": pd.Categorical(["B", "A", "B"], categories=["B", "A"])},
            index=["P1:ACGTACGTACGTACGT-1", "P8:ACGTACGTACGTACGT-1", "P8:CGTACGTACGTACGTA-1"],
        ),
        var=pd.DataFrame({"gene_ids": ["id1", "id2", "id3"]}, index=["CD3D", "CD8A", "CD4"]),
    )
    data.layers["counts"] = data.X.copy()
    data.obsm["X_umap"] = np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    return data


def _converter(path, program):
    path.write_text(f"#!{sys.executable}\n{program}\n")
    path.chmod(0o755)
    return path


def test_cloupe_calls_converter_with_correct_sparse_orientation_and_metadata(tmp_path, expression):
    executable = _converter(
        tmp_path / "louper test",
        """import sys
from pathlib import Path
import h5py
from scipy.sparse import csc_matrix
assert sys.argv[1] == "create"
source = sys.argv[2].split("=", 1)[1]
destination = sys.argv[3].split("=", 1)[1]
with h5py.File(source) as h5:
    group = h5["matrix"]
    matrix = csc_matrix((group["data"][:], group["indices"][:], group["indptr"][:]),
                        shape=tuple(group["shape"][:]))
    assert matrix.T.toarray().tolist() == [[0, 1, 0], [2, 0, 5], [0, 0, 3]]
    assert group["barcodes"].asstr()[0] == "P1:ACGTACGTACGTACGT-1"
    assert group["features/id"].asstr()[:].tolist() == ["id1", "id2", "id3"]
    assert h5["clusters/leiden/group_names"].asstr()[:].tolist() == ["B", "A"]
    assert h5["clusters/leiden/assignments"][:].tolist() == [0, 1, 0]
    assert h5["projections/X_umap/data"][:].tolist() == [[1, 3, 5], [2, 4, 6]]
    assert h5["metadata/language"].asstr()[0] == "Python"
Path(destination).write_bytes(b"mock-conversion-result")
""",
    )
    output = tmp_path / "my export.cloupe"
    assert io.write_cloupe(expression, output, executable_path=executable) == output
    assert output.read_bytes() == b"mock-conversion-result"
    assert not list(tmp_path.glob(".cellscope-loupe-*"))
    with pytest.raises(FileExistsError):
        io.write_cloupe(expression, output, executable_path=executable)


def test_cloupe_failed_conversion_keeps_existing_result(tmp_path, expression):
    executable = _converter(tmp_path / "louper", "import sys; sys.exit('converter failed')")
    output = tmp_path / "preserved.cloupe"
    output.write_bytes(b"previous")
    with pytest.raises(RuntimeError, match="converter failed"):
        io.write_cloupe(expression, output, executable_path=executable, overwrite=True)
    assert output.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".cellscope-loupe-*"))


def test_cloupe_rejects_normalized_counts(tmp_path, expression):
    expression.X = expression.X.astype(float) / 2
    with pytest.raises(ValueError, match="integer counts"):
        io.write_cloupe(expression, tmp_path / "invalid.cloupe", layer=None)


def test_louper_download_checks_license_and_checksum(tmp_path, monkeypatch):
    import hashlib
    import io as stdio

    from cellscope import _loupe

    asset = b"test binary"
    monkeypatch.delenv("CELLSCOPE_LOUPER", raising=False)
    monkeypatch.setattr(_loupe.shutil, "which", lambda _: None)
    monkeypatch.setattr(_loupe.platform, "system", lambda: "Linux")
    monkeypatch.setattr(_loupe.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(_loupe, "_SHA256", hashlib.sha256(asset).hexdigest())
    urls = []

    def fetch(url, timeout):
        urls.append(url)
        return stdio.BytesIO(asset)

    monkeypatch.setattr(_loupe, "urlopen", fetch)
    with pytest.raises(ValueError, match="accept_eula"):
        _loupe._resolve_louper(None, True, False, tmp_path)
    assert not urls

    executable = _loupe._resolve_louper(None, True, True, tmp_path)
    assert executable.read_bytes() == asset
    assert _loupe._resolve_louper(None, False, False, tmp_path) == executable
    assert urls == [_loupe._URL]
