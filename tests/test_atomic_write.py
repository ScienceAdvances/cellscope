"""Successful native round trips and failures must preserve existing files."""

import hashlib

import anndata as ad
import mudata as md
import numpy as np
import pytest

import cellscope as cs


def test_atomic_roundtrip_and_compression(tmp_path):
    data = ad.AnnData(np.ones((3, 2)))
    target = tmp_path / "rna.h5ad"
    cs.io.write(data, target, compression="gzip")
    np.testing.assert_array_equal(ad.read_h5ad(target).X, data.X)
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("exists", [False, True])
def test_failed_native_writer_preserves_destination_and_cleans_temp(tmp_path, monkeypatch, exists):
    target = tmp_path / "rna.h5ad"
    if exists:
        ad.AnnData(np.zeros((3, 2))).write_h5ad(target)
        before = target.read_bytes()

    def broken(self, filename, **kwargs):
        from pathlib import Path

        Path(filename).write_bytes(b"incomplete write")
        raise ValueError("serialization failure")

    monkeypatch.setattr(ad.AnnData, "write", broken)
    with pytest.raises(ValueError, match="serialization failure"):
        cs.io.write(ad.AnnData(np.ones((3, 2))), target)
    if exists:
        assert target.read_bytes() == before
    else:
        assert not target.exists()
    assert not list(tmp_path.glob(".rna.*"))


def test_hdf5_marker_header_overflow_preserves_previous_good_object(tmp_path):
    target = tmp_path / "rna.h5ad"
    ad.AnnData(np.ones((3, 2))).write_h5ad(target)
    before = hashlib.sha256(target.read_bytes()).hexdigest()
    bad = ad.AnnData(np.ones((3, 2)))
    names = np.empty(2, dtype=[(str(i), object) for i in range(1386)])
    for name in names.dtype.names:
        names[name] = ["G1", "G2"]
    bad.uns["rank_genes_groups"] = {"names": names}
    with pytest.raises(ValueError, match="object header message is too large"):
        cs.io.write(bad, target)
    assert hashlib.sha256(target.read_bytes()).hexdigest() == before
    assert ad.read_h5ad(target).shape == (3, 2)
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_mudata_roundtrip(tmp_path):
    data = md.MuData({"gex": ad.AnnData(np.ones((3, 2)))})
    target = tmp_path / "joint.h5mu"
    cs.io.write(data, target, compression="gzip")
    assert md.read_h5mu(target).mod["gex"].shape == (3, 2)
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_backed_object_rejected_before_modifying_destination(tmp_path):
    target = tmp_path / "rna.h5ad"
    ad.AnnData(np.ones((3, 2))).write_h5ad(target)
    before = target.read_bytes()
    backed = ad.read_h5ad(target, backed="r")
    try:
        with pytest.raises(ValueError, match="to_memory"):
            cs.io.write(backed, target)
        assert target.read_bytes() == before
    finally:
        backed.file.close()
