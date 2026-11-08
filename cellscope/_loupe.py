"""Write the Louper interchange format and invoke the official converter."""

from __future__ import annotations

import hashlib
import os
import platform
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.request import urlopen

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from . import __version__
from . import _logging as feedback
from ._core import counts, modality

_RELEASE = "1.1.5"
_ASSET = "louper-linux-x64"
_URL = f"https://github.com/10XGenomics/loupeR/releases/download/v{_RELEASE}/{_ASSET}"
# SHA-256 of the official asset after verifying the upstream manifest MD5.
_SHA256 = "dd2aa25a42363e7117d2adb0fad548c31fd64835b96daf8a35290cb54dffaa65"
_EULA = "https://10xgen.com/EULA"
_BARCODE = re.compile(r"(?:.*[:_])?(?:[ACGT]{14,}|cellid_[0-9]{9})(?:-[0-9]+)?(?:[:_].*)?")


def _matches_release(path):
    if not path.is_file():
        return False
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == _SHA256


def _resolve_louper(executable_path, auto_download, accept_eula, cache_dir):
    if executable_path is None:
        executable_path = os.environ.get("CELLSCOPE_LOUPER") or shutil.which("louper")
    if executable_path is not None:
        executable = Path(executable_path).expanduser().resolve()
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError(f"Louper path must be an executable file: {executable}")
        return executable

    if cache_dir is None:
        cache_dir = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "cellscope" / "louper"
    cache = Path(cache_dir).expanduser().resolve() / f"v{_RELEASE}"
    executable = cache / _ASSET
    if _matches_release(executable) and os.access(executable, os.X_OK):
        return executable
    if not auto_download:
        raise FileNotFoundError("Louper not found. Set executable_path or enable auto_download.")
    if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise RuntimeError("Louper automatic download supports Linux x86_64. Supply executable_path.")
    if not accept_eula:
        raise ValueError(f"Review the 10x Louper license at {_EULA}. "
                         "If you accept it, set accept_eula=True to download Louper.")
    cache.mkdir(parents=True, exist_ok=True)
    feedback.info(f"Louper | downloading official Linux x86_64 converter v{_RELEASE}")
    with tempfile.TemporaryDirectory(prefix=".louper-", dir=cache) as tmp:
        download = Path(tmp) / _ASSET
        with urlopen(_URL, timeout=60) as response, download.open("wb") as target:
            shutil.copyfileobj(response, target)
        if not _matches_release(download):
            raise RuntimeError("Downloaded Louper does not match the official release checksum")
        download.chmod(0o755)
        os.replace(download, executable)
    return executable


def _text(values, label):
    result = []
    for value in values:
        if pd.isna(value) or not str(value).strip() or "\x00" in str(value):
            raise ValueError(f"{label} must contain nonempty strings without NUL bytes")
        result.append(str(value))
    return result


def _strings(group, key, values):
    encoded = [value.encode("utf-8") for value in values]
    width = max((len(value) for value in encoded), default=1)
    group.create_dataset(key, data=np.asarray(encoded, dtype=f"S{width}"))


def _select_keys(available, requested, defaults, label):
    if requested is None:
        keys = [key for key in defaults if key in available]
    else:
        keys = [requested] if isinstance(requested, str) else list(requested)
    if not keys or len(keys) != len(set(keys)):
        raise ValueError(f"Supply at least one unique {label}")
    for key in keys:
        if key not in available:
            raise KeyError(f"{label} not found: {key}")
        if not key.strip() or key in {".", ".."} or "/" in key or "\x00" in key:
            raise ValueError(f"Invalid Loupe HDF5 group name: {key!r}")
    return keys


def _write_interchange(path, matrix, barcodes, feature_ids, names, clusters, projections):
    with h5py.File(path, "w") as handle:
        group = handle.create_group("matrix")
        # The CSR buffers of cells × genes are the CSC buffers of genes × cells.
        _strings(group, "barcodes", barcodes)
        group["data"] = matrix.data.astype(np.int32)
        group["indices"] = matrix.indices.astype(np.int32)
        group["indptr"] = matrix.indptr.astype(np.int32)
        group["shape"] = np.asarray(matrix.shape[::-1], dtype=np.int32)
        features = group.create_group("features")
        _strings(features, "name", names)
        _strings(features, "id", feature_ids)
        _strings(features, "feature_type", ["Gene Expression"] * len(names))
        _strings(features, "_all_tag_keys", [])

        group = handle.create_group("clusters")
        for key, (levels, codes) in clusters.items():
            cluster = group.create_group(key)
            _strings(cluster, "name", [key])
            _strings(cluster, "group_names", levels)
            _strings(cluster, "clustering_type", ["unknown"])
            cluster["assignments"] = codes
            cluster["score"] = np.asarray([0.0], dtype=np.float64)

        group = handle.create_group("projections")
        for key, coordinates in projections.items():
            projection = group.create_group(key)
            method = "UMAP" if "umap" in key.lower() else key
            if "tsne" in key.lower() or "t-sne" in key.lower():
                method = "t-SNE"
            _strings(projection, "name", [key])
            _strings(projection, "method", [method])
            # LoupeR's HDF5 writer stores the two coordinate vectors as rows.
            projection["data"] = coordinates.T

        group = handle.create_group("metadata")
        for key, value in {
            "tool": "cellscope", "tool_version": __version__,
            "os": platform.system(), "system": platform.machine(),
            "language": "Python", "language_version": platform.python_version(),
        }.items():
            _strings(group, key, [value])
        group.create_group("extra")


def write_cloupe(
    data, path, *, cluster_keys=None, projection_keys=None, layer="counts", mod="gex",
    barcode_key=None, feature_id_key=None, executable_path=None, auto_download=True,
    accept_eula=False, cache_dir=None, overwrite=False,
):
    """Export GEX counts and cell annotations for 10x Loupe Browser.

    Reads counts from a layer (default `counts`), or X with `layer=None`.
    Pass obs columns in cluster_keys and n_cells × 2 obsm keys in
    projection_keys. Defaults select common annotations and UMAP/t-SNE.
    Uses obs_names as barcodes by default, including valid library prefixes.
    Set executable_path to an installed 10x Louper CLI. For the official
    Linux x86_64 download, set accept_eula=True after reviewing the license.
    Returns the output Path. The input object is not changed.
    """
    path = Path(path).expanduser().absolute()
    if path.suffix.lower() != ".cloupe":
        raise ValueError("Output path must end in .cloupe")
    if not overwrite and (path.exists() or path.is_symlink()):
        raise FileExistsError(path)
    adata = modality(data, mod)
    if adata.isbacked or adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("Loupe export requires a nonempty in-memory expression matrix")

    barcodes = _text(adata.obs_names if barcode_key is None else adata.obs[barcode_key], "Barcodes")
    if len(set(barcodes)) != len(barcodes) or any(not _BARCODE.fullmatch(b) for b in barcodes):
        raise ValueError("Loupe barcodes must be unique and use 10x format, e.g. sample:ACGTACGTACGTACGT-1")
    names = _text(adata.var_names, "Feature names")
    if feature_id_key is None:
        feature_id_key = next((key for key in ("gene_ids", "gene_id") if key in adata.var), None)
    feature_ids = _text(adata.var[feature_id_key], "Feature IDs") if feature_id_key else names
    if len(set(feature_ids)) != len(feature_ids):
        raise ValueError("Loupe feature IDs must be unique")

    clusters = {}
    keys = _select_keys(
        adata.obs, cluster_keys, ("leiden", "louvain", "cell_type", "t_state", "sample_id"), "cluster key",
    )
    for key in keys:
        series = adata.obs[key]
        labels = _text(series, key)
        categories = None
        if isinstance(series.dtype, pd.CategoricalDtype):
            categories = series.cat.remove_unused_categories().cat.categories.astype(str)
        categorical = pd.Categorical(labels, categories=categories)
        if len(categorical.categories) > 32768:
            raise ValueError(f"{key}: Loupe allows up to 32768 groups")
        clusters[key] = (categorical.categories.tolist(), categorical.codes.astype(np.int32))

    projections = {}
    for key in _select_keys(adata.obsm, projection_keys, ("X_umap", "X_tsne"), "projection key"):
        coordinates = np.asarray(adata.obsm[key], dtype=np.float64)
        if coordinates.shape != (adata.n_obs, 2) or not np.isfinite(coordinates).all():
            raise ValueError(f"{key}: Loupe projections require finite coordinates of shape ({adata.n_obs}, 2)")
        projections[key] = coordinates

    observed = counts(adata, layer, integer=False)
    values = observed.data if sparse.issparse(observed) else np.asarray(observed)
    limit = np.iinfo(np.int32).max
    if np.any(values != np.floor(values)) or np.any(values > limit):
        raise ValueError("Loupe needs nonnegative integer counts within the 32-bit range")
    matrix = sparse.csr_matrix(observed, dtype=np.int64, copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if max(matrix.nnz, *matrix.shape) > limit or np.any(matrix.data > limit):
        raise ValueError("Loupe sparse matrix exceeds its 32-bit format limits")

    executable = _resolve_louper(executable_path, auto_download, accept_eula, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cellscope-loupe-", dir=path.parent) as tmp:
        interchange = Path(tmp) / "input.h5"
        output = Path(tmp) / "output.cloupe"
        _write_interchange(interchange, matrix, barcodes, feature_ids, names, clusters, projections)
        feedback.info("Louper | converting counts, clusters and projections")
        result = subprocess.run(
            [str(executable), "create", f"--input={interchange}", f"--output={output}"],
            capture_output=True, text=True, errors="replace", check=False,
        )
        if result.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
            details = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"Louper did not create a nonempty .cloupe (exit {result.returncode}): {details}")
        if overwrite:
            os.replace(output, path)
        else:
            os.link(output, path)
    return path
