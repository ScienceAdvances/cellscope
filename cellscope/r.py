"""Isolated Rscript bridge with sparse, named, orientation-preserving transport.

No R runtime starts at import time. RObject carries serialized RDS bytes, so
native fits can be passed back into R without unpickling or persistent temp files.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread, mmwrite


@dataclass(frozen=True)
class RObject:
    """A native R object serialized by saveRDS; only R reads its contents."""

    rds: bytes
    classes: tuple[str, ...] = ()

    def __repr__(self):
        return f"RObject(classes={self.classes!r}, bytes={len(self.rds)})"


def _rscript():
    command = os.environ.get("CELLSCOPE_RSCRIPT", "Rscript")
    resolved = shutil.which(command)
    if resolved is None:
        raise ImportError(
            "R methods require Rscript on PATH (or CELLSCOPE_RSCRIPT). "
            "See docs/best_practices.md for R package installation."
        )
    return resolved


def _encode(value, root, serial):
    serial[0] += 1
    stem = root / str(serial[0])
    if value is None:
        return {"kind": "null"}
    if isinstance(value, RObject):
        path = stem.with_suffix(".rds")
        path.write_bytes(value.rds)
        return {"kind": "rds", "path": str(path)}
    if isinstance(value, dict):
        return {
            "kind": "list",
            "items": {key: _encode(v, root, serial) for key, v in value.items()},
        }
    if isinstance(value, pd.DataFrame):
        if not value.index.is_unique or not value.columns.is_unique:
            raise ValueError("R data frames require unique row and column identifiers")
        # Encode columns separately to retain numeric, boolean and string types.
        return {
            "kind": "dataframe",
            "index": list(map(str, value.index)),
            "columns": {str(col): _encode(value[col], root, serial) for col in value},
        }
    if sparse.issparse(value):
        path = stem.with_suffix(".mtx")
        mmwrite(path, sparse.coo_matrix(value, dtype=float))
        return {"kind": "sparse", "path": str(path)}
    if isinstance(value, (np.ndarray, pd.Index, pd.Series, list, tuple)):
        array = np.asarray(value)
        dtype = getattr(value, "dtype", array.dtype)
        if array.ndim == 2:
            if array.dtype.kind not in "biuf":
                raise TypeError("R matrix transport requires numeric data")
            path = stem.with_suffix(".csv")
            np.savetxt(path, array, delimiter=",", fmt="%.17g")
            return {"kind": "matrix", "path": str(path), "shape": list(array.shape)}
        if array.ndim != 1:
            raise ValueError("Only vectors and two-dimensional matrices can be sent to R")
        return {
            "kind": "vector",
            "type": "logical"
            if pd.api.types.is_bool_dtype(dtype)
            else "numeric"
            if pd.api.types.is_numeric_dtype(dtype)
            else "character",
            "values": [
                None if pd.isna(v) else v.item() if isinstance(v, np.generic) else v for v in array
            ],
        }
    if isinstance(value, (str, bool, int, float, np.generic)):
        return {"kind": "scalar", "value": value.item() if isinstance(value, np.generic) else value}
    raise TypeError(f"Unsupported R argument type: {type(value).__name__}")


def _decode(value):
    kind = value["kind"]
    if kind == "null":
        return None
    if kind == "rds":
        return RObject(Path(value["path"]).read_bytes(), tuple(value["classes"]))
    if kind == "list":
        items = value["items"]
        if not items:
            return {}
        return (
            {key: _decode(item) for key, item in items.items()}
            if isinstance(items, dict)
            else [_decode(item) for item in items]
        )
    if kind == "sparse":
        return sparse.csc_matrix(mmread(value["path"]))
    if kind == "matrix":
        shape = tuple(value["shape"])
        return np.loadtxt(value["path"], delimiter=",", ndmin=2).reshape(shape)
    if kind == "dataframe":
        columns = {key: _decode(v) for key, v in (value["columns"] or {}).items()}
        columns = {key: v.array if isinstance(v, pd.Series) else v for key, v in columns.items()}
        return pd.DataFrame(columns, index=value["index"])
    if kind == "vector":
        values = value["values"]
        if value["type"] in {"double", "integer"}:
            result = np.array([np.nan if v is None else v for v in values], dtype=float)
        elif value["type"] == "logical" and None in values:
            result = pd.array(values, dtype="boolean")
        else:
            result = np.array(values, dtype=object if None in values else None)
        if value.get("names") is not None:
            return pd.Series(result, index=value["names"])
        return result
    raise ValueError(f"Unsupported R result descriptor: {kind}")


def _call(code, *, packages=(), timeout=None, **arguments):
    """Run an internal R function in a separate process; never install packages."""
    executable = _rscript()
    with TemporaryDirectory(prefix="cellscope-r-") as directory:
        root = Path(directory)
        request = {
            "packages": list(dict.fromkeys(["jsonlite", "Matrix", *packages])),
            "function": code,
            "arguments": {
                key: _encode(value, root, [i * 1000000])
                for i, (key, value) in enumerate(arguments.items())
            },
            "directory": str(root),
        }
        source = root / "request.json"
        source.write_text(json.dumps(request, allow_nan=False))
        output = root / "response.json"
        result = subprocess.run(
            [
                executable,
                "--vanilla",
                str(Path(__file__).with_name("_r_bridge.R")),
                str(source),
                str(output),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode:
            error = result.stderr[-6000:]
            if "CELLSCOPE_MISSING_PACKAGES:" in error:
                raise ImportError(error.strip())
            raise RuntimeError(f"R backend failed (exit {result.returncode}): {error.strip()}")
        if not output.exists():
            raise RuntimeError("R backend did not produce a response")
        return _decode(json.loads(output.read_text()))


def require_packages(packages):
    """Probe R namespaces and return their actual installed versions."""
    versions = _call(
        """function(names) as.list(setNames(vapply(names,
            function(p) as.character(utils::packageVersion(p)), ""), names))""",
        packages=packages,
        names=list(packages),
    )
    return {key: str(value[0]) for key, value in versions.items()}


def backend_status(
    packages=(
        "scran",
        "scry",
        "scDblFinder",
        "slingshot",
        "tradeSeq",
        "nichenetr",
        "SoupX",
        "sctransform",
        "glmpca",
        "Seurat",
    ),
):
    """Explicitly probe R; importing cellscope never invokes an R process."""
    try:
        status = _call(
            """function(names) {
            result <- as.list(setNames(lapply(names, function(p)
                if(requireNamespace(p, quietly=TRUE)) as.character(utils::packageVersion(p))
                else NULL), names))
            c(list(R=R.version.string), result)
        }""",
            names=list(packages),
        )
        return {key: None if value is None else str(value[0]) for key, value in status.items()}
    except (ImportError, RuntimeError) as error:
        return {"error": str(error)}


def to_sce(data, *, mod="gex", layer="counts"):
    """Return a serialized R SingleCellExperiment with named counts and metadata."""
    from ._core import counts, modality

    adata = modality(data, mod)
    return _call(
        """function(x, obs, var) {
        dimnames(x) <- list(rownames(var), rownames(obs))
        SingleCellExperiment::SingleCellExperiment(list(counts=x), colData=obs, rowData=var)
    }""",
        packages=["SingleCellExperiment"],
        x=counts(adata, layer).T,
        obs=adata.obs,
        var=adata.var,
    )


def from_sce(sce):
    """Convert a serialized R SingleCellExperiment to AnnData with named assays."""
    from anndata import AnnData

    if not isinstance(sce, RObject):
        raise TypeError("Expected a cellscope.r.RObject containing a SingleCellExperiment")
    result = _call(
        """function(sce) {
        stopifnot(methods::is(sce, "SingleCellExperiment"))
        list(assays=as.list(SummarizedExperiment::assays(sce)),
             obs=as.data.frame(SummarizedExperiment::colData(sce)),
             var=as.data.frame(SummarizedExperiment::rowData(sce)),
             reductions=as.list(SingleCellExperiment::reducedDims(sce)))
    }""",
        packages=["SingleCellExperiment"],
        sce=sce,
    )
    assays = result["assays"]
    first = "counts" if "counts" in assays else next(iter(assays))
    adata = AnnData(assays[first].T, obs=result["obs"], var=result["var"])
    for name, matrix in assays.items():
        adata.layers[name] = matrix.T
    for name, matrix in result["reductions"].items():
        adata.obsm[name] = matrix
    return adata


def record_r(adata, key, params, packages):
    """Persist R package versions with the ordinary cellscope result record."""
    from ._core import record

    record(adata, key, params, backend="R:" + packages[0])
    adata.uns["cellscope"][key]["r_packages"] = require_packages(packages)
