"""Lazy, per-operation device selection and transactional RAPIDS dispatch."""

from __future__ import annotations

import inspect
from functools import cache
from importlib import import_module

import numpy as np
from scipy import sparse

from . import _logging as feedback


def requested_device(device=None, legacy=None):
    if legacy is not None:
        if not isinstance(legacy, (bool, np.bool_)):
            raise ValueError("use_gpu/use_cuda must be True, False, or None")
        explicit = "gpu" if legacy else "cpu"
        if device is not None and device != explicit:
            raise ValueError("device conflicts with use_gpu/use_cuda")
        return explicit
    value = feedback.settings.device if device is None else device
    if value not in {"auto", "cpu", "gpu"}:
        raise ValueError("device must be 'auto', 'cpu', or 'gpu'")
    return value


@cache
def _probe(kind):
    try:
        if kind == "torch":
            torch = import_module("torch")
            if not torch.cuda.is_available():
                return False, "PyTorch CUDA unavailable"
            torch.ones(1, device="cuda").sum().item()
            return True, torch.cuda.get_device_name(0)
        cp = import_module("cupy")
        if cp.cuda.runtime.getDeviceCount() < 1:
            return False, "no CUDA device"
        cp.ones(1).sum().item()
        rsc = import_module("rapids_singlecell")
        for name in ("cuml", "cugraph", "cuvs"):
            import_module(name)
        if kind == "doublets":
            xgb = import_module("xgboost")
            if not xgb.build_info().get("USE_CUDA", False):
                return False, "XGBoost was built without CUDA"
        return True, f"CUDA device {cp.cuda.runtime.getDevice()} / {rsc.__name__}"
    except Exception as error:  # noqa: BLE001 - optional CUDA stacks can fail ABI initialization
        # Probe failures include driver/runtime ABI errors, not just imports.
        return False, f"{type(error).__name__}: {error}"


def device_info(*, refresh=False):
    """Probe each optional GPU stack lazily; never installs packages."""
    if refresh:
        _probe.cache_clear()
    return {
        "device": feedback.settings.device,
        **{kind: dict(zip(("available", "detail"), _probe(kind), strict=True))
           for kind in ("rapids", "doublets", "torch")},
    }


def cpu_only(operation, *, device=None):
    """Keep an existing CPU model interface instead of changing its algorithm."""
    if requested_device(device) == "gpu":
        raise RuntimeError(f"{operation} has no compatible GPU adapter; use device='cpu' or 'auto'")
    feedback.info(f"{operation} | CPU | no compatible GPU adapter")


def execution_backend(adata, namespace, name):
    return adata.uns.get("cellscope", {}).get("devices", {}).get(
        f"{namespace}.{name}", {}
    ).get("backend", "scanpy")


def select_gpu(operation, *, device=None, legacy=None, kind="rapids", reason=None, announce=True):
    mode = requested_device(device, legacy)
    if mode == "cpu":
        if announce:
            feedback.info(f"{operation} | CPU (requested)")
        return False
    available, detail = (False, reason) if reason else _probe(kind)
    if not available:
        message = f"{operation} | GPU unavailable: {detail}"
        if mode == "gpu":
            raise RuntimeError(message + "; use device='auto' for CPU fallback")
        feedback.warning(message + "; using CPU")
        return False
    if announce:
        feedback.info(f"{operation} | GPU | {detail}")
    return True


def _host(value):
    if type(value).__module__.split(".")[0] in {"cupy", "cupyx"}:
        value = value.get()
    return sparse.csr_matrix(value) if sparse.issparse(value) else value


def run_scanpy(adata, namespace, name, *, device=None, **kwargs):
    """Use equivalent RAPIDS entry points; preserve host assays and CPU returns.

    GPU execution uses an isolated copy. Only a successful call is committed.
    Unsupported explicit CPU options select CPU in auto mode and error in GPU
    mode. Algorithm/validation errors are never silently retried on CPU.
    """
    mode = requested_device(device)
    gpu_kwargs = dict(kwargs)
    reason = None
    if (kwargs.get("copy") or kwargs.get("inplace") is False
            or kwargs.get("subset") or kwargs.get("return_info")):
        reason = "copy/inplace/subset return semantics require Scanpy"
    if name == "pca" and kwargs.get("obsm"):
        reason = "PCA obsm input requires Scanpy"
    if name == "pca" and kwargs.get("svd_solver") == "arpack":
        reason = "explicit ARPACK solver requires CPU"
    if kwargs.get("use_raw"):
        reason = "use_raw=True requires CPU in this adapter"
    if name == "leiden":
        if "flavor" in kwargs or kwargs.get("directed") is True:
            reason = "explicit Leiden flavor/directed=True requires CPU"
        if kwargs.get("n_iterations", 100) < 1:
            reason = "Leiden convergence mode n_iterations<1 requires CPU"
        gpu_kwargs.pop("directed", None)
    if name == "highly_variable_genes":
        gpu_kwargs.pop("subset", None)
        gpu_kwargs.pop("inplace", None)
    # Avoid importing any CUDA dependencies when explicitly using CPU.
    use_gpu = select_gpu(f"{namespace}.{name}", device=mode, reason=reason, announce=False)
    rsc = None
    if use_gpu:
        rsc = import_module("rapids_singlecell")
        gpu_function = getattr(getattr(rsc, namespace), name, None)
        if gpu_function is None:
            reason = "installed rapids-singlecell has no matching function"
        else:
            signature = inspect.signature(gpu_function)
            unsupported = set(gpu_kwargs) - set(signature.parameters)
            if unsupported:
                reason = "GPU adapter does not support options: " + ", ".join(sorted(unsupported))
        if reason:
            use_gpu = select_gpu(f"{namespace}.{name}", device=mode, reason=reason)
    if not use_gpu:
        feedback.info(f"{namespace}.{name} | CPU / scanpy")
        sc = import_module("scanpy")
        result = getattr(getattr(sc, namespace), name)(adata, **kwargs)
        _record_backend(adata, namespace, name, "cpu", "scanpy")
        return result

    feedback.info(f"{namespace}.{name} | GPU / rapids-singlecell")
    if name == "leiden" and gpu_kwargs.get("n_iterations", 100) <= 2:
        feedback.warning("GPU Leiden n_iterations<=2 may stop before adequate optimization; "
                         "use n_iterations=100 and assess clustering stability")
    working = adata.copy()
    # Transfer only the selected assay; counts/raw_counts stay on the host.
    layer = gpu_kwargs.get("layer")
    matrix = working.X if layer is None else working.layers[layer]
    if matrix.dtype.kind not in "fc":
        matrix = matrix.astype(np.float64)
    if sparse.issparse(matrix):
        matrix = sparse.csr_matrix(matrix)
    if layer is None:
        working.X = matrix
    else:
        working.layers[layer] = matrix
    rsc.get.anndata_to_GPU(working, layer=layer)
    result = gpu_function(working, **gpu_kwargs)
    rsc.get.anndata_to_CPU(working, layer=layer)
    # Only normalized/log transformed assays are written back; PCA/HVG/
    # graph/clustering must not change the original assay's dtype or values.
    if name in {"normalize_total", "log1p"}:
        if layer is None:
            adata.X = working.X
        else:
            adata.layers[layer] = working.layers[layer]
    adata.obs = working.obs.copy()
    adata.var = working.var.copy()
    for attr in ("obsm", "varm", "obsp"):
        for key, value in getattr(working, attr).items():
            getattr(adata, attr)[key] = _host(value)
    adata.uns.update(working.uns)
    _record_backend(adata, namespace, name, "gpu", "rapids_singlecell")
    return result


def _record_backend(adata, namespace, name, device, backend):
    adata.uns.setdefault("cellscope", {}).setdefault("devices", {})[
        f"{namespace}.{name}"
    ] = {"device": device, "backend": backend}
