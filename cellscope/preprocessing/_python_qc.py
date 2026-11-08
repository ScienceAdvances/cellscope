"""In-process Python ambient-RNA and doublet adapters; no shell execution."""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from anndata import AnnData

from .. import _logging as feedback
from .._core import counts, dependency, modality, record, sync_obs
from .._device import select_gpu


def scdblfinder_py(
    data, *, mod="gex", layer="counts", batch_key=None, random_state=0,
    clusters_key=None, use_gpu=None, device=None, **kwargs,
):
    """Call scDblFinderPy independently per capture library, preserving matrices.

    Only scores/classes and per-library thresholds are copied back. The Python
    implementation is not numerically identical to R scDblFinder. For corrected
    UMI counts, choose that count layer explicitly; never pass log expression.
    """
    use_gpu = select_gpu("scDblFinderPy", device=device, legacy=use_gpu, kind="doublets")
    backend = dependency("scDblFinderPy", "doublets-python")
    adata = modality(data, mod)
    matrix = counts(adata, layer)
    kwargs.setdefault("verbose", feedback.verbose_enabled())
    forbidden = {"adata", "random_state", "clusters_col", "use_gpu", "return_type"}
    if forbidden.intersection(kwargs):
        raise ValueError("Use the adapter's explicit options; return_type is fixed to adata")
    if clusters_key is not None and (
        clusters_key not in adata.obs or adata.obs[clusters_key].isna().any()
    ):
        raise ValueError("clusters_key must identify existing, nonmissing cluster labels")
    if batch_key is None:
        for key in ("library_id", "sample_id"):
            if key in adata.obs and adata.obs[key].nunique() > 1:
                raise ValueError("Multiple libraries detected; supply batch_key explicitly")
        groups = {"all": np.arange(adata.n_obs)}
    else:
        if batch_key not in adata.obs or adata.obs[batch_key].isna().any():
            raise ValueError("batch_key must contain nonmissing capture-library labels")
        groups = adata.obs.groupby(batch_key, observed=True, sort=False).indices
    scores = pd.Series(np.nan, index=adata.obs_names, dtype=float)
    classes = pd.Series(pd.NA, index=adata.obs_names, dtype="string")
    thresholds = {}
    for batch_number, (label, positions) in enumerate(groups.items(), 1):
        with feedback.batch("scDblFinderPy", label, batch_number, len(groups), cells=len(positions)):
            if len(positions) < 20:
                raise ValueError(f"Library {label!r} has fewer than 20 cells")
            working = AnnData(
                matrix[positions].copy(), obs=adata.obs.iloc[positions].copy(), var=adata.var.copy()
            )
            if use_gpu:
                # CuPy sparse matrices reject integer data. Upstream creates
                # integer artificial doublets and concatenates them with these
                # real counts before transferring X AND layers to the GPU.
                # Float64 here makes that concatenation GPU-compatible without
                # rounding UMI values to float32 or changing the user's matrix.
                working.X = working.X.astype(np.float64)
            working.layers["counts"] = working.X.copy()
            feedback.progress(f"scDblFinderPy {label} | artificial doublets / embedding / classifier | {len(positions):,} cells")
            result = backend.compute_doublet_score(
                working, clusters_col=clusters_key, random_state=random_state,
                use_gpu=use_gpu, return_type="adata", **kwargs,
            )
            if not result.obs_names.is_unique or set(result.obs_names) != set(working.obs_names):
                raise ValueError("scDblFinderPy returned missing, duplicated or additional real cells")
            ordered = result.obs.reindex(working.obs_names)
            score = pd.to_numeric(ordered["scDblFinder_score"])
            labels = ordered["scDblFinder_class"].astype("string")
            if not np.isfinite(score).all() or not score.between(0, 1).all():
                raise ValueError("scDblFinderPy returned invalid scores")
            if labels.isna().any() or not labels.isin(["singlet", "doublet"]).all():
                raise ValueError("scDblFinderPy returned invalid classes")
            threshold = float(result.uns["scDblFinder_threshold"])
            if not np.isfinite(threshold):
                raise ValueError("scDblFinderPy returned a nonfinite threshold")
            scores.loc[working.obs_names] = score.to_numpy()
            classes.loc[working.obs_names] = labels.to_numpy()
            thresholds[str(label)] = threshold
            feedback.info(f"scDblFinderPy {label} | singlets={int(labels.eq('singlet').sum()):,} | doublets={int(labels.eq('doublet').sum()):,} | threshold={threshold:.4f}")
    adata.obs["scDblFinder_score"] = scores
    adata.obs["scDblFinder_class"] = pd.Categorical(classes)
    adata.obs["predicted_doublet"] = classes.eq("doublet").to_numpy(dtype=bool)
    adata.uns["scDblFinder_thresholds"] = thresholds
    source = inspect.getsourcefile(backend.compute_doublet_score)
    record(adata, "scdblfinder", {
        "layer": layer, "batch_key": batch_key, "random_state": random_state,
        "clusters_key": clusters_key or "", "use_gpu": use_gpu, "options": kwargs,
        "gpu_working_dtype": "float64" if use_gpu else "unchanged",
        "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest() if source else "",
        "r_numerical_equivalence_claimed": False,
    }, backend="scDblFinderPy")
    sync_obs(data, mod)
    return data


def cellbender_file(
    input_file, output_file, *, expected_cells=None, total_droplets=None,
    epochs=150, fpr=0.01, use_cuda=None, n_threads=8, options=None, device=None,
):
    """Run CellBender 0.4.x training/posterior in this Python process.

    Input must be one unfiltered droplet library, including empty droplets.
    Returns the output H5 path. Upstream Python argument defaults/validation are
    reused; no command, R process, external script, or notebook is launched.
    Only the upstream HTML/notebook report callback is disabled, not inference,
    count estimation, figures, metrics, or checkpointing. Load using
    ``cs.io.read_cellbender`` to keep observed and corrected counts separate.
    Run libraries sequentially: the backend sets process-wide RNG/torch state.
    """
    use_cuda = select_gpu("CellBender", device=device, legacy=use_cuda, kind="torch")
    dependency("cellbender", "cellbender")
    installed = version("cellbender")
    if installed.split(".")[:2] != ["0", "4"]:
        raise ImportError("This in-process adapter is validated for cellbender==0.4.0")
    from cellbender.base_cli import get_populated_argparser
    from cellbender.remove_background import consts
    from cellbender.remove_background import run as run_module
    from cellbender.remove_background.cli import CLI

    input_file, output_file = Path(input_file).resolve(), Path(output_file).resolve()
    if not input_file.is_file():
        raise FileNotFoundError(input_file)
    if "filtered" in input_file.name.lower() and "unfiltered" not in input_file.name.lower():
        raise ValueError("CellBender requires unfiltered droplets, not a filtered cell matrix")
    if input_file == output_file or output_file.suffix != ".h5":
        raise ValueError("Use a distinct output .h5 file")
    if output_file.exists():
        raise FileExistsError("Output exists; load it or select a new output directory")
    if not isinstance(epochs, int) or epochs < 1 or not 0 <= fpr < 1:
        raise ValueError("epochs must be a positive integer and fpr must be in [0, 1)")
    if expected_cells is not None and (not isinstance(expected_cells, int) or expected_cells < 1):
        raise ValueError("expected_cells must be a positive integer or None")
    if total_droplets is not None and (
        not isinstance(total_droplets, int) or total_droplets < 1
        or (expected_cells is not None and total_droplets <= expected_cells)
    ):
        raise ValueError("total_droplets must exceed expected_cells")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    parser = get_populated_argparser()
    args = parser.parse_args([
        "remove-background", "--input", str(input_file), "--output", str(output_file)
    ])
    protected = {
        "input_file", "output_file", "tool", "use_cuda", "n_threads", "epochs", "fpr",
        "expected_cell_count", "total_droplets", "input_checkpoint_tarball",
        "force_use_checkpoint",
    }
    options = dict(options or {})
    if protected.intersection(options) or set(options).difference(vars(args)):
        raise ValueError("Unknown or protected CellBender option; use explicit adapter arguments")
    for key, value in options.items():
        setattr(args, key, value)
    args.expected_cell_count = expected_cells
    args.total_droplets = total_droplets
    args.epochs, args.fpr = epochs, [fpr]
    args.use_cuda, args.n_threads = use_cuda, n_threads
    args.input_checkpoint_tarball = str(output_file.with_suffix(".ckpt.tar.gz"))
    args = CLI.validate_args(args)
    manifest = {
        "backend": "cellbender", "version": installed, "status": "running",
        "input_file": str(input_file), "output_file": str(output_file),
        "parameters": vars(args).copy(), "upstream_seed": consts.RANDOM_SEED,
        "execution": "in-process Python", "upstream_notebook_report": False,
    }
    manifest_file = output_file.with_suffix(".cellscope.json")
    manifest_file.write_text(json.dumps(manifest, indent=2, default=str))
    logger = logging.getLogger("cellbender")
    old_level = logger.level
    handler = logging.FileHandler(output_file.with_suffix(".log"), encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    def report_filter(entry):
        # Upstream unconditionally logs report success after its callback.
        return not entry.getMessage().startswith("Succeeded in writing report to ")

    logger.addFilter(report_filter)
    try:
        logger.info("cellscope: notebook/HTML report disabled; inference runs in-process")
        with feedback.relay_backend(logger, "CellBender", epochs=args.epochs), patch.object(run_module, "run_notebook_make_html", lambda **kwargs: None):
            run_module.run_remove_background(args)
        if not output_file.is_file():
            raise RuntimeError("CellBender did not produce the expected corrected H5")
        manifest["status"] = "complete"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        logger.removeFilter(report_filter)
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(old_level)
        manifest_file.write_text(json.dumps(manifest, indent=2, default=str))
    return output_file
