"""Batch-aware AnnData preprocessing: X is the default working count matrix."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .. import _logging as feedback
from .._core import counts, modality, record, sync_obs
from ._python_qc import cellbender_file, scdblfinder_py


def _groups(adata, key):
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("Preprocessing requires a nonempty cell-by-gene matrix")
    if key is None:
        return {"all": np.arange(adata.n_obs)}
    if key not in adata.obs or adata.obs[key].isna().any():
        raise ValueError("batch_key must identify nonmissing capture-library labels")
    groups = adata.obs.groupby(key, observed=True, sort=False).indices
    if len({str(k) for k in groups}) != len(groups):
        raise ValueError("Batch labels must remain unique when converted to strings")
    return {str(k): v for k, v in groups.items()}


def _value(value, batch):
    return value[batch] if isinstance(value, Mapping) else value


def cellbender(
    data, output_file=None, *, batch_key=None, raw_files=None, output_dir=None,
    layer=None, backup_layer="raw_counts", counts_layer="counts", reuse_outputs=False, device=None,
    **kwargs,
):
    """Correct merged AnnData.X per capture; preserve its cells/features/order.

    ``raw_files`` maps batch labels to unfiltered droplet H5 files. The merged
    AnnData may contain Cell Ranger filtered cells: empty droplets are obtained
    from those raw files, not invented from the filtered matrix. expected_cells
    and total_droplets in kwargs accept scalars or per-batch dictionaries.
    No matrix is changed until every batch succeeds and aligns. Original X is
    backed up once; do not overwrite an existing backup. Cell probabilities are
    annotated, not used for an implicit row deletion. Returns the same object.

    Passing file paths retains the legacy single-library interface.
    """
    if not hasattr(data, "obs") and not hasattr(data, "mod"):
        return cellbender_file(data, output_file, device=device, **kwargs)
    from .. import io
    from .._device import select_gpu

    legacy_cuda = kwargs.pop("use_cuda", None)
    kwargs["use_cuda"] = select_gpu("CellBender", device=device,
                                    legacy=legacy_cuda, kind="torch")
    adata = modality(data)
    if layer is not None:
        raise ValueError("This AnnData interface corrects X; assign the desired count matrix to X")
    matrix = counts(adata, None)
    if backup_layer in adata.layers or backup_layer == counts_layer:
        raise ValueError("Backup layer already exists or conflicts with counts_layer; never overwrite it")
    if batch_key is None or output_dir is None or not isinstance(raw_files, Mapping):
        raise ValueError("Supply batch_key, raw_files mapping, and output_dir")
    if output_file is not None:
        raise ValueError("Use output_dir for merged AnnData")
    groups = _groups(adata, batch_key)
    raw_files = {str(k): Path(v).resolve() for k, v in raw_files.items()}
    if not set(groups).issubset(raw_files) or any(not raw_files[k].is_file() for k in groups):
        raise ValueError("raw_files must supply an existing unfiltered file for every batch")
    ids_key = "gene_ids" if "gene_ids" in adata.var else "gene_id"
    if ids_key not in adata.var or adata.var[ids_key].isna().any():
        raise ValueError("Stable gene_ids or gene_id required for count alignment")
    gene_ids = pd.Index(adata.var[ids_key].astype(str))
    if not gene_ids.is_unique or "barcode" not in adata.obs or adata.obs.barcode.isna().any():
        raise ValueError("Unique gene IDs and original per-library barcode metadata required")
    if any(adata.obs.iloc[p]["barcode"].astype(str).duplicated().any() for p in groups.values()):
        raise ValueError("Duplicate barcodes within a batch; use capture library as batch_key")
    assembled, global_positions, probabilities, batch_records = [], [], [], []
    feedback.info(f"CellBender | {len(groups)} libraries | backup X -> {backup_layer}; correction -> X")
    for batch_number, (batch, positions) in enumerate(groups.items(), 1):
        with feedback.batch("CellBender", batch, batch_number, len(groups), cells=len(positions)):
            # Hashes avoid turning arbitrary metadata labels into unsafe path components.
            folder = Path(output_dir).resolve() / ("batch_" + hashlib.sha256(batch.encode()).hexdigest()[:12])
            folder.mkdir(parents=True, exist_ok=True)
            output = folder / "cellbender.h5"
            options = dict(kwargs)
            for key in ("expected_cells", "total_droplets"):
                if key in options:
                    options[key] = _value(options[key], batch)
                    if isinstance(options[key], np.integer):
                        options[key] = int(options[key])
            requested = {"expected_cells": None, "total_droplets": None, "epochs": 150,
                         "fpr": 0.01, "use_cuda": False, "n_threads": 8, "options": None}
            requested.update(options)
            feedback.progress(f"CellBender {batch} | validating raw-file SHA256")
            digest = hashlib.sha256()
            with raw_files[batch].open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            request = {"raw_file": str(raw_files[batch]), "raw_sha256": digest.hexdigest(),
                       "settings": requested}
            request_json = json.dumps(request, sort_keys=True, default=str)
            request_path = folder / "adata_request.json"
            if not output.exists():
                feedback.progress(f"CellBender {batch} | training/inference | epochs={options.get('epochs', 150)} | output={output}")
                cellbender_file(raw_files[batch], output, **options)
                request_path.write_text(request_json)
            elif not reuse_outputs:
                raise FileExistsError(f"{output}: explicitly set reuse_outputs=True to load")
            else:
                feedback.progress(f"CellBender {batch} | checking cached output provenance")
                manifest_path = output.with_suffix(".cellscope.json")
                manifest = json.loads(manifest_path.read_text())
                if manifest.get("status") != "complete" or manifest.get("input_file") != str(raw_files[batch]):
                    raise ValueError("Existing CellBender output lacks matching complete provenance")
                if not request_path.exists() or request_path.read_text() != request_json:
                    raise ValueError("Cached CellBender raw-data hash or scientific settings differ")
            feedback.progress(f"CellBender {batch} | loading corrected counts and aligning barcodes/gene IDs")
            corrected = io.read_cellbender(
                output, raw_file=raw_files[batch], library_id="alignment",
                min_cell_probability=None,
            )
            # The reader prefixes identifiers; match on original barcode within this batch.
            barcodes = pd.Index(adata.obs.iloc[positions]["barcode"].astype(str))
            if not barcodes.is_unique:
                raise ValueError(f"Duplicate original barcodes within {batch}")
            rows = pd.Index(corrected.obs["barcode"].astype(str)).get_indexer(barcodes)
            cols = pd.Index(corrected.var["gene_id"].astype(str)).get_indexer(gene_ids)
            if (rows < 0).any() or (cols < 0).any():
                raise ValueError(f"{batch}: output omits input cells/genes; increase total_droplets or check reference")
            observed = corrected.layers["raw_counts"][rows][:, cols]
            if (sparse.csr_matrix(matrix[positions]) != observed).nnz:
                raise ValueError(f"{batch}: original X differs from the specified raw file")
            assembled.append(corrected.X[rows][:, cols])
            feedback.debug(f"CellBender {batch} | aligned {len(rows):,} cells x {len(cols):,} genes")
            global_positions.extend(positions)
            probabilities.extend(corrected.obs.iloc[rows]["cell_probability"].to_numpy())
            batch_records.append({"batch": batch, "raw_file": str(raw_files[batch]), "output": str(output)})
    restore = np.argsort(global_positions)
    feedback.progress("CellBender | all batches aligned; backing up original X and committing corrected counts")
    corrected_matrix = sparse.vstack(assembled, format="csr")[restore]
    adata.layers[backup_layer] = matrix.copy()
    adata.X = corrected_matrix
    if counts_layer is not None:
        adata.layers[counts_layer] = corrected_matrix.copy()
    adata.obs["cellbender_cell_probability"] = np.asarray(probabilities)[restore]
    adata.obs["cellbender_is_cell"] = adata.obs["cellbender_cell_probability"].gt(0.5)
    record(adata, "cellbender", {
        "batch_key": batch_key, "backup_layer": backup_layer,
        "counts_layer": counts_layer or "", "batches": pd.DataFrame(batch_records),
        "row_filtering": False, "output_dir": str(Path(output_dir).resolve()),
    }, backend="cellbender")
    sync_obs(data)
    return data


def scDblFinder(
    data, *, batch_key=None, layer=None, remove_doublets=False, mod="gex", device=None, **kwargs,
):
    """Use scDblFinderPy per batch, defaulting to X; optionally return singlets.

    Always annotate all input cells first. With remove_doublets=True the returned
    object is a subset; all calls are retained in uns['scDblFinder_all_calls'].
    """
    scdblfinder_py(data, mod=mod, layer=layer, batch_key=batch_key, device=device, **kwargs)
    adata = modality(data, mod)
    columns = ["scDblFinder_score", "scDblFinder_class", "predicted_doublet"]
    if batch_key is not None:
        columns.insert(0, batch_key)
    adata.uns["scDblFinder_all_calls"] = adata.obs[list(dict.fromkeys(columns))].copy()
    if remove_doublets:
        if adata.obs["predicted_doublet"].all():
            raise ValueError("All cells were predicted doublets; inspect scores before filtering")
        removed = int(adata.obs["predicted_doublet"].sum())
        feedback.info(f"scDblFinder | removing {removed:,}/{adata.n_obs:,} predicted doublets; all calls retained in uns")
        return data[adata.obs_names[~adata.obs["predicted_doublet"]]].copy()
    return data


def basic_qc(
    data, *, mod="gex", layer=None, batch_key=None, min_genes=0, min_cells=0,
    min_counts=0, max_pct_mito=100, mito_pattern=r"(?i)^mt-", key_added="qc_pass",
    filter=False, gene_join="union",
):
    """Batch-wise minimum cell/gene QC, with explicitly union/intersection genes.

    min_genes is detected genes per cell; min_cells is cells detecting each gene
    within a library. Union retains genes passing in any remaining library and
    does not zero out observed counts in others. Iterate monotonically to a
    stable cell/feature subset. No-filter mode only annotates the proposed mask.
    """
    import scanpy as sc

    adata = modality(data, mod)
    matrix = sparse.csr_matrix(counts(adata, layer))
    groups = _groups(adata, batch_key)
    if gene_join not in {"union", "intersection"}:
        raise ValueError("gene_join must be union or intersection")
    if not 0 <= max_pct_mito <= 100:
        raise ValueError("max_pct_mito must be in [0,100]")
    for batch in groups:
        for value in (min_genes, min_cells, min_counts):
            v = _value(value, batch)
            if not isinstance(v, (int, float, np.integer, np.floating)) or not np.isfinite(v) or v < 0:
                raise ValueError("Minimum thresholds must be finite and nonnegative")
    adata.var["mt"] = adata.var_names.str.contains(mito_pattern, regex=True)
    sc.pp.calculate_qc_metrics(adata, layer=layer, qc_vars=["mt"], percent_top=None, inplace=True)
    cells = adata.obs.pct_counts_mt.le(max_pct_mito).to_numpy()
    genes = np.ones(adata.n_vars, dtype=bool)
    iteration = 0
    while True:
        iteration += 1
        next_cells = cells.copy()
        detected = np.zeros((adata.n_vars, len(groups)), dtype=np.int64)
        pass_genes = np.zeros((adata.n_vars, len(groups)), dtype=bool)
        for j, (batch, positions) in enumerate(groups.items()):
            if iteration == 1:
                feedback.info(f"QC [{j + 1}/{len(groups)}] {batch} | {len(positions):,} cells")
            feedback.progress(f"QC iteration {iteration} | [{j + 1}/{len(groups)}] {batch} | checking cell/gene support")
            block = matrix[positions][:, genes]
            n_genes = np.asarray((block > 0).sum(axis=1)).ravel()
            n_counts = np.asarray(block.sum(axis=1)).ravel()
            next_cells[positions] &= (n_genes >= _value(min_genes, batch)) & (n_counts >= _value(min_counts, batch))
            selected = positions[next_cells[positions]]
            if len(selected):
                detected[:, j] = np.asarray((matrix[selected] > 0).sum(axis=0)).ravel()
                pass_genes[:, j] = detected[:, j] >= _value(min_cells, batch)
        active = np.array([next_cells[p].any() for p in groups.values()])
        next_genes = genes & (
            (pass_genes[:, active].any(axis=1) if gene_join == "union" else pass_genes[:, active].all(axis=1))
            if active.any() else False
        )
        if np.array_equal(cells, next_cells) and np.array_equal(genes, next_genes):
            break
        cells, genes = next_cells, next_genes
    adata.obs[key_added] = cells
    feedback.debug(f"QC | stable after {iteration} iterations | gene_join={gene_join}")
    for j, (batch, positions) in enumerate(groups.items()):
        feedback.info(f"QC {batch} | cells {len(positions):,} -> {int(cells[positions].sum()):,} | {int(pass_genes[:, j].sum()):,} genes supported")
    adata.var["qc_gene_pass"] = genes
    adata.varm["qc_gene_detected_cells_by_batch"] = detected
    adata.varm["qc_gene_pass_by_batch"] = pass_genes
    summary = pd.DataFrame([
        {"batch": b, "before_cells": len(p), "after_cells": int(cells[p].sum()),
         "genes_passing_in_batch": int(pass_genes[:, j].sum())}
        for j, (b, p) in enumerate(groups.items())
    ])
    record(adata, "basic_qc", {
        "batch_key": batch_key or "", "batch_order": list(groups), "layer": layer or "X",
        "min_genes": min_genes, "min_cells": min_cells, "min_counts": min_counts,
        "max_pct_mito": max_pct_mito, "gene_join": gene_join, "summary": summary,
    }, backend="scanpy")
    sync_obs(data, mod)
    if not filter:
        return data
    if not cells.any() or not genes.any():
        raise ValueError("QC removed all cells or genes; inspect masks and thresholds")
    result = data[adata.obs_names[cells]].copy()
    if hasattr(result, "mod"):
        result.mod[mod] = result.mod[mod][:, genes].copy()
        result.update()
    else:
        result = result[:, genes].copy()
    sc.pp.calculate_qc_metrics(modality(result, mod), layer=layer, qc_vars=["mt"], percent_top=None, inplace=True)
    return result


def filter_qc(
    data, *, batch_key=None, mod="gex", layer=None, method="mad", nmads=3,
    mad_metrics=("log1p_total_counts", "log1p_n_genes_by_counts"),
    min_counts=None, max_counts=None, min_genes=None, max_genes=None,
    max_pct_mito=20, mito_pattern=r"(?i)^mt-", zero_mad="skip",
    remove_mito_genes=False, filter=True,
):
    """One entry for mitochondrial-cell and per-library MAD/fixed QC filtering.

    MAD uses unscaled median absolute deviation, independently within batches;
    log1p UMI/gene metrics are two-tailed, pct_counts_mt is upper-tailed. With
    zero_mad='skip' constant/degenerate metrics do not impose a zero-width gate;
    'strict' flags deviations from the median. High-mito cells and mitochondrial
    genes are distinct: genes are retained unless remove_mito_genes=True.
    """
    import scanpy as sc
    from scipy.stats import median_abs_deviation

    adata = modality(data, mod)
    counts(adata, layer)
    groups = _groups(adata, batch_key)
    if method not in {"mad", "fixed", "both"} or zero_mad not in {"skip", "strict"}:
        raise ValueError("method must be mad/fixed/both; zero_mad must be skip/strict")
    if not np.isfinite(nmads) or nmads <= 0:
        raise ValueError("nmads must be positive")
    if method == "mad" and any(v is not None for v in (min_counts, max_counts, min_genes, max_genes)):
        raise ValueError("Use method='both' to combine fixed UMI/gene limits with MAD")
    adata.var["mt"] = adata.var_names.str.contains(mito_pattern, regex=True)
    sc.pp.calculate_qc_metrics(adata, layer=layer, qc_vars=["mt"], percent_top=None, inplace=True)
    flags = {key: np.zeros(adata.n_obs, dtype=bool) for key in ("mito", "mad", "fixed")}
    thresholds = []
    for batch, positions in groups.items():
        feedback.progress(f"QC filtering {batch} | method={method} | {len(positions):,} cells | max_pct_mito={_value(max_pct_mito, batch) if max_pct_mito is not None else 'disabled'}")
        if max_pct_mito is not None:
            upper = _value(max_pct_mito, batch)
            if not np.isfinite(upper) or not 0 <= upper <= 100:
                raise ValueError("max_pct_mito must be in [0,100]")
            values = adata.obs.iloc[positions].pct_counts_mt.to_numpy()
            flags["mito"][positions] = ~np.isfinite(values) | (values > upper)
        if method in {"fixed", "both"}:
            for metric, lower, upper in (
                ("total_counts", min_counts, max_counts),
                ("n_genes_by_counts", min_genes, max_genes),
            ):
                lo = -np.inf if lower is None else _value(lower, batch)
                hi = np.inf if upper is None else _value(upper, batch)
                if np.isnan(lo) or np.isnan(hi) or lo > hi:
                    raise ValueError("Invalid fixed QC limits")
                values = adata.obs.iloc[positions][metric].to_numpy()
                flags["fixed"][positions] |= (values < lo) | (values > hi) | ~np.isfinite(values)
                thresholds.append({"batch": batch, "metric": metric, "method": "fixed", "lower": lo, "upper": hi})
        if method in {"mad", "both"}:
            for metric in mad_metrics:
                values = adata.obs.iloc[positions][metric].to_numpy(dtype=float)
                finite = np.isfinite(values)
                center = np.median(values[finite]) if finite.any() else np.nan
                mad = median_abs_deviation(values[finite], scale=1) if finite.any() else np.nan
                skipped = bool(mad == 0 and zero_mad == "skip")
                lower = -np.inf if metric == "pct_counts_mt" or skipped else center - nmads * mad
                upper = np.inf if skipped else center + nmads * mad
                flags["mad"][positions] |= ~finite | (values < lower) | (values > upper)
                thresholds.append({"batch": batch, "metric": metric, "method": "mad", "median": center,
                                   "mad": mad, "lower": lower, "upper": upper, "zero_mad_skipped": skipped})
    for reason, mask in flags.items():
        adata.obs[f"qc_fail_{reason}"] = mask
    keep = ~(flags["mito"] | flags["mad"] | flags["fixed"])
    for batch, positions in groups.items():
        feedback.info(f"QC filtering {batch} | keep {int(keep[positions].sum()):,}/{len(positions):,} cells")
        feedback.debug(f"QC filtering {batch} | failed: mito={int(flags['mito'][positions].sum())}, MAD={int(flags['mad'][positions].sum())}, fixed={int(flags['fixed'][positions].sum())} (reasons may overlap)")
    adata.obs["filter_qc_pass"] = keep
    summary = pd.DataFrame([
        {"batch": b, "before_cells": len(p), "after_cells": int(keep[p].sum())}
        for b, p in groups.items()
    ])
    threshold_table = pd.DataFrame(thresholds)
    if "zero_mad_skipped" in threshold_table:
        threshold_table["zero_mad_skipped"] = threshold_table["zero_mad_skipped"].astype("boolean").fillna(False).astype(bool)
    adata.uns["filter_qc_thresholds"] = threshold_table
    columns = [
        "total_counts", "n_genes_by_counts", "pct_counts_mt",
        "qc_fail_mito", "qc_fail_mad", "qc_fail_fixed", "filter_qc_pass",
    ]
    if batch_key is not None:
        columns.insert(0, batch_key)
    adata.uns["filter_qc_all_cells"] = adata.obs[list(dict.fromkeys(columns))].copy()
    record(adata, "filter_qc", {
        "batch_key": batch_key or "", "layer": layer or "X", "method": method,
        "nmads": nmads, "mad_metrics": list(mad_metrics), "max_pct_mito": max_pct_mito,
        "zero_mad": zero_mad, "remove_mito_genes": remove_mito_genes, "summary": summary,
        "metrics_before_mito_gene_removal": True,
    }, backend="scanpy")
    sync_obs(data, mod)
    if filter and not keep.any():
        raise ValueError("QC removed all cells; inspect filter_qc_all_cells and thresholds")
    result = data[adata.obs_names[keep]].copy() if filter else data
    if remove_mito_genes:
        feedback.info(f"QC | removing {int(modality(result, mod).var['mt'].sum()):,} mitochondrial genes; metrics retain pre-removal meaning")
        if modality(result, mod).var["mt"].all():
            raise ValueError("Removing mitochondrial genes would leave no genes")
        if hasattr(result, "mod"):
            result.mod[mod] = result.mod[mod][:, ~result.mod[mod].var["mt"]].copy()
            result.update()
        else:
            result = result[:, ~result.var["mt"]].copy()
    return result
