"""Clone phenotype distributions and unbalanced PhenoTrack flows.

Independent implementation of the published equations and author plotting
normalization. Cell-state labels/probabilities can come from Scanpy workflows.
No receptor identity inference or differential clonal expansion is performed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from .. import _logging as feedback
from .._core import modality

REFERENCE = "https://github.com/zsethna/PhenoTrack"
REFERENCE_SHA256 = {
    "pheno_def.py": "c65f7bee459b3705c8d83debcbd6aa735e95c299aa0da18d77be9bb3542034cc",
    "plotting_phenos.py": "d37d7d90e1e4dd2240c5d35f842d48699a17a95d317f9942b1e51c63ec9e8662",
}
_FLOW = ["donor_id", "from_time", "to_time", "clone_id", "from_phenotype",
         "to_phenotype", "outflow", "inflow"]
_FLUX = ["donor_id", "from_time", "to_time", "clone_id", "from_cells", "to_cells", "phenotypic_flux"]
_EVENT = ["donor_id", "from_time", "to_time", "clone_id", "event", "from_cells", "to_cells"]


@dataclass
class PhenoTrackResult:
    """Serializable tables, explicit denominators and method provenance."""

    composition: pd.DataFrame
    nodes: pd.DataFrame
    flows: pd.DataFrame
    flux: pd.DataFrame
    events: pd.DataFrame
    denominators: pd.DataFrame
    parameters: dict

    def as_dict(self):
        return {key: getattr(self, key) for key in self.__dataclass_fields__}

    def write(self, directory, *, overwrite=False):
        """Export CSV tables and parameters.json; overwrite requires explicit opt-in."""
        folder = Path(directory)
        if folder.exists() and any(folder.iterdir()) and not overwrite:
            raise FileExistsError(f"Output directory is not empty: {folder}")
        metadata = json.dumps(_plain(self.parameters), ensure_ascii=False, indent=2, allow_nan=False)
        folder.mkdir(parents=True, exist_ok=True)
        for key in self.__dataclass_fields__:
            if key != "parameters":
                getattr(self, key).to_csv(folder / f"{key}.csv", index=False)
        (folder / "parameters.json").write_text(metadata + "\n", encoding="utf-8")


def _order(values, supplied, label):
    observed = list(dict.fromkeys(str(x) for x in values))
    if supplied is None:
        return observed
    if isinstance(supplied, (str, set, frozenset)):
        raise TypeError(f"{label} must be an ordered sequence")
    result = [str(x) for x in supplied]
    if not result or len(set(result)) != len(result) or not set(observed).issubset(result):
        raise ValueError(f"{label} must contain unique labels covering all observed values")
    return result


def _probabilities(adata, phenotype_key, probabilities_key, phenotype_order):
    if probabilities_key is None:
        if phenotype_key not in adata.obs:
            raise KeyError(phenotype_key)
        labels = adata.obs[phenotype_key]
        if phenotype_order is None and isinstance(labels.dtype, pd.CategoricalDtype):
            observed = set(labels.dropna())
            phenotype_order = [x for x in labels.cat.categories if x in observed]
        states = _order(labels.dropna(), phenotype_order, "phenotype_order")
        weights = np.column_stack([labels.astype("string").eq(s).fillna(False).to_numpy(dtype=float)
                                   for s in states]) if states else np.empty((adata.n_obs, 0))
    else:
        source = adata.obsm[probabilities_key]
        if isinstance(source, pd.DataFrame):
            if not source.index.equals(adata.obs_names):
                raise ValueError("Probability rows must follow adata.obs_names")
            names = [str(x) for x in source.columns]
            states = _order(names, phenotype_order, "phenotype_order")
            if set(states) != set(names) or len(set(names)) != len(names):
                raise ValueError("Probability state names must be unique and match phenotype_order")
            weights = source.to_numpy(dtype=float)[:, [names.index(x) for x in states]]
        else:
            if phenotype_order is None:
                raise ValueError("Array probabilities require phenotype_order")
            states = _order(phenotype_order, phenotype_order, "phenotype_order")
            weights = np.asarray(source, dtype=float)
        if weights.shape != (adata.n_obs, len(states)) or not np.isfinite(weights).all():
            raise ValueError("Probabilities must be a finite cells-by-phenotypes matrix")
        if (weights < 0).any() or (weights > 1).any():
            raise ValueError("Phenotype probabilities must lie in [0, 1]")
        mass = weights.sum(axis=1)
        if not np.all(np.isclose(mass, 1, atol=1e-6) | (mass == 0)):
            raise ValueError("Each probability row must sum to 1, or 0 for an unassigned cell")
    if not states:
        raise ValueError("No phenotype labels available")
    return states, weights


def phenotrack(data, *, time_key, time_order=None, clone_key="clone_id",
              phenotype_key="cell_type", probabilities_key=None, phenotype_order=None,
              donor_key="donor_id", sample_key="sample_id", clones=None,
              clone_definition=None,
              denominator="all_cells", normalize=True, min_cell_count=0,
              mod="gex", key_added="phenotrack"):
    """Track supplied clone identities across annotated single-cell timepoints.

    Each input row is one cell. all_cells (default) retains the full input
    donor/time denominator before clone selection, matching author cell_list
    normalization. annotated_cells is an explicit alternative. Missing clone
    or state labels contribute no phenotype mass. Donors are never pooled.
    time_order is required unless time_key is an ordered categorical.
    min_cell_count is the author's strict > threshold for L1 flux only.
    """
    adata = modality(data, mod)
    if denominator not in {"all_cells", "annotated_cells"}:
        raise ValueError("denominator must be all_cells or annotated_cells")
    if not isinstance(normalize, (bool, np.bool_)):
        raise TypeError("normalize must be boolean")
    if not np.isfinite(min_cell_count) or min_cell_count < 0:
        raise ValueError("min_cell_count must be finite and nonnegative")
    required = [donor_key, time_key, clone_key, sample_key]
    if len(set(required)) != 4:
        raise ValueError("donor_key, time_key, clone_key and sample_key must name distinct columns")
    for key in required:
        if key not in adata.obs:
            raise KeyError(key)
    if adata.n_obs == 0:
        raise ValueError("PhenoTrack requires at least one cell")
    for key in (donor_key, time_key, sample_key):
        if adata.obs[key].isna().any():
            raise ValueError(f"{key} must contain nonmissing labels")
    if time_order is None:
        col = adata.obs[time_key]
        if not isinstance(col.dtype, pd.CategoricalDtype) or not col.cat.ordered:
            raise ValueError("Supply chronological time_order, or use an ordered time categorical")
        time_order = list(col.cat.categories)
    times = _order(adata.obs[time_key], time_order, "time_order")
    states, weights = _probabilities(adata, phenotype_key, probabilities_key, phenotype_order)
    meta = adata.obs[required].copy()
    meta.columns = ["donor_id", "time", "clone_id", "sample_id"]
    for column in ("donor_id", "time", "sample_id"):
        labels = pd.Series([str(x) for x in meta[column]], index=meta.index)
        if meta[column].nunique() != labels.nunique():
            raise ValueError(f"{column} labels collide after conversion to strings")
        meta[column] = labels
    # A capture sample may not be assigned to different people or times.
    context = meta.groupby("sample_id", observed=True)[["donor_id", "time"]].nunique()
    if context.gt(1).any().any():
        raise ValueError("Each sample_id must identify exactly one donor and time")
    if meta.clone_id.dropna().map(lambda x: isinstance(x, (list, tuple, dict, set, np.ndarray))).any():
        raise ValueError("Each cell needs one resolved clone identity; multi-candidate assignments must be resolved explicitly")
    valid_clone = meta.clone_id.notna() & ~meta.clone_id.astype(str).isin(
        ["", "nan", "NaN", "None", "not_detected"])
    meta["clone_id"] = meta.clone_id.astype(str)
    valid_state = weights.sum(axis=1) > 0
    annotated = valid_clone.to_numpy() & valid_state
    if not annotated.all():
        feedback.warning(f"PhenoTrack | {(~annotated).sum():,} cells lack a clone or phenotype; "
                         f"denominator={denominator}")
    donors = list(dict.fromkeys(meta.donor_id))
    if isinstance(clones, str):
        raise TypeError("clones must be a sequence or donor-to-sequence mapping")
    if isinstance(clones, Mapping) and set(map(str, clones)) != set(donors):
        raise ValueError("clones mapping must cover each donor exactly once")
    if isinstance(clones, Mapping):
        if len(clones) != len(donors) or any(isinstance(v, str) for v in clones.values()):
            raise ValueError("Each donor must map to a clone sequence, not a string")
        selected_by_donor = {str(k): set(map(str, v)) for k, v in clones.items()}
    elif clones is None:
        selected_by_donor = None
    else:
        ids = set(map(str, clones))
        selected_by_donor = {d: ids.copy() for d in donors}
    compositions, node_rows, denominator_rows = [], [], []
    flow_rows, flux_rows, events = [], [], []
    for number, donor in enumerate(donors, 1):
        positions = np.flatnonzero(meta.donor_id.eq(donor).to_numpy())
        with feedback.batch("PhenoTrack", donor, number, len(donors), cells=len(positions)):
            if selected_by_donor is not None and selected_by_donor[donor]:
                observed = set(meta.iloc[positions].loc[valid_clone.iloc[positions], "clone_id"])
                if not observed.intersection(selected_by_donor[donor]):
                    feedback.warning(f"PhenoTrack {donor} | no selected clone IDs found; "
                                     f"verify that candidates use obs[{clone_key!r}] identities")
            vectors, norms, sampled = {}, {}, {}
            for time in times:
                pos = positions[meta.iloc[positions].time.eq(time).to_numpy()]
                sampled[time] = len(pos) > 0
                norm = len(pos) if denominator == "all_cells" else int(annotated[pos].sum())
                norms[time] = norm
                denominator_rows.append({"donor_id": donor, "time": time, "n_cells": len(pos),
                                         "n_annotated_cells": int(annotated[pos].sum()),
                                         "denominator": norm, "sampled": sampled[time]})
                use = pos[annotated[pos]]
                if selected_by_donor is not None:
                    use = use[meta.iloc[use].clone_id.isin(selected_by_donor[donor]).to_numpy()]
                table = pd.DataFrame(weights[use], columns=states)
                table.index = meta.iloc[use].clone_id.to_numpy()
                grouped = table.groupby(level=0, sort=False).sum()
                vectors[time] = {str(clone): row.to_numpy(dtype=float)
                                 for clone, row in grouped.iterrows()}
                scale = norm if normalize else 1
                sums = grouped.sum(axis=0).to_numpy(dtype=float)
                for state, mass in zip(states, sums, strict=True):
                    node_rows.append({"donor_id": donor, "time": time, "phenotype": state,
                                      "cell_mass": float(mass), "denominator": norm,
                                      "sampled": sampled[time],
                                      "value": float(mass / scale) if scale > 0 and sampled[time] else np.nan})
                for clone, vector in vectors[time].items():
                    total = float(vector.sum())
                    for state, mass in zip(states, vector, strict=True):
                        if mass > 0:
                            compositions.append({"donor_id": donor, "time": time, "clone_id": clone,
                                                 "phenotype": state, "cell_mass": float(mass),
                                                 "clone_cell_mass": total, "denominator": norm,
                                                 "within_clone_fraction": float(mass / total),
                                                 "sample_fraction": float(mass / norm) if norm else np.nan})
            for origin, destination in pairwise(times):
                if not sampled[origin] or not sampled[destination]:
                    continue  # Missing sampling is not clonal disappearance.
                start, end = vectors[origin], vectors[destination]
                for clone in sorted(set(start) | set(end)):
                    a = start.get(clone, np.zeros(len(states)))
                    b = end.get(clone, np.zeros(len(states)))
                    na, nb = float(a.sum()), float(b.sum())
                    event = "shared" if na > 0 and nb > 0 else ("appeared" if nb > 0 else "disappeared")
                    events.append({"donor_id": donor, "from_time": origin, "to_time": destination,
                                   "clone_id": clone, "event": event, "from_cells": na, "to_cells": nb})
                    if na == 0 or nb == 0:
                        continue
                    probabilities = np.outer(a / na, b / nb)
                    out = probabilities * (na / norms[origin] if normalize else na)
                    inflow = probabilities * (nb / norms[destination] if normalize else nb)
                    for i, j in zip(*np.nonzero(probabilities), strict=True):
                        flow_rows.append([donor, origin, destination, clone, states[i], states[j],
                                          float(out[i, j]), float(inflow[i, j])])
                    if na > min_cell_count and nb > min_cell_count:
                        flux_rows.append([donor, origin, destination, clone, na, nb,
                                          float(np.abs(a / na - b / nb).sum())])
    composition_columns = ["donor_id", "time", "clone_id", "phenotype", "cell_mass",
                           "clone_cell_mass", "denominator", "within_clone_fraction", "sample_fraction"]
    parameters = {
        "reference": REFERENCE, "implementation": "cellscope.phenotrack",
        "reference_source_sha256": dict(REFERENCE_SHA256),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "clone_key": clone_key, "phenotype_key": phenotype_key,
        "clone_definition": "provided_obs_identity" if clone_definition is None else str(clone_definition),
        "probabilities_key": probabilities_key or "", "donor_key": donor_key,
        "time_key": time_key, "sample_key": sample_key, "time_order": times,
        "phenotype_order": states, "donors": donors, "denominator": denominator,
        "normalize": bool(normalize), "flux_min_cell_count_exclusive": float(min_cell_count),
        "clone_selection": "all" if selected_by_donor is None else {
            d: sorted(v) for d, v in selected_by_donor.items()},
        "denominator_before_clone_selection": True, "flow_model": "independent_state_distributions",
        "lineage_observed": False, "phenotyping": "user_supplied_labels_or_probabilities",
        "aggregation": "none", "backend": "numpy_pandas_cpu",
    }
    result = PhenoTrackResult(pd.DataFrame(compositions, columns=composition_columns),
                             pd.DataFrame(node_rows), pd.DataFrame(flow_rows, columns=_FLOW),
                             pd.DataFrame(flux_rows, columns=_FLUX), pd.DataFrame(events, columns=_EVENT),
                             pd.DataFrame(denominator_rows), parameters)
    adata.uns[key_added] = result.as_dict()
    feedback.info(f"PhenoTrack | {len(donors)} donors | {len(times)} times | {len(states)} phenotypes | "
                  f"{len(result.flows):,} clone-state flows")
    return result


def phenotrack_aggregate(result, *, missing="error"):
    """Equal-donor means, never pooled-cell or pooled-clone weighting.

    error requires a complete cohort at all times (author complete-cohort
    scenario). available is an explicit changing-cohort alternative: each
    time's nodes and each flow endpoint use that time's available donor count.
    A missing partner contributes no flow, not a biological zero measurement.
    """
    result = resolve_phenotrack(result)
    if result.parameters.get("aggregation") != "none":
        raise ValueError("This result is already aggregated")
    if missing not in {"error", "available"}:
        raise ValueError("missing must be error or available")
    available = result.denominators.assign(
        available=lambda t: t.sampled & t.denominator.gt(0))
    if missing == "error" and not available.available.all():
        raise ValueError("Equal-cohort aggregation requires every donor/time; use missing='available' explicitly")
    sizes = available.groupby("time", sort=False).available.sum().to_dict()
    nodes = result.nodes.copy()
    nodes = nodes[nodes.sampled & nodes.denominator.gt(0)]
    nodes = nodes.groupby(["time", "phenotype"], observed=True, sort=False).agg(
        value=("value", "sum"), cell_mass=("cell_mass", "sum"), denominator=("denominator", "sum"),
    ).reset_index()
    nodes["value"] /= nodes.time.map(sizes)
    nodes["donor_id"], nodes["sampled"] = "mean", True
    unknown = [{"donor_id": "mean", "time": time, "phenotype": state,
                "value": np.nan, "cell_mass": 0.0, "denominator": 0, "sampled": False}
               for time, n in sizes.items() if n == 0
               for state in result.parameters["phenotype_order"]]
    if unknown:
        nodes = pd.concat([nodes, pd.DataFrame(unknown)], ignore_index=True)
    flows = result.flows.copy()
    flows["outflow"] /= flows.from_time.map(sizes)
    flows["inflow"] /= flows.to_time.map(sizes)
    flows = flows.groupby(["from_time", "to_time", "from_phenotype", "to_phenotype"],
                          observed=True, sort=False)[["outflow", "inflow"]].sum().reset_index()
    flows["donor_id"], flows["clone_id"] = "mean", "all"
    params = {**result.parameters, "aggregation": "equal_donor_mean", "missing_donors": missing,
              "n_donors_by_time": {str(t): int(n) for t, n in sizes.items()}}
    return PhenoTrackResult(result.composition.copy(), nodes, flows, result.flux.copy(),
                            result.events.copy(), result.denominators.copy(), params)


def resolve_phenotrack(data, *, key="phenotrack", mod="gex"):
    if isinstance(data, PhenoTrackResult):
        return data
    mapping = data if isinstance(data, Mapping) else modality(data, mod).uns[key]
    result = PhenoTrackResult(**{key: mapping[key] for key in PhenoTrackResult.__dataclass_fields__})
    result.parameters = _plain(result.parameters)
    return result


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
