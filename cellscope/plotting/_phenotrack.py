"""Unbalanced clone phenotype flows with the author's Hill interpolation."""

from __future__ import annotations

from itertools import pairwise

import numpy as np

from ..tools._phenotrack import phenotrack_aggregate, resolve_phenotrack
from ._output import finish_plot


def phenotrack(data, *, key="phenotrack", mod="gex", donor=None, aggregate=False,
              missing="error", phenotype_colors=None, time_positions=None,
              interpolation="hill", state_gap=0, bar_width=0.06, alpha=0.45,
              figsize=(9, 5), ncols=2, title=None, ax=None,
              filename=None, outdir=".", formats=("pdf", "png"), dpi=300, show=False):
    """Draw static nodes and independently sized origin/destination ribbons.

    Returns Axes (one panel) or a list of Axes (several donors). Missing
    sampling is labelled, not rendered as a measured zero. Source phenotype
    determines ribbon color. Hill interpolation follows the author plot;
    linear is an explicit visual alternative. No file is written without
    filename; exporting uses cellscope's shared plotting interface.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.ticker import PercentFormatter

    result = resolve_phenotrack(data, key=key, mod=mod)
    if aggregate:
        if donor is not None:
            raise ValueError("Choose donor or aggregate, not both")
        result = phenotrack_aggregate(result, missing=missing)
    is_mean = result.parameters.get("aggregation") == "equal_donor_mean"
    if interpolation not in {"hill", "linear"}:
        raise ValueError("interpolation must be hill or linear")
    if not np.isfinite(state_gap) or state_gap < 0:
        raise ValueError("state_gap must be finite and nonnegative")
    if not 0 < bar_width < 0.5 or not 0 <= alpha <= 1:
        raise ValueError("bar_width must be in (0, .5) and alpha in [0, 1]")
    states = list(result.parameters["phenotype_order"])
    times = list(result.parameters["time_order"])
    available = list(result.nodes.donor_id.unique())
    donors = available if donor is None else [str(donor)]
    if not set(donors).issubset(available):
        raise ValueError("Unknown donor")
    if time_positions is None:
        positions = np.arange(len(times), dtype=float)
    else:
        positions = np.asarray(time_positions, dtype=float)
    if positions.shape != (len(times),) or not np.isfinite(positions).all() or (np.diff(positions) <= 0).any():
        raise ValueError("time_positions must be finite, strictly increasing and match time_order")
    if phenotype_colors is None:
        cmap = plt.get_cmap("tab20" if len(states) <= 20 else "turbo")
        colors = {state: cmap(i if len(states) <= 20 else i / max(1, len(states) - 1))
                  for i, state in enumerate(states)}
    else:
        colors = {str(k): v for k, v in phenotype_colors.items()}
        if not set(states).issubset(colors):
            raise ValueError("phenotype_colors must cover every phenotype")
    if ax is not None:
        if len(donors) != 1:
            raise ValueError("A supplied ax requires one donor or aggregate=True")
        axes = [ax]
    else:
        if not isinstance(ncols, int) or isinstance(ncols, bool) or ncols < 1:
            raise ValueError("ncols must be a positive integer")
        columns = min(ncols, len(donors))
        rows = int(np.ceil(len(donors) / columns))
        _, grid = plt.subplots(rows, columns, figsize=(figsize[0] * columns, figsize[1] * rows), squeeze=False)
        axes = list(grid.flat[:len(donors)])
        for unused in grid.flat[len(donors):]:
            unused.set_visible(False)
    width = bar_width * (np.diff(positions).min() if len(times) > 1 else 1)
    u = np.linspace(0, 1, 1000)
    curve = 1 / (1 + 10 ** (2 - 4 * u)) if interpolation == "hill" else u
    curve = (curve - curve[0]) / (curve[-1] - curve[0])
    for current, identifier in zip(axes, donors, strict=True):
        nodes = result.nodes[result.nodes.donor_id.eq(identifier)]
        bottoms, heights, sampled = {}, {}, {}
        maximum = 0.0
        for time, x in zip(times, positions, strict=True):
            part = nodes[nodes.time.eq(time)].set_index("phenotype")
            sampled[time] = bool(part.sampled.any())
            bottom = 0.0
            for state in states:
                value = float(part.loc[state, "value"]) if state in part.index else 0.0
                value = 0.0 if np.isnan(value) else value
                if value < 0:
                    raise ValueError("Node mass must be nonnegative")
                bottoms[time, state], heights[time, state] = bottom, value
                if sampled[time] and value > 0:
                    current.add_patch(Rectangle((x - width / 2, bottom), width, value,
                                                facecolor=colors[state], edgecolor="white", linewidth=0.4, zorder=3))
                bottom += value + state_gap
            maximum = max(maximum, max(0, bottom - state_gap))
            if not sampled[time]:
                current.text(x, 0.02, "not sampled", transform=current.get_xaxis_transform(),
                             ha="center", fontsize=9, color="0.45", rotation=90)
        for index, (origin, destination) in enumerate(pairwise(times)):
            if not sampled[origin] or not sampled[destination]:
                continue
            flows = result.flows[
                result.flows.donor_id.eq(identifier) & result.flows.from_time.eq(origin)
                & result.flows.to_time.eq(destination)]
            pair = flows.groupby(["from_phenotype", "to_phenotype"], observed=True)[["outflow", "inflow"]].sum()
            out_totals = pair.groupby(level=0).outflow.sum()
            in_totals = pair.groupby(level=1).inflow.sum()
            source_cursor = {s: bottoms[origin, s] for s in states}
            dest_cursor = {}
            for state in states:
                residual = heights[destination, state] - float(in_totals.get(state, 0))
                if residual < -1e-8 or float(out_totals.get(state, 0)) > heights[origin, state] + 1e-8:
                    raise ValueError("Flow exceeds its static node; check denominator and aggregation")
                # Author placement: destination shared flow sits above unmatched mass.
                dest_cursor[state] = bottoms[destination, state] + max(0, residual)
            x = positions[index] + width / 2 + (
                positions[index + 1] - positions[index] - width) * u
            for source in states:
                for target in states:
                    if (source, target) not in pair.index:
                        continue
                    out, inflow = pair.loc[(source, target), ["outflow", "inflow"]]
                    low = source_cursor[source] + (dest_cursor[target] - source_cursor[source]) * curve
                    high = source_cursor[source] + out + (
                        dest_cursor[target] + inflow - source_cursor[source] - out) * curve
                    current.fill_between(x, low, high, color=colors[source], alpha=alpha, linewidth=0)
                    source_cursor[source] += out
                    dest_cursor[target] += inflow
        current.set_xticks(positions, times)
        current.set_xlim(positions[0] - width * 3, positions[-1] + width * 3)
        current.set_ylim(0, max(maximum * 1.06, 0.01))
        if result.parameters["normalize"]:
            current.yaxis.set_major_formatter(PercentFormatter(1))
        if result.parameters["normalize"]:
            current.set_ylabel("Mean cell fraction" if is_mean else "Cell fraction")
        else:
            current.set_ylabel("Mean cell mass" if is_mean else "Cell mass")
        current.set_xlabel(result.parameters["time_key"])
        current.set_title(title or ("Equal-donor mean" if is_mean else str(identifier)), loc="left", fontweight="bold")
        current.spines[["top", "right"]].set_visible(False)
        current.legend(handles=[Patch(facecolor=colors[s], label=s) for s in states],
                       frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
        current.cellscope_parameters = {
            "donor": str(identifier), "aggregation": result.parameters["aggregation"],
            "interpolation": interpolation, "state_gap": float(state_gap),
            "time_positions": positions.tolist(), "bar_width": float(bar_width),
            "normalize": result.parameters["normalize"],
        }
    if ax is None:
        axes[0].figure.tight_layout()
    return finish_plot(axes[0] if len(axes) == 1 else axes, filename=filename,
                       outdir=outdir, formats=formats, dpi=dpi, show=show)
