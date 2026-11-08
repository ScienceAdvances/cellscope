"""Three-state clone composition and trajectories with PhenoTrack projection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..tools._phenotrack import resolve_phenotrack
from ._output import finish_plot


def phenotrack_ternary(data, *, from_time, to_time=None, donor=None, phenotypes=None,
                      clones=None, conditional=True, include_new=False,
                      clone_colors=None, color="#286D91", point_size=25, alpha=0.7,
                      line_type="arrows", grid=True, key="phenotrack", mod="gex",
                      figsize=(6, 5.5), title=None, ax=None,
                      filename=None, outdir=".", formats=("pdf", "png"), dpi=300, show=False):
    """Plot one donor's clone distributions in three states, optionally as arrows.

    Vertex order follows the author: first=top, second=left, third=right.
    conditional=True renormalizes within the three selected states, matching
    the author's projection; False rejects omitted phenotype mass. If more
    than three states exist, select three explicitly. By default only clones
    observed at the start are considered; include_new=True additionally plots
    destination-only clones. Coordinates are available in ax.cellscope_table.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    result = resolve_phenotrack(data, key=key, mod=mod)
    if result.parameters.get("aggregation") != "none":
        raise ValueError("Clone ternary trajectories require unaggregated donor-specific data")
    donors = list(result.nodes.donor_id.unique())
    if donor is None:
        if len(donors) != 1:
            raise ValueError("Select one donor for clone ternary trajectories")
        donor = donors[0]
    donor = str(donor)
    if donor not in donors:
        raise ValueError("Unknown donor")
    states = result.parameters["phenotype_order"] if phenotypes is None else list(map(str, phenotypes))
    if len(states) != 3 or len(set(states)) != 3 or not set(states).issubset(result.parameters["phenotype_order"]):
        raise ValueError("Select exactly three known, unique phenotypes")
    if line_type not in {"arrows", "lines"}:
        raise ValueError("line_type must be arrows or lines")
    if not 0 <= alpha <= 1 or point_size <= 0:
        raise ValueError("alpha must be in [0, 1] and point_size positive")
    origin, destination = str(from_time), None if to_time is None else str(to_time)
    ordered = result.parameters["time_order"]
    if origin not in ordered or (destination is not None and destination not in ordered):
        raise ValueError("Unknown from_time/to_time")
    if destination is not None and ordered.index(destination) <= ordered.index(origin):
        raise ValueError("to_time must follow from_time")
    for time in (origin,) if destination is None else (origin, destination):
        part = result.nodes[result.nodes.donor_id.eq(donor) & result.nodes.time.eq(time)]
        if not part.sampled.any():
            raise ValueError(f"Donor {donor} was not sampled at {time}")
    composition = result.composition[result.composition.donor_id.eq(donor)]
    maps = {}
    for time in (origin,) if destination is None else (origin, destination):
        table = composition[composition.time.eq(time)].pivot_table(
            index="clone_id", columns="phenotype", values="cell_mass", aggfunc="sum", fill_value=0)
        selected = table.reindex(columns=states, fill_value=0)
        if not conditional and not np.allclose(selected.sum(axis=1), table.sum(axis=1)):
            raise ValueError("Other phenotype mass exists; use conditional=True explicitly or choose a full three-state model")
        maps[time] = {}
        for clone, row in selected.iterrows():
            mass = float(row.sum())
            if mass > 0:
                p = row.to_numpy(dtype=float) / mass
                maps[time][clone] = (p, np.array([1 - p[1] - p[0] / 2, np.sqrt(3) * p[0] / 2]))
    ids = set(maps[origin])
    if destination is not None and include_new:
        ids |= set(maps[destination])
    if clones is not None:
        if isinstance(clones, str):
            raise TypeError("clones must be a sequence, not a string")
        ids &= set(map(str, clones))
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    vertices = np.array([[.5, np.sqrt(3) / 2], [0, 0], [1, 0]])
    if grid:
        for p in (.2, .4, .6, .8):
            for i in range(3):
                others = [j for j in range(3) if j != i]
                endpoints = [p * vertices[i] + (1 - p) * vertices[j] for j in others]
                ax.plot(*np.asarray(endpoints).T, color="0.8", linestyle=":", linewidth=.6, zorder=0)
    ax.plot([0, 1, .5, 0], [0, 0, np.sqrt(3) / 2, 0], color="0.25", linewidth=1.2)
    ax.text(.5, np.sqrt(3) / 2 + .04, states[0], ha="center")
    ax.text(-.03, -.04, states[1], ha="right")
    ax.text(1.03, -.04, states[2], ha="left")
    records = []
    for clone in sorted(ids):
        shade = (clone_colors or {}).get(clone, color)
        points = {}
        for time, marker in [(origin, "o")] + ([] if destination is None else [(destination, "^")]):
            if clone in maps[time]:
                p, xy = maps[time][clone]
                points[time] = xy
                ax.scatter(*xy, marker=marker, color=shade, s=point_size, alpha=alpha, zorder=3)
                records.append({"donor_id": donor, "clone_id": clone, "time": time,
                                "x": float(xy[0]), "y": float(xy[1]),
                                **{f"p_{i}": float(v) for i, v in enumerate(p)}})
        if destination is not None and origin in points and destination in points:
            a, b = points[origin], points[destination]
            if not np.allclose(a, b):
                if line_type == "arrows":
                    ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": "->", "color": shade, "alpha": alpha, "lw": .8})
                else:
                    ax.plot([a[0], b[0]], [a[1], b[1]], color=shade, alpha=alpha, lw=.8)
    ax.set_aspect("equal")
    ax.set_xlim(-.15, 1.15)
    ax.set_ylim(-.12, np.sqrt(3) / 2 + .13)
    ax.set_axis_off()
    ax.set_title(title or f"{donor}: {origin}" + ("" if destination is None else f" → {destination}"), loc="left", fontweight="bold")
    handles = [Line2D([], [], color=color, marker="o", linestyle="", label=origin)]
    if destination is not None:
        handles.append(Line2D([], [], color=color, marker="^", linestyle="", label=destination))
    ax.legend(handles=handles, frameon=False, loc="upper right")
    ax.cellscope_table = pd.DataFrame(records, columns=["donor_id", "clone_id", "time", "x", "y", "p_0", "p_1", "p_2"])
    ax.cellscope_parameters = {"phenotypes": list(states), "conditional": conditional, "include_new": include_new}
    return finish_plot(ax, filename=filename, outdir=outdir, formats=formats, dpi=dpi, show=show)
