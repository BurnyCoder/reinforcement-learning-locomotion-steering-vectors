"""Local: write audit.json, input identities and figures. Global: publish the audit's evidence in reviewable form.

Sources: Matplotlib Agg backend and pyplot API https://matplotlib.org/stable/users/index.html ;
categorical slots 1–3 of the project's reference data-viz palette (blue, orange, aqua) with gray context marks.
"""
from __future__ import annotations  # Keep annotations lightweight.

import hashlib  # Combined digest over episode files.
import json  # Compact identity listing.

import matplotlib  # Headless rendering.

matplotlib.use("Agg")  # File output only; no display needed.
import matplotlib.pyplot as plt  # noqa: E402  Figure construction.
import numpy as np  # noqa: E402  Array handling.

from rl_locomotion_steering_vectors.storage import save_json, utc_now  # noqa: E402  Atomic JSON writes and timestamps.

from .common import REPORT_DIR, SOURCE_RUN, sha256_file  # noqa: E402  Protocol paths.

INK, MUTED, GRID = "#0b0b0b", "#8a8984", "#e4e3df"  # Text, context marks and recessive gridlines.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # Primary, other label, mechanism.


def _style(axis, xlabel: str, ylabel: str) -> None:
    """Local: recessive axes. Global: let the data marks carry the comparison."""
    axis.set_xlabel(xlabel, color=INK)  # Units in labels.
    axis.set_ylabel(ylabel, color=INK)  # Units in labels.
    axis.grid(color=GRID, linewidth=0.6)  # Light grid.
    axis.set_axisbelow(True)  # Grid behind marks.
    for side in ("top", "right"):  # Remove chart junk.
        axis.spines[side].set_visible(False)  # Hide spine.


def _rank_figure(fresh: dict) -> str:
    """Local: |odd| null histogram with labelled directions. Global: show P2, S1 and M1 on one scale."""
    table = fresh["directions"]  # Per-direction parts.
    randoms = [abs(v["odd"]) for k, v in table.items() if k.startswith("random")]  # Null sample.
    shuffled = [abs(v["odd"]) for k, v in table.items() if k.startswith("shuffled")]  # Shuffle redraws.
    figure, axis = plt.subplots(figsize=(7.0, 3.6), dpi=160)  # One panel.
    axis.hist(randoms, bins=20, color=MUTED, edgecolor="white", linewidth=1.0, label=f"random directions (n = {len(randoms)})")  # Null distribution.
    axis.plot(shuffled, np.full(len(shuffled), -0.6), linestyle="none", marker="|", markersize=12, color=INK, label=f"shuffle redraws (n = {len(shuffled)})")  # Rug.
    for name, color, text in (("lateral", BLUE, "lateral (primary)"), ("turning", ORANGE, "turning"), ("heading_axis_renorm", AQUA, "perceived-heading axis"),
                              ("heading_orthogonal_renorm", AQUA, "lateral minus heading part")):  # Labelled directions.
        value = abs(table[name]["odd"])  # |odd| of this direction.
        axis.axvline(value, color=color, linewidth=2, linestyle="-" if "orthogonal" not in name else "--")  # Marker line.
        axis.annotate(text, (value, axis.get_ylim()[1] * 0.92), rotation=90, va="top", ha="right", fontsize=7, color=INK)  # Direct label.
    _style(axis, "|odd world-Y response| at |α| = 0.1, all directions at norm 5.15 (m/s)", "directions")  # Axes.
    axis.set_title("Sign-dependent world-Y response on fresh seeds 540000–540019 (n = 20 pairs)", fontsize=9, color=INK)  # Title states scope.
    axis.legend(frameon=False, fontsize=7, loc="lower right", bbox_to_anchor=(1.0, 0.15))  # Empty lower-right region, clear of the labelled lines.
    figure.tight_layout()  # Fit labels.
    path = REPORT_DIR / "fresh-odd-rank.png"  # Output path.
    figure.savefig(path)  # Write PNG.
    plt.close(figure)  # Release memory.
    return path.name  # Relative name.


def _scatter_figure(fresh: dict) -> str:
    """Local: odd versus even parts. Global: separate steering from sign-independent drift."""
    table = fresh["directions"]  # Per-direction parts.
    figure, axis = plt.subplots(figsize=(5.6, 4.2), dpi=160)  # One panel.
    groups = (("random", "o", MUTED, "random"), ("shuffled", "^", INK, "label-shuffle redraw"))  # Context groups.
    for prefix, marker, color, label in groups:  # Gray context, shape-coded.
        points = np.array([[v["odd"], v["even"]] for k, v in table.items() if k.startswith(prefix)])  # (odd, even).
        axis.scatter(points[:, 0], points[:, 1], s=14, marker=marker, facecolors="none" if prefix == "shuffled" else color, edgecolors=color, linewidths=0.8, label=label)  # Marks.
    for name, color, text, offset, align in (("lateral", BLUE, "lateral", (6, 4), "left"), ("turning", ORANGE, "turning", (6, 4), "left"),
                                             ("heading_parallel", AQUA, "heading part", (-6, 4), "right"), ("heading_axis_renorm", AQUA, "heading axis", (6, -10), "left"),
                                             ("heading_orthogonal_renorm", AQUA, "remainder (renorm.)", (-6, 6), "right")):  # Labelled directions with non-overlapping offsets.
        axis.scatter(table[name]["odd"], table[name]["even"], s=60, color=color, edgecolors="white", linewidths=1.5, zorder=3)  # Highlight.
        axis.annotate(text, (table[name]["odd"], table[name]["even"]), xytext=offset, textcoords="offset points", ha=align, fontsize=7, color=INK)  # Direct label.
    axis.axhline(0, color=INK, linewidth=0.8)  # Zero even part.
    axis.axvline(0, color=INK, linewidth=0.8)  # Zero odd part.
    _style(axis, "odd part (f(+0.1) − f(−0.1))/2 (m/s)", "even part (f(+0.1) + f(−0.1))/2 (m/s)")  # Axes.
    axis.set_title("World-Y response decomposition, fresh seeds (n = 20)", fontsize=9, color=INK)  # Title.
    axis.legend(frameon=False, fontsize=7, loc="lower left")  # Legend.
    figure.tight_layout()  # Fit labels.
    path = REPORT_DIR / "fresh-odd-even.png"  # Output path.
    figure.savefig(path)  # Write PNG.
    plt.close(figure)  # Release memory.
    return path.name  # Relative name.


def _timecourse_figure(saved: dict, fresh: dict) -> str:
    """Local: per-window torso-yaw change with 95% CIs. Global: bounded heading offset versus continuing turn."""
    series = (("confirmation, lateral −0.1", saved["kinematics_confirmation"], BLUE, "-"), ("fresh, lateral −0.1", fresh["K1"]["primary_minus"], BLUE, "--"),
              ("fresh, perceived rotation −δ", fresh["K1"]["perception_minus_delta"], AQUA, "-"))  # Series.
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), dpi=160)  # Two separate scales, two panels.
    for label, data, color, style in series:  # Each arm.
        for axis, key in zip(axes, ("delta_yaw_deg", "delta_y_velocity")):  # Same windows, two quantities.
            starts = [w["start"] for w in data["windows"]]  # Window starts.
            means = [w[key]["mean"] for w in data["windows"]]  # Means.
            low = [w[key]["percentile_ci"][0] for w in data["windows"]]  # Lower bounds.
            high = [w[key]["percentile_ci"][1] for w in data["windows"]]  # Upper bounds.
            axis.fill_between(starts, low, high, color=color, alpha=0.12, linewidth=0)  # Interval band.
            axis.plot(starts, means, color=color, linestyle=style, linewidth=2, label=label)  # Mean line.
    for axis, ylabel in zip(axes, ("Δ mean torso yaw (degrees)", "Δ mean world-Y velocity (m/s)")):  # Labels.
        axis.axhline(0, color=INK, linewidth=0.8)  # No-effect reference.
        _style(axis, "window start step (100-step windows)", ylabel)  # Axes.
    axes[0].legend(frameon=False, fontsize=7)  # Legend once.
    figure.suptitle("Treated − baseline per window, pairs with ≥2 common windows; bands are 95% percentile CIs", fontsize=9, color=INK)  # Scope.
    figure.tight_layout()  # Fit labels.
    path = REPORT_DIR / "timecourse.png"  # Output path.
    figure.savefig(path)  # Write PNG.
    plt.close(figure)  # Release memory.
    return path.name  # Relative name.


def input_identities() -> dict:
    """Local: hash every audit input. Global: let readers confirm the audit used the published artifacts."""
    files = ("manifest.json", "results.json", "selection.json", "vectors.npz", "action_biases.npz", "vector_diagnostics.json")  # Small artifacts.
    episodes = sorted((SOURCE_RUN / "episodes").rglob("*.npz"))  # Every raw episode.
    listing = [(p.relative_to(SOURCE_RUN).as_posix(), sha256_file(p)) for p in episodes]  # (path, digest).
    combined = hashlib.sha256(json.dumps(listing).encode("utf-8")).hexdigest()  # One digest over all episodes.
    return {"files": {name: sha256_file(SOURCE_RUN / name) for name in files}, "episode_files": len(listing), "episodes_combined_sha256": combined}  # Evidence.


def write_outputs(results: dict) -> dict:
    """Local: save JSON and figures. Global: the committed audit bundle."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)  # Output folder.
    figures = []  # Written figure names.
    if "fresh" in results:  # Fresh-seed figures exist only after collection.
        figures = [_rank_figure(results["fresh"]), _scatter_figure(results["fresh"]), _timecourse_figure(results["saved"], results["fresh"])]  # Figures.
    save_json(REPORT_DIR / "input-identities.json", input_identities())  # Input hashes.
    save_json(REPORT_DIR / "audit.json", {"created_utc": utc_now(), "figures": figures, **results})  # All results.
    return {"figures": figures}  # Bookkeeping.
