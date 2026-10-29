"""Measured Ant figures from the publication schema, never a simulator rerun.

Uses Matplotlib's documented plot, error-bar and layout APIs:
https://matplotlib.org/stable/users/explain/axes/arranging_axes.html
https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.errorbar.html
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, GREEN, GRAY = "#2864b7", "#b95519", "#16816a", "#6e7480"


def _color(row: dict) -> str:
    """Keep the same arm colors across trajectory, reward and comparison figures."""
    return {"baseline": GRAY, "primary": BLUE, "other_label": ORANGE, "random": "#8058aa", "shuffled": "#a25578", "action_bias": "#8e7220", "perception": "#952d2d", "ablation": GREEN}[row["kind"]]


def _cohort(evidence: dict, name: str) -> dict:
    """Select an existing cohort, failing if its evidence is absent."""
    return next(item for item in evidence["cohorts"] if item["id"] == name)


def _save(figure, path: Path) -> Path:
    """Deterministic PNG metadata and a common resolution support exact regeneration checks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, facecolor="white", metadata={"Software": "Ant saved-evidence publication"})
    plt.close(figure)
    return path


def _comparison(rows: list[dict], title: str, path: Path) -> Path:
    """Show every chosen arm with its historical or explicitly descriptive paired interval."""
    figure, axis = plt.subplots(figsize=(8.6, max(3.4, .42 * len(rows) + 1.5)), layout="constrained")
    for index, row in enumerate(rows):
        value, interval = row["pairedEffect"]["mean"], row["pairedEffect"]["ci95"]
        color = _color(row)
        axis.plot(value, index, "o", color=color, markersize=5)
        if interval is not None:
            axis.hlines(index, interval[0], interval[1], color=color, linewidth=1.5)
    labels = [f"{row['label']}  ·  failures {row['metrics']['failures']}/{row['n_pairs']}" for row in rows]
    axis.set(yticks=range(len(rows)), yticklabels=labels, xlabel="Paired change in world-Y velocity (m/s)", title=title)
    axis.invert_yaxis()
    axis.axvline(0, color=GRAY, linewidth=.8, linestyle="--")
    axis.grid(axis="x", alpha=.18)
    axis.tick_params(axis="y", labelsize=8)
    figure.supxlabel("95% intervals; exact methods and all conditions in evidence.json. Failed episodes remain included.", fontsize=8)
    return _save(figure, path)


def _strength(evidence: dict, path: Path) -> Path:
    """Keep label families apart while plotting the same measured world coordinate."""
    rows = _cohort(evidence, "validation")["conditions"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained", sharey=True)
    for axis, family in zip(axes, ("lateral", "turning")):
        vectors = sorted({row["vector"] for row in rows if row["vector"].startswith(family)})
        for vector in vectors:
            points = sorted((row for row in rows if row["vector"] == vector), key=lambda row: row["alpha"])
            line, = axis.plot([row["alpha"] for row in points], [row["pairedEffect"]["mean"] for row in points], ".-", linewidth=1.2, label=vector)
            for row in points:
                interval = row["pairedEffect"]["ci95"]
                axis.vlines(row["alpha"], interval[0], interval[1], color=line.get_color(), alpha=.35)
        axis.axhline(0, color=GRAY, linewidth=.8, linestyle="--")
        axis.set(xlabel="Activation strength (bias uses its own scale)", title=f"{family.capitalize()} extraction family")
        axis.legend(fontsize=7, frameon=False)
        axis.grid(alpha=.15)
    axes[0].set_ylabel("Paired world-Y velocity change (m/s)")
    figure.suptitle("Original validation: strength selection data, not held-out confirmation")
    return _save(figure, path)


def _specificity(evidence: dict, path: Path) -> Path:
    """Show all random/shuffle controls and mechanism directions on one common scale."""
    table = evidence["audit"]["fresh"]["directions"]
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4), layout="constrained")
    randoms = [item for name, item in table.items() if name.startswith("random")]
    shuffled = [item for name, item in table.items() if name.startswith("shuffled")]
    axes[0].hist([abs(item["odd"]) for item in randoms], bins=16, color=GRAY, edgecolor="white", label=f"Random directions (n={len(randoms)})")
    axes[0].plot([abs(item["odd"]) for item in shuffled], [-.5] * len(shuffled), "|", color=ORANGE, label=f"Shuffled labels (n={len(shuffled)})")
    for name, label, color in (("lateral", "Primary", BLUE), ("turning", "Other label", ORANGE), ("heading_axis_renorm", "Heading axis", GREEN)):
        axes[0].axvline(abs(table[name]["odd"]), color=color, label=label)
        axes[1].scatter(table[name]["odd"], table[name]["even"], color=color, label=label, s=35)
    axes[1].scatter([item["odd"] for item in randoms], [item["even"] for item in randoms], color=GRAY, s=10, alpha=.6, label="Random")
    axes[1].scatter([item["odd"] for item in shuffled], [item["even"] for item in shuffled], edgecolors=ORANGE, facecolors="none", marker="^", s=18, label="Shuffled")
    axes[0].set(xlabel="Absolute odd world-Y response (m/s)", ylabel="Direction count")
    axes[1].set(xlabel="Odd world-Y response (m/s)", ylabel="Even world-Y response (m/s)")
    axes[1].axhline(0, color=GRAY, linewidth=.7)
    axes[1].axvline(0, color=GRAY, linewidth=.7)
    for axis in axes:
        axis.legend(fontsize=7, frameon=False)
        axis.grid(alpha=.12)
    figure.suptitle("Fresh audit: sign-dependent response and sign-independent drift")
    return _save(figure, path)


def _reward(evidence: dict, path: Path) -> Path:
    """Plot recorded rewards and the actual available-episode count, with no post-failure fill."""
    figure, axes = plt.subplots(2, 2, figsize=(10, 5.2), layout="constrained", height_ratios=[3, 1], sharex="col")
    for column, cohort_id in enumerate(("confirmation", "fresh_audit")):
        rows = _cohort(evidence, cohort_id)["conditions"]
        chosen = [row for row in rows if row["id"] in ("baseline", "lateral__-0.1000", "heading_parallel__-0.1000", "lateral_shuffled__-0.2000") or (row["kind"] == "perception" and row["alpha"] < 0)]
        for row in chosen:
            windows = row["reward_windows"]
            seconds = [point["start"] * evidence["metadata"]["dt"] for point in windows]
            line, = axes[0, column].plot(seconds, [point["mean"] for point in windows], ".-", linewidth=1.4, color=_color(row), label=row["label"])
            axes[1, column].step(seconds, [point["n"] for point in windows], where="post", color=line.get_color())
        axes[0, column].set(title=_cohort(evidence, cohort_id)["label"], ylabel="Original reward / recorded transition")
        axes[0, column].legend(fontsize=7, frameon=False)
        axes[1, column].set(xlabel="Window start (simulation seconds)", ylabel="Episodes")
        for axis in axes[:, column]:
            axis.axvline(evidence["metadata"]["warmup"] * evidence["metadata"]["dt"], color=GRAY, linestyle="--", linewidth=.8)
            axis.grid(alpha=.15)
    figure.suptitle("Evaluation reward, not training curves · equal available-episode means in 100-step windows")
    return _save(figure, path)


def _trajectories(evidence: dict, root: Path, path: Path) -> Path:
    """The first predetermined reset illustrates arms without selecting for visual success."""
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    for axis, cohort_id in zip(axes, ("validation", "fresh_audit")):
        cohort = _cohort(evidence, cohort_id)
        chosen = [row for row in cohort["conditions"] if row["id"] in ("baseline", "lateral__-0.1000", "turning__-0.1000", "lateral_random1__-0.1000", "heading_parallel__-0.1000", "heading_orthogonal__-0.1000") or (row["kind"] == "perception" and row["alpha"] < 0)]
        for row in chosen:
            shard = json.loads((root / "site" / row["episodesUrl"]).read_text(encoding="utf-8"))
            arm = shard["episodes"][0]["treated"]
            style = "--" if "orthogonal" in row["vector"] else "-"
            axis.plot(arm["x"], arm["y"], label=row["label"], linewidth=1.3, color=_color(row), linestyle=style)
            axis.plot(arm["x"][-1], arm["y"][-1], "x" if arm["failed"] else "o", color=_color(row), markersize=4)
        axis.set(xlabel="World X displacement (m)", ylabel="World Y displacement (m)", title=f"{cohort_id}: predetermined seed {cohort['seeds'][0]}")
        axis.set_aspect("equal", adjustable="datalim")
        axis.legend(fontsize=6.5, frameon=False)
        axis.grid(alpha=.15)
    figure.supxlabel("Positions integrate saved velocities from reset; equal axis scale. Cross marks a failed episode endpoint.", fontsize=8)
    return _save(figure, path)


def _architecture(path: Path) -> Path:
    """Explain the actual frozen-policy data flow, not a hypothetical training architecture."""
    figure, axis = plt.subplots(figsize=(9.5, 3.2), layout="constrained")
    axis.set(xlim=(0, 10), ylim=(0, 3))
    axis.axis("off")
    labels = [(1.3, 2.1, "Saved fitting episodes\ncontrastive activation means"), (4.5, 2.1, "Frozen SAC actor\nfirst ReLU + alpha × vector"), (8, 2.1, "Saved paired evaluation\nworld-Y, reward, failures"), (4.5, .5, "Validation locks strength\nconfirmation tests the choice"), (8, .5, "Evidence + provenance\ntables, figures, explorer")]
    for x, y, text in labels:
        axis.text(x, y, text, ha="center", va="center", fontsize=9, bbox={"boxstyle": "round,pad=0.6", "fc": "#eef4fa", "ec": BLUE})
    for start, end in (((2.6, 2.1), (3.1, 2.1)), ((5.9, 2.1), (6.5, 2.1)), ((4.5, 1.55), (4.5, 1.0)), ((8, 1.55), (8, 1.0)), ((5.9, .5), (6.5, .5))):
        axis.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": GRAY})
    figure.suptitle("Ant activation steering: policy weights frozen; publication reads saved evidence only")
    return _save(figure, path)


def build_figures(evidence: dict, output_root: Path) -> list[Path]:
    """Build the complete Ant figure set from the shared data and episode shards."""
    folder = output_root / "reports/ant-publication/figures"
    confirmation = _cohort(evidence, "confirmation")["conditions"]
    fresh = _cohort(evidence, "fresh_audit")["conditions"]
    ablations = [row for row in fresh if row["kind"] in ("baseline", "primary", "other_label", "ablation", "perception") and row["alpha"] <= 0]
    paths = [_comparison(confirmation, "Original confirmation: locked conditions and their costs", folder / "confirmation.png"), _comparison(ablations, "Fresh audit: primary, other-label and mechanism interventions", folder / "ablations.png"), _strength(evidence, folder / "strength-response.png"), _specificity(evidence, folder / "specificity.png"), _reward(evidence, folder / "reward.png"), _trajectories(evidence, output_root, folder / "trajectories.png"), _architecture(folder / "architecture.png")]
    evidence["figures"] = [{"path": path.relative_to(output_root).as_posix(), "name": path.stem} for path in paths]
    return paths
