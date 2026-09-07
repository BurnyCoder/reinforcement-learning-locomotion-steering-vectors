"""Local: read explicit .env settings; global: serialize the complete protocol.

Sources: https://bbc2.github.io/python-dotenv/ ; https://docs.python.org/3.12/library/dataclasses.html
https://docs.python.org/3.12/library/math.html#math.isfinite
https://docs.python.org/3.12/library/string.html#format-specification-mini-language
"""
import os  # Only named nonsecret settings are read and logged.
from dataclasses import asdict, dataclass  # Frozen values cannot drift during evaluation.
from math import isfinite  # Reject NaN and infinities before they can enter interventions.
from pathlib import Path  # Configuration is loaded solely from the current project.

from dotenv import load_dotenv  # Reuse the established .env parser.


@dataclass(frozen=True)
class Config:
    """Local: pilot defaults; global: immutable fitting and evaluation specification."""
    torch_threads: int = 1  # Small MLP inference avoids thread startup overhead.
    rollout_workers: int = 4  # Isolated CPU processes amortize checkpoint loading across conditions.
    max_steps: int = 1000  # Match the upstream Gymnasium time limit.
    warmup: int = 100  # Steering begins after an identical unsteered prefix.
    diagnostic_episodes: int = 16  # Inspect natural variation before fitting.
    fit_episodes: int = 64  # Independent episode groups support quartile contrasts.
    validation_episodes: int = 10  # Select intervention strength outside fitting data.
    confirmation_episodes: int = 30  # Held-out pairs estimate the selected causal effect.
    replication_episodes: int = 30  # A second fresh rollout sample tests replication.
    seed_offset: int = 0  # New hypotheses can reserve entirely fresh episode partitions.
    fit_min_speed_fraction: float = .5  # A fitting-only median-relative floor excludes stationary recovery histories.
    strength_grid: str = "-0.5,-0.2,-0.1,-0.05,0.05,0.1,0.2,0.5"  # Zero remains the separate shared baseline; serialize the signed search before validation.

    def __post_init__(self) -> None:
        """Local: validate direct and restored configurations; global: fail before spending new episodes."""
        _ = self.strengths  # The same parser guards .env settings and saved manifest reconstruction.

    @property
    def strengths(self) -> tuple[float, ...]:
        """Local: parse an ordered signed grid; global: prevent nonfinite or ambiguous cached treatments."""
        try:
            values = tuple(float(item.strip()) for item in self.strength_grid.split(","))  # Standard float parsing supports ordinary decimal and scientific notation.
        except (AttributeError, ValueError) as error:
            raise ValueError("strength_grid must be a comma-separated string of numbers") from error  # Empty items and malformed numbers must not be silently skipped.
        if any(not isfinite(value) or value == 0 for value in values):  # The separate baseline already supplies zero steering.
            raise ValueError("strength_grid values must be finite and nonzero")  # Invalid additions cannot produce meaningful locomotion evidence.
        if len(set(values)) != len(values) or not (any(value < 0 for value in values) and any(value > 0 for value in values)):  # Both intervention signs receive explicit validation opportunities.
            raise ValueError("strength_grid must contain unique values and both signs")  # Duplicate floats would repeat the same comparison.
        labels = [f"{value:+.4f}" for value in values]  # Match experiment.condition_name's portable fixed-decimal directory convention.
        if len(set(labels)) != len(labels) or any(float(label) == 0 for label in labels):  # A fine grid must not collapse distinct strengths or label a nonzero addition as zero.
            raise ValueError("strength_grid values must have distinct nonzero four-decimal cache labels")  # Reject ambiguity before any condition folder is created.
        return values  # Preserve declared order rather than silently sorting the experiment.

    def as_dict(self) -> dict:
        """Local: expose serializable fields; global: record every configurable choice."""
        return asdict(self)  # Standard dataclass conversion avoids parallel field lists.


def read_config() -> Config:
    """Local: parse typed scalar and grid settings; global: keep experiments reproducible."""
    load_dotenv(Path.cwd() / ".env", override=False)  # Explicit process settings take precedence.
    defaults = Config().as_dict()  # One definition determines both defaults and supported keys.
    values = {key: type(default)(os.getenv(f"STEERING_{key.upper()}", default)) for key, default in defaults.items()}  # Preserve typed scalar settings without enumerating secrets.
    if any(value <= 0 for key, value in values.items() if key not in ("seed_offset", "fit_min_speed_fraction", "strength_grid")):  # Empty phases are not valid evidence; the grid has its own numeric parser.
        raise ValueError("All STEERING settings must be positive integers")  # Stop before artifacts are mixed.
    if values["seed_offset"] < 0 or not 0 <= values["fit_min_speed_fraction"] < 1:  # A floor must preserve a meaningful upper half of fitting motion.
        raise ValueError("seed_offset must be nonnegative and fit_min_speed_fraction must be in [0,1)")  # Explicit configuration errors are safer than silent filtering.
    if values["max_steps"] <= values["warmup"]:  # At least one steered transition is required.
        raise ValueError("max_steps must exceed warmup")  # Avoid meaningless zero-duration comparisons.
    return Config(**values)  # Freeze the resolved protocol for this invocation.
