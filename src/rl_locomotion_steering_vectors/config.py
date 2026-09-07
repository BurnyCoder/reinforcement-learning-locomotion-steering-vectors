"""Local: read explicit .env settings; global: serialize the complete protocol.

Sources: https://bbc2.github.io/python-dotenv/ ; https://docs.python.org/3.12/library/dataclasses.html
"""
import os  # Only named nonsecret settings are read and logged.
from dataclasses import asdict, dataclass  # Frozen values cannot drift during evaluation.
from pathlib import Path  # Configuration is loaded solely from the current project.

from dotenv import load_dotenv  # Reuse the established .env parser.


@dataclass(frozen=True)
class Config:
    """Local: pilot defaults; global: immutable fitting and evaluation specification."""
    torch_threads: int = 1  # Small MLP inference avoids thread startup overhead.
    max_steps: int = 1000  # Match the upstream Gymnasium time limit.
    warmup: int = 100  # Steering begins after an identical unsteered prefix.
    diagnostic_episodes: int = 16  # Inspect natural variation before fitting.
    fit_episodes: int = 64  # Independent episode groups support quartile contrasts.
    validation_episodes: int = 10  # Select intervention strength outside fitting data.
    confirmation_episodes: int = 30  # Held-out pairs estimate the selected causal effect.
    replication_episodes: int = 30  # A second fresh rollout sample tests replication.

    def as_dict(self) -> dict:
        """Local: expose serializable fields; global: record every configurable choice."""
        return asdict(self)  # Standard dataclass conversion avoids parallel field lists.


def read_config() -> Config:
    """Local: parse positive integer settings; global: keep experiments reproducible."""
    load_dotenv(Path.cwd() / ".env", override=False)  # Explicit process settings take precedence.
    defaults = Config().as_dict()  # One definition determines both defaults and supported keys.
    values = {key: int(os.getenv(f"STEERING_{key.upper()}", default)) for key, default in defaults.items()}  # Never enumerate secret environment variables.
    if any(value <= 0 for value in values.values()):  # Empty phases are not valid evidence.
        raise ValueError("All STEERING settings must be positive integers")  # Stop before artifacts are mixed.
    if values["max_steps"] <= values["warmup"]:  # At least one steered transition is required.
        raise ValueError("max_steps must exceed warmup")  # Avoid meaningless zero-duration comparisons.
    return Config(**values)  # Freeze the resolved protocol for this invocation.
