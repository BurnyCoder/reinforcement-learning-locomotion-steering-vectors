"""Local: keep test artifacts in this project; global: honor project-local work.

Source: https://docs.pytest.org/en/stable/how-to/tmp_path.html#temporary-directory-location-and-retention
"""
import os  # A process ID separates simultaneous test invocations.
from pathlib import Path  # Resolve the repository independently of shell activation.


def pytest_configure(config):
    """Local: choose a local temporary root; global: isolate tests without writing unrelated folders."""
    if config.option.basetemp is None:  # Preserve an explicit caller choice within the documented test workflow.
        config.option.basetemp = str(Path(__file__).resolve().parents[1] / ".cache" / "pytest" / str(os.getpid()))  # Pytest owns cleanup inside this process-specific project directory.
        Path(config.option.basetemp).parent.mkdir(parents=True, exist_ok=True)  # Pytest creates its final directory but requires the project-local parent to exist.
