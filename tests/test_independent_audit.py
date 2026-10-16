"""Local: test the audit's independent episode summaries. Global: the reanalysis must use the same estimand as production.

Sources: production rules in rl_locomotion_steering_vectors.runtime.summarise_episode and analysis._summary.
"""
import numpy as np  # Synthetic episodes.
import pytest  # Approximate comparisons.

from rl_locomotion_steering_vectors.analysis import _summary  # Production failure rule (private, used as an oracle only).
from rl_locomotion_steering_vectors.runtime import METRIC_NAMES, summarise_episode  # Production summary.
from scripts.ant005_independent.common import failed, post_onset_mean  # Audit reimplementation.
from scripts.ant005_independent.saved import summarize  # Audit summary.


def _episode(length: int, terminated: bool, inversion: float = 0.0, seed: int = 1) -> dict:
    """Local: build a synthetic saved episode. Global: cover normal, failed and prefix-only resets."""
    rng = np.random.default_rng(seed)  # Reproducible values.
    episode = {name: rng.normal(size=length) for name in METRIC_NAMES}  # Every recorded metric.
    episode["inversion"] = np.full(length, inversion)  # Controlled physical-failure fraction.
    episode["bad_contact"] = np.zeros(length)  # No torso contact.
    episode.update(length=np.array(length), terminated=np.array(terminated), dt=np.array(0.05), seed=np.array(seed), onset_hash=np.array("h"))  # Metadata.
    return episode  # Saved-episode schema.


@pytest.mark.parametrize("length,terminated,inversion", [(1000, False, 0.0), (541, True, 0.0), (34, True, 0.0), (1000, False, 0.2)])
def test_audit_summary_matches_production(length, terminated, inversion):
    """Local: compare estimands and failure flags. Global: reanalysis cannot drift from the recorded definitions."""
    episode = _episode(length, terminated, inversion)  # One case.
    production = summarise_episode(episode, 100)  # Runtime summary.
    audit = summarize(episode)  # Audit summary.
    for key in ("x_velocity", "y_velocity", "yaw_rate", "effort"):  # Shared metrics.
        assert audit[key] == pytest.approx(production[key])  # Same post-onset mean (zero when prefix-only).
    assert post_onset_mean(episode, "y_velocity") == pytest.approx(production["y_velocity"])  # Helper agrees.
    assert failed(episode) == bool(_summary(production, 100)["failure"])  # Same failure classification.
