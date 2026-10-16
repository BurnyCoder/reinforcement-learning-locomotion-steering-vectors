"""Inference-level statistics for the scientific-rigor audit; never simulate or select.

Local: small pure functions for randomization tests, signed-response decomposition,
fixed-horizon displacement, heading and shuffled-control alignment.
Global: test whether the recorded experiments support their interpretations,
reusing the production estimators instead of copying them.
Sources: https://arxiv.org/abs/1603.05766 (Phipson & Smyth, (b+1)/(m+1) p-values),
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html
(paired "samples" permutations are within-pair sign flips),
https://gymnasium.farama.org/environments/mujoco/ant/ (observation starts with z, then w,x,y,z quaternion),
https://arxiv.org/pdf/1307.0239 (Barnard's Monte Carlo test: exact level from the observed statistic's rank among s+1),
https://imaging.mrc-cbu.cam.ac.uk/statswiki/FAQ/power/onesamp (paired normal-approximation sample size; it slightly underestimates exact-t n),
https://numpy.org/doc/1.26/reference/generated/numpy.unwrap.html, https://docs.python.org/3.12/library/statistics.html#statistics.NormalDist.inv_cdf.
"""
import math  # Round sample sizes up to whole pairs.
from statistics import NormalDist  # Standard-library normal quantiles; avoids a SciPy dependency.
import numpy as np  # Vectorized resampling, circular means and cosines.
from rl_locomotion_steering_vectors.analysis import _contrast, _speed_bins  # Reuse the exact production contrast and effort strata.
from rl_locomotion_steering_vectors.runtime import quaternion_yaw  # Reuse the audited scalar-first quaternion-to-yaw conversion.


def odd_even(plus: float, minus: float) -> tuple[float, float]:
    """Local: split effects at +alpha and -alpha. Global: separate steerable (odd) from sign-independent (even) responses."""
    return (plus - minus) / 2, (plus + minus) / 2  # f(a)=odd+even and f(-a)=-odd+even, so both parts are exactly identified.


def sign_flip_pvalue(differences, resamples: int = 200_000, seed: int = 0) -> float:
    """Local: two-sided Monte Carlo paired randomization test. Global: an assumption-light check of the bootstrap conclusion."""
    values = np.asarray(differences, dtype=float)  # One paired treatment-minus-baseline difference per reset.
    observed = abs(values.mean())  # Two-sided statistic: the absolute mean paired difference.
    signs = np.random.default_rng(seed).choice([-1., 1.], size=(resamples, len(values)))  # Under the sharp null each pair's sign is exchangeable.
    null = np.abs(signs @ values) / len(values)  # Absolute mean after each random within-pair relabeling.
    hits = int(np.sum(null >= observed - 1e-12))  # A tiny tolerance counts floating-point ties as extreme, which is conservative.
    return (hits + 1) / (resamples + 1)  # Including the observed labeling keeps the p-value valid and never zero (Phipson & Smyth).


def percentile_interval(differences, seed: int, resamples: int = 2000) -> list[float]:
    """Local: paired percentile bootstrap. Global: mirror analysis.paired_effect for outcomes it does not summarize."""
    values = np.asarray(differences, dtype=float)  # Whole-pair differences are the resampling unit.
    indices = np.random.default_rng(seed).integers(0, len(values), size=(resamples, len(values)))  # Same index scheme as the production estimator.
    return np.quantile(values[indices].mean(axis=1), [0.025, 0.975]).tolist()  # Same 95% percentile convention as the recorded intervals.


def post_onset_displacement(episode: dict, field: str, warmup: int = 100) -> float:
    """Local: integrate one velocity after onset. Global: a fixed-horizon outcome where termination simply stops motion."""
    length = int(episode["length"])  # Actual recorded duration, including early termination.
    return float(np.sum(np.asarray(episode[field], dtype=float)[warmup:length]) * float(episode.get("dt", 0.05)))  # Sum of v*dt; a prefix-only slice is empty, giving zero.


def mean_heading(observations, length: int, warmup: int = 100) -> float | None:
    """Local: circular mean torso yaw after onset. Global: quantify how much a world-Y effect is a change of direction."""
    rows = np.asarray(observations, dtype=float)[warmup:int(length)]  # Pre-action observations of the post-onset steps.
    if not len(rows):
        return None  # Heading is undefined when the episode ended before steering began.
    yaws = np.array([quaternion_yaw(row[1:5]) for row in rows])  # Ant observation indices 1..4 hold the root's scalar-first quaternion.
    return float(np.angle(np.mean(np.exp(1j * yaws))))  # Circular mean avoids the -pi/pi branch cut.


def shuffled_alignment_null(table: dict, behavior: str, direction, draws: int, seed: int) -> np.ndarray:
    """Local: cosines of many shuffled-label contrasts with one direction. Global: test whether the shuffled control is label-free."""
    unit = np.asarray(direction, dtype=float) / np.linalg.norm(direction)  # Compare directions, not magnitudes.
    labels = np.asarray(table[behavior], dtype=float)  # Original eligible-window labels.
    bins = _speed_bins(table["speed"]) if behavior == "effort" else np.zeros(len(labels), dtype=int)  # Effort shuffles stay within speed strata, as in fit_vectors.
    rng, cosines = np.random.default_rng(seed), []  # Independent deterministic permutations.
    for _ in range(draws):
        shuffled = labels.copy()  # Keep the label distribution; only its assignment to windows changes.
        for group in np.unique(bins):
            mask = bins == group  # Permute inside each stratum.
            shuffled[mask] = rng.permutation(shuffled[mask])  # Same within-stratum window permutation as the saved control.
        raw, _ = _contrast(table, behavior, shuffled)  # Same quartile, episode-balancing and contributor rules as extraction.
        if raw is not None and np.linalg.norm(raw) > 1e-12:
            cosines.append(float(raw @ unit / np.linalg.norm(raw)))  # Cosine with the reference direction.
    return np.asarray(cosines)  # Degenerate shuffles are omitted, as extraction would skip them.


def travel_angle(episode: dict, warmup: int = 100) -> float | None:
    """Local: direction of the post-onset mean ground velocity. Global: describe a ground-path shift without torso-yaw conventions."""
    length = int(episode["length"])  # Actual recorded duration, including early termination.
    if length <= warmup:
        return None  # No post-onset path exists for a prefix-only episode.
    return float(np.arctan2(np.mean(episode["y_velocity"][warmup:length]), np.mean(episode["x_velocity"][warmup:length])))  # Mean velocity times duration is net displacement, so this is its world-frame direction.


def heading_series(observations, length: int) -> np.ndarray:
    """Local: unwrapped torso yaw per pre-action observation. Global: follow heading over time without false 2*pi jumps."""
    rows = np.asarray(observations, dtype=float)[:int(length)]  # Only actually recorded steps.
    return np.unwrap([quaternion_yaw(row[1:5]) for row in rows])  # Ant observation indices 1..4 hold the scalar-first root quaternion; unwrap removes branch-cut jumps.


def window_means(values, length: int, windows) -> np.ndarray:
    """Local: mean of each fully recorded [start, stop) window. Global: compare time courses only where both arms have data."""
    values = np.asarray(values, dtype=float)  # One sample per recorded step.
    return np.array([values[start:stop].mean() if stop <= int(length) else np.nan for start, stop in windows])  # A window cut short by termination is undefined, not partially averaged.


def rank_pvalue(observed: float, reference) -> float:
    """Local: one-sided Monte Carlo rank test. Global: compare one vector's response with exchangeable random directions."""
    reference = np.asarray(reference, dtype=float)  # Statistics of the comparison directions.
    return (1 + int(np.sum(reference >= observed - 1e-12))) / (len(reference) + 1)  # Barnard's exact-level rule; ties count as extreme, which is conservative.


def pairs_for_power(sd: float, delta: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Local: paired normal-approximation sample size. Global: check whether a confirmation split could detect a threshold effect."""
    quantile = NormalDist().inv_cdf  # Standard normal inverse CDF.
    return math.ceil(((quantile(1 - alpha / 2) + quantile(power)) * sd / delta) ** 2)  # Two-sided test of mean paired difference; sd is the SD of differences.
