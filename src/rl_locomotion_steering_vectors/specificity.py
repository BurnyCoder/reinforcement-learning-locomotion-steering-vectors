"""Local: statistics and direction builders for steering specificity. Global: compare one vector against many nulls.

Sources: rank p-value (1 + #{null >= observed}) / (k + 1), Phipson & Smyth (2010)
https://doi.org/10.2202/1544-6115.1585 ; Holm (1979) https://www.jstor.org/stable/4615733 ;
SciPy 1.14 bootstrap, permutation_test (one sample flips signs) and wilcoxon
https://docs.scipy.org/doc/scipy-1.14.1/reference/stats.html ; NumPy SeedSequence
https://numpy.org/doc/1.26/reference/random/bit_generators/generated/numpy.random.SeedSequence.html
"""
from __future__ import annotations  # Keep annotations lightweight.

import numpy as np  # Array statistics and reproducible random draws.
from scipy import stats  # Maintained bootstrap, randomization and signed-rank implementations.


def odd_even(minus, plus) -> tuple[np.ndarray, np.ndarray]:
    """Local: split ±alpha effects. Global: odd = sign-dependent steering, even = sign-independent disruption."""
    minus, plus = np.asarray(minus, dtype=float), np.asarray(plus, dtype=float)  # Accept scalars or arrays.
    return (plus - minus) / 2, (plus + minus) / 2  # plus = even + odd, minus = even − odd.


def rank_p_value(observed: float, null) -> float:
    """Local: Monte Carlo rank p-value. Global: test a statistic against exchangeable null draws."""
    null = np.asarray(null, dtype=float)  # Null statistics, larger means more extreme.
    if null.size == 0:
        raise ValueError("null draws are required")  # Nothing to rank against.
    return float((1 + np.sum(null >= observed)) / (null.size + 1))  # Ties count against the observed value.


def holm(pvalues) -> np.ndarray:
    """Local: Holm step-down adjustment. Global: control family-wise error across the primary tests."""
    p = np.asarray(pvalues, dtype=float)  # Unadjusted values.
    order = np.argsort(p)  # Smallest first.
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])  # (m − i) p_(i), made monotone.
    result = np.empty_like(p)  # Restore input order.
    result[order] = np.minimum(adjusted, 1.0)  # Cap at one.
    return result  # Adjusted p-values.


def random_directions(count: int, dimension: int, norm: float, seed: int) -> np.ndarray:
    """Local: draw isotropic Gaussian directions at a fixed norm. Global: build a magnitude-matched perturbation null."""
    rng = np.random.default_rng(np.random.SeedSequence(seed))  # Recorded, independent stream.
    draws = rng.standard_normal((count, dimension))  # Isotropic directions after normalization.
    return (draws * (norm / np.linalg.norm(draws, axis=1, keepdims=True))).astype(np.float32)  # Runtime activation dtype.


def decompose(vector, axis) -> tuple[np.ndarray, np.ndarray]:
    """Local: project onto one axis. Global: split a steering vector into a mechanism component and the rest."""
    vector, axis = np.asarray(vector, dtype=np.float64), np.asarray(axis, dtype=np.float64)  # Exact float64 algebra.
    unit = axis / np.linalg.norm(axis)  # Unit mechanism axis.
    parallel = (vector @ unit) * unit  # Component along the axis.
    return parallel, vector - parallel  # parallel + orthogonal == vector.


def renormalize(vector, norm: float) -> np.ndarray:
    """Local: rescale to a target norm. Global: compare ablated directions with the same perturbation size."""
    vector = np.asarray(vector, dtype=np.float64)  # Exact rescaling.
    size = float(np.linalg.norm(vector))  # Current magnitude.
    if size <= 1e-12:
        raise ValueError("cannot renormalize a zero vector")  # No direction to keep.
    return vector * (norm / size)  # Same direction, new magnitude.


def paired_tests(differences, seed: int, resamples: int = 10_000) -> dict:
    """Local: summarize treated-minus-baseline pairs. Global: report interval, randomization and rank evidence together."""
    d = np.asarray(differences, dtype=float)  # One difference per reset.
    percentile = stats.bootstrap((d,), np.mean, n_resamples=resamples, method="percentile", random_state=seed).confidence_interval  # Percentile interval.
    bca = stats.bootstrap((d,), np.mean, n_resamples=resamples, method="BCa", random_state=seed).confidence_interval  # Bias-corrected accelerated interval.
    flip = stats.permutation_test((d,), np.mean, permutation_type="samples", n_resamples=resamples, random_state=seed)  # Sign-flip test of symmetry about zero.
    nonzero = d[d != 0]  # Wilcoxon drops exact zeros (e.g. shared pre-onset failures).
    wilcoxon = stats.wilcoxon(nonzero).pvalue if nonzero.size else 1.0  # Signed-rank test.
    return {"n": int(d.size), "mean": float(d.mean()), "median": float(np.median(d)), "sd": float(d.std(ddof=1)),
            "percentile_ci": [float(percentile.low), float(percentile.high)], "bca_ci": [float(bca.low), float(bca.high)],
            "sign_flip_p": float(flip.pvalue), "wilcoxon_p": float(wilcoxon), "positive_pairs": int(np.sum(d > 0)),
            "negative_pairs": int(np.sum(d < 0)), "zero_pairs": int(np.sum(d == 0)), "resamples": resamples, "seed": seed}  # JSON-safe summary.


def two_way_bootstrap(matrix, seed: int, resamples: int = 10_000) -> dict:
    """Local: resample rows and columns independently. Global: propagate direction and seed uncertainty into a grand mean."""
    m = np.asarray(matrix, dtype=float)  # Rows = directions, columns = seeds.
    if m.ndim != 2:
        raise ValueError("expected a directions × seeds matrix")  # Both factors are required.
    rng = np.random.default_rng(seed)  # Reproducible resampling.
    rows = rng.integers(0, m.shape[0], size=(resamples, m.shape[0]))  # Direction resamples.
    cols = rng.integers(0, m.shape[1], size=(resamples, m.shape[1]))  # Seed resamples.
    means = np.array([m[r][:, c].mean() for r, c in zip(rows, cols)])  # Grand mean of each two-way resample.
    return {"mean": float(m.mean()), "ci": np.quantile(means, [0.025, 0.975]).tolist(), "resamples": resamples, "seed": seed}  # Percentile interval.
