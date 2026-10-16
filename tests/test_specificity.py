"""Local: test specificity statistics and direction builders. Global: keep the random-direction rank test exact.

Sources: rank p-value with k exchangeable null draws, (1 + #{null >= observed}) / (k + 1),
Phipson & Smyth (2010) https://doi.org/10.2202/1544-6115.1585 ;
Holm (1979) step-down adjustment https://www.jstor.org/stable/4615733 ;
SciPy 1.14 bootstrap/permutation_test https://docs.scipy.org/doc/scipy-1.14.1/reference/stats.html
"""
import numpy as np  # Synthetic arrays with analytic answers.
import pytest  # Approximate comparisons and failure checks.

from rl_locomotion_steering_vectors.specificity import (  # Public contract of the new module.
    decompose, holm, odd_even, paired_tests, random_directions, rank_p_value, renormalize, two_way_bootstrap,
)


def test_odd_even_reconstructs_both_arms():
    """Local: split a ±alpha pair. Global: separate sign-dependent steering from sign-independent disruption."""
    minus, plus = np.array([1.0, 3.0]), np.array([-1.0, 5.0])  # Two directions' paired effects.
    odd, even = odd_even(minus, plus)  # Decompose.
    np.testing.assert_allclose(odd, [-1.0, 1.0])  # (plus - minus) / 2.
    np.testing.assert_allclose(even, [0.0, 4.0])  # (plus + minus) / 2.
    np.testing.assert_allclose(even + odd, plus)  # Positive arm is recovered.
    np.testing.assert_allclose(even - odd, minus)  # Negative arm is recovered.


def test_rank_p_value_counts_ties_and_floor():
    """Local: compute the Monte Carlo rank p-value. Global: never report p = 0 from finite draws."""
    assert rank_p_value(5.0, np.array([1.0, 2.0, 5.0, 7.0])) == pytest.approx(3 / 5)  # Ties count against the observed value.
    assert rank_p_value(99.5, np.arange(99.0)) == pytest.approx(1 / 100)  # Smallest attainable p with 99 draws.
    with pytest.raises(ValueError):  # An empty null cannot rank anything.
        rank_p_value(1.0, np.array([]))  # Reject.


def test_holm_adjustment_is_monotone():
    """Local: adjust two or more p-values. Global: control family-wise error across primary tests."""
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])  # Hand-computed step-down values.
    np.testing.assert_allclose(holm([0.5, 0.9]), [1.0, 1.0])  # Adjusted values are capped at one.


def test_random_directions_are_reproducible_and_norm_matched():
    """Local: draw isotropic directions. Global: give the null the primary's exact magnitude."""
    first = random_directions(50, 256, norm=5.0, seed=123)  # Reproducible draw.
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 5.0, rtol=1e-6)  # Every direction matches the norm.
    np.testing.assert_array_equal(first, random_directions(50, 256, norm=5.0, seed=123))  # Same seed, same draw.
    assert not np.array_equal(first, random_directions(50, 256, norm=5.0, seed=124))  # Different seed, different draw.
    assert first.dtype == np.float32  # Matches the runtime activation dtype.
    unit = first / 5.0  # Unit directions.
    assert abs(float(np.mean(unit @ unit.T - np.eye(50)))) < 0.01  # Isotropic draws are nearly orthogonal on average.


def test_decompose_and_renormalize_are_exact():
    """Local: split a vector along an axis. Global: support additive ablation of the perceived-heading component."""
    vector, axis = np.array([3.0, 4.0, 0.0]), np.array([2.0, 0.0, 0.0])  # Known geometry.
    parallel, orthogonal = decompose(vector, axis)  # Split.
    np.testing.assert_allclose(parallel, [3.0, 0.0, 0.0])  # Projection on the axis.
    np.testing.assert_allclose(orthogonal, [0.0, 4.0, 0.0])  # Remainder.
    np.testing.assert_allclose(parallel + orthogonal, vector)  # Exact additive decomposition.
    np.testing.assert_allclose(np.linalg.norm(renormalize(orthogonal, 10.0)), 10.0)  # Rescaled magnitude.
    with pytest.raises(ValueError):  # A zero vector has no direction.
        renormalize(np.zeros(3), 1.0)  # Reject.


def test_paired_tests_on_all_positive_differences():
    """Local: summarize paired differences. Global: report interval, randomization and rank evidence together."""
    differences = np.linspace(0.2, 0.6, 20)  # Twenty strictly positive paired effects.
    result = paired_tests(differences, seed=0)  # Run every test.
    assert result["mean"] == pytest.approx(0.4)  # Arithmetic mean.
    assert result["n"] == 20  # Pair count.
    assert result["percentile_ci"][0] < 0.4 < result["percentile_ci"][1]  # Percentile interval covers the mean.
    assert result["bca_ci"][0] < 0.4 < result["bca_ci"][1]  # BCa interval covers the mean.
    assert result["sign_flip_p"] < 1e-3  # All-positive differences are extreme under symmetric sign flips.
    assert result["wilcoxon_p"] < 1e-3  # Exact signed-rank test agrees.
    assert result["positive_pairs"] == 20  # Count of positive differences.


def test_two_way_bootstrap_handles_constant_and_shape():
    """Local: resample directions and seeds. Global: propagate both sources of uncertainty into the generic effect."""
    constant = np.full((7, 5), 0.3)  # No variation in either factor.
    result = two_way_bootstrap(constant, seed=0, resamples=200)  # Bootstrap.
    assert result["mean"] == pytest.approx(0.3)  # Grand mean.
    np.testing.assert_allclose(result["ci"], [0.3, 0.3])  # Degenerate interval.
    with pytest.raises(ValueError):  # A one-dimensional input lacks the seed axis.
        two_way_bootstrap(np.ones(5), seed=0)  # Reject.
