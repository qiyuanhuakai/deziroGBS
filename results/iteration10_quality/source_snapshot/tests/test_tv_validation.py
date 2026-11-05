"""Analytic identity, conservative cutoffs, and point-oracle validation."""

import math
from types import SimpleNamespace
import unittest

import numpy as np

from gbslab.exact import NumericalProbabilityError, ThresholdModel
from gbslab.gaussian import covariance_from_interferometer, haar_unitary
from gbslab.schur_memory import SchurMemorySampler
from gbslab.surrogates import distribution_metrics
from gbslab.tv_validation import (
    GaussianThresholdPointOracle, tv_score_from_log_probabilities, validate_tv,
)


class DenseToySampler:
    """Independent small-distribution oracle, used only by validation tests."""

    def __init__(self, probabilities, order=None):
        self.q = np.asarray(probabilities, dtype=float)
        self.m = self.q.size.bit_length() - 1
        self.order = tuple(range(self.m)) if order is None else tuple(order)
        self.sample_calls = 0
        self.conditional_calls = 0

    def sample(self, n, rng):
        self.sample_calls += 1
        masks = rng.choice(len(self.q), size=n, p=self.q)
        return ((masks[:, None] >> np.arange(self.m)) & 1).astype(np.uint8)

    def conditional_click_probability(self, prefix):
        self.conditional_calls += 1
        mode = self.order[len(prefix)]
        numerator = denominator = 0.0
        for mask, probability in enumerate(self.q):
            if all(((mask >> self.order[j]) & 1) == bit for j, bit in enumerate(prefix)):
                denominator += probability
                if mask & (1 << mode):
                    numerator += probability
        if denominator == 0:
            raise NumericalProbabilityError("impossible toy history")
        return numerator / denominator


def lower_callback(p):
    def probability(sample):
        mask = sum(int(bit) << mode for mode, bit in enumerate(sample))
        return float(p[mask])
    return probability


class TVIdentityTests(unittest.TestCase):
    def test_identity_including_missing_target_support(self):
        for p, q in [([.7, .1, .1, .1], [.4, .3, .2, .1]),
                     ([.5, .5, 0, 0], [0, 0, .5, .5]),
                     ([.25, .25, .25, .25], [.5, 0, .5, 0])]:
            p, q = np.array(p), np.array(q)
            expectation = sum(
                q_i * tv_score_from_log_probabilities(-math.inf if p_i == 0 else math.log(p_i),
                                                       math.log(q_i))
                for p_i, q_i in zip(p, q) if q_i > 0)
            self.assertAlmostEqual(expectation, .5 * np.sum(np.abs(p-q)), 14)

    def test_cutoff_expectations_are_conservative_and_monotone(self):
        p = np.array([.35, .1, .1, .05, .1, .05, .05, .2])
        q = np.array([.2, .05, .05, .1, .1, .15, .15, .2])
        patterns = ((np.arange(8)[:, None] >> np.arange(3)) & 1).astype(np.uint8)
        expectations = []
        # Exhaustive deterministic score check, not an iid confidence test.
        for cutoff in range(4):
            record = validate_tv(None, DenseToySampler(q, order=(2, 0, 1)), 8, cutoff, .05,
                                 samples=patterns, probability_lower_bound=lower_callback(p),
                                 keep_scores=True)
            expectations.append(float(np.dot(q, record["scores"])))
            self.assertEqual(record["cutoff_count"], sum(mask.bit_count()>cutoff for mask in range(8)))
        self.assertTrue(all(a >= b-1e-14 for a,b in zip(expectations, expectations[1:])))
        true_tv = .5 * np.sum(np.abs(p-q))
        self.assertAlmostEqual(expectations[-1], true_tv, 13)
        self.assertTrue(all(value >= true_tv-1e-13 for value in expectations))

    def test_constant_score_has_exact_mean_and_hoeffding_margin(self):
        # Q is deterministic; this makes the numerical test non-flaky and the
        # true TV exactly .25, including target mass outside Q's support.
        q = DenseToySampler([0, 0, 0, 1], order=(1, 0))
        p = [.25, 0, 0, .75]
        record = validate_tv(None, q, 64, 2, .05, np.random.default_rng(33),
                             probability_lower_bound=lower_callback(p), keep_scores=True)
        self.assertEqual(record["status"], "ok")
        self.assertAlmostEqual(record["mean_y"], .25, 14)
        margin = math.sqrt(math.log(20)/128)
        self.assertAlmostEqual(record["statistical_margin"], margin, 14)
        self.assertAlmostEqual(record["upper"], .25+margin, 14)
        self.assertEqual(q.sample_calls, 1)
        self.assertEqual(record["unique_target_probability_evaluations"], 1)
        self.assertEqual(record["pattern_cache_hits"], 63)

    def test_failures_and_high_clicks_remain_in_denominator(self):
        calls = []
        def unresolved(sample):
            calls.append(tuple(sample))
            raise FloatingPointError("deliberately unresolved target")
        q = DenseToySampler([1, 0])
        record = validate_tv(None, q, 32, 1, .01, np.random.default_rng(7),
                             probability_lower_bound=unresolved)
        self.assertEqual(record["target_probability_failure_count"], 32)
        self.assertEqual(len(calls), 1)
        self.assertEqual(record["mean_y"], 1)
        self.assertEqual(record["upper"], 1)
        self.assertEqual(record["n_observed"], 32)
        high = DenseToySampler([0,0,0,0,0,0,0,1])
        record = validate_tv(None, high, 32, 2, .05, np.random.default_rng(7),
                             probability_lower_bound=unresolved)
        self.assertEqual(record["cutoff_count"], 32)
        self.assertEqual(high.conditional_calls, 0)
        self.assertEqual(record["unique_target_probability_evaluations"], 0)
        self.assertEqual(record["upper"], 1)

    def test_sampling_failure_is_not_retried_or_certified(self):
        class BrokenSampler(DenseToySampler):
            def sample(self, n, rng):
                self.sample_calls += 1
                raise NumericalProbabilityError("cannot draw a well-defined sample")
        q = BrokenSampler([.5,.5])
        record = validate_tv(None, q, 20, 1, .05, np.random.default_rng(1),
                             probability_lower_bound=lambda x:.5)
        self.assertEqual(q.sample_calls, 1)
        self.assertEqual(record["status"], "sampling_failed")
        self.assertIsNone(record["upper"])

    def test_reproducibility_and_explicit_qualification(self):
        p = [.1,.2,.3,.4]
        kwargs = dict(n=80, max_exact_clicks=2, alpha=.05,
                      probability_lower_bound=lower_callback(p), keep_scores=True,
                      pattern_cache_size=1)
        first = validate_tv(None, DenseToySampler([.25]*4), rng=np.random.default_rng(5), **kwargs)
        second = validate_tv(None, DenseToySampler([.25]*4), rng=np.random.default_rng(5), **kwargs)
        self.assertEqual(first["scores"], second["scores"])
        self.assertEqual(first["upper"], second["upper"])
        self.assertLessEqual(first["pattern_cache_entries"], 1)
        self.assertFalse(first["strict_floating_point_certificate"])
        self.assertIn("not simultaneous", first["confidence_scope"])


class GaussianPointOracleTests(unittest.TestCase):
    def covariance(self):
        U = haar_unitary(4, np.random.default_rng(904))
        return covariance_from_interferometer(U, [.6,.7,.8,.9], [.6,.7,.8,.9])

    def test_all_small_points_against_full_target(self):
        V = self.covariance()
        exact = ThresholdModel(V).full_distribution()
        oracle = GaussianThresholdPointOracle(V, 4)
        for mask, expected in enumerate(exact):
            sample = np.array([(mask>>j)&1 for j in range(4)])
            actual = math.exp(oracle.log_probability(sample))
            self.assertAlmostEqual(actual, expected, 12)
        self.assertLessEqual(oracle.cache_info()["maximum_small_determinant_dimension"], 8)

    def test_actual_sampler_path_and_caps(self):
        V = self.covariance()
        model = ThresholdModel(V)
        sampler = SchurMemorySampler(ThresholdModel(V), 1, order=(3, 0, 2, 1))
        record = validate_tv(model, sampler, 64, 3, .05, np.random.default_rng(110),
                             keep_scores=True)
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["n_observed"], 64)
        self.assertTrue(all(0 <= value <= 1 for value in record["scores"]))
        self.assertEqual(record["target_probability_mode"], "floating_gaussian_point_estimate")
        self.assertIn("correct", record["qualification"])
        self.assertEqual(record["cutoff_count"] + record["q_probability_failure_count"] +
                         record["target_probability_failure_count"] +
                         record["successfully_evaluated_count"], 64)
        self.assertLessEqual(record["target_oracle_diagnostics"]["maximum_all_click_terms"], 8)
        with self.assertRaises(ValueError):
            GaussianThresholdPointOracle(V, 21)
        with self.assertRaises(ValueError):
            GaussianThresholdPointOracle(V, 1).log_probability([1,1,0,0])
        with self.assertRaises(ValueError):
            validate_tv(SimpleNamespace(m=3), sampler, 10, 1, .05,
                        probability_lower_bound=lambda x:.1)


if __name__ == "__main__":
    unittest.main()
