"""Validate Gaussian conditioning and event ratios independently of histories."""

import itertools
import unittest

import numpy as np

from gbslab.exact import NumericalProbabilityError, ThresholdModel
from gbslab.gaussian import covariance_from_interferometer, haar_unitary
from gbslab.schur_memory import (
    GaussianVacuumConditioner, SchurMemorySampler,
    conditional_click_from_covariance, gaussian_vacuum_probability,
)
from gbslab.zero_memory import ZeroMemorySampler


def random_covariance(m=6):
    rng = np.random.default_rng(4417)
    U = haar_unitary(m, rng)
    return covariance_from_interferometer(
        U, np.linspace(.3, .8, m), np.linspace(.3, .9, m))


class GaussianConditioningTests(unittest.TestCase):
    def test_zero_conditioned_subsets_match_original_event_ratios(self):
        V = random_covariance()
        model = ThresholdModel(V)
        conditioner = GaussianVacuumConditioner(V)
        # Non-monotone measurement order exercises cached rank-two updates.
        zeros = ()
        for added in (4, 1, 5):
            zeros += (added,)
            remaining, covariance = conditioner.conditioned(zeros)
            self.assertEqual(remaining, tuple(i for i in range(6) if i not in zeros))
            for length in range(min(3, len(remaining)) + 1):
                for subset in itertools.combinations(remaining, length):
                    conditional_vacuum = gaussian_vacuum_probability(
                        conditioner.subcovariance(zeros, subset))
                    ratio = model.vacuum_probability(zeros + subset) / model.vacuum_probability(zeros)
                    self.assertAlmostEqual(conditional_vacuum, ratio, 12)
            self.assertFalse(covariance.flags.writeable)
        self.assertEqual(conditioner.cache_info()["rank_two_updates"], 3)

    def test_conditioned_distribution_matches_dense_target_postselection(self):
        V = random_covariance(5)
        target = ThresholdModel(V).full_distribution()
        zeros, requested = (1, 3), (4, 0, 2)
        conditional = GaussianVacuumConditioner(V).subcovariance(zeros, requested)
        prediction = ThresholdModel(conditional).full_distribution()
        expected = np.zeros(1 << len(requested))
        for mask, probability in enumerate(target):
            if any(mask & (1 << j) for j in zeros):
                continue
            local = sum(((mask >> mode) & 1) << bit for bit, mode in enumerate(requested))
            expected[local] += probability
        expected /= expected.sum()
        np.testing.assert_allclose(prediction, expected, atol=1e-13, rtol=1e-10)

    def test_tmsv_zero_projection_is_vacuum(self):
        # Exact algebra: c=3, s=sqrt(8), c-s^2/(c+1)=1.
        s = np.sqrt(8.0)
        V = np.array([[3, s, 0, 0], [s, 3, 0, 0],
                      [0, 0, 3, -s], [0, 0, -s, 3]])
        remaining, conditional = GaussianVacuumConditioner(V).conditioned((0,))
        self.assertEqual(remaining, (1,))
        np.testing.assert_allclose(conditional, np.eye(2), atol=1e-14, rtol=0)
        self.assertAlmostEqual(gaussian_vacuum_probability(conditional), 1.0, 13)

    def test_bounded_cache_direct_fallback_and_array_protection(self):
        V = random_covariance(7)
        cached = GaussianVacuumConditioner(V, cache_size=2, max_cache_bytes=3000)
        direct = GaussianVacuumConditioner(V, cache_size=0)
        for zeros in [(5,), (5, 1), (5, 1, 3), (2, 6), (0, 4), (5, 1, 3)]:
            modes_a, W_a = cached.conditioned(zeros)
            modes_b, W_b = direct.conditioned(zeros)
            self.assertEqual(modes_a, modes_b)
            np.testing.assert_allclose(W_a, W_b, rtol=0, atol=4e-14)
        info = cached.cache_info()
        self.assertLessEqual(info["cache_entries"], 2)
        self.assertLessEqual(info["peak_cache_payload_bytes"], 3000)
        self.assertGreater(info["cache_evictions"], 0)
        self.assertGreater(info["direct_factorizations"], 0)
        with self.assertRaises(ValueError):
            cached.conditioned(())[1][0, 0] = 5
        with self.assertRaises(ValueError):
            cached.subcovariance((1,), (1,))
        empty_modes, empty = cached.conditioned(tuple(range(7)))
        self.assertEqual(empty_modes, ())
        self.assertEqual(empty.shape, (0, 0))
        self.assertEqual(gaussian_vacuum_probability(empty), 1)


class SmallEventKernelTests(unittest.TestCase):
    def test_independent_thermal_click_conditional(self):
        # Three independent thermal modes with means .2, .7 and 1.2.
        nbar = np.array([.2, .7, 1.2])
        V = np.diag(np.tile(1 + 2 * nbar, 2))
        p_one, diagnostics = conditional_click_from_covariance(V)
        self.assertAlmostEqual(p_one, 1.2 / 2.2, 13)
        self.assertEqual(diagnostics["determinant_evaluations"], 7)
        self.assertEqual(diagnostics["maximum_determinant_dimension"], 6)
        self.assertAlmostEqual(diagnostics["conditioning_probability"], (.2 / 1.2) * (.7 / 1.7), 13)

    def test_ratio_against_complete_small_marginal_and_impossible_event(self):
        for m in (1, 2, 4):
            V = random_covariance(m)
            p = ThresholdModel(V).full_distribution()
            one, _ = conditional_click_from_covariance(V)
            old_clicks = (1 << (m - 1)) - 1
            expected = p[-1] / (p[old_clicks] + p[-1])
            self.assertAlmostEqual(one, expected, 11)
        # First mode is exactly vacuum; conditioning on its click is undefined.
        with self.assertRaisesRegex(NumericalProbabilityError, "zero|unresolved"):
            conditional_click_from_covariance(np.diag([1, 2, 1, 2]))


class SchurSamplerTests(unittest.TestCase):
    def test_identical_recent_and_pairwise_models(self):
        V = random_covariance()
        order = (4, 1, 5, 0, 3, 2)
        for selection in ("recent", "pairwise"):
            for budget in (0, 1, 2, 6):
                direct = ZeroMemorySampler(ThresholdModel(V), budget, order=order, selection=selection)
                schur = SchurMemorySampler(ThresholdModel(V), budget, order=order, selection=selection)
                np.testing.assert_allclose(schur.full_distribution(), direct.full_distribution(),
                                           rtol=2e-9, atol=1e-13)
                info = schur.cache_info()
                self.assertLessEqual(info["maximum_small_determinant_dimension"],
                                     2 * min(budget + 1, 6))
                self.assertGreater(info["gaussian_conditioner"]["rank_two_updates"], 0)

    def test_full_conditioning_matches_exact_target(self):
        V = random_covariance()
        target = ThresholdModel(V).full_distribution()
        schur = SchurMemorySampler(ThresholdModel(V), 6, order=(3, 0, 4, 2, 1, 5))
        np.testing.assert_allclose(schur.full_distribution(), target, rtol=2e-9, atol=2e-13)

    def test_sampling_uses_same_conditionals_and_bounded_caches(self):
        V = random_covariance()
        direct = ZeroMemorySampler(ThresholdModel(V), 2, selection="pairwise")
        schur = SchurMemorySampler(ThresholdModel(V), 2, selection="pairwise",
                                  cache_size=31, covariance_cache_size=3,
                                  covariance_cache_bytes=2000)
        a = direct.sample(500, np.random.default_rng(553))
        b = schur.sample(500, np.random.default_rng(553))
        np.testing.assert_array_equal(a, b)
        info = schur.cache_info()
        self.assertLessEqual(info["conditional_cache_entries"], 31)
        self.assertLessEqual(info["gaussian_conditioner"]["cache_entries"], 3)
        self.assertLessEqual(info["gaussian_conditioner"]["cache_payload_bytes"], 2000)
        self.assertEqual(schur.sample(0, np.random.default_rng(1)).shape, (0, 6))

    def test_vacuum_and_input_validation(self):
        sampler = SchurMemorySampler(ThresholdModel(np.eye(8)), 2)
        expected = np.zeros(16); expected[0] = 1
        np.testing.assert_array_equal(sampler.full_distribution(), expected)
        for invalid in (np.eye(3), np.ones((2, 3)), np.eye(2) * np.nan):
            with self.assertRaises(ValueError):
                GaussianVacuumConditioner(invalid)
        with self.assertRaises(ValueError):
            GaussianVacuumConditioner(np.eye(2), cache_size=-1)
        with self.assertRaises(ValueError):
            GaussianVacuumConditioner(np.eye(2), max_cache_bytes=-1)


if __name__ == "__main__":
    unittest.main()
