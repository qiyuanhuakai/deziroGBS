"""Tests for event-adaptive retention of zero outcomes and recent clicks."""

import unittest

import numpy as np

from gbslab.exact import ThresholdModel
from gbslab.gaussian import covariance_from_interferometer, haar_unitary
from gbslab.zero_memory import ZeroMemorySampler, click_tail_coupling_bound


def target_next_click(p, order, prefix):
    denominator = numerator = 0.0
    for mask, probability in enumerate(p):
        if all(((mask >> order[j]) & 1) == bit for j, bit in enumerate(prefix)):
            denominator += probability
            if mask & (1 << order[len(prefix)]):
                numerator += probability
    return numerator / denominator


class TestZeroMemorySampler(unittest.TestCase):
    def random_model(self, m=5):
        rng = np.random.default_rng(134)
        U = haar_unitary(m, rng)
        return ThresholdModel(covariance_from_interferometer(U, np.linspace(0.35, 0.65, m), 0.72))

    def test_full_conditioning_matches_reference(self):
        model = self.random_model()
        target = model.full_distribution()
        for order in (None, (4, 2, 0, 3, 1)):
            sampler = ZeroMemorySampler(model, max_clicks=5, order=order)
            np.testing.assert_allclose(sampler.full_distribution(), target, atol=3e-13, rtol=1e-10)

    def test_pairwise_full_conditioning_and_bound(self):
        model = self.random_model()
        target = model.full_distribution()
        order = (4, 2, 0, 3, 1)
        for budget in (1, 2, 5):
            sampler = ZeroMemorySampler(model, budget, order=order, selection="pairwise")
            approximate = sampler.full_distribution()
            self.assertAlmostEqual(float(np.sum(approximate)), 1.0, 12)
            self.assertGreaterEqual(float(np.min(approximate)), 0.0)
            tv = 0.5 * np.sum(np.abs(target - approximate))
            self.assertLessEqual(tv, click_tail_coupling_bound(target, budget) + 1e-12)
            if budget == 5:
                np.testing.assert_allclose(approximate, target, atol=3e-13, rtol=1e-10)
            self.assertEqual(sampler.cache_info()["pairwise_marginals_requested"], 10)

    def test_pairwise_selects_correlated_older_click(self):
        # Modes 0 and 1 are a squeezed pair; mode 2 is an independent thermal
        # mode.  Measuring 1 then 2 makes the informative click older.
        r = 0.55
        c, s = np.cosh(2 * r), np.sinh(2 * r)
        V = np.diag([c, c, 2.0, c, c, 2.0])
        V[0, 1] = V[1, 0] = s
        V[3, 4] = V[4, 3] = -s
        model = ThresholdModel(V)
        recent = ZeroMemorySampler(model, 1, order=(1, 2, 0))
        pairwise = ZeroMemorySampler(model, 1, order=(1, 2, 0), selection="pairwise")
        self.assertGreater(pairwise.pairwise_mi[0, 1], pairwise.pairwise_mi[0, 2] + 0.1)
        self.assertEqual(pairwise._retained_clicks((1, 2), 0), (1,))
        self.assertEqual(recent._retained_clicks((1, 2), 0), (2,))
        self.assertAlmostEqual(pairwise.conditional_click_probability((1, 1)), 1.0, 12)
        self.assertAlmostEqual(recent.conditional_click_probability((1, 1)), np.tanh(r)**2, 12)

    def test_pairwise_ties_and_zero_budget(self):
        # Vacuum has exactly zero pairwise scores, making tie behavior exact.
        tied = ZeroMemorySampler(ThresholdModel(np.eye(8)), 1, selection="pairwise")
        self.assertEqual(tied._retained_clicks((2, 1), 3), (1,))
        model = self.random_model(4)
        zero = ZeroMemorySampler(model, 0, selection="pairwise")
        self.assertIsNone(zero.pairwise_mi)
        self.assertEqual(zero.cache_info()["pairwise_marginals_requested"], 0)
        self.assertEqual(model.cache_info()["marginal_evaluations"], 0)
        np.testing.assert_array_equal(zero.full_distribution(),
                                      ZeroMemorySampler(model, 0).full_distribution())

    def test_exact_transitions_until_click_budget_exceeded(self):
        model = self.random_model()
        target = model.full_distribution()
        order = (3, 1, 4, 0, 2)
        for budget in range(3):
            sampler = ZeroMemorySampler(model, budget, order=order)
            for length in range(model.m):
                for mask in range(1 << length):
                    prefix = tuple((mask >> j) & 1 for j in range(length))
                    if sum(prefix) <= budget:
                        reference = target_next_click(target, order, prefix)
                        self.assertAlmostEqual(sampler.conditional_click_probability(prefix), reference, 10)

    def test_vacuum(self):
        sampler = ZeroMemorySampler(ThresholdModel(np.eye(8)), 0)
        p = sampler.full_distribution()
        expected = np.zeros(16)
        expected[0] = 1
        np.testing.assert_array_equal(p, expected)
        samples = sampler.sample(20, np.random.default_rng(2))
        np.testing.assert_array_equal(samples, np.zeros((20, 4), dtype=np.uint8))

    def test_two_mode_squeezed_clicks(self):
        r = 0.55
        c, s = np.cosh(2 * r), np.sinh(2 * r)
        V = np.array([[c, s, 0, 0], [s, c, 0, 0],
                      [0, 0, c, -s], [0, 0, -s, c]])
        model = ThresholdModel(V)
        np.testing.assert_allclose(ZeroMemorySampler(model, 1).full_distribution(),
                                   [1 / np.cosh(r)**2, 0, 0, np.tanh(r)**2], atol=2e-14)
        # k=0 discards the first click, so its second-mode click probability is
        # the unconditioned marginal.  This gives an analytic failure example.
        marginal = np.tanh(r)**2
        approximate = ZeroMemorySampler(model, 0).full_distribution()
        np.testing.assert_allclose(approximate,
                                   [1 - marginal, marginal * (1 - marginal), 0, marginal**2], atol=2e-14)

    def test_normalization_and_coupling_bound(self):
        model = self.random_model(6)
        target = model.full_distribution()
        for budget in range(4):
            approximate = ZeroMemorySampler(model, budget).full_distribution()
            self.assertAlmostEqual(float(np.sum(approximate)), 1, 12)
            self.assertGreaterEqual(float(np.min(approximate)), 0)
            tv = 0.5 * np.sum(np.abs(target - approximate))
            self.assertLessEqual(tv, click_tail_coupling_bound(target, budget) + 1e-12)

    def test_sampling_tracks_enumerated_model(self):
        sampler = ZeroMemorySampler(self.random_model(4), 1, order=(3, 1, 0, 2))
        reference = sampler.full_distribution()
        samples = sampler.sample(20000, np.random.default_rng(7))
        masks = samples @ (1 << np.arange(4))
        frequencies = np.bincount(masks, minlength=16) / samples.shape[0]
        # One fixed RNG seed and a generous statistically meaningful guard.
        self.assertLess(0.5 * np.sum(np.abs(frequencies - reference)), 0.025)
        self.assertEqual(samples.dtype, np.uint8)

    def test_cache_is_bounded_and_zero_sampling_shape(self):
        sampler = ZeroMemorySampler(self.random_model(), 1, cache_size=3)
        sampler.full_distribution()
        info = sampler.cache_info()
        self.assertLessEqual(info["conditional_cache_entries"], 3)
        self.assertLessEqual(info["event_cache_entries"], 3)
        self.assertEqual(sampler.sample(0, np.random.default_rng(2)).shape, (0, 5))

    def test_invalid_arguments(self):
        model = self.random_model(3)
        with self.assertRaises(ValueError):
            ZeroMemorySampler(model, -1)
        with self.assertRaises(ValueError):
            ZeroMemorySampler(model, 1, order=(0, 0, 1))
        with self.assertRaises(ValueError):
            ZeroMemorySampler(model, 1, selection="unimplemented")
        with self.assertRaises(ValueError):
            ZeroMemorySampler(model, 1).conditional_click_probability((0, 2))


if __name__ == "__main__":
    unittest.main()
