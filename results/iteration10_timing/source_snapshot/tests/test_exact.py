"""Analytic tests of the exact threshold reference and optics conventions."""

import unittest

import numpy as np

from gbslab.exact import ThresholdModel
from gbslab.gaussian import (
    covariance_from_interferometer,
    haar_unitary,
    local_unitary,
    mean_photon_number,
)


def product_click_distribution(click_probabilities):
    result = np.ones(1 << len(click_probabilities))
    for mask in range(result.size):
        for j, probability in enumerate(click_probabilities):
            result[mask] *= probability if mask & (1 << j) else 1.0 - probability
    return result


class TestGaussianOptics(unittest.TestCase):
    def test_unitary_generators(self):
        rng = np.random.default_rng(456)
        for U in (haar_unitary(7, rng), local_unitary(7, 5, rng)):
            np.testing.assert_allclose(U.conj().T @ U, np.eye(7), atol=2e-14)

    def test_mean_energy_and_complete_loss(self):
        rng = np.random.default_rng(12)
        squeezing = np.array([0.2, 0.4, 0.6, 0.8])
        U = haar_unitary(4, rng)
        V = covariance_from_interferometer(U, squeezing, loss=0.6)
        self.assertAlmostEqual(mean_photon_number(V), 0.6 * np.sinh(squeezing).dot(np.sinh(squeezing)), 12)
        np.testing.assert_allclose(covariance_from_interferometer(U, squeezing, 0), np.eye(8), atol=1e-15)


class TestExactThreshold(unittest.TestCase):
    def test_vacuum(self):
        model = ThresholdModel(np.eye(8))
        expected = np.zeros(16)
        expected[0] = 1
        np.testing.assert_array_equal(model.full_distribution(), expected)
        np.testing.assert_array_equal(model.marginal(()), [1.0])

    def test_independent_thermal(self):
        occupation = np.array([0.1, 0.7, 1.5, 2.0])
        V = np.diag(np.tile(2.0 * occupation + 1.0, 2))
        model = ThresholdModel(V)
        expected = product_click_distribution(occupation / (1.0 + occupation))
        np.testing.assert_allclose(model.full_distribution(), expected, atol=1e-14)

    def test_independent_lossy_squeezed(self):
        squeezing = np.array([0.1, 0.3, 0.6, 0.9])
        eta = np.array([0.25, 0.5, 0.7, 1.0])
        V = covariance_from_interferometer(np.eye(4), squeezing, eta)
        p0 = 1.0 / np.sqrt(1.0 + eta * (2.0 - eta) * np.sinh(squeezing)**2)
        expected = product_click_distribution(1.0 - p0)
        np.testing.assert_allclose(ThresholdModel(V).full_distribution(), expected, atol=2e-14)

    def test_two_mode_squeezed_pair(self):
        # Analytic state sum_n sech(r) tanh(r)^n |n,n>: clicks are equal.
        r = 0.67
        c, s = np.cosh(2 * r), np.sinh(2 * r)
        V = np.array([[c, s, 0, 0], [s, c, 0, 0],
                      [0, 0, c, -s], [0, 0, -s, c]])
        expected = [1.0 / np.cosh(r)**2, 0.0, 0.0, np.tanh(r)**2]
        np.testing.assert_allclose(ThresholdModel(V).full_distribution(), expected, atol=2e-14)

    def test_two_mode_squeezed_pair_with_loss(self):
        # Geometric pair occupation n, followed by independent binomial loss.
        r, eta_a, eta_b = 0.53, 0.4, 0.73
        c, s = np.cosh(2 * r), np.sinh(2 * r)
        V = np.array([[c, s, 0, 0], [s, c, 0, 0],
                      [0, 0, c, -s], [0, 0, -s, c]])
        eta = np.array([eta_a, eta_b, eta_a, eta_b])
        V = V * np.sqrt(eta[:, None] * eta[None, :]) + np.diag(1.0 - eta)
        t = np.tanh(r)**2
        p00 = (1.0 - t) / (1.0 - t * (1.0 - eta_a) * (1.0 - eta_b))
        no_a = (1.0 - t) / (1.0 - t * (1.0 - eta_a))
        no_b = (1.0 - t) / (1.0 - t * (1.0 - eta_b))
        expected = [p00, no_b - p00, no_a - p00, 1.0 - no_a - no_b + p00]
        np.testing.assert_allclose(ThresholdModel(V).full_distribution(), expected, atol=2e-14)

    def test_marginal_consistency_and_order(self):
        V = covariance_from_interferometer(haar_unitary(5, np.random.default_rng(4)),
                                          [0.2, 0.4, 0.3, 0.6, 0.7], loss=0.65)
        model = ThresholdModel(V)
        full = model.full_distribution()
        for modes in ((1, 3), (3, 1), (4, 0, 2), (2,)):
            expected = np.zeros(1 << len(modes))
            for mask, probability in enumerate(full):
                submask = sum(((mask >> mode) & 1) << j for j, mode in enumerate(modes))
                expected[submask] += probability
            np.testing.assert_allclose(model.marginal(modes), expected, atol=2e-14)

    def test_physical_mode_permutation(self):
        V = covariance_from_interferometer(haar_unitary(4, np.random.default_rng(9)),
                                          [0.3, 0.4, 0.6, 0.8], loss=0.8)
        permutation = (2, 0, 3, 1)
        indices = np.array(permutation + tuple(j + 4 for j in permutation))
        permuted = ThresholdModel(V[np.ix_(indices, indices)]).full_distribution()
        original_in_permuted_order = ThresholdModel(V).marginal(permutation)
        np.testing.assert_allclose(permuted, original_in_permuted_order, atol=2e-14)

    def test_invalid_inputs_and_immutability(self):
        with self.assertRaises(ValueError):
            ThresholdModel(0.5 * np.eye(4))
        with self.assertRaises(ValueError):
            covariance_from_interferometer(np.eye(2), 0.5, 1.1)
        model = ThresholdModel(np.eye(4))
        with self.assertRaises(ValueError):
            model.marginal((0, 0))
        with self.assertRaises(ValueError):
            model.full_distribution()[0] = 0

    def test_bounded_vacuum_cache_eviction_invariance(self):
        V = covariance_from_interferometer(haar_unitary(5, np.random.default_rng(44)),
                                          [0.2, 0.4, 0.3, 0.6, 0.7], loss=0.65)
        unlimited = ThresholdModel(V)
        bounded = ThresholdModel(V, vacuum_cache_size=3)
        disabled = ThresholdModel(V, vacuum_cache_size=0)
        expected = unlimited.full_distribution()
        np.testing.assert_array_equal(bounded.full_distribution(), expected)
        np.testing.assert_array_equal(disabled.full_distribution(), expected)
        # Recomputing an evicted vacuum marginal changes only work, not value.
        for subset in ((0,), (1,), (2,), (3,), (4,), (0,)):
            self.assertEqual(bounded.vacuum_probability(subset),
                             unlimited.vacuum_probability(subset))
        info = bounded.cache_info()
        self.assertGreater(info["vacuum_cache_evictions"], 0)
        self.assertLessEqual(info["vacuum_cache_peak_entries"], 3)
        self.assertLessEqual(info["vacuum_cache_entries"], 3)
        self.assertEqual(disabled.cache_info()["vacuum_cache_peak_entries"], 0)
        self.assertEqual(disabled.cache_info()["vacuum_cache_entries"], 0)
        with self.assertRaises(ValueError):
            ThresholdModel(V, vacuum_cache_size=-1)


if __name__ == "__main__":
    unittest.main()
