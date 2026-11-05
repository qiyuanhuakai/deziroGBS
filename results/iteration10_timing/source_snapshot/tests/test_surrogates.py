"""Independent probability-model checks for bounded-parent surrogate fitting."""

import unittest

import numpy as np

from gbslab.surrogates import (
    distribution_metrics,
    fit_surrogate,
    kl_chain_decomposition,
)


class DenseOracle:
    """Small explicit joint distribution; no Gaussian implementation is reused."""

    def __init__(self, p):
        self.p = np.asarray(p, dtype=float)
        self.m = int(np.log2(len(self.p)))
        self.calls = []

    def marginal(self, modes):
        self.calls.append(tuple(modes))
        out = np.zeros(1 << len(modes))
        for global_mask, probability in enumerate(self.p):
            local_mask = sum(((global_mask >> mode) & 1) << i for i, mode in enumerate(modes))
            out[local_mask] += probability
        return out


def known_chain_distribution():
    """X0 -> X1 -> X3 and an independent X2, with asymmetric probabilities."""
    p = np.zeros(16)
    for mask in range(16):
        x0, x1, x2, x3 = [(mask >> i) & 1 for i in range(4)]
        probabilities = [0.3, [0.1, 0.85][x0], 0.6, [0.2, 0.9][x1]]
        p[mask] = np.prod([a if b else 1 - a for a, b in zip(probabilities, (x0, x1, x2, x3))])
    return p


class SurrogateTests(unittest.TestCase):
    def test_full_parent_chain_rule_reconstructs_target_under_permutation(self):
        p = np.random.default_rng(41).dirichlet(np.ones(16))
        for strategy in ("previous", "pairwise", "greedy_cmi"):
            with self.subTest(strategy=strategy):
                fitted = fit_surrogate(DenseOracle(p), 3, strategy, order=(2, 0, 3, 1))
                np.testing.assert_allclose(fitted.full_distribution(), p, atol=2e-15, rtol=2e-14)

    def test_conditional_information_identity_and_bounded_queries(self):
        p = np.random.default_rng(211).dirichlet(np.full(64, 0.6))
        for strategy in ("previous", "pairwise", "greedy_cmi"):
            with self.subTest(strategy=strategy):
                oracle = DenseOracle(p)
                fitted = fit_surrogate(oracle, 2, strategy, order=(5, 1, 3, 0, 4, 2))
                self.assertLessEqual(max(map(len, oracle.calls)), 3)
                self.assertEqual(fitted.diagnostics["largest_marginal_order"], 3)
                direct = distribution_metrics(p, fitted.full_distribution())
                chain = kl_chain_decomposition(p, fitted)
                self.assertAlmostEqual(direct["kl"], chain["kl_from_conditional_entropies"], places=13)
                self.assertGreaterEqual(min(chain["omitted_information_per_mode"]), -2e-14)

    def test_parity_exposes_failure_of_pairwise_information(self):
        # This is a synthetic probability example, not asserted to be a GBS state.
        p = np.zeros(8)
        p[[0, 3, 5, 6]] = 0.25
        for strategy in ("independent", "pairwise", "greedy_cmi"):
            fitted = fit_surrogate(DenseOracle(p), 1, strategy)
            np.testing.assert_allclose(fitted.full_distribution(), np.full(8, 1 / 8))
            metrics = distribution_metrics(p, fitted.full_distribution())
            self.assertAlmostEqual(metrics["tv"], 0.5)
            self.assertAlmostEqual(metrics["kl"], np.log(2))
        exact = fit_surrogate(DenseOracle(p), 2, "greedy_cmi")
        np.testing.assert_allclose(exact.full_distribution(), p, atol=1e-15)

    def test_known_markov_structure_and_reproducible_sampling(self):
        p = known_chain_distribution()
        fitted = fit_surrogate(DenseOracle(p), 1, "greedy_cmi")
        self.assertEqual(fitted.parents[1], (0,))
        self.assertEqual(fitted.parents[3], (1,))
        np.testing.assert_allclose(fitted.full_distribution(), p, atol=1e-15)
        a = fitted.sample(30_000, np.random.default_rng(214))
        b = fitted.sample(30_000, np.random.default_rng(214))
        np.testing.assert_array_equal(a, b)
        expected_means = np.array([DenseOracle(p).marginal((i,))[1] for i in range(4)])
        np.testing.assert_allclose(a.mean(axis=0), expected_means, atol=0.012)
        empirical_masks = (a.astype(np.intp) * (1 << np.arange(4))).sum(axis=1)
        empirical = np.bincount(empirical_masks, minlength=16) / len(a)
        self.assertLess(distribution_metrics(p, empirical)["tv"], 0.018)
        self.assertEqual(fitted.sample(0, np.random.default_rng(1)).shape, (0, 4))

    def test_zero_mass_parent_histories_remain_defined(self):
        p = np.zeros(16)
        p[0], p[15] = 0.7, 0.3
        fitted = fit_surrogate(DenseOracle(p), 2, "previous")
        self.assertGreater(fitted.diagnostics["undefined_conditional_rows"], 0)
        np.testing.assert_allclose(fitted.full_distribution(), p, atol=1e-15)
        metrics = distribution_metrics(np.array([0.5, 0.5]), np.array([1.0, 0.0]))
        self.assertEqual(metrics["kl"], np.inf)
        self.assertEqual(metrics["support_missed_mass"], 0.5)

    def test_validation_rejects_invalid_inputs(self):
        oracle = DenseOracle(np.full(8, 0.125))
        with self.assertRaises(ValueError):
            fit_surrogate(oracle, 1, order=(0, 0, 2))
        with self.assertRaises(ValueError):
            fit_surrogate(oracle, -1)
        with self.assertRaises(ValueError):
            distribution_metrics(np.array([0.8, 0.8]), np.array([0.5, 0.5]))


if __name__ == "__main__":
    unittest.main()
