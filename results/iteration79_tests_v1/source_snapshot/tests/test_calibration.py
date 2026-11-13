"""Checks of exact first-event calibration independent of Gaussian formulas."""

import unittest

import numpy as np

from gbslab.calibration import CalibratedSurrogate, first_click_categories
from gbslab.surrogates import BoundedParentSurrogate, distribution_metrics, fit_surrogate


class ExplicitOracle:
    def __init__(self, p):
        self.p = np.asarray(p, dtype=float)
        self.m = int(np.log2(len(p)))
        self.vacuum_calls = []
        self.marginal_calls = []

    def marginal(self, modes):
        self.marginal_calls.append(tuple(modes))
        p = np.zeros(1 << len(modes))
        for mask, probability in enumerate(self.p):
            local_mask = sum(((mask >> mode) & 1) << i for i, mode in enumerate(modes))
            p[local_mask] += probability
        return p

    def vacuum_probability(self, modes):
        self.vacuum_calls.append(tuple(modes))
        return sum(
            probability for mask, probability in enumerate(self.p)
            if all(((mask >> mode) & 1) == 0 for mode in modes)
        )


class CalibrationTests(unittest.TestCase):
    def test_category_masses_and_kl_projection_identity(self):
        p = np.random.default_rng(674).dirichlet(np.full(32, 0.4))
        oracle = ExplicitOracle(p)
        order = (3, 0, 4, 1, 2)
        for strategy in ("independent", "pairwise", "greedy_cmi"):
            with self.subTest(strategy=strategy):
                fit = fit_surrogate(oracle, 1, strategy, order)
                prior_marginal_calls = len(oracle.marginal_calls)
                prior_vacuum_calls = len(oracle.vacuum_calls)
                calibrated = CalibratedSurrogate(oracle, fit)
                self.assertEqual(len(oracle.marginal_calls), prior_marginal_calls)
                self.assertEqual(len(oracle.vacuum_calls) - prior_vacuum_calls, oracle.m)
                self.assertEqual(oracle.vacuum_calls[-oracle.m:], [order[:i] for i in range(1, oracle.m + 1)])
                q = fit.full_distribution()
                qs = calibrated.full_distribution()
                categories = first_click_categories(oracle.m, order)
                pcat = np.bincount(categories, weights=p, minlength=oracle.m + 1)
                qcat = np.bincount(categories, weights=q, minlength=oracle.m + 1)
                qscat = np.bincount(categories, weights=qs, minlength=oracle.m + 1)
                np.testing.assert_allclose(calibrated.target_category_probabilities, pcat, atol=2e-15)
                np.testing.assert_allclose(calibrated.base_category_probabilities, qcat, atol=2e-15)
                np.testing.assert_allclose(qscat, pcat, atol=2e-15)
                improvement = distribution_metrics(p, q)["kl"] - distribution_metrics(p, qs)["kl"]
                self.assertAlmostEqual(improvement, distribution_metrics(pcat, qcat)["kl"], places=13)
                self.assertAlmostEqual(improvement, calibrated.diagnostics["predicted_kl_reduction_nats"], places=13)
                self.assertGreaterEqual(improvement, -1e-14)

    def test_sampler_matches_the_calibrated_distribution(self):
        p = np.random.default_rng(984).dirichlet(np.ones(16))
        oracle = ExplicitOracle(p)
        fit = fit_surrogate(oracle, 1, "pairwise", order=(2, 0, 3, 1))
        calibrated = CalibratedSurrogate(oracle, fit)
        a = calibrated.sample(60_000, np.random.default_rng(472))
        b = calibrated.sample(60_000, np.random.default_rng(472))
        np.testing.assert_array_equal(a, b)
        masks = (a.astype(np.intp) * (1 << np.arange(oracle.m))).sum(axis=1)
        empirical = np.bincount(masks, minlength=1 << oracle.m) / len(a)
        self.assertLess(distribution_metrics(calibrated.full_distribution(), empirical)["tv"], 0.012)
        self.assertEqual(calibrated.sample(0, np.random.default_rng(1)).shape, (0, oracle.m))

    def test_zero_categories_do_not_introduce_artificial_support(self):
        p = np.zeros(16)
        p[0], p[15] = 0.8, 0.2
        oracle = ExplicitOracle(p)
        fit = fit_surrogate(oracle, 1, "previous")
        calibrated = CalibratedSurrogate(oracle, fit)
        np.testing.assert_allclose(calibrated.full_distribution(), p, atol=1e-15)
        self.assertEqual(calibrated.diagnostics["zero_target_categories"], 3)
        self.assertEqual(calibrated.diagnostics["zero_base_categories"], 3)
        vacuum = ExplicitOracle(np.array([1.0, 0.0, 0.0, 0.0]))
        vacuum_fit = fit_surrogate(vacuum, 0, "independent")
        vacuum_calibrated = CalibratedSurrogate(vacuum, vacuum_fit)
        np.testing.assert_array_equal(vacuum_calibrated.sample(50, np.random.default_rng(9)), np.zeros((50, 2)))

    def test_support_mismatch_is_explicitly_rejected(self):
        oracle = ExplicitOracle(np.array([0.5, 0.5]))
        impossible_base = BoundedParentSurrogate(
            m=1, order=(0,), parents=((),), conditional_tables=(np.array([0.0]),), diagnostics={}
        )
        with self.assertRaisesRegex(ValueError, "positive mass"):
            CalibratedSurrogate(oracle, impossible_base)


if __name__ == "__main__":
    unittest.main()
