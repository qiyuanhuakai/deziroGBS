"""Analytic physical counterexamples, separate from floating-point kernels."""
import unittest
from math import comb
import numpy as np

from gbslab.exact import ThresholdModel
from gbslab.paired_stress import (
    paired_covariance, measurement_order, recent_tv, dense_target,
    PairedThresholdOracle, sampler_crosscheck,
)


class PairedStressTests(unittest.TestCase):
    def test_covariance_matches_independent_pair_target(self):
        for pairs in (1, 2, 3, 4):
            for p in (.2, .5):
                target = dense_target(pairs, p)
                model = ThresholdModel(paired_covariance(pairs, p))
                np.testing.assert_allclose(model.full_distribution(), target,
                                           rtol=1e-10, atol=2e-13)

    def test_recent_formula_matches_dyadic_full_enumeration(self):
        for pairs in (2, 3, 4):
            for budget in (0, 1, 2):
                with self.subTest(pairs=pairs, budget=budget):
                    row = sampler_crosscheck(pairs, budget, oracle_kind="analytic")
                    self.assertEqual(row["status"], "ok")
                    self.assertLess(row["absolute_theorem_error"], 3e-14)

    def test_order_and_partner_selection_remove_the_obstruction(self):
        for pairs in (2, 3, 4):
            for selection, order in (("pairwise", "blocked"),
                                     ("recent", "interleaved"),
                                     ("pairwise", "interleaved")):
                with self.subTest(pairs=pairs, selection=selection, order=order):
                    row = sampler_crosscheck(pairs, 1, selection=selection,
                        order_kind=order, oracle_kind="analytic")
                    self.assertEqual(row["status"], "ok")
                    self.assertEqual(row["observed_tv"], 0.0)

    def test_zero_budget_product_formula_and_large_failure(self):
        for pairs in (1, 2, 8, 32):
            for p in (.05, .3, .5, .9):
                np.testing.assert_allclose(recent_tv(pairs, 0, p),
                    1-(1-p+p*p)**pairs, atol=1e-14, rtol=1e-12)
        self.assertGreater(recent_tv(128, 2, .5), 1-1e-12)
        self.assertEqual(recent_tv(5, 5, .5), 0.0)
        self.assertEqual(recent_tv(128, 2, 0.0), 0.0)

    def test_arbitrary_order_analytic_marginal(self):
        oracle = PairedThresholdOracle(3, .5)
        np.testing.assert_array_equal(oracle.marginal((4, 1)), [.5, 0, 0, .5])
        np.testing.assert_array_equal(oracle.marginal((4, 0)), [.25]*4)
        self.assertEqual(oracle.vacuum_probability((1, 4)), .5)
        self.assertEqual(oracle.vacuum_probability((1, 3)), .25)
        self.assertEqual(measurement_order(3, "interleaved"), (0, 3, 1, 4, 2, 5))

    def test_covariance_numeric_guard_is_reported_not_hidden(self):
        # Different BLAS implementations can produce different tiny residues
        # on exact-zero support. A valid run must match the theorem; a raised
        # guard must remain an explicit recorded failure, not a fake zero TV.
        row = sampler_crosscheck(3, 2, oracle_kind="covariance")
        self.assertLess(row["covariance_reference_tv"], 2e-13)
        if row["status"] == "ok":
            self.assertLess(row["absolute_theorem_error"], 2e-12)
        else:
            self.assertEqual(row["status"], "numerical_guard")
            self.assertIn("error", row)
            self.assertNotIn("observed_tv", row)

    def test_input_and_resource_guards(self):
        for p in (-.1, 1., float("nan")):
            with self.assertRaises(ValueError):
                paired_covariance(2, p)
        with self.assertRaises(ValueError):
            recent_tv(2, -1)
        with self.assertRaises(ValueError):
            dense_target(40)
        with self.assertRaises(ValueError):
            PairedThresholdOracle(2).marginal((0, 0))


if __name__ == "__main__":
    unittest.main()
