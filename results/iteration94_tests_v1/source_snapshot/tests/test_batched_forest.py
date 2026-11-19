"""Independent event identities, ABI/resource contracts and work-count gates.

Numerical refinement budgets and canonical-history tests are independently
covered in test_batched_forest_numerics.py.  Tests never compile native code.
"""
from pathlib import Path
import os
import unittest

import numpy as np

from gbslab.batched_forest import (
    BatchedForestKernel, BatchedForestSampler, DEFAULT_LIBRARY,
    install_canonical_selector,
)
from gbslab.exact import NumericalProbabilityError, ThresholdModel
from gbslab.gaussian import covariance_from_interferometer, haar_unitary
from gbslab.positive_completion import TableEventOracle
from gbslab.positive_forest import ForestCompletionSampler


LIBRARY = Path(os.environ.get("GBSLAB_BATCHED_FOREST_LIBRARY", DEFAULT_LIBRARY))


def event(probabilities, zeros=(), ones=()):
    return sum((np.longdouble(value) for mask, value in enumerate(probabilities)
        if all(not (mask >> j) & 1 for j in zeros)
        and all((mask >> j) & 1 for j in ones)), np.longdouble(0))


@unittest.skipUnless(LIBRARY.is_file(), "build the explicit batched forest native library first")
class BatchedForestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        m = 6
        cls.V = covariance_from_interferometer(haar_unitary(m, np.random.default_rng(280813)),
            np.linspace(.45, .9, m), np.linspace(.55, .9, m))
        cls.model = ThresholdModel(cls.V)
        cls.P = cls.model.full_distribution()

    def kernel(self, V=None, **kwargs):
        kernel = BatchedForestKernel(self.V if V is None else V, library=LIBRARY, **kwargs)
        self.addCleanup(kernel.close)
        return kernel

    def sampler(self, model=None, budget=2, **kwargs):
        sampler = BatchedForestSampler(self.model if model is None else model,
            budget, library=LIBRARY, **kwargs)
        self.addCleanup(sampler.close)
        return sampler

    def test_batch_all_masses_match_independent_complete_event_sums(self):
        order = (1, 3, 0, 4, 2, 5)
        zeros, retained, omitted, edges, mode = (3,), (4, 1), (0, 2), ((0, 2),), 5
        pz = event(self.P, zeros)
        expected = [event(self.P, zeros, retained) / pz]
        for outcome in (0, 1):
            z = zeros + ((mode,) if outcome == 0 else ())
            c = retained + ((mode,) if outcome else ())
            expected.append(event(self.P, z, c) / pz)
            expected.extend(event(self.P, z, c+(j,)) / pz for j in omitted)
            expected.extend(event(self.P, z, c+edge) / pz for edge in edges)
        kernel = self.kernel(order=order)
        for precision in ("longdouble", "quad"):
            batch = kernel.query(zeros, retained, mode, omitted, edges, precision=precision)
            np.testing.assert_allclose(batch["values"], expected, atol=2e-15, rtol=3e-9)
            self.assertTrue(np.all(batch["indicators"] >= 0))
            self.assertFalse(np.any(batch["exact_zero"]))
            self.assertEqual(batch["values"].dtype, np.dtype(np.longdouble))
            self.assertAlmostEqual(float(sum(e.value for e in batch["branches"])),
                                   float(batch["baseline"].value), places=16)

    def test_zero_one_two_omissions_reduce_to_exact_conditional(self):
        kernel = self.kernel(order=(1, 3, 0, 4, 2, 5))
        retained, mode = (1, 4), 5
        for zeros, omitted, edges in (((0, 2, 3), (), ()),
            ((2, 3), (0,), ()), ((3,), (0, 2), ((0, 2),))):
            result = kernel.probability(zeros, retained, mode, omitted, edges, precision="quad")
            truth = event(self.P, zeros, retained+omitted+(mode,)) / event(self.P, zeros, retained+omitted)
            self.assertAlmostEqual(result["click_probability"], float(truth), places=10)

    def test_each_retained_subset_shares_factorization_columns_and_sparse_blocks(self):
        kernel = self.kernel(1.8*np.eye(16))
        kernel.query((0,), (1, 2, 3), 7, (4, 5, 6), ((4, 5), (5, 6)))
        counters = kernel.cache_info()["longdouble"]
        self.assertEqual(counters["subsets"], 8)
        self.assertEqual(counters["subset_factorizations"], 7)
        self.assertEqual(counters["subset_cubic_work"], 8*(3+3*8+27))
        self.assertEqual(counters["shared_solve_columns"], 7*8)
        self.assertEqual(counters["sparse_schur_blocks"], 8*(1+2*3+2))
        self.assertEqual(counters["local_determinants"], 8*(1+2*3+2*2))
        self.assertEqual(counters["event_accumulations"], 8*(1+2*(1+3+2)))
        self.assertEqual(counters["zero_updates"], 1)
        self.assertEqual(counters["state_rebuilds"], 1)
        self.assertEqual(counters["output_conversions"], 13)
        self.assertEqual(kernel.cache_info()["quad"]["queries"], 0)

    def test_common_zero_scale_is_removed_before_probability_underflow(self):
        # P(Z=0) is about 4e-600, outside binary64, while the desired current
        # conditional remains ordinary.  There are no positive constraints.
        variances = np.array([1e300, 1e300, 1.8])
        V = np.diag(np.r_[variances, variances])
        kernel = self.kernel(V)
        result = kernel.probability((0, 1), (), 2)
        self.assertEqual(result["baseline"].value, 1)
        self.assertAlmostEqual(result["click_probability"], .8/2.8, places=15)
        self.assertEqual(kernel.cache_info()["longdouble"]["zero_updates"], 2)

    def test_q_is_formed_after_exact_widening_of_binary64_input(self):
        variance = np.nextafter(1., 2.)
        # The tempting binary64 (variance+1)/2 expression rounds to 1 here.
        self.assertEqual((variance+1)/2, 1)
        kernel = self.kernel(variance*np.eye(2))
        result = kernel.probability((), (), 0, precision="quad")
        expected = float((np.longdouble(variance)-1)/(np.longdouble(variance)+1))
        self.assertGreater(result["click_probability"], 0)
        self.assertAlmostEqual(result["click_probability"] / expected, 1, places=14)
        self.assertFalse(np.any(result["exact_zero"]))

    def test_exact_isolated_vacuum_certificates_are_per_required_click_event(self):
        variances = np.array([1., 1.75, 2.25, 1.5])
        kernel = self.kernel(np.diag(np.r_[variances, variances]))
        result = kernel.query((), (1,), 3, (0, 2), ((0, 2),), precision="quad")
        # Baseline, branch, first node(vacuum), second node, edge; then branch1.
        np.testing.assert_array_equal(result["exact_zero"], [0, 0, 1, 0, 1, 0, 1, 0, 1])
        self.assertTrue(np.all(result["values"][result["exact_zero"] != 0] == 0))
        self.assertTrue(np.all(result["indicators"][result["exact_zero"] != 0] == 0))
        with self.assertRaisesRegex(NumericalProbabilityError, "both forest scores"):
            kernel.probability((), (1,), 3, (0, 2), ((0, 2),), precision="quad")
        with self.assertRaisesRegex(NumericalProbabilityError, "baseline conditioning"):
            kernel.probability((), (0,), 3, precision="quad")

    def test_schur_numeric_identity_is_not_an_analytic_zero_certificate(self):
        # Dyadic pure two-mode squeezing: a^2-c^2=1.  Conditioning mode0 on
        # vacuum makes mode1 vacuum, but neither original mode is isolated.
        V = np.array([[1.25, .75, 0, 0], [.75, 1.25, 0, 0],
                      [0, 0, 1.25, -.75], [0, 0, -.75, 1.25]])
        kernel = self.kernel(V)
        result = kernel.query((0,), (), 1, precision="quad")
        self.assertFalse(np.any(result["exact_zero"]))
        with self.assertRaises(NumericalProbabilityError):
            kernel.probability((0,), (), 1)
        self.assertEqual(kernel.cache_info()["auto_refinement_failures"], 1)

    def test_vacuum_law_and_sample_are_deterministic_without_epsilon(self):
        sampler = self.sampler(ThresholdModel(np.eye(8)), budget=0)
        law = sampler.enumerate_law()
        self.assertEqual(law["failure_mass"], 0)
        np.testing.assert_array_equal(law["distribution"], [1]+[0]*15)
        samples = sampler.sample(4, np.random.default_rng(280817))
        np.testing.assert_array_equal(samples, np.zeros((4, 4), dtype=np.uint8))
        self.assertEqual(sampler.kernel.cache_info()["auto_refinements"], 0)

    def test_kernel_arrays_are_readonly_independent_and_handle_lifetime_is_explicit(self):
        kernel = self.kernel()
        first = kernel.query((), (), 5)
        preserved = first["values"].copy()
        with self.assertRaises(ValueError):
            first["values"][0] = 0
        kernel.query((0, 1), (), 5, precision="quad")
        np.testing.assert_array_equal(first["values"], preserved)
        self.assertEqual(kernel.cache_info()["formats"]["quad_mantissa_bits"], 113)
        kernel.close()
        self.assertTrue(kernel.cache_info()["closed"])
        with self.assertRaisesRegex(RuntimeError, "closed"):
            kernel.query((), (), 5)

    def test_native_argument_errors_do_not_invoke_refinement(self):
        kernel = self.kernel()
        invalid = (
            lambda: kernel.probability((0,), (0,), 5),
            lambda: kernel.probability((), (), 5, (0, 1, 2, 3), ((0, 1), (1, 2), (2, 0))),
            lambda: kernel.probability((), (), 5, (0, 1), ((0, 2),)),
            lambda: kernel.probability((), (), 5, precision="arbitrary"),
            lambda: kernel.probability((), (), 5, max_relative_roundoff=0),
        )
        for call in invalid:
            with self.assertRaises(ValueError):
                call()
        self.assertEqual(kernel.cache_info()["auto_refinements"], 0)
        self.assertEqual(kernel.cache_info()["longdouble"]["queries"], 0)
        large = self.kernel(2*np.eye(36))
        with self.assertRaises(ValueError):
            large.probability((), tuple(range(17)), 17)

    def test_constructor_validation_and_explicit_missing_library(self):
        for covariance in (np.zeros((0, 0)), np.ones((2, 3)), [[1., np.nan], [np.nan, 1.]],
                           [[1., .1], [.2, 1.]], 1j*np.eye(2)):
            with self.assertRaises(ValueError):
                BatchedForestKernel(covariance, library=LIBRARY)
        with self.assertRaises(FileNotFoundError):
            BatchedForestKernel(np.eye(2), library=LIBRARY.parent / "missing-test-only.so")
        with self.assertRaises(ValueError):
            self.sampler(precision="arbitrary")
        with self.assertRaises(ValueError):
            self.kernel(order=(0, 1, 2))

    def test_independent_table_adapter_is_only_installed_before_evaluation(self):
        reference = ForestCompletionSampler(self.model, 2, forest="static_mi_tree",
            selection="click_influence", event_oracle=TableEventOracle(self.P))
        self.assertTrue(install_canonical_selector(reference))
        actual = self.sampler(forest="static_mi_tree")
        prefix = (1, 0, 1, 1, 1)
        first, second = reference.conditional_details(prefix), actual.conditional_details(prefix)
        self.assertEqual(first["retained_clicks"], second["retained_clicks"])
        self.assertEqual(first["forest_edges"], second["forest_edges"])
        self.assertAlmostEqual(first["click_probability"], second["click_probability"], places=11)
        with self.assertRaises(ValueError):
            install_canonical_selector(reference)


if __name__ == "__main__":
    unittest.main()
