"""Independent full-native transport/sampling checks; never compile in tests."""
from fractions import Fraction
import hashlib
import os
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class CppForestIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        explicit = os.environ.get("GBSLAB_FOREST_EXECUTABLE")
        cls.binary = Path(explicit) if explicit else ROOT / ".cache/iteration8-forest-native/batched_forest_sampler"
        if not cls.binary.is_file():
            if explicit:
                raise FileNotFoundError(cls.binary)
            raise unittest.SkipTest("Build the full native forest executable explicitly")
        from gbslab.cpp_forest import run_cpp_forest, CppForestError
        cls.run_native = staticmethod(run_cpp_forest)
        cls.Error = CppForestError
        cls.digest = hashlib.sha256(cls.binary.read_bytes()).hexdigest()

    @classmethod
    def tearDownClass(cls):
        if hashlib.sha256(cls.binary.read_bytes()).hexdigest() != cls.digest:
            raise AssertionError("Native executable changed during integration gate")

    def run_case(self, V, **kwargs):
        return self.run_native(V, executable=self.binary, **kwargs)

    def test_analytic_thermal_uniforms_raw_outputs_logq_and_full_law(self):
        rates = np.array([1.125, 1.25, 1.5, 2.])
        V = np.diag(np.r_[rates, rates])
        p = (rates - 1) / (rates + 1)
        uniforms = np.array([[.01, .9, .02, .8], [.9, .01, .9, .02], [0., 0., 0., 0.], [.9, .9, .9, .9]])
        expected = (uniforms < p).astype(int)
        exact = np.array([np.prod([p[j] if mask >> j & 1 else 1 - p[j] for j in range(4)]) for mask in range(16)])
        for tree in ("chain", "static_mi_tree"):
            with self.subTest(tree=tree):
                result = self.run_case(V, max_clicks=1, forest=tree, samples=4,
                    uniforms=uniforms, raw_samples=True, record_log_probabilities=True,
                    trace_conditionals=True, enumerate_distribution=True)
                np.testing.assert_array_equal(result["raw_samples"], expected)
                np.testing.assert_allclose(result["distribution"], exact, rtol=0, atol=2e-13)
                self.assertEqual(result["failure_mass"], 0)
                np.testing.assert_allclose(result["sample_log_probabilities"],
                    np.sum(np.where(expected, np.log(p), np.log1p(-p)), axis=1), rtol=0, atol=2e-11)
                self.assertEqual(len(result["conditional_trace"]), 16)
                self.assertEqual(len(result["enumeration_trace"]), 15)
                self.assertEqual(result["completed_samples"], 4)
                self.assertAlmostEqual(result["timings_seconds"]["cold_batch"],
                    result["timings_seconds"]["setup"] + result["timings_seconds"]["sampling"], delta=1e-10)

    def test_original_isolated_vacuum_has_exact_success_support(self):
        result = self.run_case(np.eye(8), max_clicks=2, forest="chain", samples=3,
            seed=91, raw_samples=True, enumerate_distribution=True)
        self.assertEqual(result["failure_mass"], 0)
        np.testing.assert_array_equal(result["raw_samples"], np.zeros((3, 4), dtype=int))
        np.testing.assert_array_equal(result["distribution"], np.r_[1., np.zeros(15)])

    def test_failed_batch_preserves_completed_output_and_consumed_uniforms(self):
        # Three simultaneous clicks at this brightness need over 120 bits;
        # k=0 bypasses retained-selector cancellation and exercises the batch
        # kernel's own finite precision budget. First all-zero sample completes.
        V = (1 + np.ldexp(1., -40)) * np.eye(10)
        uniforms = np.array([[.5] * 5, [0.] * 5, [.5] * 5])
        with self.assertRaises(self.Error) as caught:
            self.run_case(V, max_clicks=0, forest="chain", samples=3, uniforms=uniforms,
                raw_samples=True, record_log_probabilities=True, trace_conditionals=True)
        payload = caught.exception.payload
        self.assertEqual(payload["completed_samples"], 1)
        np.testing.assert_array_equal(payload["raw_samples"], np.zeros((1, 5), dtype=int))
        self.assertEqual(len(payload["sample_log_probabilities"]), 1)
        self.assertGreaterEqual(payload["uniforms_consumed"], 5)
        self.assertLess(payload["uniforms_consumed"], 10)
        self.assertEqual(len(payload["conditional_trace"]), payload["uniforms_consumed"])

    def test_failed_enumeration_is_complete_subprobability_plus_failure(self):
        V = (1 + np.ldexp(1., -8)) * np.eye(12)
        result = self.run_case(V, max_clicks=4, forest="chain", precision="longdouble",
            samples=0, raw_samples=False, enumerate_distribution=True)
        q = np.asarray(result["distribution"])
        failure = result["failure_mass"]
        self.assertGreater(failure, 0)
        self.assertAlmostEqual(float(q.sum()) + failure, 1., delta=1e-13)
        self.assertAlmostEqual(sum(f["mass"] for f in result["failures"]), failure, delta=1e-15)
        exact_zero = float(Fraction(512, 513) ** 6)
        self.assertAlmostEqual(q[0], exact_zero, delta=2e-13)
        self.assertEqual(result["completed_samples"], 0)
        self.assertEqual(result["uniforms_consumed"], 0)

    def test_fixed_gaussian_native_python_decisions_and_independent_table_law(self):
        from gbslab.batched_forest import BatchedForestSampler
        from gbslab.exact import ThresholdModel
        from gbslab.gaussian import covariance_from_interferometer, haar_unitary
        library = Path(os.environ.get("GBSLAB_BATCHED_FOREST_LIBRARY",
                                      ROOT / ".cache/batched-forest-native/libbatched_forest.so"))
        if not library.is_file():
            if os.environ.get("GBSLAB_BATCHED_FOREST_LIBRARY"):
                raise FileNotFoundError(library)
            self.skipTest("Build the independent Python-interface native library explicitly")
        sys.path.insert(0, str(ROOT / "experiments"))
        try:
            from iteration8_common import TracedTableForest, extended_law, total_variation
        finally:
            sys.path.pop(0)
        V = covariance_from_interferometer(haar_unitary(6, np.random.default_rng(881001)),
                                           .6, loss=.35)
        P = ThresholdModel(V, max_exact_modes=6).full_distribution()
        uniforms = np.random.default_rng(881002).random((12, 6))
        for tree in ("chain", "static_mi_tree"):
            for k in (0, 2):
                with self.subTest(tree=tree, k=k):
                    native = self.run_case(V, max_clicks=k, forest=tree, samples=12,
                        uniforms=uniforms, raw_samples=True, record_log_probabilities=True,
                        trace_conditionals=True, enumerate_distribution=True)
                    self.assertEqual(native["failure_mass"], 0)
                    self.assertEqual(len(native["enumeration_trace"]), 63)
                    with BatchedForestSampler(ThresholdModel(V, max_exact_modes=6), k,
                            forest=tree, cache_size=0, library=library) as python:
                        qp, fp, _, _ = extended_law(python, 6)
                        self.assertLess(total_variation(native["distribution"], 0, qp, fp), 2e-12)
                        for row in native["enumeration_trace"]:
                            detail = python.conditional_details(tuple(row["prefix"]))
                            self.assertEqual(tuple(row["retained_clicks"]), tuple(sorted(detail["retained_clicks"])))
                            self.assertEqual(tuple(row["omitted_clicks"]), tuple(detail["omitted_clicks"]))
                            self.assertEqual(tuple(map(tuple, row["forest_edges"])), tuple(detail["forest_edges"]))
                            self.assertAlmostEqual(row["p_one"], detail["click_probability"], delta=2e-12)
                        if tree == "static_mi_tree":
                            np.testing.assert_allclose(native["static_pairwise_mi"], python.static_pairwise_mi,
                                                       rtol=0, atol=2e-13)
                    qt, ft, _, _ = extended_law(TracedTableForest(P, native["enumeration_trace"]), 6)
                    self.assertEqual(ft, 0)
                    self.assertLess(total_variation(native["distribution"], 0, qt, ft), 2e-11)

    def test_zero_static_mi_edges_are_retained_with_physical_index_ties(self):
        rates = np.array([1.125, 1.25, 1.5, 2.])
        V = np.diag(np.r_[rates, rates])
        for tree, expected in (("chain", [[0, 1], [1, 2]]),
                               ("static_mi_tree", [[0, 1], [0, 2]])):
            with self.subTest(tree=tree):
                result = self.run_case(V, max_clicks=0, forest=tree, samples=0,
                    raw_samples=False, trace_conditionals=True, enumerate_distribution=True)
                row = next(row for row in result["enumeration_trace"] if row["prefix"] == [1, 1, 1])
                self.assertEqual(row["retained_clicks"], [])
                self.assertEqual(row["omitted_clicks"], [0, 1, 2])
                self.assertEqual(row["forest_edges"], expected)
                if tree == "static_mi_tree":
                    np.testing.assert_array_equal(result["static_pairwise_mi"], np.zeros((4, 4)))


if __name__ == "__main__":
    unittest.main()
