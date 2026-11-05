"""Independent numerical/state gates for the native batched forest backend.

No test compiles a library.  Set GBSLAB_BATCHED_FOREST_LIBRARY to the exact
already-built .so for a required run; a missing explicit path is an error.
Without an override, an absent default library skips this optional suite.

The dyadic thermal fixtures have closed-form probabilities and are fixed from
their cancellation scale, not fitted to a panel or to observed samples.
High precision refines deterministic evaluation, never sampling attempts.
"""

from fractions import Fraction
import hashlib
import importlib
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from gbslab.exact import NumericalProbabilityError, ThresholdModel
from gbslab.gaussian import covariance_from_interferometer, haar_unitary
from gbslab.positive_completion import PositiveCompletionSamplingError, TableEventOracle
from gbslab.positive_forest import ForestCompletionSampler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIBRARY = ROOT / ".cache/batched-forest-native/libbatched_forest.so"
LIBRARY_ENVIRONMENT = "GBSLAB_BATCHED_FOREST_LIBRARY"
VALIDATION_CHECKS = []


def record(name, **details):
    VALIDATION_CHECKS.append({"name": name, "passed": True, "details": details})


def thermal_fixture(modes=8, exponent=8):
    """V=(1+2^-b)I has independent click probability 1/(2^(b+1)+1).

    For b=8 and seven clicked constraints the event is about 2^-63:
    long-double inclusion-exclusion is stressed while binary128 has ample
    precision.  b=40 instead requires about 287 bits for the same event.
    Both input matrices are exactly representable in binary64.
    """
    excess = np.ldexp(1.0, -exponent)
    covariance = (1.0 + excess) * np.eye(2 * modes)
    assert Fraction(float(covariance[0, 0])) - 1 == Fraction(1, 1 << exponent)
    return ThresholdModel(covariance), Fraction(1, (1 << (exponent + 1)) + 1)


def gaussian_fixture(modes=6):
    rng = np.random.default_rng(180701 + modes)
    covariance = covariance_from_interferometer(
        haar_unitary(modes, rng), np.linspace(.45, .9, modes),
        np.linspace(.55, .9, modes))
    return ThresholdModel(covariance)


class SequenceRng:
    def __init__(self, values):
        self.values, self.calls = iter(values), 0

    def random(self):
        self.calls += 1
        return next(self.values)


def mathematical_record(details):
    """Exclude timers/counters while retaining computed probabilities/guards."""
    keys = (
        "mode", "zeros", "clicks", "retained_clicks", "omitted_clicks",
        "forest_edges", "baseline_probability", "branch_masses",
        "baseline_probabilities", "node_probabilities", "edge_probabilities",
        "log_scores_up_to_common_baseline", "log_score_indicators",
        "zero_score_reasons", "click_probability", "precision_used",
        "refinement_attempted",
    )
    return {key: details[key] for key in keys if key in details}


class BatchedForestNumericsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        explicit = os.environ.get(LIBRARY_ENVIRONMENT)
        cls.library = Path(explicit).expanduser().resolve() if explicit else DEFAULT_LIBRARY
        if not cls.library.is_file():
            if explicit:
                raise FileNotFoundError(f"explicit {LIBRARY_ENVIRONMENT} is missing: {cls.library}")
            raise unittest.SkipTest("build the native batched-forest library explicitly first")
        # An installed library with missing/broken Python glue is an error,
        # not a reason to convert a required numerical gate into a skip.
        cls.module = importlib.import_module("gbslab.batched_forest")
        cls.Sampler = cls.module.BatchedForestSampler
        cls.Kernel = cls.module.BatchedForestKernel
        cls.library_sha256 = hashlib.sha256(cls.library.read_bytes()).hexdigest()
        record("loaded_library_binding", path=str(cls.library),
               sha256=cls.library_sha256, explicit_path=bool(explicit))

    @classmethod
    def tearDownClass(cls):
        if hashlib.sha256(cls.library.read_bytes()).hexdigest() != cls.library_sha256:
            raise AssertionError("the tested native library changed during validation")

    def sampler(self, model, budget, **kwargs):
        sampler = self.Sampler(model, budget, library=self.library, **kwargs)
        self.addCleanup(sampler.kernel.close)
        return sampler

    def test_dyadic_thermal_auto_refines_once_without_reselecting_structure(self):
        model, exact = thermal_fixture()
        prefix = (1,) * 7
        options = dict(selection="recent", forest="chain", cache_size=8)
        low = self.sampler(model, 4, precision="longdouble", **options)
        with self.assertRaises(NumericalProbabilityError):
            low.conditional_click_probability(prefix)

        high = self.sampler(model, 4, precision="quad", **options)
        high_details = high.conditional_details(prefix)
        automatic = self.sampler(model, 4, precision="auto", **options)
        before = automatic.kernel.cache_info()
        with patch.object(automatic._selector, "_retained_clicks",
                          wraps=automatic._selector._retained_clicks) as select, \
                patch.object(automatic, "_choose_edges", wraps=automatic._choose_edges) as edges:
            details = automatic.conditional_details(prefix)
        after = automatic.kernel.cache_info()
        self.assertEqual(select.call_count, 1)
        self.assertEqual(edges.call_count, 1)
        self.assertEqual(details["retained_clicks"], (3, 4, 5, 6))
        self.assertEqual(details["forest_edges"], ((0, 1), (1, 2)))
        self.assertEqual(details["click_probability"], high_details["click_probability"])
        self.assertAlmostEqual(details["click_probability"], float(exact), delta=4e-13)
        self.assertTrue(details["refinement_attempted"])
        self.assertEqual(details["precision_used"], "quad")
        self.assertTrue(details["first_error"])
        self.assertEqual(after["auto_refinements"] - before["auto_refinements"], 1)
        self.assertEqual(after["auto_refinement_successes"] - before["auto_refinement_successes"], 1)
        self.assertEqual(after["auto_refinement_failures"] - before["auto_refinement_failures"], 0)
        self.assertEqual(automatic.conditional_details(prefix), details)
        self.assertEqual(automatic.kernel.cache_info()["auto_refinements"], after["auto_refinements"])
        record("thermal_real_refinement", covariance="(1+2^-8) I", modes=8,
               budget=4, exact_click_probability=str(exact),
               actual_click_probability=details["click_probability"],
               first_error=details["first_error"], attempts=2,
               retained=list(details["retained_clicks"]),
               edges=[list(edge) for edge in details["forest_edges"]])

    def test_quad_budget_exhaustion_is_not_a_zero_certificate_or_global_budget(self):
        model, exact = thermal_fixture(exponent=40)
        self.assertGreater(float(exact), 0)
        self.assertLess(float(exact), 1)
        for precision in ("quad", "auto"):
            with self.subTest(precision=precision):
                sampler = self.sampler(model, 4, selection="recent", precision=precision,
                                       cache_size=0)
                for _ in range(2):
                    with self.assertRaises(NumericalProbabilityError):
                        sampler.conditional_click_probability((1,) * 7)
                info = sampler.kernel.cache_info()
                self.assertEqual(info["auto_refinements"], 2 if precision == "auto" else 0)
                self.assertEqual(info["auto_refinement_successes"], 0)
                self.assertEqual(info["auto_refinement_failures"], 2 if precision == "auto" else 0)
        record("positive_thermal_event_beyond_quad_budget_fails",
               covariance="(1+2^-40) I", modes=8, budget=4,
               exact_click_probability=str(exact), repeated_attempts=2,
               zero_certificate_claimed=False)

    def test_real_numerical_failure_consumes_no_extra_uniform_or_refill(self):
        model, _ = thermal_fixture(exponent=40)
        sampler = self.sampler(model, 4, selection="recent", precision="auto", cache_size=0)
        rng = SequenceRng([0.] * 16)
        with self.assertRaises(PositiveCompletionSamplingError) as caught:
            sampler.sample(2, rng)
        failure = caught.exception
        self.assertEqual(failure.sample_index, 0)
        self.assertEqual(failure.successful_samples.shape, (0, 8))
        self.assertGreater(len(failure.failed_prefix), 0)
        self.assertLess(len(failure.failed_prefix), 8)
        self.assertEqual(failure.failed_prefix, (1,) * len(failure.failed_prefix))
        self.assertEqual(rng.calls, len(failure.failed_prefix))
        self.assertGreater(sampler.kernel.cache_info()["auto_refinement_failures"], 0)
        record("real_failure_stops_before_uniform", failed_prefix=list(failure.failed_prefix),
               rng_calls=rng.calls, completed_outputs=0, requested_outputs=2)

    def test_numerical_failure_mass_is_kept_in_extended_law(self):
        model, exact = thermal_fixture(modes=6)
        sampler = self.sampler(model, 4, selection="recent", precision="longdouble", cache_size=0)
        law = sampler.enumerate_law()
        self.assertGreater(law["failure_mass"], 0)
        self.assertGreater(law["failure_count"], 0)
        self.assertEqual(law["output_space"], "binary_patterns_plus_failure")
        self.assertFalse(law["conditional_on_success"])
        self.assertAlmostEqual(sum(item["mass"] for item in law["failures"]),
                               law["failure_mass"], delta=1e-16)
        self.assertAlmostEqual(float(np.sum(law["extended_distribution"], dtype=np.longdouble)),
                               1., delta=5e-15)
        # This ordinary, successful atom must retain its original chain mass;
        # renormalizing successes after a rare failure would move it.
        expected_zero_atom = float((1 - exact) ** model.m)
        self.assertAlmostEqual(law["distribution"][0], expected_zero_atom, delta=2e-13)
        for failure in law["failures"]:
            prefix = failure["prefix"]
            masks = [mask for mask in range(1 << model.m)
                     if all(((mask >> sampler.order[j]) & 1) == bit
                            for j, bit in enumerate(prefix))]
            np.testing.assert_array_equal(law["distribution"][masks], np.zeros(len(masks)))
        with self.assertRaises(NumericalProbabilityError):
            sampler.full_distribution()
        record("extended_law_preserves_failure_mass", failure_mass=law["failure_mass"],
               failure_count=law["failure_count"], zero_atom=law["distribution"][0],
               analytic_zero_atom=expected_zero_atom, success_renormalization=False)

    def test_final_binary64_endpoint_is_not_an_analytic_branch(self):
        # Both branches are strictly positive, but p1 rounds to one in
        # binary64.  More accurate event evaluation cannot fix that API limit.
        model = ThresholdModel(np.ldexp(1., 60) * np.eye(2))
        for precision in ("longdouble", "quad", "auto"):
            with self.subTest(precision=precision):
                sampler = self.sampler(model, 0, selection="recent", precision=precision)
                with self.assertRaises(NumericalProbabilityError):
                    sampler.conditional_click_probability(())
                if precision == "auto":
                    info = sampler.kernel.cache_info()
                    self.assertEqual(info["auto_refinements"], 1)
                    self.assertEqual(info["auto_refinement_failures"], 1)
        record("positive_branches_binary64_endpoint_remains_failure",
               covariance="2^60 I", modes=1)

    def test_history_order_and_cache_capacity_do_not_change_conditional(self):
        model = gaussian_fixture(8)
        order = (6, 2, 0, 7, 3, 5, 1, 4)
        targets = ((1, 0, 1, 1, 0, 1, 1), (0, 1, 1, 1, 1, 0, 1), (1,) * 7)
        warmups = ((), (0,), (0, 1), (0, 1, 0), (1,),
                   (1, 1, 0, 1), targets[1], targets[0], targets[2])
        comparisons = 0
        for precision in ("longdouble", "quad", "auto"):
            for forest in ("chain", "static_mi_tree"):
                options = dict(order=order, forest=forest, precision=precision,
                               selection="click_influence")
                expected = {}
                for prefix in targets:
                    cold = self.sampler(model, 2, cache_size=0, **options)
                    expected[prefix] = mathematical_record(cold.conditional_details(prefix))
                for cache_size in (0, 1, 32):
                    warm = self.sampler(model, 2, cache_size=cache_size, **options)
                    for prefix in warmups + tuple(reversed(warmups)):
                        warm.conditional_click_probability(prefix)
                    for prefix in targets + tuple(reversed(targets)):
                        with self.subTest(precision=precision, forest=forest,
                                          cache_size=cache_size, prefix=prefix):
                            self.assertEqual(mathematical_record(warm.conditional_details(prefix)),
                                             expected[prefix])
                            comparisons += 1
        record("canonical_state_history_order_cache_invariance", order=list(order),
               precisions=["longdouble", "quad", "auto"], capacities=[0, 1, 32],
               exact_conditional_comparisons=comparisons)

    def test_same_uniform_stream_after_refinement_with_cold_and_warm_caches(self):
        model, exact = thermal_fixture()
        rows = np.array([[0.] * 8, [.99, 0., 0., .99, 0., .99, 0., 0.], [0.] * 8])
        expected = (rows < float(exact)).astype(np.uint8)
        baseline_trace = None
        refinements = {}
        for capacity in (0, 1, 128):
            sampler = self.sampler(model, 4, selection="recent", precision="auto",
                                   cache_size=capacity)
            original = sampler.conditional_click_probability
            trace = []

            def observed(prefix):
                probability = original(prefix)
                trace.append((tuple(prefix), probability))
                return probability

            rng = SequenceRng(rows.flat)
            with patch.object(sampler, "conditional_click_probability", side_effect=observed):
                samples = sampler.sample(len(rows), rng)
            np.testing.assert_array_equal(samples, expected)
            self.assertEqual(rng.calls, rows.size)
            if baseline_trace is None:
                baseline_trace = trace
            else:
                self.assertEqual(trace, baseline_trace)
            refinements[str(capacity)] = sampler.kernel.cache_info()["auto_refinements"]
            self.assertGreater(refinements[str(capacity)], 0)
        record("same_uniform_refinement_cache_invariance", samples=len(rows), modes=8,
               conditional_trace_entries=len(baseline_trace), refinements=refinements,
               rng_calls_per_run=int(rows.size))

    def test_gaussian_table_crosscheck_uses_identical_retained_set_and_edges(self):
        model = gaussian_fixture(6)
        target = model.full_distribution()
        order = (4, 0, 5, 2, 1, 3)
        maximum_conditional_difference = 0.
        maximum_law_tv = 0.
        conditionals = 0
        with patch.object(model, "full_distribution", side_effect=AssertionError("no target access")):
            for forest in ("chain", "static_mi_tree"):
                for budget in (0, 2, 4):
                    native = self.sampler(model, budget, forest=forest, selection="click_influence",
                                          order=order, precision="auto", cache_size=128)
                    # The reference receives the native run's already selected
                    # R/E, so only its event arithmetic is compared here.
                    table = ForestCompletionSampler(model, budget, forest=forest,
                        selection="recent", order=order, cache_size=0,
                        event_oracle=TableEventOracle(target))
                    retained_by_query, edges_by_omitted = {}, {}
                    for length in range(model.m):
                        for mask in range(1 << length):
                            prefix = tuple((mask >> j) & 1 for j in range(length))
                            actual = native.conditional_details(prefix)
                            retained_by_query[(tuple(actual["clicks"]), actual["mode"])] = actual["retained_clicks"]
                            edges_by_omitted[tuple(actual["omitted_clicks"])] = actual["forest_edges"]
                            with patch.object(table._selector, "_retained_clicks", return_value=actual["retained_clicks"]), \
                                    patch.object(table, "_choose_edges", return_value=actual["forest_edges"]):
                                reference = table.conditional_details(prefix)
                            self.assertEqual(actual["retained_clicks"], reference["retained_clicks"])
                            self.assertEqual(actual["forest_edges"], reference["forest_edges"])
                            difference = abs(actual["click_probability"] - reference["click_probability"])
                            self.assertLess(difference, 5e-9)
                            maximum_conditional_difference = max(maximum_conditional_difference, difference)
                            conditionals += 1
                    with patch.object(table._selector, "_retained_clicks",
                            side_effect=lambda clicks, mode: retained_by_query[(tuple(clicks), mode)]), \
                            patch.object(table, "_choose_edges",
                            side_effect=lambda omitted: edges_by_omitted[tuple(omitted)]):
                        reference_law = table.enumerate_law()
                    actual_law = native.enumerate_law()
                    self.assertEqual(actual_law["failure_mass"], 0)
                    self.assertEqual(reference_law["failure_mass"], 0)
                    tv = .5 * float(np.sum(np.abs(actual_law["extended_distribution"]
                                                   - reference_law["extended_distribution"])))
                    self.assertLess(tv, 2e-11)
                    maximum_law_tv = max(maximum_law_tv, tv)
        record("gaussian_table_same_R_E_full_law", modes=6, seed=180707,
               order=list(order), budgets=[0, 2, 4], forests=["chain", "static_mi_tree"],
               conditional_comparisons=conditionals,
               maximum_conditional_difference=maximum_conditional_difference,
               maximum_extended_law_tv=maximum_law_tv)

    def test_program_argument_and_resource_errors_are_not_numeric_failures(self):
        model = gaussian_fixture(3)
        for error_type in (RuntimeError, ValueError, MemoryError):
            for entrypoint in ("conditional", "sample", "enumerate"):
                with self.subTest(error_type=error_type.__name__, entrypoint=entrypoint):
                    sampler = self.sampler(model, 0, selection="recent", precision="auto")
                    failure = error_type("independent injected nonnumerical error")
                    rng = SequenceRng([.5] * 3)
                    with patch.object(sampler.kernel, "probability", side_effect=failure):
                        with self.assertRaises(error_type) as caught:
                            if entrypoint == "conditional":
                                sampler.conditional_click_probability(())
                            elif entrypoint == "sample":
                                sampler.sample(1, rng)
                            else:
                                sampler.enumerate_law()
                    self.assertIs(caught.exception, failure)
                    self.assertEqual(rng.calls, 0)
                    self.assertEqual(sampler.kernel.cache_info()["auto_refinements"], 0)
        record("nonnumerical_errors_propagate", error_types=["RuntimeError", "ValueError", "MemoryError"],
               entrypoints=["conditional", "sample", "enumerate"], swallowed_errors=0)

    def test_numeric_error_retains_failure_symbol_and_does_not_draw(self):
        model = gaussian_fixture(3)
        sampler = self.sampler(model, 0, selection="recent", precision="auto")
        failure = NumericalProbabilityError("independent unresolved numeric event")
        rng = SequenceRng([.5] * 3)
        with patch.object(sampler.kernel, "probability", side_effect=failure):
            with self.assertRaises(PositiveCompletionSamplingError) as caught:
                sampler.sample(1, rng)
            self.assertEqual(caught.exception.failed_prefix, ())
            self.assertEqual(caught.exception.successful_samples.shape, (0, 3))
            self.assertEqual(rng.calls, 0)
            law = sampler.enumerate_law()
        np.testing.assert_array_equal(law["distribution"], np.zeros(8))
        self.assertEqual(law["failure_mass"], 1.)
        self.assertEqual(law["failure_count"], 1)
        self.assertFalse(law["conditional_on_success"])
        record("injected_numeric_failure_extends_output_space", failure_mass=1., rng_calls=0)


if __name__ == "__main__":
    unittest.main()
