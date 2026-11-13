"""Independent small-law and failure-boundary checks for positive completion."""

import unittest

import numpy as np

from gbslab.conditional_memory import ConditionalMemorySampler
from gbslab.exact import NumericalProbabilityError, ThresholdModel
from gbslab.gaussian import covariance_from_interferometer, haar_unitary
from gbslab.positive_completion import (
    EventEstimate,
    PositiveCompletionSampler,
    PositiveCompletionSamplingError,
    TableEventOracle,
)
from gbslab.zero_memory import ZeroMemorySampler
from positive_event_fixtures import ConditionalIndependentEvents


def event_from_table(p, zeros=(), ones=()):
    return sum((np.longdouble(value) for mask, value in enumerate(p)
                if all(not (mask >> j) & 1 for j in zeros)
                and all((mask >> j) & 1 for j in ones)), np.longdouble(0))


def target_next(p, order, prefix):
    zeros = tuple(order[j] for j, bit in enumerate(prefix) if not bit)
    ones = tuple(order[j] for j, bit in enumerate(prefix) if bit)
    return float(event_from_table(p, zeros, ones + (order[len(prefix)],))
                 / event_from_table(p, zeros, ones))


def independent_table(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.longdouble)
    return np.array([np.prod([p if (mask >> j) & 1 else 1 - p
                             for j, p in enumerate(probabilities)], dtype=np.longdouble)
                     for mask in range(1 << probabilities.size)], dtype=np.longdouble)


class AlteredOracle:
    """Inject faults/indicators without changing the table or the selector."""

    def __init__(self, p, transform):
        self.table = TableEventOracle(p)
        self.m = self.table.m
        self.transform = transform
        self.requests = []

    def event_probability(self, zeros, ones):
        key = (tuple(sorted(zeros)), tuple(sorted(ones)))
        self.requests.append(key)
        estimate = self.table.event_probability(zeros, ones)
        return self.transform(key, estimate)


class SequenceRng:
    def __init__(self, values):
        self.values = iter(values)
        self.calls = 0

    def random(self):
        self.calls += 1
        return next(self.values)


class PositiveCompletionTests(unittest.TestCase):
    @staticmethod
    def gaussian(m=6, *, seed=609, pure=False):
        rng = np.random.default_rng(seed)
        covariance = covariance_from_interferometer(haar_unitary(m, rng),
            np.linspace(.45, .9, m), 1.0 if pure else np.linspace(.55, .9, m))
        return ThresholdModel(covariance)

    def test_fixed_partition_and_formula_match_independent_dense_sums(self):
        p = np.random.default_rng(312).dirichlet(np.ones(128))
        order = (5, 2, 0, 4, 1, 6, 3)
        prefix = (1, 1, 1, 1, 1, 0)
        zeros, retained, mode = (6,), (4, 1), 3
        for completion in ("product", "pair_block"):
            for block_order in ("measurement", "physical"):
                sampler = PositiveCompletionSampler(TableEventOracle(p), 2,
                    order=order, completion=completion, block_order=block_order)
                details = sampler.conditional_details(prefix)
                omitted = (5, 2, 0) if block_order == "measurement" else (0, 2, 5)
                size = 1 if completion == "product" else 2
                blocks = tuple(omitted[j:j + size] for j in range(0, 3, size))
                self.assertEqual(details["retained_clicks"], retained)
                self.assertEqual(details["omitted_clicks"], omitted)
                self.assertEqual(details["blocks"], blocks)
                scores = []
                for value in (0, 1):
                    z = zeros + ((mode,) if value == 0 else ())
                    c = retained + ((mode,) if value else ())
                    branch = event_from_table(p, z, c)
                    score = branch / event_from_table(p, zeros, retained)
                    for block in blocks:
                        score *= event_from_table(p, z, c + block) / branch
                    scores.append(score)
                self.assertAlmostEqual(details["click_probability"],
                    float(scores[1] / sum(scores)), places=14)

    def test_exact_empty_or_single_block_boundaries(self):
        p = np.random.default_rng(42).dirichlet(np.full(64, 1.3))
        order = (4, 0, 5, 2, 1, 3)
        for completion, extra in (("product", 1), ("pair_block", 2)):
            for budget in (0, 2, 4):
                sampler = PositiveCompletionSampler(TableEventOracle(p), budget,
                    order=order, completion=completion)
                for length in range(6):
                    for mask in range(1 << length):
                        prefix = tuple((mask >> j) & 1 for j in range(length))
                        if sum(prefix) <= budget + extra:
                            self.assertAlmostEqual(sampler.conditional_click_probability(prefix),
                                                   target_next(p, order, prefix), places=13)
                if budget + extra >= 5:
                    np.testing.assert_allclose(sampler.full_distribution(), p,
                                               atol=2e-16, rtol=2e-14)

    def test_independent_modes_are_exact_for_orders_budgets_and_blocks(self):
        p = independent_table((.12, .21, .4, .61, .75, .87))
        for order in (tuple(range(6)), (5, 0, 2, 4, 1, 3)):
            for budget in (0, 2, 4):
                for completion in ("product", "pair_block"):
                    sampler = PositiveCompletionSampler(TableEventOracle(p), budget,
                        completion=completion, order=order)
                    np.testing.assert_allclose(sampler.full_distribution(), p,
                                               atol=3e-16, rtol=3e-14)

    def test_gaussian_and_table_event_oracles_agree_with_identical_selectors(self):
        model = self.gaussian()
        p = model.full_distribution()
        order = (4, 0, 5, 2, 1, 3)
        for selection in ("recent", "click_influence"):
            for budget in (0, 2, 4):
                for completion in ("product", "pair_block"):
                    options = dict(selection=selection, completion=completion, order=order)
                    direct = PositiveCompletionSampler(model, budget, **options)
                    table = PositiveCompletionSampler(model, budget,
                        event_oracle=TableEventOracle(p), **options)
                    law = direct.enumerate_law()
                    reference = table.enumerate_law()
                    self.assertEqual(law["failure_mass"], 0)
                    self.assertEqual(reference["failure_mass"], 0)
                    np.testing.assert_allclose(law["distribution"], reference["distribution"],
                                               atol=8e-13, rtol=4e-9)
                    for prefix in ((1, 1, 1, 0), (1, 1, 1, 1, 0)):
                        self.assertEqual(direct.conditional_details(prefix)["retained_clicks"],
                                         table.conditional_details(prefix)["retained_clicks"])
                    if budget == 4:
                        np.testing.assert_allclose(law["distribution"], p,
                                                   atol=8e-13, rtol=4e-9)
                    self.assertGreater(direct.cache_info()["event_backend"]["vacuum_probability_calls"], 0)

    def test_existing_selectors_are_reused_including_nonidentity_order(self):
        model = self.gaussian()
        order = (5, 2, 4, 0, 1, 3)
        prefix = (1, 1, 0, 1, 1)
        clicks, mode = (5, 2, 0, 1), 3
        for selection in ("recent", "pairwise", "click_influence"):
            cls = ConditionalMemorySampler if selection == "click_influence" else ZeroMemorySampler
            original = cls(model, 2, order=order, selection=selection)
            for completion in ("product", "pair_block"):
                sampler = PositiveCompletionSampler(model, 2, order=order,
                    selection=selection, completion=completion)
                self.assertEqual(sampler.conditional_details(prefix)["retained_clicks"],
                                 original._retained_clicks(clicks, mode))

    def test_table_analytic_zero_branches_and_nonadjacent_squeezed_pairs(self):
        # Analytic threshold law of independent pairs (0,3) and (1,2).
        p = np.zeros(16)
        for first in (0, 1):
            for second in (0, 1):
                mask = first * 9 + second * 6
                p[mask] = (.23 if first else .77) * (.41 if second else .59)
        for order in ((0, 1, 2, 3), (3, 0, 2, 1)):
            for completion in ("product", "pair_block"):
                sampler = PositiveCompletionSampler(TableEventOracle(p), 0,
                    order=order, completion=completion)
                np.testing.assert_allclose(sampler.full_distribution(), p, atol=2e-16)
                samples = sampler.sample(20, np.random.default_rng(11))
                np.testing.assert_array_equal(samples[:, 0], samples[:, 3])
                np.testing.assert_array_equal(samples[:, 1], samples[:, 2])
        p = np.array([0., .4, 0., .6])  # First outcome is analytically one.
        sampler = PositiveCompletionSampler(TableEventOracle(p), 0)
        self.assertEqual(sampler.conditional_click_probability(()), 1.)
        self.assertEqual(sampler.conditional_details(())["factor_probabilities"][0], ())
        self.assertEqual(sampler.conditional_details(())["zero_score_reasons"][0],
                         "analytic_zero_baseline_branch")
        np.testing.assert_allclose(sampler.full_distribution(), p, atol=1e-16)

    def test_gaussian_vacuum_is_analytic_but_tmsv_roundoff_is_not_a_zero_certificate(self):
        vacuum = PositiveCompletionSampler(ThresholdModel(np.eye(8)), 0)
        expected = np.zeros(16); expected[0] = 1
        np.testing.assert_array_equal(vacuum.full_distribution(), expected)
        self.assertGreater(vacuum.cache_info()["structural_zero_returns"], 0)
        self.assertEqual(vacuum.conditional_click_probability(()), 0.)
        c, s = np.cosh(1.1), np.sinh(1.1)
        V = np.array([[c, s, 0, 0], [s, c, 0, 0],
                      [0, 0, c, -s], [0, 0, -s, c]])
        for completion in ("product", "pair_block"):
            sampler = PositiveCompletionSampler(ThresholdModel(V), 0, completion=completion)
            law = sampler.enumerate_law()
            self.assertGreater(law["failure_mass"], 0)
            self.assertAlmostEqual(float(law["extended_distribution"].sum()), 1., places=14)
            self.assertTrue(any("zero" in f["error"] or "indicator" in f["error"]
                                for f in law["failures"]))

    def test_generic_pure_gaussian_crosscheck_does_not_request_full_target(self):
        model = self.gaussian(5, pure=True)
        before = model.cache_info()["marginal_evaluations"]
        direct = PositiveCompletionSampler(model, 2, completion="pair_block",
                                            selection="click_influence")
        law = direct.enumerate_law()
        self.assertEqual(model.cache_info()["marginal_evaluations"], before)
        self.assertEqual(law["failure_mass"], 0)
        p = model.full_distribution()
        # At m=5,k=2 the pair method has at most one omitted block throughout.
        np.testing.assert_allclose(law["distribution"], p, atol=8e-13, rtol=5e-9)

    def test_general_discrete_parity_can_have_false_support(self):
        p = np.zeros(16)
        for first_three in range(8):
            parity = first_three.bit_count() % 2
            p[first_three | (parity << 3)] = 1 / 8
        for completion in ("product", "pair_block"):
            sampler = PositiveCompletionSampler(TableEventOracle(p), 0, completion=completion)
            self.assertAlmostEqual(sampler.conditional_click_probability((1, 1, 1)), .5)
            self.assertEqual(target_next(p, sampler.order, (1, 1, 1)), 1.)
            law = sampler.enumerate_law()
            self.assertEqual(law["failure_mass"], 0.)
            self.assertGreater(float(law["distribution"][p == 0].sum()), 0.)
            # This generic discrete counterexample is not a Gaussian state.

    def test_two_zero_scores_fail_without_epsilon_or_redraw(self):
        p = np.zeros(8); p[0] = p[3] = .5
        for completion in ("product", "pair_block"):
            sampler = PositiveCompletionSampler(TableEventOracle(p), 0, completion=completion)
            with self.assertRaisesRegex(NumericalProbabilityError, "both completion scores"):
                sampler.conditional_click_probability((1, 0))
            np.testing.assert_allclose(sampler.full_distribution(), p, atol=1e-16)

    def test_failure_mass_and_sampling_abort_are_unconditional(self):
        p = np.full(4, .25)

        def fault(key, estimate):
            if key == ((0,), (1,)):
                raise NumericalProbabilityError("injected fixed event failure")
            return estimate

        sampler = PositiveCompletionSampler(TableEventOracle(p), 0,
                                             event_oracle=AlteredOracle(p, fault))
        law = sampler.enumerate_law()
        np.testing.assert_array_equal(law["distribution"], [0, .25, 0, .25])
        self.assertEqual(law["failure_mass"], .5)
        self.assertEqual(law["failures"][0]["prefix"], (0,))
        self.assertFalse(law["conditional_on_success"])
        with self.assertRaisesRegex(NumericalProbabilityError, "refuses failure mass"):
            sampler.full_distribution()
        rng = SequenceRng([.1, .1, .9, .1, .1, .1])
        with self.assertRaises(PositiveCompletionSamplingError) as caught:
            sampler.sample(3, rng)
        error = caught.exception
        self.assertEqual(error.completed_samples, 1)
        self.assertEqual(error.failed_prefix, (0,))
        np.testing.assert_array_equal(error.successful_samples, [[1, 1]])
        self.assertEqual(rng.calls, 3)  # No uniform consumed by failed conditional.

    def test_uncertified_zeros_invalid_values_and_inconsistent_events_fail_closed(self):
        p = np.array([1., 0.])
        oracle = AlteredOracle(p, lambda key, e: (e.value, e.roundoff_indicator))
        sampler = PositiveCompletionSampler(TableEventOracle(p), 0, event_oracle=oracle)
        self.assertEqual(sampler.enumerate_law()["failure_mass"], 1.)
        with self.assertRaisesRegex(NumericalProbabilityError, "uncertified"):
            sampler.conditional_click_probability(())
        for invalid in (np.nan, np.inf, -1e-30, 1.000001):
            oracle = AlteredOracle([.5, .5], lambda key, e: EventEstimate(invalid)
                                   if key == ((), (0,)) else e)
            sampler = PositiveCompletionSampler(TableEventOracle([.5, .5]), 0,
                                                 event_oracle=oracle)
            with self.assertRaisesRegex(NumericalProbabilityError, "invalid event"):
                sampler.conditional_click_probability(())
        oracle = AlteredOracle(np.full(4, .25), lambda key, e: EventEstimate(.75)
                               if key == ((1,), (0,)) else e)
        sampler = PositiveCompletionSampler(TableEventOracle(np.full(4, .25)), 0,
                                             event_oracle=oracle)
        with self.assertRaisesRegex(NumericalProbabilityError, "event nesting"):
            sampler.conditional_click_probability((1,))
        oracle = AlteredOracle([.5, .5], lambda key, e: EventEstimate(.4)
                               if key == ((), (0,)) else e)
        sampler = PositiveCompletionSampler(TableEventOracle([.5, .5]), 0, event_oracle=oracle)
        with self.assertRaisesRegex(NumericalProbabilityError, "do not sum"):
            sampler.conditional_click_probability(())

    def test_propagated_roundoff_guard_checks_whole_product(self):
        p = np.full(16, 1 / 16)
        oracle = AlteredOracle(p, lambda key, e: EventEstimate(e.value, e.value * 4e-7))
        sampler = PositiveCompletionSampler(TableEventOracle(p), 0, event_oracle=oracle)
        with self.assertRaisesRegex(NumericalProbabilityError, "log-score roundoff"):
            sampler.conditional_click_probability((1, 1, 1))
        self.assertLess(sampler.cache_info()["maximum_event_relative_indicator"], 1e-6)
        self.assertGreater(sampler.cache_info()["maximum_log_score_indicator"], 1e-6)

    def test_log_scores_avoid_intermediate_product_underflow(self):
        p = independent_table([np.longdouble("1e-100")] * 5 + [np.longdouble(".3")])
        sampler = PositiveCompletionSampler(TableEventOracle(p), 0)
        details = sampler.conditional_details((1, 1, 1, 1, 1))
        self.assertAlmostEqual(details["click_probability"], .3, places=13)
        self.assertTrue(all(np.isfinite(x) and x < -1100
                            for x in details["log_scores_up_to_common_baseline"]))
        self.assertEqual(float(np.prod(np.array(details["factor_probabilities"][0]))), 0.)

    def test_final_positive_branch_rounding_is_not_certified_zero(self):
        for log_odds in (40, 800, -800):
            oracle = ConditionalIndependentEvents(log_odds)
            sampler = PositiveCompletionSampler(TableEventOracle(np.full(64,1/64)), 0, event_oracle=oracle)
            with self.assertRaisesRegex(NumericalProbabilityError, "Bernoulli endpoint"):
                sampler.conditional_click_probability((1,)*5)
            self.assertTrue(oracle.queried and all(e[2]>0 for e in oracle.queried))
        p = np.array([1, np.exp(np.longdouble(-40))], dtype=np.longdouble)
        self.assertGreater(PositiveCompletionSampler(TableEventOracle(p), 0)
                           .conditional_click_probability(()), 0.)

    def test_bounded_caches_preserve_law_samples_and_returned_details(self):
        p = np.random.default_rng(71).dirichlet(np.ones(32))
        options = dict(completion="pair_block", order=(4, 1, 0, 3, 2))
        cached = PositiveCompletionSampler(TableEventOracle(p), 0, cache_size=3, **options)
        uncached = PositiveCompletionSampler(TableEventOracle(p), 0, cache_size=0, **options)
        np.testing.assert_array_equal(cached.full_distribution(), uncached.full_distribution())
        np.testing.assert_array_equal(cached.sample(100, np.random.default_rng(19)),
                                      uncached.sample(100, np.random.default_rng(19)))
        prefix = (1, 0, 1)
        details = cached.conditional_details(prefix)
        expected = details["click_probability"]
        before = cached.cache_info()["event_oracle_calls"]
        details["click_probability"] = -1
        self.assertEqual(cached.conditional_click_probability(prefix), expected)
        self.assertEqual(cached.cache_info()["event_oracle_calls"], before)
        info = cached.cache_info()
        self.assertLessEqual(info["event_cache_entries"], 3)
        self.assertLessEqual(info["conditional_cache_entries"], 3)
        self.assertGreater(info["event_cache_evictions"], 0)
        self.assertGreater(info["conditional_cache_evictions"], 0)
        self.assertEqual(info["event_oracle_calls"], info["event_backend"]["event_calls"])

    def test_samples_match_enumerated_surrogate(self):
        p = np.random.default_rng(117).dirichlet(np.ones(16))
        sampler = PositiveCompletionSampler(TableEventOracle(p), 0,
            completion="product", order=(2, 0, 3, 1))
        q = sampler.full_distribution()
        samples = sampler.sample(6000, np.random.default_rng(921))
        masks = samples @ (1 << np.arange(4))
        frequencies = np.bincount(masks, minlength=16) / len(samples)
        self.assertLess(.5 * np.abs(frequencies - q).sum(), .035)
        self.assertEqual(samples.dtype, np.uint8)
        self.assertEqual(sampler.sample(0, np.random.default_rng(1)).shape, (0, 4))

    def test_invalid_arguments_are_rejected(self):
        model = TableEventOracle(np.full(8, .125))
        for options in ({"completion": "adaptive"}, {"selection": "fit"},
                        {"block_order": "learned"}, {"cache_size": -1},
                        {"order": (0, 0, 2)}, {"max_relative_roundoff": 0},
                        {"selection": "click_influence"}):
            with self.assertRaises(ValueError):
                PositiveCompletionSampler(model, 0, **options)
        with self.assertRaises(ValueError):
            PositiveCompletionSampler(model, -1)
        sampler = PositiveCompletionSampler(model, 0)
        for prefix in ((0, 1, 0), (2,), (np.nan,)):
            with self.assertRaises(ValueError):
                sampler.conditional_click_probability(prefix)
        with self.assertRaises(ValueError):
            sampler.enumerate_law(max_modes=2)
        for p in ([.4, .4], [1., -.1], [np.nan, 0.], [1.], [1., 0., 0.]):
            with self.assertRaises(ValueError):
                TableEventOracle(p)
        with self.assertRaises(ValueError):
            model.event_probability((0,), (0,))


if __name__ == "__main__":
    unittest.main()
