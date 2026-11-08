"""Dense-table, graph, Gaussian-oracle and failure-boundary forest checks."""

from itertools import combinations
import unittest
from unittest.mock import patch

import numpy as np

from gbslab.conditional_memory import ConditionalMemorySampler
from gbslab.exact import NumericalProbabilityError, ThresholdModel
from gbslab.gaussian import covariance_from_interferometer, haar_unitary
from gbslab.positive_completion import (
    EventEstimate, PositiveCompletionSampler, PositiveCompletionSamplingError,
    TableEventOracle,
)
from gbslab.positive_forest import ForestCompletionSampler
from gbslab.zero_memory import ZeroMemorySampler
from positive_event_fixtures import ConditionalIndependentEvents


def event(p, zeros=(), ones=()):
    return sum((np.longdouble(value) for mask, value in enumerate(p)
        if all(not ((mask >> j) & 1) for j in zeros)
        and all((mask >> j) & 1 for j in ones)), np.longdouble(0))


def target_next(p, order, prefix):
    zeros = tuple(order[j] for j, bit in enumerate(prefix) if not bit)
    ones = tuple(order[j] for j, bit in enumerate(prefix) if bit)
    return float(event(p, zeros, ones + (order[len(prefix)],)) / event(p, zeros, ones))


def independent(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.longdouble)
    return np.array([np.prod([p if (mask >> j) & 1 else 1-p
        for j, p in enumerate(probabilities)], dtype=np.longdouble)
        for mask in range(1 << len(probabilities))], dtype=np.longdouble)


def chain_table(modes):
    p = np.zeros(1 << modes, dtype=np.longdouble)
    for mask in range(len(p)):
        first = mask & 1
        value = np.longdouble(.38 if first else .62)
        for j in range(1, modes):
            probability = (.73 + .015*j) if (mask >> (j-1)) & 1 else (.09 + .025*j)
            value *= probability if (mask >> j) & 1 else 1-probability
        p[mask] = value
    return p


def class_conditional_tree(parents):
    """Generic binary features form the chosen tree given the final class bit."""
    count = len(parents)
    p = np.zeros(1 << (count+1), dtype=np.longdouble)
    for mask in range(len(p)):
        label = (mask >> count) & 1
        value = np.longdouble(.37 if label else .63)
        for j, parent in enumerate(parents):
            if parent is None:
                probability = .63 if label else .18
            elif (mask >> parent) & 1:
                probability = (.88 if label else .69) + .005*j
            else:
                probability = (.31 if label else .11) + .008*j
            value *= probability if (mask >> j) & 1 else 1-probability
        p[mask] = value
    return p


class AlteredOracle:
    def __init__(self, p, transform):
        self.table, self.transform = TableEventOracle(p), transform
        self.m = self.table.m
        self.requests = []

    def event_probability(self, zeros, ones):
        key = (tuple(sorted(zeros)), tuple(sorted(ones)))
        self.requests.append(key)
        return self.transform(key, self.table.event_probability(zeros, ones))


class SequenceRng:
    def __init__(self, values):
        self.values, self.calls = iter(values), 0

    def random(self):
        self.calls += 1
        return next(self.values)


class ForestCompletionTests(unittest.TestCase):
    @staticmethod
    def gaussian(modes=6, *, seed=1701, pure=False):
        rng = np.random.default_rng(seed)
        V = covariance_from_interferometer(haar_unitary(modes, rng),
            np.linspace(.45, .9, modes), 1.0 if pure else np.linspace(.55, .9, modes))
        return ThresholdModel(V)

    @staticmethod
    def generic(p, budget=0, *, forest="chain", order=None, **kwargs):
        # For static-tree algebra tests, the auxiliary independent Gaussian
        # supplies only a deterministic all-tied graph, not the table law.
        if forest == "static_mi_tree":
            model = ThresholdModel(1.8*np.eye(2*(len(p).bit_length()-1)))
            return ForestCompletionSampler(model, budget, forest=forest,
                selection="recent", order=order, event_oracle=TableEventOracle(p), **kwargs)
        return ForestCompletionSampler(TableEventOracle(p), budget, forest=forest,
            selection="recent", order=order, **kwargs)

    def test_forest_formula_matches_independent_dense_event_sums(self):
        p = np.random.default_rng(1721).dirichlet(np.ones(128))
        order, prefix = (5, 2, 0, 4, 1, 6, 3), (1, 1, 1, 1, 1, 0)
        for forest in ("chain", "static_mi_tree"):
            sampler = self.generic(p, 1, forest=forest, order=order)
            d = sampler.conditional_details(prefix)
            self.assertEqual(d["omitted_clicks"], (5, 2, 0, 4))
            self.assertEqual(d["retained_clicks"], (1,))
            edges = d["forest_edges"]
            self.assertEqual(len(edges), 3)
            self.assertEqual(set(v for edge in edges for v in edge), {5, 2, 0, 4})
            scores = []
            for bit in (0, 1):
                z, c = (6,) + ((3,) if not bit else ()), (1,) + ((3,) if bit else ())
                w = event(p, z, c)
                likelihood = np.longdouble(1)
                singles = {j: event(p, z, c+(j,))/w for j in d["omitted_clicks"]}
                for value in singles.values():
                    likelihood *= value
                for first, second in edges:
                    likelihood *= event(p, z, c+(first, second))/w/(singles[first]*singles[second])
                scores.append(w*likelihood)
            self.assertAlmostEqual(d["click_probability"], float(scores[1]/sum(scores)), places=13)
            self.assertEqual(d["branch_log_coefficient"], 0)
            self.assertLessEqual(sampler.cache_info()["forest_maximum_local_event_width"], 4)

    def test_zero_one_two_omissions_are_exact_for_both_trees(self):
        p = np.random.default_rng(1733).dirichlet(np.full(64, 1.3))
        order = (4, 0, 5, 2, 1, 3)
        for forest in ("chain", "static_mi_tree"):
            for budget in (0, 2, 4):
                sampler = self.generic(p, budget, forest=forest, order=order)
                for length in range(6):
                    for mask in range(1 << length):
                        prefix = tuple((mask >> j) & 1 for j in range(length))
                        if sum(prefix) <= budget+2:
                            self.assertAlmostEqual(sampler.conditional_click_probability(prefix),
                                target_next(p, order, prefix), places=13)
                if budget >= 4:
                    np.testing.assert_allclose(sampler.full_distribution(), p, atol=3e-16, rtol=3e-14)

    def test_generic_markov_chain_full_law_is_exact(self):
        p = chain_table(6)
        for order in (tuple(range(6)), tuple(reversed(range(6)))):
            for budget in (0, 2):
                sampler = self.generic(p, budget, order=order)
                np.testing.assert_allclose(sampler.full_distribution(), p, atol=4e-16, rtol=4e-14)

    def test_class_conditional_chain_and_star_posteriors_are_exact(self):
        for forest, parents in (("chain", (None, 0, 1, 2, 3)),
                                ("static_mi_tree", (None, 0, 0, 0, 0))):
            p = class_conditional_tree(parents)
            sampler = self.generic(p, forest=forest)
            prefix = (1,)*5
            actual = sampler.conditional_click_probability(prefix)
            self.assertAlmostEqual(actual, target_next(p, sampler.order, prefix), places=13)
            product = PositiveCompletionSampler(TableEventOracle(p), 0)
            self.assertGreater(abs(product.conditional_click_probability(prefix)-actual), 1e-3)

    def test_gaussian_table_full_laws_share_retained_selection_and_tree(self):
        model = self.gaussian()
        p = model.full_distribution()
        order = (4, 0, 5, 2, 1, 3)
        with patch.object(model, "full_distribution", side_effect=AssertionError("no target oracle access")):
            for forest in ("chain", "static_mi_tree"):
                for budget in (0, 2, 4):
                    options = dict(forest=forest, selection="click_influence", order=order)
                    direct = ForestCompletionSampler(model, budget, **options)
                    table = ForestCompletionSampler(model, budget,
                        event_oracle=TableEventOracle(p), **options)
                    q, qt = direct.enumerate_law(), table.enumerate_law()
                    self.assertEqual(q["failure_mass"], 0)
                    self.assertEqual(qt["failure_mass"], 0)
                    np.testing.assert_allclose(q["distribution"], qt["distribution"], atol=8e-13, rtol=5e-9)
                    for prefix in ((1, 1, 1, 0), (1, 1, 0, 1, 1)):
                        a, b = direct.conditional_details(prefix), table.conditional_details(prefix)
                        self.assertEqual(a["retained_clicks"], b["retained_clicks"])
                        self.assertEqual(a["forest_edges"], b["forest_edges"])
                    if budget == 4:
                        np.testing.assert_allclose(q["distribution"], p, atol=8e-13, rtol=5e-9)
                    info = direct.cache_info()
                    self.assertLessEqual(info["forest_maximum_event_ones"], budget+3)
                    self.assertLessEqual(info["forest_maximum_local_event_width"], budget+3)

    def test_retained_selectors_and_pure_gaussian_boundary_are_unchanged(self):
        model = self.gaussian(5, pure=True)
        order, prefix = (4, 0, 3, 1, 2), (1, 1, 1, 1)
        for selection in ("recent", "pairwise", "click_influence"):
            old_cls = ConditionalMemorySampler if selection == "click_influence" else ZeroMemorySampler
            old = old_cls(model, 2, order=order, selection=selection)
            for forest in ("chain", "static_mi_tree"):
                sampler = ForestCompletionSampler(model, 2, order=order,
                    selection=selection, forest=forest)
                self.assertEqual(sampler.conditional_details(prefix)["retained_clicks"],
                    old._retained_clicks(order[:4], order[4]))
                np.testing.assert_allclose(sampler.full_distribution(), model.full_distribution(),
                    atol=1e-12, rtol=6e-9)

    def test_static_mi_reuses_corrected_policy_and_tree_has_maximum_weight(self):
        model = self.gaussian(5, seed=1747)
        order = (3, 0, 4, 1, 2)
        sampler = ForestCompletionSampler(model, 0, forest="static_mi_tree", order=order)
        old = ZeroMemorySampler(model, 1, selection="pairwise")
        np.testing.assert_array_equal(sampler.static_pairwise_mi, old.pairwise_mi)
        self.assertFalse(sampler.static_pairwise_mi.flags.writeable)
        d = sampler.conditional_details((1, 1, 1, 1))
        nodes = set(order[:4])
        possible = list(combinations(sorted(nodes), 2))
        weights = []
        for edges in combinations(possible, 3):
            reached = {min(nodes)}
            for _ in range(3):
                for u, v in edges:
                    if u in reached or v in reached:
                        reached.update((u, v))
            if reached == nodes:
                weights.append(sum(old.pairwise_mi[u, v] for u, v in edges))
        self.assertAlmostEqual(sum(old.pairwise_mi[u, v] for u, v in d["forest_edges"]), max(weights), places=14)
        info = sampler.cache_info()
        self.assertEqual(info["forest_weight_lookups"], 6)
        self.assertEqual(info["forest_update_comparisons"], 3)
        self.assertEqual(info["forest_cut_candidates_examined"], 6)
        self.assertEqual(info["static_pairwise_precompute"]["pairwise_marginals_requested"], 10)
        self.assertEqual(info["static_pairwise_precompute"]["pairwise_mi_policy"], old.cache_info()["pairwise_mi_policy"])
        self.assertGreater(info["static_mi_setup_seconds"], 0)
        self.assertGreater(info["forest_selection_seconds"], 0)

    def test_zero_weight_edges_are_kept_and_ties_use_physical_indices(self):
        model = ThresholdModel(1.8*np.eye(10))
        order = (2, 4, 0, 1, 3)
        sampler = ForestCompletionSampler(model, 0, forest="static_mi_tree", order=order)
        d = sampler.conditional_details((1, 1, 1))
        np.testing.assert_array_equal(sampler.static_pairwise_mi, np.zeros((5, 5)))
        self.assertEqual(d["forest_edges"], ((0, 2), (0, 4)))
        chain = ForestCompletionSampler(model, 0, forest="chain", order=order)
        self.assertEqual(chain.conditional_details((1, 1, 1))["forest_edges"], ((2, 4), (4, 0)))
        self.assertEqual(chain.cache_info()["forest_weight_lookups"], 0)

    def test_singleton_analytic_zero_precedes_edge_queries(self):
        p = np.zeros(16)
        for label in (0, 1):
            for middle in range(4):
                p[label | (middle << 1) | (label << 3)] = (.4 if label else .6)/4

        def transform(key, estimate):
            if key[0] == (3,) and len(key[1]) >= 2:
                raise RuntimeError("a node-zero branch must not query an edge")
            return estimate

        sampler = ForestCompletionSampler(TableEventOracle(p), 0, selection="recent",
            event_oracle=AlteredOracle(p, transform))
        d = sampler.conditional_details((1, 1, 1))
        self.assertEqual(d["click_probability"], 1.)
        self.assertEqual(d["zero_score_reasons"][0], "analytic_zero_forest_node")
        self.assertEqual(d["edge_probabilities"][0], ())

    def test_edge_analytic_zero_and_two_zero_scores_are_explicit(self):
        p = np.zeros(8); p[1] = p[2] = .3; p[4] = p[7] = .2
        sampler = self.generic(p)
        d = sampler.conditional_details((1, 1))
        self.assertEqual(d["click_probability"], 1.)
        self.assertEqual(d["zero_score_reasons"][0], "analytic_zero_forest_edge")
        p = np.zeros(8); p[[1, 2, 5, 6]] = .25
        sampler = self.generic(p)
        with self.assertRaisesRegex(NumericalProbabilityError, "both forest scores"):
            sampler.conditional_click_probability((1, 1))
        np.testing.assert_allclose(sampler.full_distribution(), p, atol=2e-16)

    def test_failure_mass_no_redraw_and_program_errors_propagate(self):
        p = np.full(8, 1/8)

        def numerical(key, estimate):
            return EventEstimate(0, 0, False) if key == ((2,), (0, 1)) else estimate

        sampler = ForestCompletionSampler(TableEventOracle(p), 0, selection="recent",
            event_oracle=AlteredOracle(p, numerical))
        law = sampler.enumerate_law()
        self.assertEqual(law["failure_mass"], .25)
        self.assertAlmostEqual(law["distribution"].sum(), .75, places=15)
        self.assertFalse(law["conditional_on_success"])
        with self.assertRaisesRegex(NumericalProbabilityError, "refuses failure mass"):
            sampler.full_distribution()
        rng = SequenceRng([.9, .9, .9, .1, .1, .5])
        with self.assertRaises(PositiveCompletionSamplingError) as caught:
            sampler.sample(3, rng)
        self.assertEqual(caught.exception.completed_samples, 1)
        self.assertEqual(caught.exception.failed_prefix, (1, 1))
        np.testing.assert_array_equal(caught.exception.successful_samples, [[0, 0, 0]])
        self.assertEqual(rng.calls, 5)

        def programming(key, estimate):
            if key == ((2,), (0, 1)):
                raise RuntimeError("injected programming fault")
            return estimate

        broken = ForestCompletionSampler(TableEventOracle(p), 0, selection="recent",
            event_oracle=AlteredOracle(p, programming))
        with self.assertRaisesRegex(RuntimeError, "programming fault"):
            broken.enumerate_law()

    def test_merged_exponents_cancel_error_terms_but_negative_powers_count(self):
        p = np.full(16, 1/16)
        oracle = AlteredOracle(p, lambda key, e: EventEstimate(e.value, 4e-7*e.value))
        sampler = ForestCompletionSampler(TableEventOracle(p), 0,
            selection="recent", event_oracle=oracle)
        two = sampler.conditional_details((1, 1))
        self.assertEqual(two["branch_log_coefficient"], 0)
        self.assertEqual(two["node_log_coefficients"], {0: 0, 1: 0})
        self.assertLess(max(two["log_score_indicators"]), 5e-7)
        with self.assertRaisesRegex(NumericalProbabilityError, "log-score roundoff"):
            sampler.conditional_click_probability((1, 1, 1))
        self.assertLess(sampler.cache_info()["maximum_event_relative_indicator"], 1e-6)
        self.assertGreater(sampler.cache_info()["maximum_log_score_indicator"], 1e-6)

    def test_log_domain_survives_products_below_float64_range(self):
        p = independent([np.longdouble("1e-100")]*5 + [np.longdouble(".3")])
        for forest in ("chain", "static_mi_tree"):
            sampler = self.generic(p, forest=forest)
            d = sampler.conditional_details((1,)*5)
            self.assertAlmostEqual(d["click_probability"], .3, places=13)
            self.assertTrue(all(np.isfinite(v) and v < -1100 for v in d["log_scores_up_to_common_baseline"]))

    def test_endpoint_rounding_is_failure_but_analytic_vacuum_is_exact(self):
        for log_odds in (40, 800, -800):
            for forest in ("chain", "static_mi_tree"):
                oracle = ConditionalIndependentEvents(log_odds)
                sampler = ForestCompletionSampler(ThresholdModel(1.8*np.eye(12)),0,
                    selection="recent",forest=forest,event_oracle=oracle)
                with self.assertRaisesRegex(NumericalProbabilityError, "Bernoulli endpoint"):
                    sampler.conditional_click_probability((1,)*5)
                self.assertTrue(oracle.queried and all(e[2]>0 for e in oracle.queried))
        for forest in ("chain", "static_mi_tree"):
            sampler = ForestCompletionSampler(ThresholdModel(np.eye(8)), 0, forest=forest)
            expected = np.zeros(16); expected[0] = 1
            np.testing.assert_array_equal(sampler.full_distribution(), expected)
            self.assertGreater(sampler.cache_info()["structural_zero_returns"], 0)

    def test_cache_changes_cost_not_law_and_details_are_defensive_copies(self):
        p = np.random.default_rng(1759).dirichlet(np.ones(32))
        order = (4, 1, 0, 3, 2)
        cached = self.generic(p, order=order, cache_size=3)
        uncached = self.generic(p, order=order, cache_size=0)
        np.testing.assert_array_equal(cached.full_distribution(), uncached.full_distribution())
        np.testing.assert_array_equal(cached.sample(40, np.random.default_rng(1777)),
                                      uncached.sample(40, np.random.default_rng(1777)))
        d = cached.conditional_details((1, 1, 1))
        before = cached.cache_info()["forest_selections"]
        d["node_log_coefficients"][1] = 99
        self.assertNotEqual(cached.conditional_details((1, 1, 1))["node_log_coefficients"].get(1), 99)
        self.assertEqual(cached.cache_info()["forest_selections"], before)
        self.assertLessEqual(cached.cache_info()["event_cache_entries"], 3)

    def test_generic_high_order_parity_is_not_a_tree_exactness_claim(self):
        p = np.zeros(32)
        for features in range(16):
            p[features | ((features.bit_count() % 2) << 4)] = 1/16
        for forest in ("chain", "static_mi_tree"):
            sampler = self.generic(p, forest=forest)
            self.assertAlmostEqual(sampler.conditional_click_probability((1,)*4), .5)
            self.assertEqual(target_next(p, sampler.order, (1,)*4), 0.)
            law = sampler.enumerate_law()
            self.assertEqual(law["failure_mass"], 0.)
            self.assertGreater(law["distribution"][p == 0].sum(), 0.)

    def test_invalid_forest_and_static_gaussian_contract(self):
        table = TableEventOracle(np.full(4, .25))
        with self.assertRaises(ValueError):
            ForestCompletionSampler(table, 0, forest="unknown", selection="recent")
        with self.assertRaisesRegex(ValueError, "original Gaussian"):
            ForestCompletionSampler(table, 0, forest="static_mi_tree", selection="recent")
        with self.assertRaisesRegex(ValueError, "original Gaussian"):
            ForestCompletionSampler(table, 0)
        with self.assertRaises(ValueError):
            ForestCompletionSampler(ThresholdModel(1.8*np.eye(4)), -1)


if __name__ == "__main__":
    unittest.main()
