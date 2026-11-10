"""Conditional forest completion of every omitted positive threshold outcome.

For B=(all past zeros, retained clicks), F the omitted clicks, and a forest E,

    L_a = prod_j P(j=1|a,B) prod_(j,l) P(j=l=1|a,B) /
                                      (P(j=1|a,B) P(l=1|a,B)),
    score_a = P(a|B) L_a.

The two implemented rules are connected trees whenever F is nonempty:
``chain`` follows measurement order; ``static_mi_tree`` is a deterministic
maximum spanning tree on F with fixed, unconditional input-Gaussian pair MI.
Zero-weight edges remain present.  The static tree is not optimized for the
branch-conditioned distribution.  This is a TAN/Chow--Liu-style application,
not a novel GBS algorithm or a guarantee of lower global TV.

Events, numerical guards, LRU caches, sampling, and failure-symbol enumeration
are inherited from the frozen iteration-6 PositiveCompletionSampler.  The
small log-score normalizer below copies its stable logistic/endpoint policy.
The default event backend is still direct inclusion-exclusion: determinants
include every observed zero.  No small-k Schur performance optimization is
implemented or claimed here.  Tree selection and static MI setup are counted.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from .exact import NumericalProbabilityError
from .positive_completion import PositiveCompletionSampler
from .zero_memory import ZeroMemorySampler


def _positive_log_ratio(first, second):
    """Log ratio without subtracting two large, almost equal logarithms."""
    with np.errstate(over="ignore", under="ignore"):
        ratio = first / second
    if np.isfinite(ratio) and ratio > 0:
        return np.log(ratio)
    a, ea = np.frexp(first)
    b, eb = np.frexp(second)
    return np.log(a / b) + (int(ea) - int(eb)) * np.log(np.longdouble(2))


def _normalized_click_probability(log_scores, *, log_odds=None):
    """Copied normalization policy from PositiveCompletionSampler, unchanged.

    None denotes a separately certified zero score.  A finite log score is
    strictly positive, so rounding its normalized branch to 0 or 1 fails.
    """
    if log_scores[0] is None and log_scores[1] is None:
        raise NumericalProbabilityError("both forest scores have zero support")
    if log_scores[0] is None:
        return 1.0
    if log_scores[1] is None:
        return 0.0
    difference = log_scores[0] - log_scores[1] if log_odds is None else log_odds
    with np.errstate(under="ignore"):
        if difference >= 0:
            small = np.exp(-difference)
            probability = float(small / (1 + small))
        else:
            small = np.exp(difference)
            probability = float(1 / (1 + small))
    if not np.isfinite(probability) or not 0 < probability < 1:
        raise NumericalProbabilityError(
            "strictly positive forest branches rounded to a Bernoulli endpoint")
    return probability


class ForestCompletionSampler(PositiveCompletionSampler):
    """Complete F with a chain or a fixed-weight maximum spanning tree.

    Parameters shared with PositiveCompletionSampler preserve its meanings.
    ``selection='click_influence'`` is the default.  A generic TableEventOracle
    can be the model for ``chain`` with ``selection='recent'``; a static MI
    tree requires the original Gaussian model, even when an independent
    ``event_oracle=TableEventOracle(P)`` is supplied for the event cross-check.

    Static Gaussian pair MI is obtained by the existing ZeroMemorySampler
    pairwise precomputation, including its stable corrected MI policy.  Prim's
    algorithm takes O(|F|**2) work per uncached conditional.  Tied edges use
    their sorted physical indices, without a numerical tie tolerance.  There
    is no fitting to generated samples and no omitted-node or zero-edge cutoff.
    """

    def __init__(self, model, max_clicks, *, forest="chain",
                 selection="click_influence", order=None, cache_size=8192,
                 max_relative_roundoff=1e-6, event_oracle=None):
        started = perf_counter()
        if forest not in ("chain", "static_mi_tree"):
            raise ValueError("forest must be 'chain' or 'static_mi_tree'")
        if forest == "static_mi_tree" and not hasattr(model, "V"):
            raise ValueError("static_mi_tree requires the original Gaussian model")
        super().__init__(model, max_clicks, completion="product", selection=selection,
            order=order, block_order="measurement", cache_size=cache_size,
            max_relative_roundoff=max_relative_roundoff, event_oracle=event_oracle)
        self.forest, self.completion = forest, "forest"
        self._static_builder = None
        self.static_pairwise_mi = None
        static_started = perf_counter()
        if forest == "static_mi_tree":
            # A positive budget activates the existing corrected pairwise-MI
            # precompute even if the actual retained budget is zero.  This
            # object is only a graph-weight builder, never the retained selector.
            self._static_builder = ZeroMemorySampler(model, 1, order=self.order,
                selection="pairwise", cache_size=0,
                max_relative_roundoff=max_relative_roundoff)
            self.static_pairwise_mi = self._static_builder.pairwise_mi
            if (self.static_pairwise_mi.shape != (self.m, self.m)
                    or not np.all(np.isfinite(self.static_pairwise_mi))
                    or np.any(self.static_pairwise_mi < 0)):
                raise NumericalProbabilityError("invalid static Gaussian MI weights")
        static_seconds = perf_counter() - static_started if self._static_builder else 0.0
        self.diagnostics.update({
            "static_mi_setup_seconds": static_seconds,
            "forest_selections": 0, "forest_selection_seconds": 0.0,
            "forest_vertices_processed": 0, "forest_edges_selected": 0,
            "forest_maximum_vertices": 0, "forest_maximum_edges": 0,
            "forest_weight_lookups": 0, "forest_cut_candidates_examined": 0,
            "forest_update_comparisons": 0,
            "forest_singleton_requests": 0, "forest_edge_event_requests": 0,
            "forest_maximum_local_event_width": 0,
            "forest_maximum_event_ones": 0, "forest_maximum_event_constraints": 0,
            "initialization_seconds": perf_counter() - started,
        })

    def _event(self, zeros, ones):
        self.diagnostics["forest_maximum_event_ones"] = max(
            self.diagnostics["forest_maximum_event_ones"], len(ones))
        self.diagnostics["forest_maximum_event_constraints"] = max(
            self.diagnostics["forest_maximum_event_constraints"], len(zeros) + len(ones))
        return super()._event(zeros, ones)

    def _choose_edges(self, omitted):
        started = perf_counter()
        self.diagnostics["forest_selections"] += 1
        self.diagnostics["forest_vertices_processed"] += len(omitted)
        self.diagnostics["forest_maximum_vertices"] = max(
            self.diagnostics["forest_maximum_vertices"], len(omitted))
        try:
            if len(omitted) < 2:
                edges = ()
            elif self.forest == "chain":
                edges = tuple(zip(omitted[:-1], omitted[1:]))
            else:
                vertices = tuple(sorted(omitted))
                root = vertices[0]

                def edge_rank(first, second):
                    self.diagnostics["forest_weight_lookups"] += 1
                    low, high = min(first, second), max(first, second)
                    return (-float(self.static_pairwise_mi[first, second]), low, high)

                best = {node: (edge_rank(root, node), root) for node in vertices[1:]}
                selected = []
                while best:
                    self.diagnostics["forest_cut_candidates_examined"] += len(best)
                    node = min(best, key=lambda j: best[j][0])
                    rank, parent = best.pop(node)
                    selected.append((min(parent, node), max(parent, node)))
                    for other in best:
                        candidate = edge_rank(node, other)
                        self.diagnostics["forest_update_comparisons"] += 1
                        if candidate < best[other][0]:
                            best[other] = (candidate, node)
                edges = tuple(selected)
            if len(edges) != max(0, len(omitted) - 1):
                raise RuntimeError("Forest construction did not span every omitted click")
            self.diagnostics["forest_edges_selected"] += len(edges)
            self.diagnostics["forest_maximum_edges"] = max(
                self.diagnostics["forest_maximum_edges"], len(edges))
            return edges
        finally:
            self.diagnostics["forest_selection_seconds"] += perf_counter() - started

    def _compute_conditional(self, prefix):
        mode = self.order[len(prefix)]
        zeros = tuple(self.order[j] for j, bit in enumerate(prefix) if not bit)
        clicks = tuple(self.order[j] for j, bit in enumerate(prefix) if bit)
        started = perf_counter()
        self.diagnostics["selector_calls"] += 1
        try:
            retained = self._selector._retained_clicks(clicks, mode)
        finally:
            self.diagnostics["selector_seconds"] += perf_counter() - started
        selected = set(retained)
        omitted = tuple(j for j in clicks if j not in selected)
        edges = self._choose_edges(omitted)
        degrees = {j: 0 for j in omitted}
        for first, second in edges:
            degrees[first] += 1
            degrees[second] += 1
        branch_coefficient = 1 - len(omitted) + len(edges)
        node_coefficients = {j: 1 - degrees[j] for j in omitted}
        width = len(retained) + 1 + min(2, len(omitted))
        self.diagnostics["forest_maximum_local_event_width"] = max(
            self.diagnostics["forest_maximum_local_event_width"], width)
        self.diagnostics["maximum_factor_count"] = max(
            self.diagnostics["maximum_factor_count"], len(omitted) + len(edges))
        baseline = self._event(zeros, retained)
        if baseline.value == 0:
            raise NumericalProbabilityError("baseline conditioning event has zero support")
        branches = (self._event(zeros + (mode,), retained),
                    self._event(zeros, retained + (mode,)))
        for branch in branches:
            self._nested(branch, baseline, "baseline branch")
        consistency = abs(branches[0].value + branches[1].value - baseline.value)
        allowance = (baseline.roundoff_indicator + branches[0].roundoff_indicator
                     + branches[1].roundoff_indicator
                     + 8 * np.finfo(np.longdouble).eps * baseline.value)
        if consistency > allowance:
            raise NumericalProbabilityError("branch masses do not sum to baseline event mass")

        log_scores, log_indicators = [None, None], [0.0, 0.0]
        score_terms = [None, None]
        node_probabilities, edge_probabilities = [[], []], [[], []]
        zero_reasons = [None, None]
        for outcome, branch in enumerate(branches):
            if branch.value == 0:
                zero_reasons[outcome] = "analytic_zero_baseline_branch"
                self.diagnostics["analytic_zero_score_branches"] += 1
                continue
            branch_zeros = zeros + (mode,) if outcome == 0 else zeros
            branch_ones = retained if outcome == 0 else retained + (mode,)
            nodes, pairs = {}, {}
            # Node zeros are checked before any edge division/log, including
            # nodes whose algebraically combined exponent is zero.
            for node in omitted:
                self.diagnostics["forest_singleton_requests"] += 1
                estimate = self._event(branch_zeros, branch_ones + (node,))
                self._nested(estimate, branch, "forest singleton")
                nodes[node] = estimate
                node_probabilities[outcome].append((node, float(estimate.value / branch.value)))
                if estimate.value == 0:
                    zero_reasons[outcome] = "analytic_zero_forest_node"
                    break
            if zero_reasons[outcome] is None:
                for edge in edges:
                    self.diagnostics["forest_edge_event_requests"] += 1
                    estimate = self._event(branch_zeros, branch_ones + edge)
                    self._nested(estimate, nodes[edge[0]], "forest edge at first node")
                    self._nested(estimate, nodes[edge[1]], "forest edge at second node")
                    pairs[edge] = estimate
                    edge_probabilities[outcome].append((edge, float(estimate.value / branch.value)))
                    if estimate.value == 0:
                        zero_reasons[outcome] = "analytic_zero_forest_edge"
                        break
            if zero_reasons[outcome] is not None:
                self.diagnostics["analytic_zero_score_branches"] += 1
                continue

            # Combining identical event exponents first is essential: for one
            # omitted node the score is its joint event u; for two it is the
            # joint edge v.  The cancelled w/node terms add no log error.
            terms = ([(branch_coefficient, branch)]
                     + [(node_coefficients[j], nodes[j]) for j in omitted]
                     + [(1, pairs[edge]) for edge in edges])
            log_score = np.longdouble(0)
            log_indicator = np.longdouble(0)
            absolute_log_sum = np.longdouble(0)
            used_terms = 0
            for coefficient, estimate in terms:
                if coefficient == 0:
                    continue
                term = coefficient * np.log(estimate.value)
                log_score += term
                absolute_log_sum += abs(term)
                log_indicator += abs(coefficient) * self._log_indicator(estimate)
                used_terms += 1
            log_indicator += (16 * np.finfo(np.longdouble).eps
                              * (1 + abs(log_score) + absolute_log_sum + used_terms))
            self.diagnostics["maximum_log_score_indicator"] = max(
                self.diagnostics["maximum_log_score_indicator"], float(log_indicator))
            if (not np.isfinite(log_score) or not np.isfinite(log_indicator)
                    or log_indicator > self.max_relative_roundoff):
                raise NumericalProbabilityError("forest log-score roundoff guard failed")
            log_scores[outcome] = log_score
            log_indicators[outcome] = float(log_indicator)
            score_terms[outcome] = [(coefficient, estimate.value) for coefficient, estimate in terms if coefficient]

        # Pair the same events across branches before logarithmic summation.
        # Common tiny event scales then cancel in ratios, not in two ~-1000 logs.
        log_odds, odds_indicator = None, None
        if all(score is not None for score in log_scores):
            differences = []
            for (coefficient, first), (other, second) in zip(*score_terms):
                if coefficient != other:
                    raise RuntimeError("Forest branch exponents differ")
                differences.append(coefficient * _positive_log_ratio(first, second))
            log_odds = np.sum(np.asarray(differences, dtype=np.longdouble), dtype=np.longdouble)
            odds_indicator = (sum(log_indicators) + 16 * np.finfo(np.longdouble).eps
                * (1 + abs(log_odds) + sum(abs(x) for x in differences) + len(differences)))
            if not np.isfinite(log_odds) or not np.isfinite(odds_indicator) or odds_indicator > self.max_relative_roundoff:
                raise NumericalProbabilityError("forest log-odds roundoff guard failed")

        return {"prefix": prefix, "mode": mode, "zeros": zeros, "clicks": clicks,
            "retained_clicks": retained, "omitted_clicks": omitted,
            "forest": self.forest, "forest_edges": edges, "node_degrees": degrees,
            "branch_log_coefficient": branch_coefficient,
            "node_log_coefficients": node_coefficients,
            "baseline_probability": float(baseline.value),
            "branch_masses": tuple(float(e.value) for e in branches),
            "baseline_probabilities": tuple(float(e.value / baseline.value) for e in branches),
            "node_probabilities": tuple(tuple(v) for v in node_probabilities),
            "edge_probabilities": tuple(tuple(v) for v in edge_probabilities),
            "log_scores_up_to_common_baseline": tuple(None if v is None else float(v) for v in log_scores),
            "log_score_indicators": tuple(log_indicators), "zero_score_reasons": tuple(zero_reasons),
            "log_odds_from_paired_events": None if log_odds is None else float(log_odds),
            "log_odds_roundoff_indicator": None if odds_indicator is None else float(odds_indicator),
            "click_probability": _normalized_click_probability(log_scores, log_odds=log_odds),
            "log_score_common_factor_omitted": "1/P(B)"}

    def cache_info(self):
        info = super().cache_info()
        info.update({"forest": self.forest, "block_order": "not_applicable",
            "tree_scope": "all omitted clicks, including zero-weight spanning edges",
            "static_pairwise_weight_kind": ("unconditional_input_gaussian_threshold_MI"
                if self._static_builder else "not_used"),
            "static_pairwise_payload_bytes": (self.static_pairwise_mi.nbytes
                if self.static_pairwise_mi is not None else 0),
            "static_pairwise_precompute": (self._static_builder.cache_info()
                if self._static_builder else None),
            "tree_tie_rule": "negative MI then sorted physical edge indices",
            "tree_fitted_to_samples": False,
            "local_event_width_limit": self.max_clicks + 3,
            "event_evaluator_scope": "direct determinants include all zeros; no small-k Schur event optimization",
            "tree_optimization_scope": "static unconditional MI, not conditional-likelihood or global-TV optimization"})
        return info
