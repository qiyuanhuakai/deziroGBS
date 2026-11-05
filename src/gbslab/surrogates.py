"""Normalized bounded-parent approximations to threshold distributions.

These are established directed graphical-model approximations, used here as
exploratory classical baselines rather than claimed as a new GBS algorithm.
Fitting queries only exact marginals on at most ``max_parents + 1`` modes
(or two modes for a nonzero pairwise budget).  No full target distribution is
used to choose the model.  Dense enumeration is a separate small-instance
validation operation.

All distributions use little-endian bit indexing: local bit zero is the first
mode in the supplied mode tuple, and a one denotes a detector click.  Entropies
and relative entropies are in nats.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np


class MarginalOracle(Protocol):
    m: int

    def marginal(self, modes: tuple[int, ...]) -> np.ndarray: ...


def _validated_distribution(probabilities: np.ndarray, *, name: str) -> np.ndarray:
    """Check a probability vector without concealing substantive errors."""
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or not len(p) or not np.all(np.isfinite(p)):
        raise ValueError(f"{name} must be a nonempty finite probability vector")
    if np.min(p) < -1e-12:
        raise ValueError(f"{name} contains a negative probability")
    if not np.isclose(np.sum(p), 1.0, rtol=0.0, atol=2e-9):
        raise ValueError(f"{name} must sum to one; got {np.sum(p):.17g}")
    # Permit only tiny numerical defects of an otherwise valid oracle output.
    p = np.maximum(p, 0.0)
    return p / np.sum(p)


def entropy(probabilities: np.ndarray) -> float:
    """Shannon entropy in nats; zero-probability terms contribute zero."""
    p = _validated_distribution(probabilities, name="probabilities")
    positive = p > 0.0
    return float(-np.dot(p[positive], np.log(p[positive])))


def _entropy_unchecked(p: np.ndarray) -> float:
    p = np.ravel(p)
    positive = p > 0.0
    return float(-np.dot(p[positive], np.log(p[positive])))


class _MarginalCache:
    """Per-fit cache keyed by unordered mode subsets, with explicit reindexing."""

    def __init__(self, model: MarginalOracle):
        self.model = model
        self.cache: dict[tuple[int, ...], np.ndarray] = {(): np.array([1.0])}
        self.requests = 0
        self.oracle_calls = 0
        self.max_order = 0
        self.returned_states = 0

    def marginal(self, modes: Sequence[int]) -> np.ndarray:
        requested = tuple(int(i) for i in modes)
        if len(set(requested)) != len(requested):
            raise ValueError("marginal mode tuple must not contain duplicates")
        if any(i < 0 or i >= self.model.m for i in requested):
            raise ValueError("marginal mode index is out of range")
        canonical = tuple(sorted(requested))
        self.requests += 1
        if canonical not in self.cache:
            p = _validated_distribution(
                self.model.marginal(canonical), name=f"marginal {canonical}"
            )
            if len(p) != 1 << len(canonical):
                raise ValueError("oracle returned a marginal with the wrong length")
            self.cache[canonical] = p
            self.oracle_calls += 1
            self.max_order = max(self.max_order, len(canonical))
            self.returned_states += len(p)
        p = self.cache[canonical]
        if canonical == requested:
            return p
        requested_masks = np.arange(len(p), dtype=np.uint64)
        canonical_masks = np.zeros(len(p), dtype=np.uint64)
        for local, mode in enumerate(requested):
            canonical_masks |= ((requested_masks >> local) & 1) << canonical.index(mode)
        return p[canonical_masks.astype(np.intp)]


def _conditional_mutual_information(
    cache: _MarginalCache, target: int, candidate: int, parents: Sequence[int]
) -> float:
    """I(target; candidate | parents), from one small exact joint marginal."""
    modes = tuple(parents) + (target, candidate)
    joint = cache.marginal(modes).reshape(2, 2, 1 << len(parents))
    target_parent = joint.sum(axis=0)
    candidate_parent = joint.sum(axis=1)
    parent = joint.sum(axis=(0, 1))
    information = (
        _entropy_unchecked(target_parent)
        + _entropy_unchecked(candidate_parent)
        - _entropy_unchecked(parent)
        - _entropy_unchecked(joint)
    )
    if information < -2e-9:
        raise ArithmeticError(f"negative conditional mutual information: {information}")
    return max(0.0, float(information))


@dataclass
class BoundedParentSurrogate:
    """A normalized binary Bayesian network with original-mode indexing.

    ``parents[i]`` is the ordered tuple of parents of mode i.
    ``conditional_tables[i][s]`` is P(X_i=1 | parent bitmask s).
    ``order`` is a permutation in which parents always precede their children.
    """

    m: int
    order: tuple[int, ...]
    parents: tuple[tuple[int, ...], ...]
    conditional_tables: tuple[np.ndarray, ...]
    diagnostics: dict[str, Any]

    def __post_init__(self) -> None:
        if sorted(self.order) != list(range(self.m)):
            raise ValueError("order must be a permutation of all modes")
        if len(self.parents) != self.m or len(self.conditional_tables) != self.m:
            raise ValueError("one parent tuple and conditional table are required per mode")
        position = {mode: i for i, mode in enumerate(self.order)}
        for mode in self.order:
            pa = self.parents[mode]
            if len(set(pa)) != len(pa) or any(
                p not in position or position[p] >= position[mode] for p in pa
            ):
                raise ValueError("all distinct parents must precede their child")
            table = np.asarray(self.conditional_tables[mode], dtype=float)
            if table.shape != (1 << len(pa),):
                raise ValueError("conditional table has the wrong shape")
            if not np.all(np.isfinite(table)) or np.any((table < 0) | (table > 1)):
                raise ValueError("conditional probabilities must be finite and in [0, 1]")

    @property
    def cpts(self) -> tuple[np.ndarray, ...]:
        """Alias for the conditional click-probability tables."""
        return self.conditional_tables

    def full_distribution(self, *, max_states: int = 1 << 24) -> np.ndarray:
        """Enumerate Q for small-instance validation, not for scalable sampling."""
        count = 1 << self.m
        if count > max_states:
            raise ValueError("dense enumeration exceeds max_states; use sample instead")
        masks = np.arange(count, dtype=np.uint64)
        probabilities = np.ones(count, dtype=float)
        for mode in self.order:
            parent_masks = np.zeros(count, dtype=np.uint64)
            for local, parent in enumerate(self.parents[mode]):
                parent_masks |= ((masks >> parent) & 1) << local
            click = self.conditional_tables[mode][parent_masks.astype(np.intp)]
            probabilities *= np.where((masks >> mode) & 1, click, 1.0 - click)
        if not np.isclose(probabilities.sum(), 1.0, rtol=0.0, atol=2e-10):
            raise ArithmeticError("a Bayesian-network distribution failed normalization")
        return probabilities

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Generate n independent bit strings, shape (n, m), in original mode order."""
        if not isinstance(n, (int, np.integer)) or n < 0:
            raise ValueError("n must be a nonnegative integer")
        rng = np.random.default_rng() if rng is None else rng
        samples = np.zeros((n, self.m), dtype=np.uint8)
        for mode in self.order:
            parent_masks = np.zeros(n, dtype=np.intp)
            for local, parent in enumerate(self.parents[mode]):
                parent_masks |= samples[:, parent].astype(np.intp) << local
            click = self.conditional_tables[mode][parent_masks]
            samples[:, mode] = rng.random(n) < click
        return samples


def fit_surrogate(
    model: MarginalOracle,
    max_parents: int,
    strategy: str = "pairwise",
    order: Sequence[int] | None = None,
) -> BoundedParentSurrogate:
    """Fit from exact low-order marginals without using the full target.

    Strategies are ``independent``, ``previous``, ``pairwise`` and ``greedy_cmi``.
    ``independent`` ignores max_parents and uses no parents.  Pairwise selection
    ranks I(child; candidate); greedy selection ranks I(child; candidate | the
    already selected parents).  Greedy selection fills the entire budget even
    if its next gain is zero, because zero pairwise information does not rule
    out higher-order dependence.  Ties within 1e-12 are broken by mode index.

    Zero-mass parent configurations receive a conditional click probability
    of 1/2.  This defines a valid network everywhere without pretending those
    undefined target conditionals are known.
    """
    if not isinstance(max_parents, (int, np.integer)) or max_parents < 0:
        raise ValueError("max_parents must be a nonnegative integer")
    if strategy not in {"independent", "previous", "pairwise", "greedy_cmi"}:
        raise ValueError(f"unknown parent selection strategy: {strategy}")
    m = int(model.m)
    if m < 1:
        raise ValueError("model must contain at least one mode")
    mode_order = tuple(range(m)) if order is None else tuple(int(i) for i in order)
    if sorted(mode_order) != list(range(m)):
        raise ValueError("order must be a permutation of all modes")
    effective_budget = 0 if strategy == "independent" else int(max_parents)
    cache = _MarginalCache(model)
    parent_tuples: list[tuple[int, ...]] = [()] * m
    tables: list[np.ndarray | None] = [None] * m
    traces: list[dict[str, Any]] = []
    undefined_rows = 0
    position = {mode: i for i, mode in enumerate(mode_order)}

    for i, target in enumerate(mode_order):
        preceding = list(mode_order[:i])
        budget = min(effective_budget, len(preceding))
        selected: list[int] = []
        gains: list[float] = []
        if budget and strategy == "previous":
            selected = preceding[-budget:]
        elif budget and strategy == "pairwise":
            scores = {
                candidate: _conditional_mutual_information(cache, target, candidate, ())
                for candidate in preceding
            }
            # Explicit tie handling avoids a platform-dependent ordering of
            # nearly equal floating-point scores.
            remaining = list(preceding)
            for _ in range(budget):
                best = max(scores[candidate] for candidate in remaining)
                chosen = min(c for c in remaining if scores[c] >= best - 1e-12)
                selected.append(chosen)
                gains.append(scores[chosen])
                remaining.remove(chosen)
        elif budget and strategy == "greedy_cmi":
            remaining = list(preceding)
            for _ in range(budget):
                scores = {
                    candidate: _conditional_mutual_information(
                        cache, target, candidate, selected
                    )
                    for candidate in remaining
                }
                best = max(scores.values())
                chosen = min(c for c in remaining if scores[c] >= best - 1e-12)
                selected.append(chosen)
                gains.append(scores[chosen])
                remaining.remove(chosen)
        parents = tuple(sorted(selected, key=position.__getitem__))
        joint = cache.marginal(parents + (target,))
        width = 1 << len(parents)
        no_click, click = joint[:width], joint[width:]
        denominator = no_click + click
        cpt = np.full(width, 0.5, dtype=float)
        np.divide(click, denominator, out=cpt, where=denominator > 0.0)
        undefined_rows += int(np.count_nonzero(denominator == 0.0))
        parent_tuples[target] = parents
        tables[target] = cpt
        traces.append({
            "mode": target,
            "selection_order": selected,
            "selected_information_nats": gains,
        })

    diagnostics: dict[str, Any] = {
        "strategy": strategy,
        "requested_max_parents": int(max_parents),
        "effective_max_parents": min(effective_budget, m - 1),
        "oracle_calls": cache.oracle_calls,
        "marginal_requests": cache.requests,
        "unique_cached_marginals": len(cache.cache) - 1,
        "largest_marginal_order": cache.max_order,
        "total_oracle_probability_entries": cache.returned_states,
        "undefined_conditional_rows": undefined_rows,
        "selection_trace": traces,
    }
    return BoundedParentSurrogate(
        m=m,
        order=mode_order,
        parents=tuple(parent_tuples),
        conditional_tables=tuple(tables),  # all entries populated above
        diagnostics=diagnostics,
    )


def distribution_metrics(p: np.ndarray, q: np.ndarray) -> dict[str, float]:
    """Exact dense-distribution metrics; KL is infinity for lost target support."""
    p = _validated_distribution(p, name="p")
    q = _validated_distribution(q, name="q")
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    positive = p > 0
    missed = positive & (q == 0)
    kl = (
        float("inf") if np.any(missed)
        else float(np.dot(p[positive], np.log(p[positive]) - np.log(q[positive])))
    )
    if np.isfinite(kl) and kl < 0 and kl > -1e-12:
        kl = 0.0
    return {
        "tv": float(0.5 * np.abs(p - q).sum()),
        "kl": kl,
        "max_abs_error": float(np.max(np.abs(p - q))),
        "hellinger_squared": float(0.5 * np.square(np.sqrt(p) - np.sqrt(q)).sum()),
        "support_missed_mass": float(p[missed].sum()),
        "p_entropy": _entropy_unchecked(p),
        "q_entropy": _entropy_unchecked(q),
    }


def _dense_marginal(p: np.ndarray, modes: Sequence[int]) -> np.ndarray:
    masks = np.arange(len(p), dtype=np.uint64)
    local_masks = np.zeros(len(p), dtype=np.uint64)
    for local, mode in enumerate(modes):
        local_masks |= ((masks >> mode) & 1) << local
    return np.bincount(local_masks.astype(np.intp), weights=p, minlength=1 << len(modes))


def kl_chain_decomposition(
    p: np.ndarray, surrogate: BoundedParentSurrogate
) -> dict[str, Any]:
    """Validate the omitted-conditional-information identity on a small target.

    If surrogate CPTs equal the corresponding P-conditionals, then
    KL(P || Q) = sum_i H_P(X_i | parents_i) - H_P(X_all).
    This routine requires the full exact target and therefore is NOT a
    scalable certificate of the quality of a large-mode approximation.
    """
    p = _validated_distribution(p, name="p")
    if len(p) != 1 << surrogate.m:
        raise ValueError("p length does not match the surrogate mode count")
    conditional_entropies: list[float] = []
    omitted_information: list[float] = []
    prefix: tuple[int, ...] = ()
    previous_entropy = 0.0
    for mode in surrogate.order:
        parents = surrogate.parents[mode]
        conditional = _entropy_unchecked(_dense_marginal(p, parents + (mode,)))
        conditional -= _entropy_unchecked(_dense_marginal(p, parents))
        prefix = prefix + (mode,)
        prefix_entropy = _entropy_unchecked(_dense_marginal(p, prefix))
        full_conditional = prefix_entropy - previous_entropy
        conditional_entropies.append(conditional)
        omitted_information.append(conditional - full_conditional)
        previous_entropy = prefix_entropy
    return {
        "target_entropy": _entropy_unchecked(p),
        "conditional_entropies": conditional_entropies,
        "omitted_information_per_mode": omitted_information,
        "kl_from_conditional_entropies": float(sum(conditional_entropies) - _entropy_unchecked(p)),
    }
