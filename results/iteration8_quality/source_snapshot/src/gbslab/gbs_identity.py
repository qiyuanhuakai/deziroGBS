"""Evidence-graded validation against a specified threshold-GBS target.

This module does not classify a histogram as "Gaussian".  A physical Gaussian
covariance defines a target state; threshold detection generally produces a
non-Gaussian discrete distribution.  Agreement with that *specified* target
is a separate question from physicality, detector scope, and sampler speed.

Evidence levels are complete small-system laws, a conservative global-TV
upper bound from an iid validation batch and path densities, and necessary
bounded-observable witnesses.  Passing witnesses never proves global
agreement.  All Gaussian probabilities here are guarded floating-point
estimates, not outward-rounded interval certificates.

Failure outcomes are retained mathematically.  Let Q_ext be the output of one
sampling attempt on bitstrings plus a symbol bottom for an explicit numerical
abort, and let P_ext(bottom)=0.  A complete bitstring has the unconditional
product of its branch probabilities, even if the masses on bitstrings sum to
less than one.  Then

    TV(P_ext,Q_ext) = E_Q_ext[(1-P_ext(X)/Q_ext(X))_+].

The score at bottom is one.  A fixed-N protocol that aborts validation at its
first failed attempt may report a bound only when all N attempts complete.
The event of a false acceptance is then a subset of the usual unconditional
Hoeffding tail event.  This does NOT give coverage conditional on obtaining a
successful batch, and retrying a failed batch until one succeeds invalidates
the protocol.  No assumption of zero unobserved abort mass is required.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
from time import perf_counter
from typing import Callable

import numpy as np

from .exact import NumericalProbabilityError, ThresholdModel
from .tv_validation import GaussianThresholdPointOracle, tv_score_from_log_probabilities


SCHEMA_VERSION = 1
_NUMERICAL_FAILURES = (ArithmeticError, ValueError, np.linalg.LinAlgError)
_QUALIFICATION = (
    "Gaussian probabilities and the confidence arithmetic use guarded floating "
    "point, not outward-rounded intervals. Statistical statements assume a "
    "target and sampler fixed before validation, iid attempts under the random "
    "number model, unconditional path densities, and a first fixed-size batch "
    "with no retries, filtering, or postselection.")


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def covariance_sha256(covariance):
    """Hash the canonical real symmetric float64 matrix in C order."""
    covariance = np.asarray(covariance, dtype=float)
    canonical = np.ascontiguousarray(covariance * .5 + covariance.T * .5)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


@dataclass(frozen=True)
class ConfidencePlan:
    """A comparison family fixed before samples; unused budgets are not reused.

    Each comparison receives alpha / number_of_comparisons.  Half belongs to
    the global-TV upper bound; half to all predeclared witnesses together.
    The fixed family can reuse a sample batch across comparisons; the union
    bound does not require independence *between* comparisons.  Within each
    comparison the sampling assumptions remain necessary.
    """

    comparison_ids: tuple[str, ...]
    familywise_alpha: float = .05

    def __post_init__(self):
        ids = tuple(self.comparison_ids)
        if (not ids or any(not isinstance(x, str) or not x for x in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError("comparison_ids must be nonempty unique strings")
        if not math.isfinite(self.familywise_alpha) or not 0 < self.familywise_alpha < 1:
            raise ValueError("familywise_alpha must be strictly between zero and one")
        object.__setattr__(self, "comparison_ids", ids)

    def as_dict(self):
        record = {"comparison_ids": list(self.comparison_ids),
                  "planned_comparisons": len(self.comparison_ids),
                  "familywise_alpha": float(self.familywise_alpha),
                  "familywise_nominal_confidence": 1-float(self.familywise_alpha),
                  "per_comparison_alpha": float(self.familywise_alpha)/len(self.comparison_ids),
                  "channel_split": {"global_tv_upper": .5, "simultaneous_witnesses": .5},
                  "schedule_rule": "fixed before samples; no reuse of unused alpha or posthoc selection"}
        record["plan_sha256"] = _json_hash(record)
        return record

    def allocation(self, comparison_id):
        if comparison_id not in self.comparison_ids:
            raise ValueError("comparison_id is not in the predeclared confidence family")
        alpha = float(self.familywise_alpha)/len(self.comparison_ids)
        return {**self.as_dict(), "comparison_id": comparison_id,
                "alpha_global_tv": alpha/2, "alpha_witnesses": alpha/2}


def default_witness_subsets(modes):
    """Deterministic O(m) no-click events chosen without seeing samples.

    All single modes, nearest-neighbor pairs, opposing half-pairs, and
    disjoint blocks of up to four modes are included.  The opposing pairs
    also catch the blocked two-mode-squeezed fixture; this is not a claim
    that the schedule is complete for arbitrary Gaussian distributions.
    """
    if isinstance(modes, (bool, np.bool_)) or not isinstance(modes, (int, np.integer)) or modes < 1:
        raise ValueError("modes must be a positive integer")
    m = int(modes)
    subsets = [(j,) for j in range(m)]
    subsets += [(j, j+1) for j in range(m-1)]
    subsets += [(j, j+m//2) for j in range(m//2) if j != j+m//2]
    subsets += [tuple(range(j, min(m, j+4))) for j in range(0, m, 4)]
    return tuple(dict.fromkeys(tuple(sorted(x)) for x in subsets))


def fixed_batch_protocol(requested_draws, *, source="caller_attested"):
    """Construct an explicit protocol attestation, not a proof of provenance.

    Call this only when the actual experiment uses its first prespecified
    batch, no retry/selection, and a target/sampler fixed in advance.  It
    never attests that unseen numerical guards have zero probability.
    """
    if (isinstance(requested_draws, (bool, np.bool_))
            or not isinstance(requested_draws, (int, np.integer)) or requested_draws <= 0):
        raise ValueError("requested_draws must be a positive integer")
    return {"requested_draws": int(requested_draws), "target_and_sampler_fixed_before_samples": True,
            "iid_attempts": True, "first_batch_no_retry": True, "no_postselection": True,
            "density_is_unconditional_path_mass": True, "source": str(source),
            "unseen_failure_mass_assumed_zero": False,
            "randomness_qualification": "iid mathematical model; finite PRNG streams are not independent random real numbers"}


def _protocol_check(protocol, n):
    required = ("target_and_sampler_fixed_before_samples", "iid_attempts",
                "first_batch_no_retry", "no_postselection")
    if not isinstance(protocol, dict):
        return False, ["No fixed-iid-first-batch protocol attestation was supplied"]
    missing = [key for key in required if protocol.get(key) is not True]
    if protocol.get("requested_draws") != n:
        missing.append("requested_draws must equal the complete observed batch size")
    return not missing, missing


def _binary_samples(samples, modes):
    arr = np.asarray(samples)
    if (arr.ndim != 2 or arr.shape[1] != modes or len(arr) == 0
            or not np.all((arr == 0) | (arr == 1))):
        raise ValueError("samples must be a nonempty binary array with one physical-mode column per mode")
    return np.ascontiguousarray(arr, dtype=np.uint8)


def _probability_vector(values, length):
    q = np.asarray(values, dtype=float)
    if (q.shape != (length,) or not np.all(np.isfinite(q)) or np.any(q < 0)
            or np.any(q > 1)):
        raise ValueError("candidate distribution must be a finite nonnegative probability vector")
    mass = float(np.sum(q, dtype=np.longdouble))
    if abs(mass-1) > 1e-10:
        raise ValueError("complete candidate distribution is not normalized")
    # Expose this numerically negligible normalization instead of hiding it.
    return q/mass, {"supplied_mass": mass, "normalization_l1_change": abs(mass-1)}


def _density_check(samples, log_probabilities):
    logs = np.asarray(log_probabilities, dtype=float)
    if logs.shape != (len(samples),) or not np.all(np.isfinite(logs)) or np.any(logs > 0):
        raise ValueError("successful-path log probabilities must be finite, nonpositive, and match samples")
    seen = {}
    for sample, logq in zip(samples, logs):
        key = np.packbits(sample, bitorder="little").tobytes()
        if key in seen and abs(seen[key]-float(logq)) > 2e-10:
            raise ValueError("the same bitstring has inconsistent Q density in a fixed sampler")
        seen[key] = float(logq)
    distinct_mass = math.fsum(math.exp(value) for value in seen.values())
    if distinct_mass > 1+1e-10:
        raise ValueError("observed distinct-bitstring Q masses already exceed one")
    return logs, {"distinct_patterns": len(seen), "observed_distinct_mass": distinct_mass,
                  "normalization_scope": "necessary partial-mass check only; total law follows from the declared output-process construction"}


def sample_tv_upper(samples, sample_log_probabilities, target_log_probability: Callable,
                    *, alpha, max_exact_clicks=6, keep_scores=False, cache_size=8192):
    """One conservative upper bound, under the module's fixed-batch protocol.

    The target callback returns a log probability, or log lower bound.  Values
    above the click cap and unresolved callback evaluations receive score one.
    They are never removed from the denominator.  Malformed Q densities are
    invalid evidence rather than a reason to silently resample.

    An actual sampling failure is handled *before* this function by aborting
    assessment.  The bound covers the failure-inclusive output distribution
    unconditionally over prespecified attempts, not conditional on success.
    """
    if not callable(target_log_probability):
        raise ValueError("target_log_probability must be callable")
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")
    if (isinstance(max_exact_clicks, (bool, np.bool_))
            or not isinstance(max_exact_clicks, (int, np.integer)) or max_exact_clicks < 0):
        raise ValueError("max_exact_clicks must be a nonnegative integer")
    if not isinstance(cache_size, (int, np.integer)) or cache_size < 0:
        raise ValueError("cache_size must be nonnegative")
    arr = np.asarray(samples)
    if arr.ndim != 2 or arr.shape[1] == 0:
        raise ValueError("samples must be a two-dimensional binary array")
    arr = _binary_samples(arr, arr.shape[1])
    logs, density_diagnostics = _density_check(arr, sample_log_probabilities)
    n = len(arr)
    scores = np.ones(n, dtype=float)
    click_counts = arr.sum(axis=1)
    cache = OrderedDict()
    reasons = Counter()
    cutoff_count = failure_count = successful_count = cache_hits = evaluations = 0
    started = perf_counter()
    for j, (sample, logq) in enumerate(zip(arr, logs)):
        if click_counts[j] > max_exact_clicks:
            cutoff_count += 1
            continue
        key = np.packbits(sample, bitorder="little").tobytes()
        if key in cache:
            value, error = cache[key]
            cache.move_to_end(key)
            cache_hits += 1
        else:
            evaluations += 1
            try:
                logp = float(target_log_probability(sample))
                value = tv_score_from_log_probabilities(logp, float(logq))
                error = None
            except _NUMERICAL_FAILURES as exc:
                value, error = 1.0, f"{type(exc).__name__}: {exc}"
            if cache_size:
                if len(cache) >= cache_size:
                    cache.popitem(last=False)
                cache[key] = (value, error)
        scores[j] = value
        if error is None:
            successful_count += 1
        else:
            failure_count += 1
            reasons[error] += 1
    margin = math.sqrt(math.log(1/alpha)/(2*n))
    result = {"status": "ok", "n_requested": n, "n_observed": n,
              "alpha": float(alpha), "nominal_confidence": 1-float(alpha),
              "mean_score": float(scores.mean()), "statistical_margin": margin,
              "upper": min(1.0, float(scores.mean())+margin),
              "max_exact_clicks": int(max_exact_clicks), "cutoff_count": cutoff_count,
              "target_probability_failure_count": failure_count,
              "successfully_evaluated_count": successful_count,
              "unique_target_probability_evaluations": evaluations, "pattern_cache_hits": cache_hits,
              "failure_reasons": dict(reasons), "density_check": density_diagnostics,
              "output_space": "binary_patterns_plus_failure",
              "target_failure_probability": 0.0, "observed_sampling_failures": 0,
              "unseen_failure_mass_assumed_zero": False,
              "confidence_scope": "unconditional false-accept guarantee for first fixed-N iid attempts; validation aborts on any sampling failure",
              "conditional_on_success_coverage_claimed": False,
              "conditional_success_law_tv_also_bounded": True,
              "strict_floating_point_certificate": False,
              "qualification": _QUALIFICATION,
              "elapsed_seconds": perf_counter()-started}
    if keep_scores:
        result.update(scores=scores.tolist(), click_counts=click_counts.tolist())
    return result


def _witnesses(model, samples, subsets, *, alpha, statistical_protocol_valid):
    m, n = model.m, len(samples)
    normalized = []
    for subset in subsets:
        if (not isinstance(subset, (list, tuple)) or not subset
                or any(isinstance(j, (bool, np.bool_)) or not isinstance(j, (int, np.integer))
                       or j < 0 or j >= m for j in subset)
                or len(set(subset)) != len(subset)):
            raise ValueError("witness subsets must contain distinct valid physical mode IDs")
        normalized.append(tuple(sorted(int(j) for j in subset)))
    if len(set(normalized)) != len(normalized):
        raise ValueError("duplicate witnesses are not allowed")
    # One additional bounded observable is mean total clicks divided by m.
    count = len(normalized)+1
    radius = math.sqrt(math.log(2*count/alpha)/(2*n)) if statistical_protocol_valid else None
    rows, failures = [], []
    for subset in normalized:
        try:
            target = float(model.vacuum_probability(subset))
            observed = float(np.all(samples[:, subset] == 0, axis=1).mean())
            discrepancy = abs(observed-target)
            rows.append({"kind": "no_click_subset", "modes": list(subset),
                         "target": target, "observed": observed, "absolute_difference": discrepancy,
                         "tv_lower": max(0.0, discrepancy-radius) if radius is not None else None})
        except _NUMERICAL_FAILURES as exc:
            failures.append({"kind": "no_click_subset", "modes": list(subset),
                             "error_type": type(exc).__name__, "error": str(exc)})
    target_mean = math.fsum(1-float(model.vacuum_probability((j,))) for j in range(m))
    observed_mean = float(samples.sum(axis=1).mean())
    difference = abs(observed_mean-target_mean)/m
    rows.append({"kind": "normalized_total_clicks", "normalization": m,
                 "target": target_mean/m, "observed": observed_mean/m,
                 "absolute_difference": difference,
                 "tv_lower": max(0.0, difference-radius) if radius is not None else None})
    return {"status": "ok" if not failures else "partial_numerical_failure",
            "planned_witnesses": count, "evaluated_witnesses": len(rows),
            "witness_schedule_sha256": _json_hash([list(s) for s in normalized]),
            "predeclared_subsets": [list(s) for s in normalized],
            "alpha": float(alpha), "simultaneous_hoeffding_radius": radius,
            "tv_lower": max(r["tv_lower"] for r in rows) if radius is not None else None,
            "target_mean_clicks": target_mean, "observed_mean_clicks": observed_mean,
            "rows": rows, "numerical_failures": failures,
            "scope": "necessary bounded-observable tests; passing does not certify the full law",
            "output_space": "binary_patterns_plus_failure",
            "failure_symbol_observable_value": 0.0,
            "strict_floating_point_certificate": False}


def assess_gbs_identity(
    covariance, *, candidate_distribution=None, samples=None, sample_log_probabilities=None,
    epsilon=.1, plan: ConfidencePlan | None = None, comparison_id=None,
    witness_subsets=None, max_exact_clicks=6, protocol=None,
    detector_model="threshold", state_model="zero_mean_gaussian", displacement=None,
    quadrature_order="xxpp", hbar=2, target_id=None, full_law_max_modes=14,
    max_model_modes=256, keep_scores=False, target_log_probability=None,
    target_probability_mode=None, sampling_failure=None,
):
    """Assess agreement with one specified GBS target and a fixed TV tolerance.

    ``protocol`` is an explicit experiment attestation (see
    :func:`fixed_batch_protocol`), not inferred from a batch happening to
    finish.  Missing attestation permits descriptive witnesses only.  A
    complete candidate law can still be compared numerically without an iid
    assumption.  A supplied target callback owns its probability/lower-bound
    claim; the default is the existing guarded Gaussian point oracle.
    """
    if not math.isfinite(epsilon) or not 0 <= epsilon <= 1:
        raise ValueError("epsilon must lie in [0,1]")
    if not isinstance(full_law_max_modes, (int, np.integer)) or not 0 <= full_law_max_modes <= 20:
        raise ValueError("full_law_max_modes must be between zero and20")
    if plan is not None and not isinstance(plan, ConfidencePlan):
        raise ValueError("plan must be a ConfidencePlan")
    allocation = plan.allocation(comparison_id) if plan is not None else None
    result = {"schema_version": SCHEMA_VERSION, "decision": "insufficient_evidence",
              "evidence_scope": "none", "epsilon": float(epsilon),
              "confidence_plan": allocation, "model_check": {}, "target": {
                  "target_id": target_id, "detector_model": detector_model, "state_model": state_model,
                  "quadrature_order": quadrature_order, "hbar": hbar,
                  "covariance_convention": "xxpp;hbar=2;vacuum=I"},
              "full_law": None, "global_tv": None, "witnesses": None,
              "sampling_protocol": protocol, "strict_floating_point_certificate": False,
              "qualification": _QUALIFICATION,
              "evidence_trust_boundary": "A provided full law or provenance hash is evidence to verify, not cryptographic proof of an arbitrary sampler's honesty; implementation-equivalence checks and sample consistency remain required",
              "identity_scope": "agreement with the specified target, not existence of some Gaussian model that could fit the data",
              "reasons": []}
    started = perf_counter()
    if detector_model != "threshold" or state_model != "zero_mean_gaussian" or quadrature_order != "xxpp" or hbar != 2:
        result.update(decision="out_of_scope", model_check={"valid": None, "in_scope": False})
        result["reasons"].append("This evaluator supports zero-mean Gaussian threshold GBS in hbar2/xxpp only; other GBS variants are not declared invalid")
        return result
    try:
        if np.iscomplexobj(covariance):
            raise ValueError("covariance must be real in xxpp quadratures")
        arr = np.asarray(covariance, dtype=float)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] == 0 or arr.shape[0] % 2:
            raise ValueError("covariance must be a nonempty even-dimensional square matrix")
        modes = arr.shape[0]//2
        if modes > max_model_modes:
            result.update(decision="out_of_scope", model_check={"valid": None, "in_scope": False})
            result["reasons"].append(f"Covariance physicality resource guard is {max_model_modes} modes")
            return result
        if displacement is not None:
            mean = np.asarray(displacement, dtype=float)
            if mean.shape != (2*modes,) or not np.all(np.isfinite(mean)):
                raise ValueError("displacement must be a finite 2m vector")
            if np.any(mean != 0):
                result.update(decision="out_of_scope", model_check={"valid": None, "in_scope": False})
                result["reasons"].append("Nonzero displacement is outside this evaluator, although displaced Gaussian sampling is a valid separate model")
                return result
        model = ThresholdModel(arr, max_exact_modes=full_law_max_modes, vacuum_cache_size=8192)
    except _NUMERICAL_FAILURES as exc:
        result.update(decision="invalid_model", model_check={"valid": False, "in_scope": True,
                      "error_type": type(exc).__name__, "error": str(exc)})
        result["reasons"].append("The supplied covariance failed the specified physical-input checks")
        return result
    result["target"].update(modes=modes, covariance_sha256=covariance_sha256(model.V),
                            mean_photons=float((np.trace(model.V)-2*modes)/4))
    result["model_check"] = {"valid": True, "in_scope": True,
                             "minimum_uncertainty_eigenvalue": model.minimum_uncertainty_eigenvalue,
                             "physicality_is_distribution_validation": False,
                             "resource_max_modes": int(max_model_modes)}
    if sampling_failure is not None:
        result.update(decision="evaluation_failed", sampling_failure=sampling_failure)
        result["reasons"].append("Sampling failed or the fixed batch was incomplete; no retry, filtering, or acceptance is permitted")
        return result
    q = p = observed = logs = None
    try:
        if candidate_distribution is not None:
            if modes > full_law_max_modes:
                raise ValueError("full candidate-law evaluation exceeds the explicit resource cap")
            q, normalization = _probability_vector(candidate_distribution, 1 << modes)
            p = model.full_distribution()
            tv = .5*float(np.sum(np.abs(p-q), dtype=np.longdouble))
            result["full_law"] = {"tv": tv, "within_tolerance": bool(tv <= epsilon),
                                   "patterns": int(len(q)), "epsilon": float(epsilon),
                                   "candidate_normalization": normalization,
                                   "target_diagnostics": model.cache_info(),
                                   "scope": "all bitstrings, numerically enumerated; complete provided Q has no declared failure mass",
                                   "strict_floating_point_certificate": False}
            result["evidence_scope"] = "full_distribution"
        if samples is not None:
            observed = _binary_samples(samples, modes)
            result["samples_sha256"] = hashlib.sha256(observed.tobytes()).hexdigest()
            result["n_observed"] = len(observed)
            if sample_log_probabilities is not None:
                logs, check = _density_check(observed, sample_log_probabilities)
                result["density_check"] = check
            if q is not None:
                masks = observed.astype(np.int64) @ (1 << np.arange(modes))
                if np.any(q[masks] <= 0):
                    raise ValueError("sample has zero probability under the provided complete Q")
                full_logs = np.log(q[masks])
                if logs is not None and not np.allclose(logs, full_logs, atol=2e-10, rtol=0):
                    raise ValueError("sample log densities disagree with the complete normalized Q")
                if logs is None:
                    logs = full_logs
                    result["density_check"] = {"source": "provided_complete_Q"}
                result["density_check"]["checked_against_complete_Q"] = True
        elif sample_log_probabilities is not None:
            raise ValueError("sample log densities were supplied without samples")
    except _NUMERICAL_FAILURES as exc:
        result.update(decision="invalid_evidence", evidence_error={"error_type": type(exc).__name__, "error": str(exc)})
        result["reasons"].append("Probability or sample evidence failed consistency checks")
        return result
    if observed is not None:
        valid_protocol, protocol_errors = _protocol_check(protocol, len(observed))
        statistical = valid_protocol and allocation is not None
        result["statistical_protocol_valid"] = statistical
        result["statistical_protocol_issues"] = protocol_errors + ([] if allocation else ["No predeclared confidence plan"])
        subsets = default_witness_subsets(modes) if witness_subsets is None else witness_subsets
        alpha_w = allocation["alpha_witnesses"] if allocation else .025
        try:
            result["witnesses"] = _witnesses(model, observed, subsets, alpha=alpha_w,
                                            statistical_protocol_valid=statistical)
            if result["evidence_scope"] == "none":
                result["evidence_scope"] = "witnesses_only"
        except _NUMERICAL_FAILURES as exc:
            result["witnesses"] = {"status": "evaluation_failed", "tv_lower": None,
                                   "error_type": type(exc).__name__, "error": str(exc)}
        density_attested = (q is not None or (isinstance(protocol, dict)
                            and protocol.get("density_is_unconditional_path_mass") is True))
        if statistical and logs is not None and density_attested:
            oracle = None
            try:
                if target_log_probability is None:
                    oracle = GaussianThresholdPointOracle(model.V, min(int(max_exact_clicks), modes))
                    callback = oracle.log_probability
                    mode = "floating_gaussian_point_estimate"
                else:
                    callback = target_log_probability
                    mode = target_probability_mode or "caller_supplied_target_log_probability"
                result["global_tv"] = sample_tv_upper(
                    observed, logs, callback, alpha=allocation["alpha_global_tv"],
                    max_exact_clicks=max_exact_clicks, keep_scores=keep_scores)
                result["global_tv"]["target_probability_mode"] = mode
                result["global_tv"]["target_oracle_diagnostics"] = oracle.cache_info() if oracle else None
                result["global_tv"]["fixed_iid_no_retry_assumptions"] = dict(protocol)
                if result["evidence_scope"] != "full_distribution":
                    result["evidence_scope"] = "global_tv_upper"
            except _NUMERICAL_FAILURES as exc:
                result["global_tv"] = {"status": "evaluation_failed", "upper": None,
                                       "error_type": type(exc).__name__, "error": str(exc)}
        elif logs is not None:
            result["reasons"].append("A global bound requires a fixed confidence plan, valid first-batch protocol, and unconditional density provenance")
    full = result["full_law"]
    global_tv = result["global_tv"] or {}
    witnesses = result["witnesses"] or {}
    lower = witnesses.get("tv_lower")
    upper = global_tv.get("upper")
    if full is not None:
        result["decision"] = "accepted_within_tolerance_conditional" if full["within_tolerance"] else "rejected"
        result["decision_basis"] = "complete_numerical_law"
        result["reasons"].append("Full numerical TV is within the preset tolerance" if full["within_tolerance"]
                                 else "Full numerical TV exceeds the preset tolerance")
    elif lower is not None and lower > epsilon:
        result["decision"] = "rejected"
        result["decision_basis"] = "simultaneous_witness_lower_bound"
        result["reasons"].append("A predeclared bounded observable gives a TV lower bound above tolerance")
    elif upper is not None and upper <= epsilon:
        result["decision"] = "accepted_within_tolerance_conditional"
        result["decision_basis"] = "global_statistical_upper_bound"
        result["reasons"].append("The failure-inclusive global TV upper bound is within the preset tolerance, under the recorded assumptions")
    else:
        result["decision"] = "insufficient_evidence"
        result["decision_basis"] = "no_global_acceptance_or_rejection_bound"
        result["reasons"].append("Passing witnesses or obtaining an upper bound above tolerance does not decide full-distribution agreement")
    if (full is not None and lower is not None and lower > full["tv"]+1e-10):
        result["sample_law_consistency_warning"] = (
            "The witness confidence lower bound exceeds the provided full-law TV; "
            "this can be a statistical tail event or indicate that samples are not from the declared law")
        result["decision"] = "invalid_evidence"
        result["reasons"].append("Samples conflict with the declared complete law at the allocated witness confidence; do not accept")
    if upper is not None and lower is not None and lower > upper:
        result["statistical_evidence_conflict"] = True
        result["decision"] = "invalid_evidence"
        result["reasons"].append("Simultaneous statistical lower and upper bounds conflict; do not accept")
    result["elapsed_seconds"] = perf_counter()-started
    return result


def assess_native_samples(covariance, native_result, *, protocol=None, **options):
    """Assess a completed native wrapper result, preserving failure semantics.

    The caller supplies the prespecified fixed-batch protocol; the presence
    of an apparently successful native JSON object is not that attestation.
    Native log masses remain unconditional successful-path probabilities.
    """
    if not isinstance(native_result, dict) or native_result.get("status") != "ok":
        return assess_gbs_identity(covariance, protocol=protocol,
            sampling_failure=native_result if isinstance(native_result, dict) else {"error": str(native_result)},
            **options)
    if (native_result.get("raw_samples") is None
            or native_result.get("sample_log_probabilities") is None
            or native_result.get("log_probability_enabled") is not True):
        raise ValueError("native assessment requires raw_samples and opted-in sample log probabilities")
    m = np.asarray(covariance).shape[0]//2
    if native_result.get("modes") != m:
        raise ValueError("native result modes do not match the target")
    expected_hash = covariance_sha256(covariance)
    if native_result.get("covariance_sha256") != expected_hash:
        raise ValueError("native result is not bound to the supplied target covariance")
    if len(native_result["raw_samples"]) != native_result.get("sample_count"):
        raise ValueError("native result does not contain the complete requested batch")
    if isinstance(protocol, dict) and protocol.get("requested_draws") != native_result.get("sample_count"):
        return assess_gbs_identity(covariance, protocol=protocol,
            sampling_failure={"error": "observed native batch does not match prespecified requested_draws"},
            **options)
    result = assess_gbs_identity(covariance, candidate_distribution=native_result.get("distribution"),
        samples=native_result["raw_samples"], sample_log_probabilities=native_result["sample_log_probabilities"],
        protocol=protocol, **options)
    result["native_provenance"] = {key: native_result.get(key) for key in (
        "executable_sha256", "build_cache_key", "build_metadata_matches_binary", "covariance_sha256",
        "selector", "k", "vacuum_kernel", "seed", "sample_count", "order", "uniform_source",
        "sample_log_probability_semantics", "uniforms_sha256")}
    result["native_provenance"]["binding_scope"] = (
        "Traceability to the declared covariance and executable/selector/kernel; "
        "not cryptographic verification of implementation correctness")
    return result
