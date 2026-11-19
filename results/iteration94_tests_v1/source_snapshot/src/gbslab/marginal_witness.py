"""Held-out marginal-event witnesses for a specified threshold-GBS law.

A pilot prefix chooses an event A in each small, predeclared mode window:
the patterns whose pilot frequencies exceed their exact Gaussian marginal
probabilities.  Only the disjoint held-out suffix estimates Q(A).  Thus

    TV(P, Q) >= Q(A) - P(A)

can be bounded without a full-system point probability or sampler logQ.
The event's direction is NEVER flipped after looking at held-out counts.
Passing this test does not upper-bound TV, even within a tested window.

All target probabilities are guarded floating-point calculations.  The
statistical guarantees are conditional on those calculations being correct,
on the declared fixed split and sampling protocol, and on iid attempts.
They cover the failure-inclusive output process unconditionally; retrying
until a batch succeeds or conditioning the coverage claim on success is
invalid.  The evaluator produces no bound after an observed failed attempt.
"""

from __future__ import annotations

import hashlib
import json
import math
from time import perf_counter

import numpy as np

from .exact import ThresholdModel


SCHEMA_VERSION = 1
_SPLIT_RULE = "first_pilot_rows_then_all_remaining_rows"
_NUMERICAL_FAILURES = (ArithmeticError, ValueError, np.linalg.LinAlgError)


def _integer(value, name, *, minimum=0):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _probability(value, name, *, open_interval=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite real probability")
    if np.iscomplexobj(value) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite real probability")
    value = float(value)
    if not (0 < value < 1 if open_interval else 0 <= value <= 1):
        raise ValueError(f"{name} is outside its probability range")
    return value


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _covariance_sha256(covariance):
    # Same canonical bytes as gbs_identity.covariance_sha256.  Keep this tiny
    # helper local so the main discriminator can import this optional witness
    # without introducing a circular module dependency.
    canonical = np.ascontiguousarray(covariance*.5+covariance.T*.5)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _normalize_windows(windows, *, modes=None, max_window_modes=None):
    if not isinstance(windows, (list, tuple)) or not windows:
        raise ValueError("windows must be a nonempty list or tuple")
    normalized, seen = [], set()
    for window in windows:
        if not isinstance(window, (list, tuple)) or not window:
            raise ValueError("each window must be a nonempty list or tuple")
        window = tuple(_integer(j, "mode index") for j in window)
        if len(set(window)) != len(window):
            raise ValueError("a window cannot repeat a mode")
        if modes is not None and any(j >= modes for j in window):
            raise ValueError("window mode index is outside the covariance")
        if max_window_modes is not None and len(window) > max_window_modes:
            raise ValueError("window exceeds max_window_modes")
        key = tuple(sorted(window))
        if key in seen:
            raise ValueError("duplicate mode windows are not allowed, including reordered duplicates")
        seen.add(key)
        normalized.append(window)
    return tuple(normalized)


def default_marginal_windows(modes, width=8):
    """Early, middle, late and evenly spaced windows, fixed by dimension only.

    Duplicate windows at small dimensions are removed.  Bit j of a marginal
    mask refers to the j-th physical-mode entry of that window.
    """
    modes = _integer(modes, "modes", minimum=1)
    width = min(_integer(width, "width", minimum=1), modes)
    first, middle, last = 0, (modes-width)//2, modes-width
    windows = [tuple(range(start, start+width)) for start in (first, middle, last)]
    # Integer arithmetic fixes endpoint rounding and gives distinct entries
    # because modes >= width.  A width-one sparse window coincides with early.
    sparse = (0,) if width == 1 else tuple(j*(modes-1)//(width-1) for j in range(width))
    windows.append(sparse)
    return tuple(dict.fromkeys(windows))


def fixed_split_protocol(requested_draws, pilot_size, *, windows, source="caller_attested"):
    """Record an actual prespecified experiment; this is an attestation only.

    Freeze and save this record BEFORE sampling.  The sampler must remain
    fixed after the pilot: this split trains a witness, never the sampler.
    Samples are split internally by row position, with no user-supplied
    indices that could overlap or select favorable held-out observations.
    """
    requested_draws = _integer(requested_draws, "requested_draws", minimum=2)
    pilot_size = _integer(pilot_size, "pilot_size", minimum=1)
    if pilot_size >= requested_draws:
        raise ValueError("pilot_size must leave a nonempty held-out suffix")
    windows = _normalize_windows(windows)
    if not isinstance(source, str) or not source:
        raise ValueError("source must be a nonempty string")
    return {
        "requested_draws": requested_draws, "pilot_size": pilot_size,
        "heldout_size": requested_draws-pilot_size, "split_rule": _SPLIT_RULE,
        "split_fixed_before_samples": True, "windows_fixed_before_samples": True,
        "windows_sha256": _json_hash([list(w) for w in windows]),
        "target_and_sampler_fixed_before_samples": True,
        "pilot_not_used_to_modify_sampler": True, "iid_attempts": True,
        "first_batch_no_retry": True, "no_postselection": True,
        "source": source, "unseen_failure_mass_assumed_zero": False,
        "randomness_qualification": "iid mathematical model; finite PRNG streams are not independent random real numbers",
    }


def _protocol_errors(protocol, *, observed, pilot_size, windows_sha256):
    if not isinstance(protocol, dict):
        return ["No fixed-split iid first-batch protocol attestation was supplied"]
    required = ("split_fixed_before_samples", "windows_fixed_before_samples",
                "target_and_sampler_fixed_before_samples", "pilot_not_used_to_modify_sampler",
                "iid_attempts", "first_batch_no_retry", "no_postselection")
    errors = [key+" must be true" for key in required if protocol.get(key) is not True]
    for key, expected in (("requested_draws", observed), ("pilot_size", pilot_size),
                          ("heldout_size", observed-pilot_size)):
        value = protocol.get(key)
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or value != expected):
            errors.append(key+" must be an integer matching the complete fixed split")
    if protocol.get("split_rule") != _SPLIT_RULE:
        errors.append("split_rule must use the pilot prefix and complete disjoint held-out suffix")
    if protocol.get("windows_sha256") != windows_sha256:
        errors.append("windows_sha256 does not match the predeclared windows")
    return errors


def hoeffding_event_tv_lower(successes, draws, target_probability, *, alpha):
    """One-sided lower bound Q(A)-P(A), for an event fixed before these draws.

    This primitive does not check how the event or the sample batch was
    chosen.  ``assess_marginal_witness`` supplies those protocol checks.
    ``alpha`` is the allocation for this ONE event, after all multiplicity.
    """
    draws = _integer(draws, "draws", minimum=1)
    successes = _integer(successes, "successes")
    if successes > draws:
        raise ValueError("successes cannot exceed draws")
    target_probability = _probability(target_probability, "target_probability")
    alpha = _probability(alpha, "alpha", open_interval=True)
    observed = successes/draws
    radius = math.sqrt(-math.log(alpha)/(2*draws))
    q_lower = max(0.0, observed-radius)
    return {"heldout_successes": successes, "heldout_draws": draws,
            "heldout_event_probability": observed,
            "target_event_probability": target_probability,
            "signed_discrepancy": observed-target_probability,
            "one_sided_hoeffding_radius": radius, "alpha": alpha,
            "q_event_lower": q_lower,
            "tv_lower": max(0.0, q_lower-target_probability)}


def _masks(samples, window):
    # The same supplied order is passed to ThresholdModel.marginal.  In its
    # xxpp covariance indexing, mode j selects quadratures j and j+m.
    return samples[:, window].astype(np.int64) @ (1 << np.arange(len(window), dtype=np.int64))


def assess_marginal_witness(
    covariance, samples=None, *, windows=None, pilot_size, alpha=.05,
    epsilon=.1, protocol=None, sampling_failure=None, target_id=None,
    max_window_modes=12, max_model_modes=256,
):
    """Return a rejection witness, never a global or marginal TV acceptance.

    ``alpha`` is the total budget assigned to THIS model/sampler comparison
    and THIS witness suite.  The caller must allocate it across other
    comparisons and evidence channels before sampling.  It is divided by
    the complete predeclared window count, including failed evaluations.

    Missing protocol allows descriptive pilot-selected event statistics but
    no lower bound.  A conflicting protocol, incomplete batch, or reported
    sampling failure never gets a statistical bound.  Exactly the first
    ``pilot_size`` rows train events; every remaining row is held out.
    """
    started = perf_counter()
    alpha = _probability(alpha, "alpha", open_interval=True)
    epsilon = _probability(epsilon, "epsilon")
    pilot_size = _integer(pilot_size, "pilot_size", minimum=1)
    max_window_modes = _integer(max_window_modes, "max_window_modes", minimum=1)
    max_model_modes = _integer(max_model_modes, "max_model_modes", minimum=1)
    if max_window_modes > 20:
        raise ValueError("max_window_modes cannot exceed the explicit cap of 20")
    result = {
        "schema_version": SCHEMA_VERSION, "decision": "insufficient_evidence",
        "status": "ok", "evidence_scope": "heldout_marginal_events_only",
        "epsilon": epsilon, "alpha": alpha, "tv_lower": None,
        "statistical_protocol_valid": False, "sampling_protocol": protocol,
        "target": {"target_id": target_id, "detector_model": "threshold",
                   "state_model": "zero_mean_gaussian", "quadrature_order": "xxpp",
                   "hbar": 2},
        "rows": [], "numerical_failures": [], "reasons": [],
        "strict_floating_point_certificate": False,
        "output_space": "binary_patterns_plus_failure",
        "failure_symbol_observable_value": 0.0,
        "unseen_failure_mass_assumed_zero": False,
        "conditional_on_success_coverage_claimed": False,
        "confidence_scope": "unconditional first fixed-N iid attempts; pilot and held-out suffix disjoint; no bound after any observed sampling failure",
        "global_tv_acceptance_supported": False,
        "qualification": "Target marginal probabilities and confidence arithmetic use guarded floating point, not outward-rounded intervals; passing these events does not prove full-law or marginal-law closeness.",
    }

    def finish():
        result["elapsed_seconds"] = perf_counter()-started
        return result

    if sampling_failure is not None:
        result.update(decision="evaluation_failed", status="sampling_failed",
                      sampling_failure=sampling_failure)
        result["reasons"].append("A fixed-batch attempt failed; no retry, filtering, or success-only bound is permitted")
        return finish()

    try:
        if np.iscomplexobj(covariance):
            raise ValueError("covariance must be real in xxpp quadratures")
        V = np.asarray(covariance, dtype=float)
        if V.ndim != 2 or V.shape[0] != V.shape[1] or V.shape[0] == 0 or V.shape[0] % 2:
            raise ValueError("covariance must be a nonempty even-dimensional square matrix")
        modes = V.shape[0]//2
        if modes > max_model_modes:
            result.update(decision="out_of_scope", status="model_resource_limit")
            result["reasons"].append(f"Covariance physicality resource guard is {max_model_modes} modes")
            return finish()
        model = ThresholdModel(V, max_exact_modes=max_window_modes, vacuum_cache_size=8192)
    except _NUMERICAL_FAILURES as exc:
        result.update(decision="invalid_model", status="invalid_model",
                      model_error={"error_type": type(exc).__name__, "error": str(exc)})
        return finish()
    result["target"].update(modes=modes, covariance_sha256=_covariance_sha256(model.V))
    result["model_check"] = {"valid": True,
                             "minimum_uncertainty_eigenvalue": model.minimum_uncertainty_eigenvalue,
                             "max_model_modes": max_model_modes}

    try:
        windows = _normalize_windows(default_marginal_windows(modes) if windows is None else windows,
                                     modes=modes, max_window_modes=max_window_modes)
        arr = np.asarray(samples)
        if (arr.ndim != 2 or arr.shape[1] != modes or len(arr) <= pilot_size
                or not np.all((arr == 0) | (arr == 1))):
            raise ValueError("samples must be a complete binary N-by-m array with nonempty pilot and held-out parts")
        arr = np.array(arr, dtype=np.uint8, order="C", copy=True)
        arr.setflags(write=False)
    except (TypeError, *_NUMERICAL_FAILURES) as exc:
        result.update(decision="invalid_evidence", status="invalid_evidence",
                      evidence_error={"error_type": type(exc).__name__, "error": str(exc)})
        return finish()

    n = len(arr)
    window_lists = [list(w) for w in windows]
    windows_hash = _json_hash(window_lists)
    result.update(planned_windows=len(windows), windows=window_lists,
                  windows_sha256=windows_hash, n_observed=n,
                  pilot_size=pilot_size, heldout_size=n-pilot_size,
                  split_rule=_SPLIT_RULE,
                  samples_sha256=hashlib.sha256(arr.tobytes()).hexdigest(),
                  pilot_samples_sha256=hashlib.sha256(arr[:pilot_size].tobytes()).hexdigest(),
                  heldout_samples_sha256=hashlib.sha256(arr[pilot_size:].tobytes()).hexdigest())
    errors = _protocol_errors(protocol, observed=n, pilot_size=pilot_size,
                              windows_sha256=windows_hash)
    result["statistical_protocol_issues"] = errors
    if protocol is not None and errors:
        requested = protocol.get("requested_draws") if isinstance(protocol, dict) else None
        incomplete = (not isinstance(requested, (bool, np.bool_))
                      and isinstance(requested, (int, np.integer)) and requested > n)
        result.update(decision="evaluation_failed" if incomplete else "invalid_evidence",
                      status="incomplete_batch" if incomplete else "invalid_protocol")
        result["reasons"].extend(errors)
        return finish()
    statistical = not errors
    result["statistical_protocol_valid"] = statistical
    per_window_alpha = alpha/len(windows)
    if per_window_alpha == 0:
        raise ValueError("per-window alpha underflows floating point")
    result["confidence_allocation"] = {
        "comparison_witness_alpha": alpha, "planned_windows": len(windows),
        "per_window_alpha": per_window_alpha,
        "method": "one-sided Hoeffding with Bonferroni across all predeclared windows",
        "other_comparisons": "caller must budget this alpha across models, samplers, and other evidence channels in advance",
        "unused_alpha_reallocated": False,
    }

    for index, window in enumerate(windows):
        try:
            target = model.marginal(window)
            pilot_masks = _masks(arr[:pilot_size], window)
            pilot_counts = np.bincount(pilot_masks, minlength=len(target))
            pilot_frequencies = pilot_counts/pilot_size
            # No held-out arrays or counts have entered the event selection.
            event = pilot_frequencies > target
            event_masks = np.flatnonzero(event).tolist()
            target_event = float(np.sum(target[event], dtype=np.longdouble))
            test_masks = _masks(arr[pilot_size:], window)
            heldout_counts = np.bincount(test_masks, minlength=len(target))
            successes = int(heldout_counts[event].sum())
            row = {
                "window_index": index, "modes": list(window),
                "pattern_encoding": "little-endian; bit j is modes[j]",
                "event_rule": "pilot frequency strictly greater than target; ties excluded; direction fixed",
                "event_masks": event_masks,
                "event_sha256": _json_hash({"modes": list(window), "event_masks": event_masks}),
                "target_probabilities": target.tolist(),
                "pilot_counts": pilot_counts.tolist(), "heldout_counts": heldout_counts.tolist(),
                "pilot_size": pilot_size,
                "pilot_empirical_tv_descriptive_only": .5*float(np.abs(pilot_frequencies-target).sum()),
                "pilot_event_probability": int(pilot_counts[event].sum())/pilot_size,
                **hoeffding_event_tv_lower(successes, n-pilot_size, target_event,
                                           alpha=per_window_alpha),
            }
            if not statistical:
                row.update(tv_lower=None, q_event_lower=None, one_sided_hoeffding_radius=None)
            result["rows"].append(row)
        except _NUMERICAL_FAILURES as exc:
            result["numerical_failures"].append({"window_index": index, "modes": list(window),
                                                 "error_type": type(exc).__name__, "error": str(exc)})
    result["target_diagnostics"] = model.cache_info()
    result["evaluated_windows"] = len(result["rows"])
    if result["numerical_failures"]:
        result["status"] = "partial_numerical_failure" if result["rows"] else "evaluation_failed"
    if not result["rows"]:
        result["decision"] = "evaluation_failed"
    elif statistical:
        result["tv_lower"] = max(row["tv_lower"] for row in result["rows"])
        if result["tv_lower"] > epsilon:
            result["decision"] = "rejected"
            result["reasons"].append("A held-out marginal event lower-bounds full-system TV above epsilon")
        else:
            result["reasons"].append("No declared event rejects the tolerance; no TV upper bound is supplied")
    else:
        result["reasons"].append("Without a valid fixed-split sampling protocol these are descriptive statistics only")
    return finish()
