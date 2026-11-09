"""Post-hoc exact-binomial refinement of frozen marginal-event witnesses.

The PRIMARY Hoeffding reports are never edited. Events, held-out rows, target
probabilities and alpha allocations are frozen; only the one-sided confidence
bound changes. This method was selected after examining primary results.

For k>0 let T_k(p)=Pr_p[Bin(n,p)>=k]. The Clopper-Pearson lower endpoint C
solves T_k(C)=a. If H=max(0,k/n-sqrt(log(1/a)/(2n))) is positive, Hoeffding
gives T_k(H)<=a, hence H<=C by monotonicity. For H=0 the result is immediate.
Thus reporting either H or C, even after observing the data, is dominated
pointwise by C. The original union budget across the SAME frozen events
covers both rules; this does not license searching over arbitrary methods,
events, directions, sample sizes or alpha values.

"Exact" names the binomial confidence construction. SciPy quantiles and
target marginals remain floating-point calculations, not interval proofs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import scipy
from scipy.stats import beta, binom

from .exact import ThresholdModel


SCHEMA_VERSION = 1
_SPLIT = "first_pilot_rows_then_all_remaining_rows"
_EVENT_RULE = "pilot frequency strictly greater than target; ties excluded; direction fixed"
_ENCODING = "little-endian; bit j is modes[j]"
_METHOD = "one-sided Clopper-Pearson binomial-tail inversion"
_REQUIRED_FLAGS = (
    "split_fixed_before_samples", "windows_fixed_before_samples",
    "target_and_sampler_fixed_before_samples", "pilot_not_used_to_modify_sampler",
    "iid_attempts", "first_batch_no_retry", "no_postselection",
)


class RefinementError(ValueError):
    """Frozen evidence is inconsistent or the numerical refinement failed."""


def _int(value, name, minimum=0):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < minimum):
        raise RefinementError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _prob(value, name, *, interior=False):
    if (isinstance(value, (bool, np.bool_)) or np.iscomplexobj(value)
            or not isinstance(value, (int, float, np.number))):
        raise RefinementError(f"{name} must be a real probability")
    value = float(value)
    if not math.isfinite(value) or not (0 < value < 1 if interior else 0 <= value <= 1):
        raise RefinementError(f"{name} is outside its probability range")
    return value


def _hash_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _hash_json(value):
    return _hash_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).encode())


def _hash_file(path):
    return _hash_bytes(Path(path).read_bytes())


def _same(actual, expected, name):
    if actual != expected:
        raise RefinementError(f"Frozen {name} does not match")


def _same_float(actual, expected, name, atol=2e-14):
    if (isinstance(actual, (bool, np.bool_)) or not isinstance(actual, (int, float, np.number))
            or not math.isfinite(float(actual))
            or not math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=atol)):
        raise RefinementError(f"Frozen {name} does not match")


def clopper_pearson_event_tv_lower(successes, draws, target_probability, *, alpha):
    """Refine a single FIXED event using its already allocated one-sided alpha.

    This primitive cannot establish event selection or iid provenance; use
    ``refine_frozen_marginal_report`` for an archived fixed-split report.
    """
    n = _int(draws, "draws", 1)
    k = _int(successes, "successes")
    if k > n:
        raise RefinementError("successes cannot exceed draws")
    p = _prob(target_probability, "target_probability")
    a = _prob(alpha, "alpha", interior=True)
    radius = math.sqrt(-math.log(a)/(2*n))
    h = max(0.0, k/n-radius)
    if k == 0:
        c, tail, residual = 0.0, None, None
    else:
        c = float(beta.ppf(a, k, n-k+1))
        if not math.isfinite(c) or not 0 < c < 1:
            raise RefinementError("Binomial lower-quantile evaluation is unresolved")
        # Survival avoids subtracting a CDF near one. This is a numerical
        # consistency check, NOT a rigorous outward-rounded certificate.
        tail = float(binom.sf(k-1, n, c))
        residual = abs(tail-a)/a
        if not math.isfinite(tail) or not math.isfinite(residual) or residual > 1e-8:
            raise RefinementError("Binomial quantile failed its tail-inversion check")
    if c < h:
        raise RefinementError("Numerically computed CP endpoint did not dominate Hoeffding")
    return {
        "method": _METHOD, "heldout_successes": k, "heldout_draws": n,
        "alpha": a, "target_event_probability": p,
        "heldout_event_probability": k/n,
        "primary_hoeffding_q_lower": h,
        "primary_hoeffding_tv_lower": max(0.0, h-p),
        "q_event_lower": c, "tv_lower": max(0.0, c-p),
        "pointwise_dominates_primary": True,
        "binomial_tail_at_quantile": tail,
        "tail_inversion_relative_residual": residual,
        "strict_floating_point_certificate": False,
    }


def _windows(values, modes):
    if not isinstance(values, (list, tuple)) or not values:
        raise RefinementError("Frozen windows must be nonempty")
    result, seen = [], set()
    for value in values:
        if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= 12:
            raise RefinementError("Frozen window size must lie in [1,12]")
        window = [_int(j, "window mode") for j in value]
        if any(j >= modes for j in window) or len(set(window)) != len(window):
            raise RefinementError("Frozen window contains an invalid or repeated mode")
        key = tuple(sorted(window))
        if key in seen:
            raise RefinementError("Frozen window is duplicated")
        seen.add(key)
        result.append(window)
    return result


def _counts(values, size, total, name):
    if not isinstance(values, list) or len(values) != size:
        raise RefinementError(f"{name} has the wrong shape")
    values = [_int(v, name) for v in values]
    if sum(values) != total:
        raise RefinementError(f"{name} has the wrong total")
    return np.asarray(values, dtype=np.int64)


def refine_frozen_marginal_report(
    report, covariance, samples, *, expected_windows, expected_protocol,
    expected_alpha, expected_epsilon, expected_target_id,
):
    """Validate saved evidence and return a separate secondary CP report.

    Every saved event and count is replayed from the same pilot/held-out
    rows; replay never changes event membership. Recorded target probabilities
    are preserved verbatim after a small-marginal consistency re-evaluation.
    Failed windows retain their slots and receive no replacement event.
    Invalid/failed/incomplete batches raise instead of gaining a bound.
    """
    if not isinstance(report, dict):
        raise RefinementError("Primary report must be an object")
    if (report.get("schema_version") != 1
            or report.get("status") not in ("ok", "partial_numerical_failure")
            or report.get("statistical_protocol_valid") is not True
            or report.get("decision") not in ("insufficient_evidence", "rejected")
            or report.get("sampling_failure") is not None):
        raise RefinementError("Primary report has no valid completed fixed-batch witness")
    for flag in ("unseen_failure_mass_assumed_zero", "conditional_on_success_coverage_claimed"):
        if report.get(flag) is not False:
            raise RefinementError("Primary failure-inclusive qualification is inconsistent")
    _same(report.get("output_space"), "binary_patterns_plus_failure", "failure output space")
    _same(report.get("evidence_scope"), "heldout_marginal_events_only", "evidence scope")
    _same(_prob(report.get("failure_symbol_observable_value"), "failure observable"), 0.0,
          "failure observable")
    a = _prob(expected_alpha, "expected_alpha", interior=True)
    eps = _prob(expected_epsilon, "expected_epsilon")
    _same(_prob(report.get("alpha"), "report alpha", interior=True), a, "comparison alpha")
    _same(_prob(report.get("epsilon"), "report epsilon"), eps, "tolerance")
    if not isinstance(expected_target_id, str) or not expected_target_id:
        raise RefinementError("Expected target ID is required")
    if np.iscomplexobj(covariance):
        raise RefinementError("Covariance must be real")
    v = np.asarray(covariance, dtype=float)
    if (v.ndim != 2 or v.shape[0] != v.shape[1] or not v.shape[0]
            or v.shape[0] % 2 or v.shape[0] > 512 or not np.all(np.isfinite(v))):
        raise RefinementError("Invalid covariance dimensions or values")
    m = len(v)//2
    v_hash = _hash_bytes(np.ascontiguousarray(v*.5+v.T*.5).tobytes())
    target = report.get("target")
    if not isinstance(target, dict):
        raise RefinementError("Primary target binding is absent")
    for key, value in dict(target_id=expected_target_id, covariance_sha256=v_hash,
            modes=m, detector_model="threshold", state_model="zero_mean_gaussian",
            quadrature_order="xxpp", hbar=2).items():
        _same(target.get(key), value, "target "+key)
    if (report.get("model_check") or {}).get("valid") is not True:
        raise RefinementError("Primary target was not validated")
    windows = _windows(expected_windows, m)
    j = len(windows)
    windows_hash = _hash_json(windows)
    _same(report.get("windows"), windows, "window list")
    _same(report.get("windows_sha256"), windows_hash, "window hash")
    _same(_int(report.get("planned_windows"), "planned_windows", 1), j, "planned window count")
    _same(report.get("sampling_protocol"), expected_protocol, "sampling protocol")
    if not isinstance(expected_protocol, dict) or any(
            expected_protocol.get(key) is not True for key in _REQUIRED_FLAGS):
        raise RefinementError("Frozen iid/split/no-retry protocol is incomplete")
    _same(expected_protocol.get("split_rule"), _SPLIT, "protocol split rule")
    _same(expected_protocol.get("windows_sha256"), windows_hash, "protocol window hash")
    _same(report.get("split_rule"), _SPLIT, "report split rule")
    arr = np.asarray(samples)
    if (np.iscomplexobj(arr) or arr.ndim != 2 or arr.shape[1] != m
            or not np.all((arr == 0) | (arr == 1))):
        raise RefinementError("Samples must be the complete binary N-by-m array")
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    n = len(arr)
    pilot = _int(report.get("pilot_size"), "pilot_size", 1)
    heldout = n-pilot
    if heldout <= 0:
        raise RefinementError("The held-out suffix must be nonempty")
    for key, value in (("n_observed", n), ("heldout_size", heldout)):
        _same(_int(report.get(key), key, 1), value, key)
    for key, value in (("requested_draws", n), ("pilot_size", pilot), ("heldout_size", heldout)):
        _same(_int(expected_protocol.get(key), "protocol "+key, 1), value, "protocol "+key)
    sample_hashes = dict(samples_sha256=_hash_bytes(arr.tobytes()),
        pilot_samples_sha256=_hash_bytes(arr[:pilot].tobytes()),
        heldout_samples_sha256=_hash_bytes(arr[pilot:].tobytes()))
    for key, value in sample_hashes.items():
        _same(report.get(key), value, key)
    allocation = report.get("confidence_allocation") or {}
    per_event = a/j
    if per_event == 0:
        raise RefinementError("Per-event alpha underflow")
    _same(allocation.get("comparison_witness_alpha"), a, "allocation alpha")
    _same(_int(allocation.get("planned_windows"), "allocation planned_windows", 1), j, "allocation windows")
    _same(allocation.get("per_window_alpha"), per_event, "per-event alpha")
    _same(allocation.get("method"), "one-sided Hoeffding with Bonferroni across all predeclared windows", "primary method")
    if allocation.get("unused_alpha_reallocated") is not False:
        raise RefinementError("Unused alpha must not be reallocated")
    rows, failures = report.get("rows"), report.get("numerical_failures")
    if not isinstance(rows, list) or not rows or not isinstance(failures, list):
        raise RefinementError("Primary rows/failures are missing")
    indices = []
    for row in rows+failures:
        if not isinstance(row, dict):
            raise RefinementError("Primary window record is invalid")
        index = _int(row.get("window_index"), "window_index")
        if index >= j or row.get("modes") != windows[index]:
            raise RefinementError("Window record differs from frozen schedule")
        indices.append(index)
    if sorted(indices) != list(range(j)):
        raise RefinementError("Rows and failures must partition every planned window exactly once")
    _same(_int(report.get("evaluated_windows"), "evaluated_windows"), len(rows), "evaluated window count")
    _same(report.get("status"), "partial_numerical_failure" if failures else "ok", "window evaluation status")
    # Physicality and the complete covariance were checked by the PRIMARY
    # evaluator. The byte binding above is retained; this avoids repeating a
    # large eigensolve merely to recheck small saved probability vectors.
    model = ThresholdModel(v, check_physicality=False, max_exact_modes=12, vacuum_cache_size=8192)
    refined = []
    for row in rows:
        index = row["window_index"]
        window = windows[index]
        count = 1 << len(window)
        p = np.asarray(row.get("target_probabilities"), dtype=float)
        if (p.shape != (count,) or not np.all(np.isfinite(p))
                or np.any(p < 0) or np.any(p > 1)
                or abs(float(np.sum(p, dtype=np.longdouble))-1) > 2e-10):
            raise RefinementError("Frozen target marginal is not a probability vector")
        rechecked = model.marginal(tuple(window))
        target_error = float(np.max(np.abs(rechecked-p)))
        if target_error > 2e-12:
            raise RefinementError("Frozen target probabilities do not match their covariance")
        pilot_counts = _counts(row.get("pilot_counts"), count, pilot, "pilot counts")
        heldout_counts = _counts(row.get("heldout_counts"), count, heldout, "heldout counts")
        masks = arr[:, window].astype(np.int64) @ (1 << np.arange(len(window), dtype=np.int64))
        if (not np.array_equal(pilot_counts, np.bincount(masks[:pilot], minlength=count))
                or not np.array_equal(heldout_counts, np.bincount(masks[pilot:], minlength=count))):
            raise RefinementError("Saved counts do not match the frozen sample split")
        event = np.flatnonzero(pilot_counts/pilot > p).tolist()
        saved_event = row.get("event_masks")
        if not isinstance(saved_event, list):
            raise RefinementError("Frozen event is missing")
        saved_event = [_int(mask, "event mask") for mask in saved_event]
        _same(saved_event, event, "pilot-selected event")
        event_hash = _hash_json(dict(modes=window, event_masks=event))
        _same(row.get("event_sha256"), event_hash, "event hash")
        _same(row.get("event_rule"), _EVENT_RULE, "event direction/rule")
        _same(row.get("pattern_encoding"), _ENCODING, "event encoding")
        _same(_int(row.get("pilot_size"), "row pilot_size", 1), pilot, "row pilot size")
        k = int(heldout_counts[event].sum())
        _same(_int(row.get("heldout_successes"), "heldout_successes"), k, "success count")
        _same(_int(row.get("heldout_draws"), "heldout_draws", 1), heldout, "held-out count")
        _same(_prob(row.get("alpha"), "row alpha", interior=True), per_event, "row alpha")
        target_event = float(np.sum(p[event], dtype=np.longdouble))
        _same_float(row.get("target_event_probability"), target_event, "target event probability")
        _same_float(row.get("heldout_event_probability"), k/heldout, "held-out event probability")
        # Preserve the exact previously saved scalar P(A), after checking its
        # relation to the saved vector. It is not replaced by a new estimate.
        result = clopper_pearson_event_tv_lower(k, heldout, row["target_event_probability"], alpha=per_event)
        _same_float(row.get("q_event_lower"), result["primary_hoeffding_q_lower"], "primary q lower")
        _same_float(row.get("tv_lower"), result["primary_hoeffding_tv_lower"], "primary TV lower")
        if result["tv_lower"] < row["tv_lower"]:
            raise RefinementError("Computed refinement is below the saved primary bound")
        refined.append(dict(window_index=index, modes=window, event_masks=event,
            event_sha256=event_hash, target_probabilities_sha256=_hash_bytes(np.ascontiguousarray(p).tobytes()),
            target_recheck_max_abs_error=target_error, **result))
    primary_lower = max(row["tv_lower"] for row in rows)
    _same_float(report.get("tv_lower"), primary_lower, "primary suite lower")
    _same(report.get("decision"), "rejected" if primary_lower > eps else "insufficient_evidence", "primary decision")
    lower = max(row["tv_lower"] for row in refined)
    return {
        "schema_version": SCHEMA_VERSION, "status": "ok", "method": _METHOD,
        "analysis_role": "secondary_posthoc_refinement_of_unchanged_frozen_events",
        "method_chosen_after_primary_results": True, "method_pre_registered": False,
        "primary_report_canonical_sha256": _hash_json(report),
        "primary_decision": report["decision"], "primary_tv_lower": primary_lower,
        "decision": "rejected" if lower > eps else "insufficient_evidence",
        "tv_lower": lower, "epsilon": eps, "alpha": a,
        "target": dict(target), **sample_hashes,
        "n_observed": n, "pilot_size": pilot, "heldout_size": heldout,
        "windows_sha256": windows_hash, "planned_windows": j, "evaluated_windows": len(rows),
        "per_window_alpha": per_event, "rows": refined,
        "unchanged_numerical_failures": failures,
        "all_frozen_event_target_count_and_split_bindings_verified": True,
        "same_alpha_covers_both_methods_by_pointwise_dominance": True,
        "unused_alpha_reallocated": False, "global_tv_acceptance_supported": False,
        "output_space": "binary_patterns_plus_failure", "failure_symbol_observable_value": 0.0,
        "unseen_failure_mass_assumed_zero": False, "conditional_on_success_coverage_claimed": False,
        "strict_floating_point_certificate": False, "scipy_version": scipy.__version__,
        "confidence_scope": "Unconditional first fixed-N iid attempts; event selected only from disjoint pilot; failed batches receive no bound; original event-family alpha allocation unchanged.",
        "qualification": "The binomial confidence construction is exact in real arithmetic. Quantiles, tail checks and frozen Gaussian targets use floating point, not outward-rounded intervals. This secondary method was selected after examining primary results and is not claimed pre-registered.",
    }


def _case_plan(config):
    cases = []
    for m in config["synthetic_modes"]:
        for profile in config["profiles"]:
            cases.append(dict(id=f"m{m}_{profile['name']}", modes=m, profile=profile,
                synthetic=True, method="candidate", count=config["sample_count_by_modes"][str(m)],
                pilot=config["pilot_size_by_modes"][str(m)], seed=config["sample_seed_base"]+len(cases)))
    jz = config["jiuzhang1"]
    for method in ("candidate", "frugal"):
        cases.append(dict(id=f"jiuzhang1_{method}", modes=100, synthetic=False,
            method=method, count=jz["sample_count"], pilot=jz["pilot_size"], seed=jz["sample_seed"]))
    return cases


def refine_archive(quality_directory, output_directory):
    """Refine every planned comparison into a FRESH sibling archive, no draws."""
    directory, output = Path(quality_directory).resolve(), Path(output_directory).resolve()
    if output == directory or directory in output.parents:
        raise RefinementError("Secondary archive must be outside the primary archive")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite an existing secondary archive")
    hashes = {}

    def read(path):
        path = Path(path)
        raw = path.read_bytes()
        hashes[str(path)] = _hash_bytes(raw)
        return json.loads(raw)

    config = read(directory/"config.json")
    plan = read(directory/"comparison_plan.json")
    _same(plan, _case_plan(config), "comparison plan")
    _same(_int(config.get("planned_comparisons"), "planned_comparisons", 1), len(plan), "comparison count")
    family_alpha = _prob(config.get("familywise_alpha"), "familywise_alpha", interior=True)
    marginal_alpha = _prob(config.get("marginal_suite_alpha"), "marginal_suite_alpha", interior=True)
    original_alpha = _prob(config.get("old_identity_suite_alpha"), "old_identity_suite_alpha", interior=True)
    if marginal_alpha+original_alpha > family_alpha+1e-15:
        raise RefinementError("Frozen confidence channels exceed their family budget")
    schedules = read(directory/"witness_schedules.json")
    protocols = read(directory/"split_protocols.json")
    schedule_manifest = read(directory/"witness_schedule_manifest.json")
    _same(schedule_manifest.get("config_sha256"), hashes[str(directory/"config.json")], "schedule config hash")
    artifact_hashes = schedule_manifest.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict) or set(artifact_hashes) != {
            "witness_schedules.json", "split_protocols.json", "confidence_plan.json"}:
        raise RefinementError("Frozen schedule artifact hashes are incomplete")
    for name, expected in artifact_hashes.items():
        if name not in ("witness_schedules.json", "split_protocols.json", "confidence_plan.json"):
            raise RefinementError("Unexpected evidence-plan artifact name")
        path = directory/name
        actual = _hash_file(path)
        hashes[str(path)] = actual
        _same(actual, expected, "schedule artifact hash "+name)
    ids = [case["id"] for case in plan]
    if len(set(ids)) != len(ids) or any(re.fullmatch(r"[A-Za-z0-9_]+", name) is None for name in ids):
        raise RefinementError("Comparison IDs must be unique safe names")
    _same(set(schedules), set(ids), "schedule comparison IDs")
    _same(set(protocols), set(ids), "protocol comparison IDs")
    original_plan = read(directory/"confidence_plan.json")
    _same(original_plan.get("comparison_ids"), ids, "original confidence-plan IDs")
    _same(original_plan.get("planned_comparisons"), len(plan), "original confidence-plan size")
    _same(original_plan.get("familywise_alpha"), original_alpha, "original confidence channel")
    per_comparison = marginal_alpha/len(plan)
    summaries, reports = [], {}
    root = Path(__file__).resolve().parents[2]
    for case in plan:
        name = case["id"]
        folder = directory/name
        primary_path = folder/"marginal_witness.json"
        if not primary_path.is_file():
            summaries.append(dict(case_id=name, status="primary_report_not_available", tv_lower=None))
            continue
        primary = read(primary_path)
        try:
            native = read(folder/"native.json")
            completion = read(folder/"completion.json")
            attempt = read(folder/"attempt.json")
            build = read(folder/"build.json")
            _same(attempt.get("case"), case, "sample attempt case")
            _same(attempt.get("config_sha256"), hashes[str(directory/"config.json")], "attempt config hash")
            _same(completion.get("native_sha256"), hashes[str(folder/"native.json")], "native completion hash")
            _same(completion.get("status"), native.get("status"), "native completion status")
            _same(build.get("executable_sha256"), attempt.get("executable_sha256"), "build executable hash")
            if native.get("status") != "ok":
                raise RefinementError("Observed native failure; no success-only refinement")
            for key, value in dict(executable_sha256=attempt["executable_sha256"],
                    covariance_sha256=attempt["covariance_sha256"], modes=case["modes"],
                    sample_count=case["count"], seed=case["seed"]).items():
                _same(native.get(key), value, "native "+key)
            if case["synthetic"]:
                path = folder/"input.npz"
                hashes[str(path)] = _hash_file(path)
                with np.load(path, allow_pickle=False) as saved:
                    covariance = saved["covariance"]
            else:
                path = root/config["jiuzhang1"]["input"]
                hashes[str(path)] = _hash_file(path)
                _same(hashes[str(path)], config["jiuzhang1"]["input_file_sha256"], "author covariance file hash")
                covariance = np.load(path, allow_pickle=False)
            covariance_hash = _hash_bytes(np.ascontiguousarray(covariance*.5+covariance.T*.5).tobytes())
            _same(covariance_hash, attempt["covariance_sha256"], "attempt covariance hash")
            _same(schedules[name].get("marginal_comparison_alpha"), per_comparison, "scheduled comparison alpha")
            secondary = refine_frozen_marginal_report(primary, covariance, native.get("raw_samples"),
                expected_windows=schedules[name]["marginal_windows"],
                expected_protocol=protocols[name]["marginal"], expected_alpha=per_comparison,
                expected_epsilon=config["epsilon"], expected_target_id=name)
            secondary["archive_binding"] = {
                "primary_report_file_sha256": hashes[str(primary_path)],
                "native_file_sha256": hashes[str(folder/"native.json")],
                "attempt_file_sha256": hashes[str(folder/"attempt.json")],
                "config_file_sha256": hashes[str(directory/"config.json")],
                "executable_sha256": attempt["executable_sha256"],
            }
            reports[name] = secondary
            summaries.append(dict(case_id=name, status="refined", primary_decision=primary["decision"],
                primary_tv_lower=primary["tv_lower"], decision=secondary["decision"],
                tv_lower=secondary["tv_lower"], alpha=per_comparison))
        except (ValueError, KeyError, TypeError, OSError, ArithmeticError, np.linalg.LinAlgError) as exc:
            reports[name] = dict(status="not_refined", decision="invalid_or_unavailable_evidence",
                tv_lower=None, error_type=type(exc).__name__, error=str(exc),
                primary_report_file_sha256=hashes[str(primary_path)])
            summaries.append(dict(case_id=name, status="not_refined", tv_lower=None,
                                  reason=str(exc), alpha=per_comparison))
    if any(_hash_file(path) != expected for path, expected in hashes.items()):
        raise RefinementError("A primary artifact changed during secondary analysis")
    summary = {
        "schema_version": SCHEMA_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": "secondary_posthoc_refinement", "method": _METHOD,
        "method_pre_registered": False, "method_chosen_after_primary_results": True,
        "primary_reports_modified": False, "new_draws": 0,
        "new_event_searches": 0, "sample_split_changed": False,
        "planned_comparisons": len(plan), "rows": summaries,
        "familywise_alpha": family_alpha, "original_suite_alpha": original_alpha,
        "marginal_suite_alpha": marginal_alpha, "per_comparison_alpha": per_comparison,
        "same_alpha_covers_primary_and_secondary_by_pointwise_dominance": True,
        "unused_alpha_reallocated": False,
        "primary_artifacts_unchanged_during_analysis": True,
        "primary_artifact_sha256": hashes,
        "source_sha256": _hash_file(__file__), "scipy_version": scipy.__version__,
        "strict_floating_point_certificate": False,
        "reporting_requirement": "Keep primary Hoeffding results. Describe CP as a post-hoc refinement on identical pilot-frozen events and validation rows, not a pre-registered analysis or an additional independently sampled experiment.",
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, report in reports.items():
        (output/(name+".json")).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = refine_archive(args.quality_directory, args.output)
    print(json.dumps({"planned_comparisons": result["planned_comparisons"], "rows": result["rows"]}, indent=2))


if __name__ == "__main__":
    main()
