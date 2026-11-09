"""Evidence gates for comparison with a specified Jiuzhang sampling task.

This module audits supplied evidence; it does not validate arbitrary input
claims cryptographically, compute a new fidelity bound, or assign a readiness
percentage.  The curated reference catalog is the trust boundary.  A matching
mode count never establishes a matching distribution.  Numerical certificates
remain conditional on the qualifications returned by the identity validator.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = ROOT / "configs" / "jiuzhang_reference.json"
TARGET_KINDS = {"published_gaussian_model", "empirical_hardware_samples"}


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def _mapping(value):
    return value if isinstance(value, Mapping) else {}


def _positive_number(value, *, zero=False):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and (value >= 0 if zero else value > 0))


def _count(value, *, zero=False):
    return (not isinstance(value, bool) and isinstance(value, int)
            and (value >= 0 if zero else value > 0))


def load_reference_catalog(path=None):
    """Read the versioned primary-source catalog, without network access."""
    def reject_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    catalog = json.loads(Path(path or DEFAULT_CATALOG).read_text(),
                         parse_constant=reject_constant)
    if catalog.get("schema_version") != 1 or not isinstance(catalog.get("targets"), dict):
        raise ValueError("unsupported Jiuzhang reference catalog")
    return catalog


def _gate(status, reason, *, observed=None, required=None, next_action=None):
    result = {"status": status, "reason": reason}
    if observed is not None:
        result["observed"] = observed
    if required is not None:
        result["required"] = required
    if next_action:
        result["next_action"] = next_action
    return result


def _global_evidence_consistency(evidence, global_tv):
    """Check D1's actual fixed-batch and familywise-allocation record.

    This is structural consistency, not a proof that attestations are true.
    Full enumeration does not need this statistical protocol.
    """
    requested, observed = global_tv.get("n_requested"), global_tv.get("n_observed")
    if not _count(requested) or not _count(observed) or observed != requested:
        return _gate("fail", "Global TV evidence uses an incomplete or invalid sample batch.")
    if evidence.get("statistical_protocol_valid") is not True:
        status = "unknown" if evidence.get("statistical_protocol_valid") is None else "fail"
        return _gate(status, "Global TV evidence lacks a valid fixed-batch statistical protocol.")
    protocol = _mapping(evidence.get("sampling_protocol"))
    required_flags = ("target_and_sampler_fixed_before_samples", "iid_attempts",
                      "first_batch_no_retry", "no_postselection",
                      "density_is_unconditional_path_mass")
    if not protocol or any(key not in protocol for key in required_flags + ("requested_draws",)):
        return _gate("unknown", "The identity evidence's fixed-batch protocol is incomplete.")
    if (any(protocol[key] is not True for key in required_flags)
            or not _count(protocol["requested_draws"])
            or protocol["requested_draws"] != requested):
        return _gate("fail", "The identity evidence contradicts its IID, first-batch, unconditional-density or sample-count assumptions.")
    bound_protocol = _mapping(global_tv.get("fixed_iid_no_retry_assumptions"))
    if not bound_protocol:
        return _gate("unknown", "The global bound does not retain its fixed-batch protocol.")
    if bound_protocol != protocol:
        return _gate("fail", "The global bound and identity record declare different sampling protocols.")
    plan = _mapping(evidence.get("confidence_plan"))
    plan_fields = ("comparison_ids", "planned_comparisons", "familywise_alpha",
                   "familywise_nominal_confidence", "per_comparison_alpha",
                   "channel_split", "schedule_rule")
    allocation_fields = ("comparison_id", "alpha_global_tv", "alpha_witnesses", "plan_sha256")
    if not plan or any(key not in plan for key in plan_fields + allocation_fields):
        return _gate("unknown", "The global bound lacks a complete predeclared confidence allocation.")
    ids, count, family = plan["comparison_ids"], plan["planned_comparisons"], plan["familywise_alpha"]
    if (not isinstance(ids, list) or not ids or any(not isinstance(x, str) or not x for x in ids)
            or len(set(ids)) != len(ids) or not _count(count) or count != len(ids)
            or plan["comparison_id"] not in ids or not _positive_number(family) or family >= 1):
        return _gate("fail", "The confidence family has invalid identifiers, comparison count or failure probability.")
    expected = {"per_comparison_alpha": family / count,
                "alpha_global_tv": family / (2 * count),
                "alpha_witnesses": family / (2 * count),
                "familywise_nominal_confidence": 1 - family}
    if (plan["channel_split"] != {"global_tv_upper": .5, "simultaneous_witnesses": .5}
            or not isinstance(plan["schedule_rule"], str) or not plan["schedule_rule"].strip()
            or any(not _positive_number(plan[key])
                   or not math.isclose(plan[key], value, rel_tol=1e-12, abs_tol=0)
                   for key, value in expected.items())
            or not _positive_number(global_tv.get("alpha"))
            or not math.isclose(global_tv["alpha"], expected["alpha_global_tv"], rel_tol=1e-12, abs_tol=0)):
        return _gate("fail", "The global bound's alpha is inconsistent with its familywise comparison allocation.")
    if plan["plan_sha256"] != _json_digest({key: plan[key] for key in plan_fields}):
        return _gate("fail", "The confidence-plan digest does not match its declared allocation schedule.")
    return None


def _quality_gate(run, evidence, target_kind, epsilon, instance):
    """Never promote moments, XEB, subsystem tests, or wrong-target TV to TV."""
    if not evidence:
        return _gate("unknown", "No distribution-quality evidence was supplied.",
                     next_action="Run the identity validator on this exact target and sampler.")
    if evidence.get("decision") in {"evaluation_failed", "invalid_model", "invalid_evidence"}:
        return _gate("fail", f"Identity validator returned {evidence['decision']}.")
    if target_kind == "empirical_hardware_samples":
        # An empirical witness comparison is useful, but accepting finitely
        # many witnesses cannot certify the underlying hardware law in TV.
        empirical = _mapping(evidence.get("empirical_hardware"))
        expected_data = instance.get("hardware_dataset_sha256")
        if not expected_data:
            return _gate("unknown", "No cataloged hardware dataset and calibration binding exists.",
                         observed={"empirical_evidence_present": bool(empirical)},
                         next_action="Import the actual samples, calibration, selection rule and held-out analysis protocol.")
        if empirical.get("hardware_dataset_sha256") != expected_data:
            return _gate("fail", "Empirical evidence is not bound to the required hardware dataset.")
        return _gate("unknown", "Finite hardware samples or passed witnesses alone do not certify full-law TV.",
                     next_action="State the empirical benchmark separately; supply a defensible full-distribution error bound for a full-law claim.")
    bound_target = _mapping(evidence.get("target"))
    expected_hash = run.get("covariance_sha256")
    if not expected_hash or not bound_target.get("covariance_sha256") or not bound_target.get("modes"):
        return _gate("unknown", "Quality evidence lacks an identified covariance/output size.",
                     next_action="Bind the quality result to this run's exact covariance SHA-256 and modes.")
    if (bound_target.get("covariance_sha256") != expected_hash
            or bound_target.get("modes") != run.get("modes")):
        return _gate("fail", "Quality evidence belongs to a different covariance/output size.",
                     next_action="Bind the quality result to this run's exact covariance SHA-256 and modes.")
    if not bound_target.get("detector_model") or not run.get("detector_model"):
        return _gate("unknown", "Quality evidence or timed run lacks a detector-model declaration.")
    if bound_target.get("detector_model") != run.get("detector_model"):
        return _gate("fail", "Quality evidence and timed run use different detector models.")
    timing_sampler = _mapping(run.get("native_provenance"))
    quality_sampler = _mapping(evidence.get("native_provenance"))
    rule_fields = ["executable_sha256", "selector", "k", "order", "vacuum_kernel", "covariance_sha256"]
    missing_rule = [k for k in rule_fields if timing_sampler.get(k) is None or quality_sampler.get(k) is None]
    if missing_rule:
        return _gate("unknown", "Quality is not bound to the exact timed sampler implementation and rule.",
                     required=missing_rule,
                     next_action="Record matching native_provenance from timing and held-out validation batches; seeds/counts may differ.")
    if any(timing_sampler[k] != quality_sampler[k] for k in rule_fields):
        return _gate("fail", "Quality evidence uses a different binary, covariance, selection rule, order or numerical kernel.")
    if timing_sampler["covariance_sha256"] != expected_hash:
        return _gate("fail", "Sampler provenance and run covariance hashes disagree.")
    if _mapping(evidence.get("model_check")).get("valid") is not True:
        return _gate("unknown", "A physical target-model validation is missing or did not pass.")
    decision = evidence.get("decision")
    if decision in {"rejected", "evaluation_failed", "out_of_scope"}:
        return _gate("fail", f"Identity validator returned {decision}.",
                     observed={"evidence_scope": evidence.get("evidence_scope")})
    scope = evidence.get("evidence_scope")
    value = None
    qualifications = []
    if scope == "full_distribution":
        law = _mapping(evidence.get("full_law"))
        value = law.get("tv")
        qualifications.append("Full-law result is subject to its declared floating-point/reference qualifications.")
    elif scope == "global_tv_upper":
        global_tv = _mapping(evidence.get("global_tv"))
        value = global_tv.get("upper")
        alpha = global_tv.get("alpha")
        if not _positive_number(alpha) or alpha >= 1:
            return _gate("unknown", "Global TV bound lacks a valid failure probability.")
        inconsistent = _global_evidence_consistency(evidence, global_tv)
        if inconsistent is not None:
            return inconsistent
        if global_tv.get("output_space") != "binary_patterns_plus_failure":
            return _gate("unknown", "Global TV bound does not explicitly include algorithmic failure mass.")
        qualifications.extend([
            f"Statistical failure probability alpha={alpha}; familywise allocation must match the declared comparison plan.",
            "Fixed independent draws, no retries, target evaluations and logged sampler probabilities remain assumptions.",
        ])
        if global_tv.get("strict_floating_point_certificate") is not True:
            qualifications.append("Numerically qualified bound; not a rigorous floating-point enclosure.")
    else:
        return _gate("unknown", "Witnesses, moments or timing alone cannot accept global distributional closeness.",
                     observed={"evidence_scope": scope},
                     next_action="Obtain same-target full-law TV or a valid failure-inclusive global TV upper bound.")
    if not _positive_number(value, zero=True) or value > 1:
        return _gate("unknown", "TV evidence is absent, nonfinite or outside [0,1].")
    if value > epsilon:
        # An upper bound above epsilon is inconclusive, not evidence that TV
        # itself exceeds epsilon.  A computed full-law TV above epsilon fails.
        status = "fail" if scope == "full_distribution" else "unknown"
        return _gate(status, "The supplied evidence does not establish the requested TV tolerance.",
                     observed={"value": value, "scope": scope}, required={"epsilon": epsilon})
    if decision != "accepted_within_tolerance_conditional":
        return _gate("unknown", "The identity validator did not accept the bound under its own policy.")
    return _gate("pass", "Same-target global error evidence meets the declared tolerance, conditionally.",
                 observed={"value": value, "scope": scope, "qualifications": qualifications},
                 required={"epsilon": epsilon})


def assess_jiuzhang_gap(run, evidence=None, *, reference_id="jiuzhang4_l1024",
                       target_kind="published_gaussian_model", catalog=None,
                       epsilon=0.1):
    """Audit one run against one experiment/model, returning JSON-safe axes.

    ``run`` is a normalized manifest. Required fields are described in
    ``research/iteration4_jiuzhang.md``. Missing evidence becomes ``unknown``;
    a demonstrated mismatch becomes ``fail``. Only every passed hard gate can
    unlock an explicitly scoped runtime comparison. The current reference
    catalog deliberately contains no invented matched hardware timing record.
    """
    if not isinstance(run, Mapping) or (evidence is not None and not isinstance(evidence, Mapping)):
        raise ValueError("run and evidence must be mappings")
    if target_kind not in TARGET_KINDS:
        raise ValueError(f"target_kind must be one of {sorted(TARGET_KINDS)}")
    if not _positive_number(epsilon, zero=True) or epsilon > 1:
        raise ValueError("epsilon must be finite and in [0,1]")
    catalog = load_reference_catalog() if catalog is None else catalog
    if reference_id not in catalog["targets"]:
        raise ValueError(f"unknown reference_id: {reference_id}")
    ref = catalog["targets"][reference_id]
    instance_id = run.get("instance_id")
    instance = _mapping(ref.get("instances", {}).get(instance_id))
    sampling = _mapping(run.get("sampling"))
    timings = _mapping(run.get("timings"))
    resources = _mapping(run.get("resources"))
    gates = {}

    if run.get("synthetic") is True:
        gates["physical_instance"] = _gate("fail", "The run uses a synthetic circuit, not this experimental instance.",
            next_action="Import a cited experimental calibration or an explicitly registered published-model covariance.")
    elif not instance:
        gates["physical_instance"] = _gate("unknown", "No registered instance/calibration matches the supplied instance identifier.",
            next_action="Register a verified covariance/calibration hash and its provenance; a generation name is insufficient.")
    elif target_kind == "published_gaussian_model":
        expected = instance.get("covariance_sha256")
        actual = run.get("covariance_sha256")
        if expected and actual == expected:
            gates["physical_instance"] = _gate("pass", "The covariance hash matches this registered published-model artifact.",
                observed={"instance_id": instance_id, "provenance_scope": instance.get("provenance_scope")})
        elif expected and actual:
            gates["physical_instance"] = _gate("fail", "The covariance hash differs from the registered model.")
        else:
            gates["physical_instance"] = _gate("unknown", "The model covariance has not been identified by hash.")
    else:
        expected = instance.get("calibration_sha256")
        actual = run.get("calibration_sha256")
        if expected and actual == expected and instance.get("hardware_dataset_sha256"):
            gates["physical_instance"] = _gate("pass", "A cataloged hardware calibration and dataset are bound to this run.")
        elif expected and actual and actual != expected:
            gates["physical_instance"] = _gate("fail", "Hardware calibration hashes differ.")
        else:
            gates["physical_instance"] = _gate("unknown", "A model covariance alone does not identify the empirical hardware distribution.",
                next_action="Bind the run to the actual hardware dataset, calibration version and sampling selection rule.")

    expected_modes = ref["optical_output_modes"]
    run_modes = run.get("modes")
    required_encoding = ref["measurement"]["output_encoding"]
    expected_bins = ref["measurement"]["binary_detection_bins"]
    # A PPnR target may be represented either by its complete bin record or
    # grouped per-main-mode counts, but that mapping must be explicit.
    binary_expanded = (required_encoding == "grouped_fanout_click_counts"
                       and run.get("output_encoding") == "binary_threshold_bins"
                       and run.get("modes") == expected_bins
                       and run.get("fanout_grouping") == ref["measurement"].get("fanout_grouping"))
    required_run_modes = expected_bins if binary_expanded else expected_modes
    if not _count(run_modes):
        gates["full_output_extent"] = _gate("unknown", "Output mode count is missing or invalid.")
    elif not _count(run.get("output_bins")):
        gates["full_output_extent"] = _gate("unknown", "The number of represented binary detection bins is missing or invalid.")
    elif run_modes != required_run_modes or run.get("output_bins") != expected_bins:
        gates["full_output_extent"] = _gate("fail", "The full output extent differs from the target task.",
            observed={"modes": run_modes, "output_bins": run.get("output_bins")},
            required={"optical_modes": expected_modes, "binary_detection_bins": expected_bins})
    else:
        gates["full_output_extent"] = _gate("pass", "All target output positions are represented; counts alone do not identify the circuit.")
    detector_matches = (run.get("detector_model") == ref["measurement"]["detector_model"]
                        and run.get("output_encoding") == required_encoding)
    if binary_expanded and run.get("detector_model") == "threshold":
        detector_matches = True
    if not run.get("detector_model") or not run.get("output_encoding"):
        gates["measurement_alphabet"] = _gate("unknown", "Detector model or output alphabet is unspecified.")
    elif detector_matches:
        gates["measurement_alphabet"] = _gate("pass", "Declared detector alphabet matches; calibrated response is a separate gate.")
    else:
        gates["measurement_alphabet"] = _gate("fail", "Detector/readout alphabet differs: threshold clicks, true photons and pseudo-PNR are not interchangeable.",
            observed={"detector_model": run.get("detector_model"), "encoding": run.get("output_encoding")},
            required=ref["measurement"])

    required_noise = instance.get("model_noise_features", ref["required_model_features"])
    if target_kind == "empirical_hardware_samples":
        required_noise = ref["required_hardware_features"]
    represented_noise = run.get("noise_features")
    if not isinstance(represented_noise, list) or not all(isinstance(x, str) for x in represented_noise):
        gates["imperfections"] = _gate("unknown", "The represented noise and detector response are unspecified.", required=required_noise)
    else:
        missing_noise = sorted(set(required_noise) - set(represented_noise))
        if missing_noise:
            gates["imperfections"] = _gate("fail", "Required model/response features are omitted or unsupported.",
                observed=represented_noise, required=missing_noise,
                next_action="Implement the selected target's calibrated noise/response; a uniform-loss surrogate is insufficient.")
        elif gates["physical_instance"]["status"] != "pass":
            gates["imperfections"] = _gate("unknown", "Feature names are present, but their calibration is not bound to the target instance.")
        else:
            gates["imperfections"] = _gate("pass", "Declared model features are bound to the registered artifact; unmodeled hardware effects are outside this target.")
    maximum_modes = _mapping(run.get("implementation")).get("maximum_modes")
    if not _count(maximum_modes):
        gates["implementation_capacity"] = _gate("unknown", "Supported implementation size is unspecified.")
    elif maximum_modes < required_run_modes:
        gates["implementation_capacity"] = _gate("fail", "The current implementation cannot represent the full target extent.",
            observed={"maximum_modes": maximum_modes}, required={"modes": required_run_modes})
    else:
        gates["implementation_capacity"] = _gate("pass", "The declared mode limit permits this extent; this does not predict runtime or quality.")

    local_quality = _quality_gate(run, evidence or {}, target_kind, epsilon, instance)
    bound_task_gates = ["physical_instance", "full_output_extent", "measurement_alphabet", "imperfections"]
    if all(gates[k]["status"] == "pass" for k in bound_task_gates):
        gates["distribution_quality"] = local_quality
    else:
        gates["distribution_quality"] = _gate("unknown", "Local validation does not establish quality for an unbound or different Jiuzhang target.",
            observed={"local_quality_status": local_quality["status"],
                      "blocking_task_axes": [k for k in bound_task_gates if gates[k]["status"] != "pass"]},
            next_action="First bind the physical instance, full extent, detector alphabet and imperfections; then validate that same law.")
    if run.get("status") not in {"ok", "complete"}:
        gates["sampling_protocol"] = _gate("fail" if run.get("status") in {"failed", "error"} else "unknown",
                                          "The run is failed, incomplete, planned, or lacks completion metadata.")
    elif sampling.get("count_requested") is None or sampling.get("count_observed") is None:
        gates["sampling_protocol"] = _gate("unknown", "Requested or observed sample counts are missing.")
    elif (not _count(sampling.get("count_requested"))
          or not _count(sampling.get("count_observed"))
          or sampling.get("count_observed") != sampling.get("count_requested")):
        gates["sampling_protocol"] = _gate("fail", "Requested and observed sample counts do not form one complete positive-size batch.")
    elif sampling.get("iid") is not True or sampling.get("no_retries") is not True:
        gates["sampling_protocol"] = _gate("unknown", "IID sampling and a no-retry/failure accounting policy are not established.")
    elif sampling.get("postselection") is None:
        gates["sampling_protocol"] = _gate("unknown", "Selection/conditioning rules are missing.")
    elif sampling.get("postselection") != instance.get("postselection", ref["sampling_protocol"]["postselection"]):
        gates["sampling_protocol"] = _gate("fail", "Selection/conditioning rules differ.")
    else:
        gates["sampling_protocol"] = _gate("pass", "A complete declared IID batch uses the target selection rule without retries.")

    timing_fields = ["setup_seconds", "sampling_seconds", "wall_seconds"]
    invalid_times = [k for k in timing_fields if not _positive_number(timings.get(k), zero=k == "setup_seconds")]
    if not isinstance(timings.get("scope"), str) or not timings["scope"].strip():
        invalid_times.append("scope")
    required_resource_fields = ["cpu", "threads", "gpus", "peak_memory_bytes"]
    invalid_resources = [k for k in required_resource_fields if resources.get(k) is None]
    if not isinstance(resources.get("cpu"), str) or not resources["cpu"].strip():
        invalid_resources.append("cpu")
    if not _count(resources.get("threads")):
        invalid_resources.append("threads")
    if not _count(resources.get("gpus"), zero=True):
        invalid_resources.append("gpus")
    if not _count(resources.get("peak_memory_bytes")):
        invalid_resources.append("peak_memory_bytes")
    timing_record = _mapping(instance.get("matched_timing_reference"))
    if invalid_times or invalid_resources:
        gates["resource_timing_comparability"] = _gate("unknown", "Complete timing boundaries and resource measurements are missing or invalid.",
            observed={"missing_or_invalid_times": invalid_times, "missing_or_invalid_resources": sorted(set(invalid_resources))},
            next_action="Measure setup, sampling, complete wall time, threads/accelerators and peak memory under one documented protocol.")
    elif timings["wall_seconds"] + max(1e-6, 1e-9 * (timings["setup_seconds"] + timings["sampling_seconds"])) < timings["setup_seconds"] + timings["sampling_seconds"]:
        gates["resource_timing_comparability"] = _gate("fail", "Complete wall time cannot be shorter than the recorded serial setup and sampling stages.",
            observed={key: timings[key] for key in timing_fields},
            required={"contract": "nonoverlapping setup and sampling inside complete wall time",
                      "tolerance_seconds": max(1e-6, 1e-9 * (timings["setup_seconds"] + timings["sampling_seconds"]))})
    elif not timing_record:
        gates["resource_timing_comparability"] = _gate("unknown", "The paper's sample interval is not a matched sample-count/timing protocol.",
            next_action="Register an explicitly comparable target timing record, including sample count, preparation scope and selection rule.")
    elif (not _count(timing_record.get("sample_count"))
          or not _count(sampling.get("count_observed"))
          or timing_record.get("sample_count") != sampling.get("count_observed")
          or not isinstance(timing_record.get("timing_scope"), str)
          or not timing_record["timing_scope"].strip()
          or timing_record.get("timing_scope") != timings.get("scope")
          or timing_record.get("postselection") != sampling.get("postselection")
          or timing_record.get("target_kind") != target_kind
          or timing_record.get("covariance_sha256") != run.get("covariance_sha256")
          or not _positive_number(timing_record.get("wall_seconds"))):
        gates["resource_timing_comparability"] = _gate("fail", "The registered timing reference differs in target, batch size, selection rule or timing scope.")
    else:
        gates["resource_timing_comparability"] = _gate("pass", "Task, sample count, timing scope and declared resources support a scoped timing comparison.")

    all_passed = all(g["status"] == "pass" for g in gates.values())
    task_gates = ["physical_instance", "full_output_extent", "measurement_alphabet", "imperfections"]
    if any(gates[k]["status"] == "fail" for k in task_gates):
        verdict = "different_or_incomplete_task"
    elif gates["distribution_quality"]["status"] == "fail":
        verdict = "quality_not_accepted"
    elif all_passed:
        verdict = "comparable_for_declared_target_conditionally"
    else:
        verdict = "insufficient_evidence"
    raw_rate = None
    if _count(sampling.get("count_observed")) and _positive_number(timings.get("sampling_seconds")):
        raw_rate = sampling["count_observed"] / timings["sampling_seconds"]
    timing_ratio = (timings["wall_seconds"] / timing_record["wall_seconds"]) if all_passed else None
    return {
        "schema_version": 1, "run_id": run.get("run_id"), "reference_id": reference_id,
        "target_kind": target_kind, "target_label": ref["label"], "epsilon": epsilon,
        "catalog_checked_as_of": catalog.get("checked_as_of"), "catalog_sha256": _json_digest(catalog),
        "run_manifest_sha256": _json_digest(run), "evidence_sha256": _json_digest(evidence) if evidence else None,
        "verdict": verdict, "gates": gates, "local_distribution_quality": local_quality,
        "scale": {"local_modes": run_modes, "target_optical_modes": expected_modes,
                  "local_squeezed_input_states": run.get("source_count"),
                  "target_squeezed_input_states": ref["squeezed_input_states"],
                  "target_binary_detection_bins": expected_bins,
                  "target_hardware_detector_channels": ref["measurement"]["hardware_detector_channels"],
                  "local_sample_count": sampling.get("count_observed"),
                  "local_click_statistics": run.get("click_statistics"),
                  "target_click_statistics": ref["reported_detection_statistics"],
                  "dense_float64_covariance_bytes_at_target_extent": 32 * required_run_modes ** 2,
                  "dense_memory_note": "One dense real 2m-by-2m covariance only; excludes copies, complex validation, workspaces and sampling state."},
        "timing": {"local_sampling_draws_per_second": raw_rate,
                   "published_hardware_sample_period_seconds": ref["sampling_protocol"].get("reported_sample_period_seconds"),
                   "classical_wall_over_reference_wall": timing_ratio,
                   "comparison_authorized_by_evidence": all_passed,
                   "historical_advantage_claim_is_not_a_current_classical_lower_bound": True},
        "blockers": [{"axis": name, **gate} for name, gate in gates.items() if gate["status"] != "pass"],
        "source_ids": ref["source_ids"],
        "limitations": [
            "No scalar readiness score or defeat-of-Jiuzhang conclusion is produced.",
            "Model equivalence and empirical hardware agreement are separate targets.",
            "Observed maximum clicks are not mean/typical event hardness or a full-distribution quality certificate.",
            "The checker trusts curated hashes and supplied validation records; it does not establish their truth by assertion.",
        ],
    }


def format_gap_markdown(report):
    """Compact reviewable report; the full JSON retains all qualifications."""
    lines = [f"# Jiuzhang gap: {report['run_id']}", "",
             f"Target: **{report['target_label']}**, `{report['target_kind']}`.", "",
             f"Verdict: **{report['verdict']}**. No readiness percentage is assigned.", "",
             "| Axis | Status | Finding |", "|---|---|---|"]
    for name, gate in report["gates"].items():
        reason = gate["reason"].replace("|", "\\|")
        lines.append(f"| {name} | {gate['status']} | {reason} |")
    lines += ["", "## Scale and timing", "", "```json",
              json.dumps({"scale": report["scale"], "timing": report["timing"]}, indent=2), "```", "",
              "All source definitions and unknowns remain in the versioned reference catalog. ",
              "A null timing ratio means that matched experimental comparison is not supported.", ""]
    return "\n".join(lines)
