"""Audit the frozen iteration-10 evidence and summarize construction quality/cost.

This is postprocessing only: no new targets, amplitudes, sampling trajectories,
Gaussian integrals, or native experiments are computed.  Core quality refers to
the ideal-channel law of the exported/rounded/normalized coefficient vector;
retained quality is the available enumerated native law with its failure mass.
The two scopes must not be described as equal native error certificates.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from iteration10_common import ROOT, save, sha, utc, validate_law


CORE_SCOPE = "construction_reference: exported normalized coefficients under ideal Gaussian displacement and threshold POVM"
RETAINED_SCOPE = "enumerated native threshold law extended by a numerical-failure outcome"
SUCCESS = "complete"


def read_json(path, bindings):
    path = Path(path)
    bindings[str(path)] = sha(path)
    return json.loads(path.read_text())


def checked_file(path, expected, bindings):
    path = Path(path)
    actual = sha(path)
    if not isinstance(expected, str) or actual != expected:
        raise ValueError("Evidence hash mismatch: " + str(path))
    bindings[str(path)] = actual
    return path


def finite(value, label, *, nonnegative=False):
    extended = np.longdouble(value)
    number = float(extended)
    if not np.isfinite(extended) or not math.isfinite(number) or (extended != 0 and number == 0) or (nonnegative and number < 0):
        raise ValueError("Invalid finite number: " + label)
    return number


def close(actual, expected, label, atol=1e-12, rtol=0.0):
    a, b = finite(actual, label), finite(expected, label)
    if abs(a-b) > atol + rtol*abs(b):
        raise ValueError(f"Bound metric differs from raw evidence ({label}): {a} vs {b}")


def profile_name(value):
    return value["name"] if isinstance(value, dict) else value


def case_id(modes, profile, seed_base):
    return f"m{modes}_{profile}_s{seed_base}"


def quality_plan(config):
    inputs, methods = {}, {}
    for modes in config["primary_modes"]:
        for base in config["seed_bases"]:
            for profile in config["profiles"]:
                key = case_id(modes, profile["name"], base)
                inputs[key] = {"case_id": key, "modes": modes, "seed_base": base,
                    "seed": base+1000*modes, "profile": profile["name"], "panel": "primary"}
                methods[key] = [("core_k"+str(k), "coherent_core", k) for k in config["core_cutoffs"]]
                methods[key] += [("retained_k"+str(k), "retained", k) for k in config["retained_budgets"]]
    supplement = config["supplement"]
    key = case_id(supplement["modes"], supplement["profile"], supplement["seed_base"])
    if key in inputs:
        raise ValueError("Supplement duplicates a primary input")
    inputs[key] = {"case_id": key, "modes": supplement["modes"], "seed_base": supplement["seed_base"],
        "seed": supplement["seed_base"]+1000*supplement["modes"],
        "profile": supplement["profile"], "panel": "supplement"}
    methods[key] = [("core_k"+str(k), "coherent_core", k) for k in supplement["core_cutoffs"]]
    methods[key] += [("retained_k"+str(k), "retained", k) for k in supplement["retained_budgets"]]
    if sum(i["panel"] == "primary" for i in inputs.values()) != config["planned_primary_inputs"]:
        raise ValueError("Primary input count in configuration is inconsistent")
    for family, field in (("coherent_core", "planned_primary_core_laws"), ("retained", "planned_primary_retained_laws")):
        count = sum(f == family for key, specs in methods.items() if inputs[key]["panel"] == "primary" for _, f, _ in specs)
        if count != config[field]:
            raise ValueError("Configured primary law count is inconsistent")
    return inputs, methods


def timing_plan(config):
    inputs, jobs = {}, {}
    for case in config["timing"]["cases"]:
        cid = case_id(case["modes"], case["profile"], case["seed_base"])
        if cid in inputs:
            raise ValueError("Timing case is duplicated")
        inputs[cid] = dict(case, case_id=cid, seed=case["seed_base"]+1000*case["modes"])
        specs = [("core_k"+str(k), "coherent_core", k) for k in case["core_cutoffs"]]
        specs += [("retained_k"+str(k), "retained", k) for k in case["retained_budgets"]]
        for method, family, budget in specs:
            for samples in case["sample_counts"]:
                for replicate in range(config["timing"]["repeats"]):
                    key = (cid, method, samples, replicate)
                    jobs[key] = {"case_id": cid, "method": method, "family": family,
                        "budget": budget, "samples": samples, "replicate": replicate,
                        "seed": config["timing"]["sample_seed_base"]+replicate}
    if len(jobs) != config["timing"]["planned_runs"]:
        raise ValueError("Configured timing run count is inconsistent")
    return inputs, jobs


def unique_index(rows, keys, label):
    index = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        if key in index:
            raise ValueError("Duplicate " + label + " key: " + str(key))
        index[key] = row
    return index


def panel_metadata(directory, bindings):
    metadata = read_json(directory/"metadata.json", bindings)
    if metadata.get("status") not in ("complete", "complete_with_failures", "investigation_required"):
        raise ValueError("The panel has not completed: " + str(directory))
    if metadata.get("complete_matrix_keys_verified") is not True or metadata.get("sources_and_binaries_stable") is not True:
        raise ValueError("Panel completeness/source stability was not established")
    checked_file(directory/"config.json", metadata["config_sha256"], bindings)
    checked_file(directory/"validation_gate.json", metadata["validation_sha256"], bindings)
    config = read_json(directory/"config.json", bindings)
    gate = read_json(directory/"validation_gate.json", bindings)
    if gate.get("passed") is not True or gate.get("skipped") != 0 or gate.get("sources_and_binaries_stable_during_tests") is not True:
        raise ValueError("The archived complete gate is not a stable zero-skip pass")
    if not metadata["source_hashes"] or metadata["source_hashes"] != gate["source_hashes"]:
        raise ValueError("Panel scientific sources differ from the tested sources")
    if not metadata.get("artifact_hashes"):
        raise ValueError("Panel artifact manifest is missing")
    for name, expected in metadata["artifact_hashes"].items():
        checked_file(directory/name,expected,bindings)
    for name, expected in metadata["source_hashes"].items():
        checked_file(directory/"source_snapshot"/name, expected, bindings)
        if sha(ROOT/name) != expected:
            raise ValueError("A frozen scientific source changed before analysis: " + name)
    for family in ("core", "retained"):
        build = read_json(directory/(family+"_build.json"), bindings)
        if build.get("executable_sha256") != metadata["binary_hashes"][family] or gate["binary_hashes"][family] != metadata["binary_hashes"][family]:
            raise ValueError("Build/gate/panel binary identities do not match")
        for name, expected in build.get("source_hashes", {}).items():
            if metadata["source_hashes"].get(name) != expected:
                raise ValueError("Build source is not the frozen scientific source: " + name)
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
        if metadata["thread_environment"].get(variable) != "1":
            raise ValueError("Panel was not run with the declared single-thread environment")
    return metadata, config


def target_arrays(directory, spec, row, bindings):
    path = checked_file(directory/spec["case_id"]/"target.npz", row["input_sha256"], bindings)
    with np.load(path, allow_pickle=False) as stored:
        U, V, P = stored["unitary"], stored["covariance"], stored["target"]
    m = spec["modes"]
    if U.shape != (m, m) or V.shape != (2*m, 2*m) or not np.all(np.isfinite(U)) or not np.all(np.isfinite(V)):
        raise ValueError("Target input arrays have invalid dimensions or values")
    if np.max(np.abs(U.conj().T@U-np.eye(m))) > 1e-10 or np.max(np.abs(V-V.T)) > 1e-12:
        raise ValueError("Saved input is not a unitary/symmetric covariance within tolerance")
    return U, V, validate_law(P, m)


def saved_mobius(marginals):
    result = np.asarray(marginals, dtype=np.longdouble).copy()
    count = len(result)
    indices = np.arange(count)
    for bit in range(count.bit_length()-1):
        left = indices[(indices & (1 << bit)) == 0]
        result[left] -= result[left | (1 << bit)]
    return result[(count-1)^indices]


def core_evidence(directory, row, U, V, native, config, bindings):
    cid, method = row["case_id"], row["method"]
    coeff_path = checked_file(directory/cid/(method+"_coefficients.npz"), row["coefficient_sha256"], bindings)
    with np.load(coeff_path, allow_pickle=False) as stored:
        coeff = {name: stored[name] for name in stored.files}
    native_occ, reference_occ = coeff["native_occupations"], coeff["reference_occupations"]
    native_amp = np.asarray(coeff["native_amplitudes"], dtype=np.clongdouble)
    reference_amp = np.asarray(coeff["reference_amplitudes"], dtype=np.complex128)
    if len(native_amp) != len(native_occ) or len(reference_amp) != len(reference_occ):
        raise ValueError("Coefficient arrays have inconsistent lengths")
    native_map = {tuple(n): a for n, a in zip(native_occ, native_amp)}
    if len(native_map) != len(native_occ) or set(native_map) != set(map(tuple, reference_occ)):
        raise ValueError("Native/independent occupation bases do not match")
    aligned = np.asarray([native_map[tuple(n)] for n in reference_occ], dtype=np.clongdouble)
    if not np.array_equal(aligned, coeff["native_aligned_amplitudes"]):
        raise ValueError("Saved native coefficient alignment is incorrect")
    if not np.all(np.isfinite(aligned)) or not np.all(np.isfinite(reference_amp)):
        raise ValueError("Nonfinite coefficient vector")
    exported = native["state"]
    json_amp = np.asarray(exported["amplitudes_real"], dtype=np.longdouble) + np.clongdouble(1j)*np.asarray(exported["amplitudes_imag"], dtype=np.longdouble)
    json_map = {tuple(n): z for n,z in zip(exported["occupations"],json_amp)}
    if set(json_map) != set(native_map):
        raise ValueError("Native JSON and full-precision NPZ occupations differ")
    if max(abs(json_map[n]-z) for n,z in native_map.items()) > 2e-15:
        raise ValueError("Native JSON coefficient values differ from their NPZ evidence")
    native_norm = float(np.sum(np.abs(aligned)**2, dtype=np.longdouble))
    reference_norm = float(np.sum(np.abs(reference_amp.astype(np.clongdouble))**2,dtype=np.longdouble))
    max_error = float(np.max(np.abs(aligned-reference_amp.astype(np.clongdouble))))
    l2_error = float(np.sqrt(np.sum(np.abs(aligned-reference_amp.astype(np.clongdouble))**2, dtype=np.longdouble)))
    settings = config["reference"]
    if max_error > settings["coefficient_comparison_max_absolute_tolerance"] or max(abs(native_norm-1),abs(reference_norm-1),l2_error) > settings["coefficient_norm_comparison_tolerance"]:
        raise ValueError("Archived coefficient vectors fail the fixed comparison gate")
    rounded = aligned.astype(np.complex128)
    before = float(np.linalg.norm(rounded))
    if before <= 0 or not math.isfinite(before):
        raise ValueError("Exported native coefficient vector cannot be normalized")
    normalized = rounded/before
    saved_input = np.asarray(coeff["reference_input_amplitudes"], dtype=np.complex128)
    np.testing.assert_allclose(saved_input, normalized, rtol=0, atol=2e-15)
    if np.any(saved_input[np.asarray(reference_occ).sum(axis=1) % 2 == 1] != 0):
        raise ValueError("Initial core has nonzero odd occupations")
    after = float(np.sum(np.abs(saved_input)**2, dtype=np.longdouble))
    close(coeff["reference_input_norm_before"],before,"saved conversion norm",atol=2e-15)
    close(coeff["reference_input_norm_after"],math.sqrt(after),"saved normalized input norm",atol=2e-15)
    close(row["coefficient_comparison"]["maximum_absolute_error"],max_error,"coefficient max error",atol=2e-15)
    close(row["coefficient_comparison"]["l2_error"],l2_error,"coefficient L2 error",atol=2e-15)
    if abs(after-1) > settings["coefficient_norm_comparison_tolerance"]:
        raise ValueError("Reference input is not normalized")
    np.testing.assert_allclose(coeff["target_covariance"], V, rtol=0, atol=2e-12)
    np.testing.assert_allclose(coeff["pure_covariance"]+coeff["displacement_covariance"], V, rtol=0, atol=2e-12)
    profile = next(p for p in config["profiles"] if p["name"] == profile_name(row["profile"]))
    r, eta = profile["squeezing"], profile["transmissivity"]
    a = 1-eta+eta*math.exp(-2*r)
    delta = 4*eta*(1-eta)*math.sinh(r)**2/a
    O = np.block([[U.real, -U.imag], [U.imag, U.real]])
    expected_C = delta*(O[:, len(U):]@O[:, len(U):].T)
    np.testing.assert_allclose(coeff["displacement_covariance"], expected_C, rtol=0, atol=2e-12)
    tail = finite(coeff["native_tail_probability"], "native tail", nonnegative=True)
    reference_tail = finite(coeff["reference_tail_probability"], "reference tail", nonnegative=True)
    if tail > 1 or abs(tail-reference_tail) > settings["native_tail_comparison_absolute_tolerance"]:
        raise ValueError("Archived core tail comparison fails")
    nominal_bound = math.sqrt(tail)
    close(native["tail_probability"], tail, "native tail JSON/NPZ", atol=2e-15)
    close(row["nominal_trace_bound"], nominal_bound, "nominal core bound")
    return {"native_norm_squared": native_norm, "coefficient_max_absolute_error": max_error,
        "coefficient_l2_error": l2_error, "reference_input_norm_before": before,
        "reference_input_norm_after_squared": after, "native_tail_probability": tail,
        "nominal_trace_bound": nominal_bound, "coefficient_file": str(coeff_path)}


def audit_quality(directory, config, metadata, bindings):
    expected_inputs, methods = quality_plan(config)
    input_rows = read_json(directory/"inputs.json", bindings)
    row_index = unique_index(read_json(directory/"metrics.json", bindings), ("case_id", "method"), "quality")
    inputs = unique_index(input_rows, ("case_id",), "quality input")
    if set(inputs) != {(k,) for k in expected_inputs}:
        raise ValueError("The complete fixed input matrix is not present")
    expected_keys = {(cid, method) for cid, specs in methods.items() for method, _, _ in specs}
    if set(row_index) != expected_keys:
        raise ValueError("Quality method matrix is missing, duplicated, or expanded")
    audited, targets = [], {}
    for cid, spec in expected_inputs.items():
        input_row = inputs[(cid,)]
        for field in ("modes", "seed_base", "seed"):
            if input_row[field] != spec[field]:
                raise ValueError("Quality input identity differs from the fixed matrix")
        if profile_name(input_row["profile"]) != spec["profile"]:
            raise ValueError("Quality input profile changed")
        U, V, P = target_arrays(directory, spec, input_row, bindings)
        targets[cid] = {"unitary": U, "covariance": V, "target": P,
            "input_sha256": input_row["input_sha256"], "spec": spec}
        for method, family, budget in methods[cid]:
            row = row_index[cid, method]
            if row["family"] != family or row["input_sha256"] != input_row["input_sha256"] or row["panel"] != spec["panel"]:
                raise ValueError("Quality row family/input/panel binding mismatch")
            for field in ("modes", "seed", "seed_base"):
                if row[field] != spec[field]:
                    raise ValueError("Quality row case identity changed")
            if profile_name(row["profile"]) != spec["profile"] or row["cutoff" if family == "coherent_core" else "retained_budget"] != budget:
                raise ValueError("Quality row profile/budget changed")
            record = {**spec, "method": method, "family": family, "budget": budget,
                "status": row["status"], "quality_scope": CORE_SCOPE if family == "coherent_core" else RETAINED_SCOPE,
                "construction_reference_tv": None, "extended_tv": None,
                "quality_value": None, "failure_mass": None, "nominal_trace_bound": None}
            if row["status"] != SUCCESS:
                if row.get("construction_reference_tv") is not None or row.get("extended_tv") is not None:
                    raise ValueError("Unavailable quality row claims a complete TV")
                for field,suffix in (("native_result_sha256","_native.json"),("coefficient_sha256","_coefficients.npz"),("law_sha256",".npz")):
                    if row.get(field) is not None:
                        checked_file(directory/cid/(method+suffix),row[field],bindings)
                record.update(error=row.get("error"), resource_diagnostics=row.get("resource_diagnostics"))
                audited.append(record)
                continue
            native_path = checked_file(directory/cid/(method+"_native.json"), row["native_result_sha256"], bindings)
            native = read_json(native_path, bindings)
            if native.get("status") != "ok":
                raise ValueError("Completed quality row has a failed native result")
            binary_family = "core" if family == "coherent_core" else "retained"
            if native.get("executable_sha256") != metadata["binary_hashes"][binary_family] or native.get("modes") != spec["modes"]:
                raise ValueError("Quality native result used a different binary or mode count")
            if native["cutoff" if family == "coherent_core" else "k"] != budget:
                raise ValueError("Native quality budget differs from the fixed row")
            law_path = checked_file(directory/cid/(method+".npz"), row["law_sha256"], bindings)
            with np.load(law_path, allow_pickle=False) as stored:
                law = {name: stored[name] for name in stored.files}
            if family == "coherent_core":
                if row.get("extended_tv") is not None or row.get("failure_mass") is not None or "failure_mass" in law:
                    raise ValueError("Construction reference incorrectly claims a full native failure law")
                q = validate_law(law["distribution"], spec["modes"])
                raw = np.asarray(law["raw_distribution"], dtype=float)
                marginals = np.asarray(law["vacuum_marginals"], dtype=float)
                if raw.shape != q.shape or marginals.shape != q.shape or not np.all(np.isfinite(raw)) or not np.all(np.isfinite(marginals)):
                    raise ValueError("Malformed raw construction reference")
                settings = config["reference"]
                correction = float(np.sum(np.abs(q.astype(np.longdouble)-raw.astype(np.longdouble)), dtype=np.longdouble))
                raw_error = float(abs(np.sum(raw, dtype=np.longdouble)-1))
                if raw.min() < -settings["negative_atom_tolerance"] or raw_error > settings["sum_error_tolerance"] or correction > settings["maximum_correction_l1"]:
                    raise ValueError("Construction reference exceeds frozen repair limits")
                marginal_difference = float(np.max(np.abs(saved_mobius(marginals)-raw)))
                if marginal_difference > settings["sum_error_tolerance"]:
                    raise ValueError("Saved raw law does not agree with saved vacuum marginals")
                close(row["reference_diagnostics"]["correction_l1"], correction, "reference repair", atol=2e-15)
                evidence = core_evidence(directory, row, U, V, native, config, bindings)
                tv = float(np.sum(np.abs(q.astype(np.longdouble)-P.astype(np.longdouble)), dtype=np.longdouble)/2)
                close(row["construction_reference_tv"], tv, "construction_reference_tv")
                nominal_pass = tv <= evidence["nominal_trace_bound"] + settings["nominal_trace_bound_check_absolute_tolerance"]
                if not nominal_pass or row["nominal_bound_check_passed"] is not True:
                    raise ValueError("Complete construction row fails its nominal state-budget check")
                record.update(construction_reference_tv=tv, quality_value=tv,
                    nominal_trace_bound=evidence["nominal_trace_bound"], coefficient_audit=evidence,
                    reference_correction_l1=correction, raw_reference_sum_error=raw_error,
                    saved_marginal_mobius_max_difference=marginal_difference,
                    native_global_failure_probability="not integrated or certified")
            else:
                if row.get("construction_reference_tv") is not None:
                    raise ValueError("Retained law has the wrong quality scope")
                failure = finite(law["failure_mass"], "retained failure mass", nonnegative=True)
                q = validate_law(law["distribution"], spec["modes"], failure)
                tv = float((np.sum(np.abs(q.astype(np.longdouble)-P.astype(np.longdouble)), dtype=np.longdouble)+failure)/2)
                close(row["extended_tv"], tv, "retained extended_tv")
                close(row["failure_mass"], failure, "retained failure_mass")
                np.testing.assert_allclose(native["distribution"], q, rtol=0, atol=2e-15)
                close(native.get("failure_mass", 0), failure, "retained native/NPZ failure mass", atol=2e-15)
                record.update(extended_tv=tv, quality_value=tv, failure_mass=failure)
            audited.append(record)
    checks = read_json(directory/"crosschecks.json", bindings)
    check_index=unique_index(checks,("case_id","method","kind"),"quality crosscheck")
    expected_checks={}
    for (cid,method),row in row_index.items():
        if row.get("coefficient_comparison") is not None:
            expected_checks[cid,method,"native_independent_coefficients"]=row["coefficient_comparison"]["passed"]
        if "nominal_bound_check_passed" in row:
            expected_checks[cid,method,"nominal_trace_budget_consistency"]=row["nominal_bound_check_passed"]
        if row["family"]=="retained" and row["status"]==SUCCESS:
            expected_checks[cid,method,"retained_extended_law_mass"]=True
    if set(check_index)!=set(expected_checks):
        raise ValueError("Quality crosscheck matrix is incomplete or expanded")
    if any(check_index[key].get("passed") is not passed for key,passed in expected_checks.items()):
        raise ValueError("Quality crosscheck verdict disagrees with its row")
    return audited, targets, checks


def timing_key(row):
    return (row["case_id"], row["method"], row["samples"] if "samples" in row else row["requested_samples"], row["replicate"])


def audit_timing(directory, config, quality_targets, metadata, bindings):
    expected_inputs, expected_jobs = timing_plan(config)
    input_rows = read_json(directory/"inputs.json", bindings)
    inputs = unique_index(input_rows, ("case_id",), "timing input")
    if set(inputs) != {(cid,) for cid in expected_inputs}:
        raise ValueError("Timing input matrix differs from the plan")
    for cid, case in expected_inputs.items():
        row = inputs[(cid,)]
        if row["input_path"] != cid+"/input.npz":
            raise ValueError("Unexpected timing input path")
        path = checked_file(directory/row["input_path"], row["input_sha256"], bindings)
        with np.load(path, allow_pickle=False) as stored:
            U, V = stored["unitary"], stored["covariance"]
        m = case["modes"]
        if U.shape != (m,m) or V.shape != (2*m,2*m) or not np.all(np.isfinite(U)) or not np.all(np.isfinite(V)):
            raise ValueError("Malformed timing input")
        if cid in quality_targets:
            np.testing.assert_array_equal(U, quality_targets[cid]["unitary"])
            np.testing.assert_array_equal(V, quality_targets[cid]["covariance"])
            if row["quality_target_sha256"] != quality_targets[cid]["input_sha256"]:
                raise ValueError("Timing input does not bind the quality target")
        elif row.get("quality_case_id") is not None:
            raise ValueError("Unmeasured timing input claims a quality law")
    rows = read_json(directory/"metrics.json", bindings)
    order = read_json(directory/"job_order.json", bindings)
    if len(rows) != len(expected_jobs) or len(order) != len(expected_jobs):
        raise ValueError("Timing/job-order count is incomplete")
    indexed = {}
    for row in rows:
        key = timing_key(row)
        if key in indexed:
            raise ValueError("Duplicate timing cell")
        indexed[key] = row
    if set(indexed) != set(expected_jobs) or {timing_key(job) for job in order} != set(expected_jobs):
        raise ValueError("Timing/job-order keys differ from the fixed complete plan")
    fixed_order=[]
    for case in config["timing"]["cases"]:
        cid=case_id(case["modes"],case["profile"],case["seed_base"])
        methods=["core_k"+str(k) for k in case["core_cutoffs"]]+["retained_k"+str(k) for k in case["retained_budgets"]]
        for samples in case["sample_counts"]:
            for replicate in range(config["timing"]["repeats"]):
                fixed_order.extend((cid,method,samples,replicate) for method in methods)
    np.random.default_rng(config["timing"]["shuffle_seed"]).shuffle(fixed_order)
    if [timing_key(job) for job in order]!=fixed_order:
        raise ValueError("Timing order differs from the fixed seeded shuffle")
    ordinals = {row["ordinal"] for row in rows}
    if ordinals != set(range(len(expected_jobs))):
        raise ValueError("Timing ordinals are not a complete permutation")
    groups, audited_rows = defaultdict(list), []
    for key, expected in expected_jobs.items():
        row = indexed[key]
        if timing_key(order[row["ordinal"]]) != key:
            raise ValueError("Timing ordinal does not match the saved predetermined job order")
        cid, method, samples, replicate = key
        for field in ("family", "seed"):
            if row[field] != expected[field]:
                raise ValueError("Timing family or fixed seed differs")
        budget_field="cutoff" if expected["family"]=="coherent_core" else "retained_budget"
        if row[budget_field]!=expected["budget"]:
            raise ValueError("Timing method budget changed")
        if row["input_path"] != cid+"/input.npz" or row["input_sha256"] != inputs[(cid,)]["input_sha256"]:
            raise ValueError("Timing row input binding differs")
        if row["requested_samples"] != samples:
            raise ValueError("Requested batch size was changed")
        raw_path = checked_file(directory/f"run_{row['ordinal']:03d}.json", row["raw_result_sha256"], bindings)
        raw = read_json(raw_path, bindings)
        complete = row["status"] == SUCCESS
        record = {"ordinal": row["ordinal"], "case_id": cid, "method": method,
            "family": expected["family"], "samples": samples, "replicate": replicate,
            "status": row["status"], "completed_samples": row["completed_samples"],
            "cold_batch": None, "setup": None, "sampling": None,
            "native_input_seconds":None,"native_input_setup_sampling_seconds":None,
            "wrapper_seconds": finite(row["wrapper_seconds"], "wrapper cost", nonnegative=True)}
        if complete:
            if raw.get("status") != "ok" or row["completed_samples"] != samples:
                raise ValueError("A completed timing row does not contain a full successful batch")
            family = "core" if expected["family"] == "coherent_core" else "retained"
            if raw.get("executable_sha256") != metadata["binary_hashes"][family]:
                raise ValueError("Timing raw output used a different executable")
            if raw["seed"] != expected["seed"] or raw["modes"] != expected_inputs[cid]["modes"]:
                raise ValueError("Native timing request identity differs")
            if raw["cutoff" if expected["family"]=="coherent_core" else "k"]!=expected["budget"]:
                raise ValueError("Native timing method budget changed")
            native_count = raw.get("completed_samples", raw.get("sample_count"))
            if native_count != samples:
                raise ValueError("Raw timing output does not contain the requested complete batch")
            bits = np.asarray(raw["raw_samples"])
            if bits.shape != (samples, expected_inputs[cid]["modes"]) or np.any((bits != 0)&(bits != 1)):
                raise ValueError("Timing raw sample shape/binary entries are invalid")
            for name in ("cold_batch", "setup", "sampling"):
                value = finite(raw["timings_seconds"][name], "native "+name, nonnegative=True)
                close(row["timings_seconds"][name], value, "timing "+name)
                record[name] = value
            close(record["cold_batch"], record["setup"]+record["sampling"], "native complete batch cost", atol=1e-10)
            record["native_input_seconds"] = finite(raw["timings_seconds"]["input"], "native input", nonnegative=True)
            record["native_input_setup_sampling_seconds"] = record["native_input_seconds"]+record["setup"]+record["sampling"]
            close(row["native_input_setup_sampling_seconds"],record["native_input_setup_sampling_seconds"],"native input/setup/sampling")
            for name in ("wrapper_input_validation_seconds","wrapper_process_wall_seconds"):
                record[name] = finite(row[name],name,nonnegative=True)
                close(record[name],raw[name],name)
            if record["cold_batch"] <= 0 or record["wrapper_seconds"]+1e-9 < record["cold_batch"]:
                raise ValueError("Complete native/wrapper timing is inconsistent")
            if expected["family"] == "coherent_core":
                record["nominal_trace_bound"] = finite(raw["trace_distance_bound"], "timing nominal budget", nonnegative=True)
                record["full_basis"] = raw["resources"]["full_basis"]
                record["numeric_failures_observed"] = raw["diagnostics"]["numeric_failures"]
                m=expected_inputs[cid]["modes"]
                if raw["resources"]["full_basis"]!=math.comb(m+expected["budget"],expected["budget"]):
                    raise ValueError("Native basis dimension differs from total-photon cutoff")
                for name,count in (("normal_variates",samples*m),("displacement_evaluations",samples),("quantum_uniforms",2*samples*m)):
                    if raw["diagnostics"][name]!=count:
                        raise ValueError("Complete core batch has unexpected displacement/measurement draw counts")
            elif (raw["selector"]!=config["timing"]["retained_selector"] or raw["vacuum_kernel"]!=config["timing"]["retained_vacuum_kernel"] or raw["prefix_cache"]["requested_max_entries"]!=0 or raw["prefix_cache"]["enabled"] is not False):
                raise ValueError("Retained timing options differ from the fixed no-cache baseline")
        else:
            count = row["completed_samples"]
            if count is not None and (type(count) is not int or not 0 <= count <= samples):
                raise ValueError("Failed batch has an invalid preserved completion count")
            record.update(error=row.get("error"), native_failure_is_global_probability=False,
                completed_sample_count_known=count is not None,
                completed_count_scope="preserved native evidence" if count is not None else "unknown; timeout/missing native payload is not zero completed samples")
        groups[cid, method, samples].append(record)
        audited_rows.append(record)
    aggregated = []
    repeats = config["timing"]["repeats"]
    for (cid, method, samples), group in groups.items():
        if len(group) != repeats or {row["replicate"] for row in group} != set(range(repeats)):
            raise ValueError("The three predetermined timing repetitions are incomplete")
        good = [row for row in group if row["status"] == SUCCESS]
        all_good = len(good) == repeats
        entry = {"case_id": cid, "modes": expected_inputs[cid]["modes"], "method": method,
            "family": group[0]["family"], "samples": samples, "repetitions": repeats,
            "successful_repetitions": len(good), "all_repetitions_complete": all_good,
            "median_cold_seconds": float(np.median([r["cold_batch"] for r in good])) if all_good else None,
            "minimum_cold_seconds": min((r["cold_batch"] for r in good), default=None),
            "maximum_cold_seconds": max((r["cold_batch"] for r in good), default=None),
            "median_setup_seconds": float(np.median([r["setup"] for r in good])) if all_good else None,
            "median_sampling_seconds": float(np.median([r["sampling"] for r in good])) if all_good else None,
            "median_input_setup_sampling_seconds":float(np.median([r["native_input_setup_sampling_seconds"] for r in good])) if all_good else None,
            "median_native_input_seconds":float(np.median([r["native_input_seconds"] for r in good])) if all_good else None,
            "median_wrapper_seconds": float(np.median([r["wrapper_seconds"] for r in group])) if all_good else None,
            "ordinals": sorted(r["ordinal"] for r in group)}
        if group[0]["family"] == "coherent_core":
            bounds = [r["nominal_trace_bound"] for r in good]
            entry["nominal_trace_bound"] = bounds[0] if bounds else None
            entry["full_basis"] = good[0]["full_basis"] if good else None
            if bounds and max(bounds)-min(bounds) > 1e-13:
                raise ValueError("Identical timing cores report different state budgets")
        aggregated.append(entry)
    boundary = read_json(directory/"resource_boundary.json", bindings)
    planned=config["resource_boundary"]
    if boundary["modes"]!=planned["modes"] or boundary["cutoff"]!=planned["cutoff"] or boundary["profile"]!=planned["profile"]:
        raise ValueError("Resource boundary input changed")
    if boundary["preflight_rejected"] is not True or any(boundary[key] is not False for key in ("backend_invoked","state_allocated","is_numerical_failure","is_replacement_timing_input")):
        raise ValueError("Resource boundary was not a pre-allocation rejection")
    if boundary["resources"]["feasible"] is not False or boundary["resources"]["full_basis"]!=math.comb(planned["modes"]+planned["cutoff"],planned["cutoff"]):
        raise ValueError("Resource boundary combinatorial count differs")
    return audited_rows, aggregated, boundary


def comparisons(quality, timings, config):
    qindex = {(r["case_id"],r["method"]): r for r in quality}
    tindex = {(r["case_id"],r["method"],r["samples"]): r for r in timings}
    quality_pairs, cost_pairs = [], []
    for cid in dict.fromkeys(r["case_id"] for r in quality):
        cores = [r for r in quality if r["case_id"] == cid and r["family"] == "coherent_core"]
        retained = [r for r in quality if r["case_id"] == cid and r["family"] == "retained"]
        for core in cores:
            for baseline in retained:
                available = core["quality_value"] is not None and baseline["quality_value"] is not None
                quality_pairs.append({"case_id":cid,"core_method":core["method"],"retained_method":baseline["method"],
                    "core_construction_tv":core["quality_value"],"retained_extended_tv":baseline["quality_value"],
                    "tv_difference":core["quality_value"]-baseline["quality_value"] if available else None,
                    "scope":"descriptive comparison of construction-reference and extended-native laws"})
    for case in config["timing"]["cases"]:
        cid = case_id(case["modes"],case["profile"],case["seed_base"])
        for cutoff in case["core_cutoffs"]:
            for retained_budget in case["retained_budgets"]:
                core, baseline = "core_k"+str(cutoff), "retained_k"+str(retained_budget)
                for samples in case["sample_counts"]:
                    a,b = tindex[cid,core,samples],tindex[cid,baseline,samples]
                    costs_available = a["median_cold_seconds"] is not None and b["median_cold_seconds"] is not None
                    qa,qb = qindex.get((cid,core)),qindex.get((cid,baseline))
                    quality_available = bool(qa and qb and qa["quality_value"] is not None and qb["quality_value"] is not None)
                    difference = qa["quality_value"]-qb["quality_value"] if quality_available else None
                    ratio = a["median_cold_seconds"]/b["median_cold_seconds"] if costs_available else None
                    cost_pairs.append({"case_id":cid,"modes":case["modes"],"samples":samples,
                        "core_method":core,"retained_method":baseline,"core_cold_seconds":a["median_cold_seconds"],
                        "retained_cold_seconds":b["median_cold_seconds"],"time_ratio":ratio,
                        "quality_available":quality_available,"core_construction_tv":qa["quality_value"] if qa else None,
                        "retained_extended_tv":qb["quality_value"] if qb else None,"tv_difference":difference,
                        "construction_reference_not_worse_and_faster": difference <= 0 and ratio < 1 if quality_available and costs_available else None,
                        "finite_native_quality_certificate":False,
                        "scope":"cost only; no full Q" if not quality_available else "construction-reference quality versus extended-native baseline; no global native error certificate"})
    return quality_pairs,cost_pairs


def display(value, precision=7):
    return "—" if value is None else f"{value:.{precision}g}"


def write_tables(path, summary):
    lines = ["# Iteration 10: constructed-law quality and native batch cost", "",
        "Core TV is the ideal-channel construction reference of the exported normalized coefficient vector. Retained TV includes its enumerated native failure mass. These are different evidence scopes; neither the reference comparison nor a finite batch with zero observed failures certifies all native core sampling errors.", "",
        f"Verified planned rows: {summary['quality_rows']} quality laws and {summary['timing_runs']} cold processes. Successful quality rows: {summary['successful_quality_rows']}; successful timing runs: {summary['successful_timing_runs']}.", "",
        "## Complete fixed quality panel", "",
        "| Panel | Input | Method | Status | Construction-reference TV | Retained extended TV | Nominal sqrt(tail) | Retained failure mass |",
        "|---|---|---|---|---:|---:|---:|---:|"]
    for row in summary["quality_rows_detail"]:
        lines.append(f"| {row['panel']} | {row['case_id']} | {row['method']} | {row['status']} | {display(row['construction_reference_tv'])} | {display(row['extended_tv'])} | {display(row['nominal_trace_bound'])} | {display(row['failure_mass'])} |")
    lines += ["", "## Native complete batch costs", "",
        "Each median requires all three predetermined repetitions to finish. The cost includes native setup plus sampling; component medians need not sum to the median total. Missing medians retain failed repetitions.", "",
        "| Input | Method | N | Complete repetitions | Median cold batch (s) | Min–max (s) | Median input+setup+sampling (s) | Median wrapper (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary["timing_aggregates"]:
        lines.append(f"| {row['case_id']} | {row['method']} | {row['samples']} | {row['successful_repetitions']}/{row['repetitions']} | {display(row['median_cold_seconds'])} | {display(row['minimum_cold_seconds'])}–{display(row['maximum_cold_seconds'])} | {display(row['median_input_setup_sampling_seconds'])} | {display(row['median_wrapper_seconds'])} |")
    lines += ["", "## Descriptive paired cost/quality values", "",
        "Time ratio = core/retained, from medians of three cold processes. TV difference = core construction-reference TV minus retained extended-native TV. No tie threshold is introduced. At m=32 only timing and a nominal core-state budget exist; no quality-cost victory is assigned.", "",
        "| Input | N | Core | Retained | Time ratio | TV difference (different scopes) | Construction-reference not worse and faster |",
        "|---|---:|---|---|---:|---:|---|"]
    for row in summary["cost_comparisons"]:
        claim = row["construction_reference_not_worse_and_faster"]
        lines.append(f"| {row['case_id']} | {row['samples']} | {row['core_method']} | {row['retained_method']} | {display(row['time_ratio'])} | {display(row['tv_difference'])} | {'unavailable' if claim is None else str(claim).lower()} |")
    lines += ["", "## Numerical evidence", "",
        f"Maximum core coefficient difference: {display(summary['maximum_coefficient_absolute_error'])}; maximum saved reference repair L1: {display(summary['maximum_reference_correction_l1'])}.", "",
        "The ideal projection theorem and the floating construction reference are separate from finite-native rounding, Gaussian RNG, and global failure probability. Floating tails, reference repairs, and the nominal bound checks are not outward-rounded certificates.", "",
        "All source, input, law, coefficient, raw timing, and gate hashes re-read by the analyzer are stored in summary.json. The analyzer does not rerun the formal matrix or recompute targets/integrals."]
    path.write_text("\n".join(lines)+"\n")


def plot_quality_cost(output, quality, timings, config):
    plt.rcParams.update({"font.family":"DejaVu Sans","svg.fonttype":"none","font.size":9})
    expected_inputs,_ = quality_plan(config)
    cases = list(expected_inputs)
    methods = ["core_k0","core_k2","core_k4","retained_k2","retained_k4","retained_k6"]
    qindex = {(r["case_id"],r["method"]):r for r in quality}
    matrix = np.full((len(methods),len(cases)),np.nan)
    labels = np.full(matrix.shape,"n/a",dtype=object)
    for y,method in enumerate(methods):
        for x,cid in enumerate(cases):
            row = qindex.get((cid,method))
            if row:
                value = row["quality_value"]
                labels[y,x] = f"{value:.2g}" if value is not None else "unavail."
                if value is not None and value > 0:
                    matrix[y,x] = math.log10(value)
    present = matrix[np.isfinite(matrix)]
    lower,upper = (float(present.min()),float(present.max())) if len(present) else (-1.,0.)
    if upper == lower: upper=lower+1
    fig = plt.figure(figsize=(14,9.4))
    grid = fig.add_gridspec(2,3,height_ratios=[1.25,1.],hspace=.58,wspace=.30)
    ax=fig.add_subplot(grid[0,:])
    cmap=plt.get_cmap("viridis_r").copy();cmap.set_bad("#eceff2")
    im=ax.imshow(np.ma.masked_invalid(matrix),aspect="auto",cmap=cmap,norm=Normalize(lower,upper))
    ax.set_xticks(range(len(cases)),[f"m{i['modes']} {i['profile']}\ns{str(i['seed_base'])[-1]}"+(" *" if i['panel']=="supplement" else "") for i in expected_inputs.values()])
    ax.set_yticks(range(len(methods)),[m.replace("core_k","Core K=").replace("retained_k","Retained k=") for m in methods])
    ax.axhline(2.5,color="white",linewidth=3)
    for y in range(len(methods)):
        for x in range(len(cases)):
            color="white" if np.isfinite(matrix[y,x]) and (matrix[y,x]-lower)/(upper-lower)>.55 else "#15252f"
            ax.text(x,y,labels[y,x],ha="center",va="center",fontsize=8.5,color=color)
    ax.set_title("A  Complete-law distances: core construction reference / retained extended native law",loc="left",fontweight="bold",pad=12)
    fig.colorbar(im,ax=ax,pad=.015,fraction=.025,label="log10 numerical TV")
    colors={"core_k0":"#127c98","core_k2":"#0b5173","core_k4":"#142c52","retained_k2":"#d3a52d","retained_k4":"#d76b1d","retained_k6":"#9b2f25"}
    for column,case in enumerate(config["timing"]["cases"]):
        axis=fig.add_subplot(grid[1,column])
        cid=case_id(case["modes"],case["profile"],case["seed_base"])
        selected=[row for row in timings if row["case_id"]==cid]
        method_names=list(dict.fromkeys(row["method"] for row in selected))
        for method in method_names:
            group=sorted((row for row in selected if row["method"]==method),key=lambda row:row["samples"])
            x=np.array([row["samples"] for row in group])
            y=np.array([row["median_cold_seconds"] if row["median_cold_seconds"] is not None else np.nan for row in group])
            good=np.isfinite(y)
            label=method.replace("core_k","Core K=").replace("retained_k","Retained k=")
            axis.plot(x,y,marker="o" if method.startswith("core") else "s",linestyle="-" if method.startswith("core") else "--",color=colors[method],label=label)
            if good.any():
                low=np.array([row["minimum_cold_seconds"] if row["median_cold_seconds"] is not None else np.nan for row in group])
                high=np.array([row["maximum_cold_seconds"] if row["median_cold_seconds"] is not None else np.nan for row in group])
                axis.errorbar(x[good],y[good],yerr=[y[good]-low[good],high[good]-y[good]],fmt="none",ecolor=colors[method],capsize=3)
        axis.set_xscale("log");axis.set_yscale("log")
        axis.set_xticks(case["sample_counts"],[str(n) for n in case["sample_counts"]])
        axis.set_xlabel("Requested samples")
        if column==0:axis.set_ylabel("Setup + sampling seconds")
        axis.grid(True,which="major",alpha=.22)
        axis.set_title(f"{chr(66+column)}  m={case['modes']}, {case['profile']}"+("\nNo full Q evaluated" if case['modes']==32 else "\nQuality values above"),loc="left",fontweight="bold")
        axis.legend(fontsize=8,loc="best",framealpha=.9)
    fig.suptitle("Coherent truncated cores: constructed-state quality and full native sampling cost",fontsize=16,fontweight="bold",y=.975)
    fig.subplots_adjust(top=.90,bottom=.15,left=.09,right=.95)
    fig.text(.06,.077,"Core TV excludes finite-native RNG/rounding/global failure integration. Retained TV includes its enumerated failure mass. * Separate m=12 supplement.",fontsize=9)
    fig.text(.06,.047,"Cost: median of all 3 predetermined cold processes; whiskers show min–max. Shared host, one thread. Missing cells and failed runs are not replaced.",fontsize=9)
    fig.text(.06,.018,"m=32 shows resource/cost feasibility and a nominal state-tail budget only; no large-mode quality-cost or exact-GBS efficiency claim.",fontsize=9)
    for extension in ("png","svg"):
        fig.savefig(output/("quality_cost."+extension),dpi=180,facecolor="white")
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality",type=Path,default=ROOT/"results/iteration10_quality")
    parser.add_argument("--timing",type=Path,default=ROOT/"results/iteration10_timing")
    parser.add_argument("--output",type=Path,default=ROOT/"results/iteration10_analysis")
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Analysis requires a fresh output directory")
    bindings={}
    qmeta,config=panel_metadata(args.quality,bindings)
    tmeta,tconfig=panel_metadata(args.timing,bindings)
    if config!=tconfig or qmeta["config_sha256"]!=tmeta["config_sha256"] or qmeta["validation_sha256"]!=tmeta["validation_sha256"] or qmeta["binary_hashes"]!=tmeta["binary_hashes"]:
        raise ValueError("Quality and timing do not share the frozen experiment/binaries")
    quality,targets,crosschecks=audit_quality(args.quality,config,qmeta,bindings)
    timing_rows,aggregates,boundary=audit_timing(args.timing,config,targets,tmeta,bindings)
    quality_pairs,cost_pairs=comparisons(quality,aggregates,config)
    successful_core=[r for r in quality if r["family"]=="coherent_core" and r["status"]==SUCCESS]
    successful_retained=[r for r in quality if r["family"]=="retained" and r["status"]==SUCCESS]
    summary={"created_utc":utc(),"analysis_source_sha256":sha(Path(__file__)),
        "common_source_sha256":sha(Path(__file__).with_name("iteration10_common.py")),
        "status":"complete" if all(r["status"]==SUCCESS for r in quality+timing_rows) else "complete_matrix_with_unavailable_or_failed_rows",
        "quality_rows":len(quality),"primary_quality_rows":sum(r["panel"]=="primary" for r in quality),
        "supplement_quality_rows":sum(r["panel"]=="supplement" for r in quality),
        "quality_inputs":len(targets),"successful_quality_rows":len(successful_core)+len(successful_retained),
        "core_construction_reference_laws":len(successful_core),"retained_extended_native_laws":len(successful_retained),
        "unavailable_quality_rows":sum(r["status"]!=SUCCESS for r in quality),
        "timing_runs":len(timing_rows),"successful_timing_runs":sum(r["status"]==SUCCESS for r in timing_rows),
        "failed_timing_runs":sum(r["status"]!=SUCCESS for r in timing_rows),
        "timing_cells":len(aggregates),"timing_cells_with_all_three_runs":sum(r["all_repetitions_complete"] for r in aggregates),
        "maximum_reference_correction_l1":max((r["reference_correction_l1"] for r in successful_core),default=None),
        "maximum_coefficient_absolute_error":max((r["coefficient_audit"]["coefficient_max_absolute_error"] for r in successful_core),default=None),
        "retained_rows_with_failure_mass":sum(r["failure_mass"]>0 for r in successful_retained),
        "maximum_retained_failure_mass":max((r["failure_mass"] for r in successful_retained),default=None),
        "quality_rows_detail":quality,"quality_comparisons":quality_pairs,
        "timing_rows_audit":timing_rows,"timing_aggregates":aggregates,"cost_comparisons":cost_pairs,
        "crosschecks":crosschecks,"resource_boundary":boundary,
        "quality_crosscheck_count":len(crosschecks),"quality_crosschecks_passed":all(c["passed"] for c in crosschecks),
        "evidence_files":bindings,"config_sha256":qmeta["config_sha256"],
        "quality_metadata_status":qmeta["status"],"timing_metadata_status":tmeta["status"],
        "core_quality_scope":CORE_SCOPE,"retained_quality_scope":RETAINED_SCOPE,
        "native_core_global_failure_mass":None,"large_mode_full_law_available":False,
        "analysis_recomputed_targets_or_integrals":False,"outward_rounded_certificate":False,
        "scope":"Fixed inputs and shared-host timings. Complete saved P/Q arrays, coefficient maps, source/gate/hash bindings and all cost repetitions re-read; no formal experiment rerun. Continuous differences are descriptive, without an invented tie rule."}
    args.output.mkdir(parents=True)
    save(args.output/"summary.json",summary)
    write_tables(args.output/"summary_tables.md",summary)
    plot_quality_cost(args.output,quality,aggregates,config)
    print(json.dumps({key:summary[key] for key in ("status","quality_rows","successful_quality_rows","timing_runs","successful_timing_runs","maximum_reference_correction_l1","maximum_coefficient_absolute_error")},indent=2))


if __name__=="__main__":
    main()
