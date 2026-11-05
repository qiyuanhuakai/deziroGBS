"""Frozen, exhaustive small-system comparison of conditional forest completion.

This is a probability-model experiment. Python elapsed times are descriptive
and are never compared with native performance as algorithmic speedups.
Unresolved reachable branches retain their mass at a failure symbol.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter

import numpy as np
import scipy

from gbslab.conditional_memory import ConditionalMemorySampler
from gbslab.cpp_click import run_cpp_click
from gbslab.exact import NumericalProbabilityError, ThresholdModel
from gbslab.gaussian import covariance_from_interferometer, haar_unitary
from gbslab.positive_completion import PositiveCompletionSampler, TableEventOracle
from gbslab.positive_forest import ForestCompletionSampler

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def save(path, value):
    path.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")


def extended_law(sampler, modes):
    """Identical traversal for both families, with no success conditioning."""
    q = np.zeros(1 << modes, dtype=np.longdouble)
    failures = []
    stack = [((), 0, np.longdouble(1))]
    started = perf_counter()
    while stack:
        prefix, mask, weight = stack.pop()
        if weight == 0:
            continue
        if len(prefix) == modes:
            q[mask] += weight
            continue
        try:
            p = float(sampler.conditional_click_probability(prefix))
            if not np.isfinite(p) or not 0 <= p <= 1:
                raise NumericalProbabilityError("Invalid conditional probability")
        except NumericalProbabilityError as error:
            stored_weight = float(weight)
            if weight > 0 and stored_weight == 0:
                raise ArithmeticError("Enumeration failure mass cannot be represented in float64")
            failures.append({"prefix": list(prefix), "mass": stored_weight,
                "error_type": type(error).__name__, "error": str(error)})
            continue
        pos = len(prefix)
        for bit,factor in ((1,np.longdouble(p)),(0,np.longdouble(1)-np.longdouble(p))):
            if factor > 0:
                child=weight*factor
                if child == 0 or not np.isfinite(child):
                    # An enumeration arithmetic failure is not an invented
                    # sampler failure. Abort instead of dropping the branch.
                    raise ArithmeticError("Positive enumeration branch underflowed")
                stack.append((prefix+(bit,),mask | ((1<<pos) if bit else 0),child))
    failure_mass = float(sum(r["mass"] for r in failures))
    cast_q=q.astype(np.float64)
    if np.any((q>0)&(cast_q==0)):
        raise ArithmeticError("Positive output mass cannot be represented in float64")
    q=cast_q
    normalization = float(q.sum() + failure_mass)
    if abs(normalization-1) > 2e-11:
        raise ArithmeticError("Extended sequential law failed normalization")
    return q, failure_mass, failures, perf_counter()-started


def make_sampler(V, spec, config, *, target=None):
    model = ThresholdModel(V, max_exact_modes=config["full_law_max_modes"],
                           vacuum_cache_size=config["cache_size"])
    common = {"selection": config["selector"], "cache_size": config["cache_size"],
              "max_relative_roundoff": config["max_relative_roundoff"]}
    if spec["completion"] == "none":
        return ConditionalMemorySampler(model, spec["k"], **common), model
    event_oracle=None if target is None else TableEventOracle(target)
    if spec["completion"] in ("chain", "static_mi_tree"):
        sampler=ForestCompletionSampler(model,spec["k"],forest=spec["completion"],
            event_oracle=event_oracle,**common)
    else:
        sampler=PositiveCompletionSampler(model,spec["k"],completion=spec["completion"],
            block_order=config["block_order"],event_oracle=event_oracle,**common)
    return sampler, model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/iteration7_forest.json")
    parser.add_argument("--output", type=Path, default=ROOT/"results/iteration7_forest")
    parser.add_argument("--validation", type=Path, default=ROOT/"results/iteration7_forest_validation/validation.json")
    parser.add_argument("--native-executable", type=Path, default=ROOT/".cache/click-prefix-native/click_sampler")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output directory, including after partial runs")
    config = json.loads(args.config.read_text())
    if config.get("order") != "natural physical-mode order":
        raise ValueError("This fixed runner implements natural physical-mode order only")
    method_ids = [spec["id"] for spec in config["methods"]]
    if len(method_ids) != len(set(method_ids)) or "retained_k4" not in method_ids:
        raise ValueError("Unique method IDs and the native retained_k4 baseline are required")
    gate = json.loads(args.validation.read_text())
    if gate.get("passed") is not True:
        raise ValueError("Completion validation must have passed before this experiment")
    gate_sources = gate.get("source_hashes", {})
    required_gate_sources = {"src/gbslab/positive_forest.py", "tests/test_positive_forest.py",
        "src/gbslab/positive_completion.py", "src/gbslab/conditional_memory.py",
        "src/gbslab/zero_memory.py", "src/gbslab/schur_memory.py", "src/gbslab/exact.py",
        "src/gbslab/gaussian.py"}
    if not required_gate_sources.issubset(gate_sources):
        raise ValueError("Validation gate must bind every scientific dependency")
    for source, expected_hash in gate_sources.items():
        if expected_hash != sha(ROOT/source):
            raise ValueError("Completion dependency/test does not match its validation gate: " + source)
    source_paths = [Path(__file__), ROOT/"src/gbslab/positive_forest.py", ROOT/"src/gbslab/positive_completion.py",
        ROOT/"src/gbslab/conditional_memory.py", ROOT/"src/gbslab/zero_memory.py",
        ROOT/"src/gbslab/schur_memory.py", ROOT/"src/gbslab/exact.py",
        ROOT/"src/gbslab/gaussian.py", ROOT/"src/gbslab/cpp_click.py"]
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    binary_sha = sha(args.native_executable)
    native_gate = json.loads((ROOT/"results/iteration5_prefix_validation/validation.json").read_text())
    if native_gate.get("passed") is not True or native_gate.get("executable_sha256") != binary_sha:
        raise ValueError("Native baseline must match the frozen validated binary")
    args.output.mkdir(parents=True)
    shutil.copy2(args.config, args.output/"config.json")
    shutil.copy2(args.validation, args.output/"validation_gate.json")
    for p in source_paths:
        destination = args.output/"source_snapshot"/p.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p,destination)
    metadata = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
        "base_commit": subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "source_hashes": source_hashes, "config_sha256":sha(args.config),
        "validation_sha256":sha(args.validation), "native_executable_sha256":binary_sha,
        "python":sys.version, "numpy":np.__version__, "scipy":scipy.__version__,
        "platform":platform.platform(), "dedicated_host":False,
        "scope":config["claim_scope"]}
    save(args.output/"metadata.json",metadata)
    rows, checks, inputs = [], [], []
    overall_start = perf_counter()
    with (args.output/"records.jsonl").open("w") as journal:
        for m in config["modes"]:
            for seed_base in config["seeds_per_mode"]:
                seed = seed_base + 1000*m
                U = haar_unitary(m, np.random.default_rng(seed))
                for profile in config["profiles"]:
                    case_id=f"m{m}_{profile['name']}_s{seed_base}"
                    directory=args.output/case_id
                    directory.mkdir()
                    V=covariance_from_interferometer(U,profile["squeezing"],loss=profile["transmissivity"])
                    oracle=ThresholdModel(V,max_exact_modes=m)
                    P=oracle.full_distribution()
                    np.savez_compressed(directory/"target.npz",covariance=V,target=P)
                    entry={"case_id":case_id,"modes":m,"seed":seed,"profile":profile,
                           "input_sha256":sha(directory/"target.npz"),"target_diagnostics":oracle.diagnostics}
                    inputs.append(entry)
                    local={}
                    click_counts=np.asarray([i.bit_count() for i in range(1<<m)])
                    print(json.dumps({"starting":case_id}),flush=True)
                    for method_index,spec in enumerate(config["methods"]):
                        started=perf_counter()
                        sampler,model=make_sampler(V,spec,config)
                        setup_seconds=perf_counter()-started
                        q,failure,failures,enumeration_seconds=extended_law(sampler,m)
                        tv=float(.5*(np.abs(q-P).sum()+failure))
                        np.savez_compressed(directory/(spec["id"]+".npz"),distribution=q,
                            failure_mass=np.asarray(failure))
                        row={"case_id":case_id,"modes":m,"profile":profile["name"],
                            "seed":seed,"method":spec["id"],"completion":spec["completion"],"k":spec["k"],
                            "status":"complete" if failure==0 else "complete_with_failure_mass",
                            "extended_tv":tv,"failure_mass":failure,"unresolved_prefixes":len(failures),
                            "target_mean_clicks":float(P@click_counts),
                            "success_subprobability_mean_clicks":float(q@click_counts),
                            "setup_seconds":setup_seconds,"enumeration_seconds":enumeration_seconds,
                            "diagnostics":sampler.cache_info(),"target_oracle_diagnostics":model.diagnostics,
                            "law_sha256":sha(directory/(spec["id"]+".npz")),
                            "strict_floating_point_certificate":False}
                        save(directory/(spec["id"]+".json"),{**row,"failures":failures})
                        rows.append(row);local[spec["id"]]=q
                        journal.write(json.dumps(plain(row),allow_nan=False)+"\n");journal.flush()
                        print(json.dumps({"case":case_id,"method":spec["id"],"tv":tv,
                            "failure_mass":failure,"seconds":enumeration_seconds}),flush=True)
                        if m==8 and spec["completion"]!="none":
                            ref,_=make_sampler(V,spec,config,target=P)
                            qt,ft,failt,_=extended_law(ref,m)
                            delta=float(.5*(np.abs(q-qt).sum()+abs(failure-ft)))
                            check={"case_id":case_id,"method":spec["id"],"kind":"independent_table_events_same_selector",
                                "law_tv_difference":delta,"reference_failure_mass":ft,"passed":delta<2e-8}
                            checks.append(check)
                            save(directory/(spec["id"]+"_table_check.json"),check)
                    native=run_cpp_click(V,selection="click_influence",max_clicks=4,samples=0,
                        seed=0,executable=args.native_executable,vacuum_kernel="product",
                        prefix_cache_entries=0,enumerate_distribution=True,timeout=180)
                    nq=np.asarray(native["distribution"],dtype=float)
                    baseline_failure=next(r["failure_mass"] for r in reversed(rows)
                        if r["case_id"]==case_id and r["method"]=="retained_k4")
                    delta=float(.5*(np.abs(nq-local["retained_k4"]).sum()+baseline_failure))
                    checks.append({"case_id":case_id,"method":"retained_k4","kind":"independent_native_baseline",
                        "law_tv_difference":delta,"passed":delta<2e-8})
                    save(directory/"native_k4.json",native)
    expected_cases = {f"m{m}_{profile['name']}_s{seed}" for m in config["modes"]
        for seed in config["seeds_per_mode"] for profile in config["profiles"]}
    expected_rows = {(case_id, method_id) for case_id in expected_cases for method_id in method_ids}
    actual_rows = {(row["case_id"], row["method"]) for row in rows}
    if (len(inputs) != config["planned_instances"]
            or {case["case_id"] for case in inputs} != expected_cases
            or len(rows) != config["planned_method_evaluations"]
            or len(rows) != len(expected_rows) or actual_rows != expected_rows):
        raise RuntimeError("Formal matrix has missing, duplicate, or unexpected evaluations")
    expected_checks = {(case_id, "retained_k4", "independent_native_baseline")
        for case_id in expected_cases}
    expected_checks.update((case["case_id"], spec["id"], "independent_table_events_same_selector")
        for case in inputs if case["modes"] == 8 for spec in config["methods"]
        if spec["completion"] != "none")
    actual_checks = {(check["case_id"], check["method"], check["kind"]) for check in checks}
    if len(checks) != len(expected_checks) or actual_checks != expected_checks:
        raise RuntimeError("Formal cross-check matrix is missing or duplicated")
    stable=all(sha(ROOT/p)==h for p,h in source_hashes.items()) and sha(args.native_executable)==binary_sha
    all_checks=all(c["passed"] for c in checks)
    metadata.update(status="complete" if stable and all_checks else "investigation_required",
        completed_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=perf_counter()-overall_start,sources_and_binary_stable=stable,
        instances=len(inputs),method_evaluations=len(rows),crosschecks_passed=all_checks,
        crosscheck_count=len(checks),expected_crosscheck_count=len(expected_checks),
        complete_matrix_keys_verified=True)
    save(args.output/"inputs.json",inputs);save(args.output/"metrics.json",rows)
    save(args.output/"crosschecks.json",checks);save(args.output/"metadata.json",metadata)
    print(json.dumps({k:metadata[k] for k in ("status","instances","method_evaluations","crosschecks_passed","sources_and_binary_stable","elapsed_seconds")}),flush=True)
    if not stable or not metadata["crosschecks_passed"]:
        raise SystemExit("Investigation required; do not interpret this as a validated comparison")


if __name__=="__main__":
    main()
