"""Matched native C++ probes: quality, cold latency, throughput, and larger cost.

Inputs and workloads are fixed in cpp_matched_benchmark.json. Validation and
sampling binaries must match, and neither validation nor full Q enumeration
warms a sampler before its timed draws. Larger cases are explicitly cost-only.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
from time import perf_counter

import numpy as np

from gbslab.cpp_click import run_cpp_click,CppClickError
from gbslab.frugal_scaling import frugal_native_sample
from gbslab.surrogates import distribution_metrics
from gbslab.experiment import json_safe

ROOT=Path(__file__).resolve().parents[1]


def digest_file(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def bind_binary(executable,validation_path):
    build=json.loads((executable.parent/"build.json").read_text())
    validation=json.loads(validation_path.read_text())
    sha=digest_file(executable)
    if sha!=build["executable_sha256"] or sha!=validation["executable_sha256"] or not validation["passed"]:
        raise RuntimeError("Binary/build/validation mismatch")
    command=build.get("command",[])
    for flag in ["-O3","-march=native","-ffp-contract=off"]:
        if flag not in command:raise RuntimeError("Required compiler flag missing: "+flag)
    if any(flag in command for flag in ["-ffast-math","-Ofast"]):raise RuntimeError("Fast math disallowed")
    return {"executable":str(executable),"sha256":sha,"build":build,"validation_path":str(validation_path),
            "validation_sha256":digest_file(validation_path)}


def run(args):
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    if (output/"metadata.json").exists() or (output/"records.jsonl").exists():raise FileExistsError("Use fresh output")
    config=json.loads(args.config.read_text())
    cpp=bind_binary(args.cpp_executable,args.cpp_validation)
    frugal=bind_binary(args.frugal_executable,args.frugal_validation)
    if cpp["build"]["compiler_sha256"]!=frugal["build"]["compiler_sha256"]:
        raise RuntimeError("Candidate and baseline use different compiler binaries")
    for name,item in [("cpp",cpp),("frugal",frugal)]:
        (output/f"{name}_build.json").write_text(json.dumps(item["build"],indent=2)+"\n")
        shutil.copy2(item["validation_path"],output/f"{name}_validation.json")
    source_dir=output/"source_snapshot";source_dir.mkdir()
    for p in [Path(__file__),ROOT/"experiments/click_sampler.cpp",ROOT/"experiments/build_click_sampler.py",ROOT/"experiments/frugal_sampling_adapter.cpp",ROOT/"experiments/build_frugal_sampling.py",ROOT/"src/gbslab/cpp_click.py",ROOT/"src/gbslab/frugal_scaling.py"]:
        shutil.copy2(p,source_dir/p.name)
    (output/"config.json").write_text(json.dumps(config,indent=2)+"\n")
    reference_sources={}
    phase=config[args.phase]
    cases=[]
    if args.phase in ("cold","steady"):
        for item in phase["cases"]:
            p=ROOT/item["input"]
            if digest_file(p)!=item["input_sha256"]:raise RuntimeError("Input archive hash mismatch")
            cases.append({**item,"path":p,"samples":phase["samples"],"repeats":phase["repeats"]})
    else:
        for m in phase["modes"]:
            for seed in phase["seeds"]:
                p=ROOT/phase["input_directory"]/f"m{m}_seed{seed}_input.npz"
                cases.append({"case_id":f"m{m}_seed{seed}","path":p,"input":str(p.relative_to(ROOT)),
                    "input_sha256":digest_file(p),"suite":"cpp_large_cost","spec":{"modes":m,"seed":seed,"topology":"haar","squeezing":.6,"transmissivity":.35},
                    "samples":phase["samples"],"repeats":phase["repeats_at_64"] if m==64 else phase["repeats_up_to_32"]})
    if args.case_limit:cases=cases[:args.case_limit]
    if args.samples is not None:
        for case in cases:case["samples"]=args.samples
    if args.repeats is not None:
        for case in cases:case["repeats"]=args.repeats
    if args.phase=="cold":
        for case in cases:
            corrected=args.pairwise_reference/f"{case['case_id']}.npz"
            historical=ROOT/"results"/(case["suite"]+"_k4_extension")/f"surrogates_{case['case_id']}.npz"
            if not corrected.exists():
                raise FileNotFoundError("Corrected pairwise reference required: "+str(corrected))
            reference_sources[case["case_id"]]={
                "cpp_pairwise":{"path":str(corrected),"sha256":digest_file(corrected),"key":"distribution",
                                "policy":"Stable Gaussian binary MI; exact structural-zero handling; physical-ID tie break"},
                "cpp_click_influence":{"path":str(historical),"sha256":digest_file(historical),"key":"click_influence_k4",
                                       "policy":"Unchanged historical click-influence rule"}}
    metadata={"status":"running","phase":args.phase,"started_utc":datetime.now(timezone.utc).isoformat(),
              "cpp_binary":cpp,"frugal_binary":frugal,"planned_cases":len(cases),"methods":config["methods"],
              "pilot_overrides":{"case_limit":args.case_limit,"samples":args.samples,"repeats":args.repeats},
              "timing_scope":"Single-thread native C++ setup and draws; wrapper validation/I/O separately measured; full Q only after timed sampling",
              "wrapper_timing_scope":"Individual wrappers have different internal timer/validation scopes. Harness_call_seconds times each complete wrapper invocation with the same outer clock. Cold repeat0 includes post-sampling full-Q enumeration; exclude it from sampling-only harness latency. Validation instrumentation still differs, so native core times are the primary comparison.",
              "large_quality_scope":"No full target or matched-error claim above14 modes",
              "reference_sources":reference_sources,
              "host":{"platform":platform.platform(),"machine":platform.machine(),
                      "cpu_affinity":sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None,
                      "thread_environment":{key:os.environ.get(key) for key in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]}}}
    (output/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    rows=[];failures=[];qualities={};started=perf_counter()
    with (output/"records.jsonl").open("w") as journal:
        for index,case in enumerate(cases):
            with np.load(case["path"]) as d:
                V=d["covariance"];p=d["target"] if "target" in d else None
            q_arrays={}
            for repeat in range(case["repeats"]):
                methods=config["methods"]
                offset=(index+repeat)%len(methods)
                ordered=methods[offset:]+methods[:offset]
                for method in ordered:
                    name=method["method"]
                    row={"phase":args.phase,"case_id":case["case_id"],"suite":case["suite"],**case["spec"],
                         "method":name,"k":method.get("k"),"K":method.get("K"),"samples":case["samples"],"repeat":repeat,
                         "seed_sampling":910000+index*101+repeat,"input_sha256":case["input_sha256"]}
                    enumerate_q=args.phase=="cold" and repeat==0
                    row["wrapper_includes_enumeration"]=enumerate_q
                    try:
                        harness_started=perf_counter()
                        if name=="frugal_native":
                            out=frugal_native_sample(V,method["K"],case["samples"],row["seed_sampling"],
                                 enumerate_distribution=enumerate_q,executable=args.frugal_executable,timeout=args.timeout)
                            t=out["timings_seconds"];setup=t["layout"]+t["target"]
                        else:
                            out=run_cpp_click(V,selection="pairwise" if name=="cpp_pairwise" else "click_influence",
                                 max_clicks=method["k"],samples=case["samples"],seed=row["seed_sampling"],
                                 enumerate_distribution=enumerate_q,executable=args.cpp_executable,
                                 raw_samples=False,profile=False,timeout=args.timeout)
                            t=out["timings_seconds"];setup=t["setup"]
                        harness_elapsed=perf_counter()-harness_started
                        row.update(setup_seconds=setup,sampling_seconds=t["sampling"],cold_seconds=setup+t["sampling"],
                                   draws_per_second=case["samples"]/t["sampling"],wrapper_wall_seconds=out["wrapper_wall_seconds"],
                                   harness_call_seconds=harness_elapsed,
                                   enumeration_seconds=t.get("enumeration",0),status="ok")
                        if enumerate_q:
                            q=np.asarray(out.pop("distribution"));q_arrays[name]=q
                            metrics=distribution_metrics(p,q)
                            if name!="frugal_native":
                                reference=reference_sources[case["case_id"]][name]
                                with np.load(reference["path"]) as old:
                                    oldq=old[reference["key"]]
                                metrics["tv_to_python_same_rule"]=float(.5*np.abs(q-oldq).sum())
                                row.update(metrics)
                                if metrics["tv_to_python_same_rule"]>2e-8:
                                    raise ArithmeticError("C++ full law differs materially from declared Python same-rule reference")
                            qualities[(case["case_id"],name)]=metrics
                        row.update(qualities.get((case["case_id"],name),{}))
                        out.pop("build_metadata",None)
                        record={"status":"ok","metrics":row,"native_result":out}
                        rows.append(row)
                    except (CppClickError,RuntimeError,ArithmeticError,ValueError,OSError,subprocess.TimeoutExpired) as exc:
                        record={"status":"failed","request":row,"error_type":type(exc).__name__,"error":str(exc),
                                "payload":getattr(exc,"payload",None)}
                        failures.append(record)
                    journal.write(json.dumps(json_safe(record))+"\n");journal.flush()
            if q_arrays:np.savez_compressed(output/f"laws_{case['case_id']}.npz",**q_arrays)
            print(f"{args.phase} {index+1}/{len(cases)} {case['case_id']} rows={len(rows)} failures={len(failures)}",flush=True)
    if digest_file(args.cpp_executable)!=cpp["sha256"] or digest_file(args.frugal_executable)!=frugal["sha256"]:
        raise RuntimeError("Binary changed during benchmark")
    columns=list(dict.fromkeys(key for row in rows for key in row))
    with (output/"metrics.csv").open("w",newline="") as handle:
        w=csv.DictWriter(handle,fieldnames=columns);w.writeheader();w.writerows(rows)
    (output/"failures.json").write_text(json.dumps(failures,indent=2)+"\n")
    metadata.update(status="complete" if not failures else "complete_with_failures",completed_utc=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=perf_counter()-started,rows=len(rows),failed_rows=len(failures))
    (output/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps({k:metadata[k] for k in ["status","phase","rows","failed_rows","elapsed_seconds"]}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=["cold","steady","large"],required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/cpp_matched_benchmark.json")
    parser.add_argument("--cpp-executable",type=Path,default=ROOT/".cache/click-native/click_sampler")
    parser.add_argument("--cpp-validation",type=Path,required=True)
    parser.add_argument("--frugal-executable",type=Path,default=ROOT/".cache/frugal-native/frugal_sampling_adapter")
    parser.add_argument("--frugal-validation",type=Path,default=ROOT/"results/cpp_frugal_validation/validation.json",
                        help="validation JSON bound to the chosen Frugal executable")
    parser.add_argument("--pairwise-reference",type=Path,default=ROOT/"results/cpp_pairwise_reference")
    parser.add_argument("--timeout",type=float,default=300)
    parser.add_argument("--case-limit",type=int)
    parser.add_argument("--samples",type=int)
    parser.add_argument("--repeats",type=int)
    run(parser.parse_args())
