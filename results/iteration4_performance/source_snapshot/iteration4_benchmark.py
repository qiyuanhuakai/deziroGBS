"""Fixed cross-mode C++ log/product implementation comparison.

No large-system target-quality claim follows from timing or same-seed counts.
Complete new small-system laws are checked after the timed sampling call.
"""
from pathlib import Path
from datetime import datetime,timezone
from time import perf_counter
import argparse,csv,hashlib,json,os,platform,shutil,subprocess
import numpy as np

from gbslab.cpp_click import run_cpp_click,CppClickError
from gbslab.experiment import json_safe
from gbslab.surrogates import distribution_metrics

ROOT=Path(__file__).resolve().parents[1]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def run(args):
    output=args.output
    if output.exists() and any(output.iterdir()):raise FileExistsError("Use a fresh output directory")
    config=json.loads(args.config.read_text());inputs=json.loads(args.inputs.read_text())
    build=json.loads((args.executable.parent/"build.json").read_text())
    validation=json.loads(args.validation.read_text());binary=digest(args.executable)
    if binary!=build["executable_sha256"] or binary!=validation["executable_sha256"] or not validation["passed"] or not validation["product_equivalence_passed"]:
        raise RuntimeError("Frozen implementation/equivalence gate mismatch")
    if inputs["config_sha256"]!=digest(args.config):raise RuntimeError("Input protocol has changed")
    if not all(f in build["flags"] for f in ["-O3","-march=native","-ffp-contract=off"]):raise RuntimeError("Unmatched build settings")
    output.mkdir(parents=True);snapshot=output/"source_snapshot";snapshot.mkdir()
    for p in [Path(__file__),ROOT/"experiments/click_sampler.cpp",ROOT/"src/gbslab/cpp_click.py",ROOT/"experiments/build_click_sampler.py"]:shutil.copy2(p,snapshot/p.name)
    for p,name in [(args.config,"config.json"),(args.inputs,"inputs_manifest.json"),(args.validation,"kernel_validation.json"),(args.executable.parent/"build.json","build.json")]:shutil.copy2(p,output/name)
    metadata={"status":"running","started_utc":datetime.now(timezone.utc).isoformat(),"executable_sha256":binary,
              "source_hashes":{p.name:digest(p) for p in snapshot.iterdir()},"config_sha256":digest(args.config),
              "kernel_validation_sha256":digest(args.validation),"planned_cases":len(inputs["cases"]),
              "host":{"platform":platform.platform(),"threads":1,"environment":{k:os.environ.get(k) for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS"]}},
              "scope":"Same mathematical approximation, same binary, different determinant evaluation. Synthetic inputs only. No large-mode quality claim.",
              "timing":"Native setup+draw and draw-only; profile/logQ disabled. Full Q follows timed draws at first repeat for small inputs. Harness wall then includes enumeration."}
    (output/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    rows=[];failures=[];law_checks=[];started=perf_counter()
    with (output/"records.jsonl").open("w") as journal:
        for index,case in enumerate(inputs["cases"]):
            path=ROOT/case["input"]
            if digest(path)!=case["sha256"]:raise RuntimeError("Input bytes changed")
            with np.load(path) as d:V=d["covariance"];target=d["target"] if "target" in d else None
            m=case["spec"]["modes"];n=config["performance"]["samples_at_most_64" if m<=64 else "samples_above_64"]
            laws={};per_case=[]
            for repeat in range(config["performance"]["repetitions"]):
                variants=config["kernel_variants"];shift=(index+repeat)%len(variants);ordered=variants[shift:]+variants[:shift]
                for kernel in ordered:
                    row={"case_id":case["case_id"],**case["spec"],"kernel":kernel,"repeat":repeat,"samples":n,
                         "seed":550000+index*101+repeat,"input_sha256":case["sha256"],"target_mean_clicks":case["physical_target_mean_clicks"]}
                    enumerate_q=target is not None and repeat==0
                    try:
                        outer=perf_counter()
                        out=run_cpp_click(V,selection=config["selector"],max_clicks=config["retained_clicks"],samples=n,seed=row["seed"],
                            executable=args.executable,vacuum_kernel=kernel,record_log_probabilities=False,raw_samples=False,
                            enumerate_distribution=enumerate_q,timeout=config["performance"]["timeout_seconds"])
                        elapsed=perf_counter()-outer;t=out["timings_seconds"]
                        row.update(status="ok",setup_seconds=t["setup"],sampling_seconds=t["sampling"],cold_seconds=t["setup"]+t["sampling"],
                                   microseconds_per_draw=t["sampling"]/n*1e6,harness_call_seconds=elapsed,harness_includes_enumeration=enumerate_q,
                                   sample_mean_clicks=float(np.dot(np.arange(m+1),out["click_count_histogram"])/n))
                        if enumerate_q:
                            laws[kernel]=np.asarray(out.pop("distribution"));row.update(distribution_metrics(target,laws[kernel]))
                        out.pop("build_metadata",None)
                        record={"status":"ok","metrics":row,"native_result":out};rows.append(row);per_case.append(row)
                    except (CppClickError,ArithmeticError,ValueError,RuntimeError,OSError,subprocess.TimeoutExpired) as exc:
                        record={"status":"failed","request":row,"error_type":type(exc).__name__,"error":str(exc),"payload":getattr(exc,"payload",None)}
                        failures.append(record)
                    journal.write(json.dumps(json_safe(record))+"\n");journal.flush()
            if laws:
                np.savez_compressed(output/f"laws_{case['case_id']}.npz",**laws)
                if set(laws)=={"log","product"}:
                    tv=float(.5*np.abs(laws["log"]-laws["product"]).sum())
                    check={"case_id":case["case_id"],"tv_product_to_log":tv,"gate":2e-8,"passed":tv<=2e-8};law_checks.append(check)
                    for row in per_case:row["kernel_full_q_tv_difference"]=tv
            print(f"{index+1}/{len(inputs['cases'])} {case['case_id']} rows={len(rows)} failures={len(failures)}",flush=True)
    if digest(args.executable)!=binary:raise RuntimeError("Binary changed during performance run")
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (output/"metrics.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    (output/"failures.json").write_text(json.dumps(failures,indent=2)+"\n")
    (output/"new_input_law_checks.json").write_text(json.dumps(law_checks,indent=2)+"\n")
    metadata.update(status="complete" if not failures and all(x["passed"] for x in law_checks) else "complete_with_failures",
                    completed_utc=datetime.now(timezone.utc).isoformat(),rows=len(rows),failed_rows=len(failures),
                    full_q_checks=len(law_checks),all_new_full_q_checks_passed=all(x["passed"] for x in law_checks),elapsed_seconds=perf_counter()-started)
    (output/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps({k:metadata[k] for k in ["status","rows","failed_rows","full_q_checks","elapsed_seconds"]}))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=ROOT/"configs/iteration4_cross_mode.json")
    p.add_argument("--inputs",type=Path,default=ROOT/"results/iteration4_inputs/manifest.json")
    p.add_argument("--executable",type=Path,default=ROOT/".cache/click-iteration4-native/click_sampler")
    p.add_argument("--validation",type=Path,default=ROOT/"results/iteration4_kernel_validation/first_candidate/metadata.json")
    p.add_argument("--output",type=Path,default=ROOT/"results/iteration4_performance")
    run(p.parse_args())
