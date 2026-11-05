"""Validate optional log Q and a guarded product kernel without timing claims.

Run this from any directory; input/binary paths are resolved from this script.
All failures are retained. A product variant is not accepted by merely passing
its own normalization: its complete law must match the frozen logarithmic law.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
Q_TV_GATE=2e-8
LOG_Q_GATE=2e-10


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def invoke(executable,V,uniforms,*,selector,k,order=None,kernel=None,logging=False,enumerate_q=False,trace=False):
    m=len(V)//2
    with tempfile.TemporaryDirectory(prefix="gbs-kernel-check-") as directory:
        directory=Path(directory)
        np.savetxt(directory/"V.csv",V,delimiter=",",fmt="%.17g")
        np.savetxt(directory/"u.csv",uniforms,delimiter=",",fmt="%.17g")
        command=[str(executable),"--input",str(directory/"V.csv"),"--uniforms",str(directory/"u.csv"),
                 "--modes",str(m),"--samples",str(len(uniforms)),"--selector",selector,"--k",str(k),"--raw-samples"]
        if order is not None:command += ["--order",",".join(map(str,order))]
        if kernel is not None:command += ["--vacuum-kernel",kernel]
        if logging:command += ["--record-log-probabilities"]
        if enumerate_q:command += ["--enumerate"]
        if trace:command += ["--trace-conditionals"]
        run=subprocess.run(command,text=True,capture_output=True,timeout=240)
        value=json.loads(run.stdout)
        if run.returncode and value.get("status")!="error":raise RuntimeError("Nonzero exit without explicit error")
        value["_returncode"]=run.returncode
        value["_stderr"]=run.stderr
        return value


def payload_without_arrays(value):
    return {k:v for k,v in value.items() if k not in ["distribution","raw_samples","sample_log_probabilities","conditional_trace"]}


def assert_logq(value):
    if value["status"]!="ok":return {"passed":False,"error":"native failure","details":value}
    logs=np.asarray(value["sample_log_probabilities"],dtype=float)
    samples=np.asarray(value["raw_samples"],dtype=np.uint8)
    n,m=value["sample_count"],value["modes"]
    errors=[]
    if logs.shape!=(n,) or not np.all(np.isfinite(logs)) or np.any(logs>0):errors.append("invalid logQ array")
    result={"passed":not errors,"errors":errors}
    if value["distribution"] is not None:
        q=np.asarray(value["distribution"],dtype=float)
        masks=samples.astype(np.uint64) @ (np.uint64(1)<<np.arange(m,dtype=np.uint64))
        point=q[masks]
        if np.any(point<=0):errors.append("sampled full-Q zero")
        else:
            residual=float(np.max(np.abs(logs-np.log(point)),initial=0))
            result["maximum_absolute_logq_vs_fullq_error"]=residual
            if residual>LOG_Q_GATE:errors.append("logQ/full-Q mismatch")
        if abs(float(q.sum(dtype=np.longdouble))-1)>1e-10 or np.any(q<0) or not np.all(np.isfinite(q)):
            errors.append("full-Q invalid")
    if value["conditional_trace"] is not None:
        traced=np.zeros(n,dtype=np.longdouble)
        for step in value["conditional_trace"]:
            p=np.longdouble(step["p_one"])
            chosen=samples[step["sample_index"],step["mode"]]
            traced[step["sample_index"]] += np.log(p) if chosen else np.log1p(-p)
        residual=float(np.max(np.abs(logs-traced),initial=0))
        result["maximum_absolute_logq_vs_trace_error"]=residual
        if residual>LOG_Q_GATE:errors.append("logQ/trace mismatch")
    result["passed"]=not errors
    return result


def main(args):
    output=args.output.resolve()
    if output.exists() and any(output.iterdir()):raise FileExistsError("Use new output")
    output.mkdir(parents=True,exist_ok=True)
    executable=args.executable.resolve()
    old=(ROOT/".cache/click-native/click_sampler").resolve()
    build=json.loads((executable.parent/"build.json").read_text())
    sha=digest(executable)
    if sha!=build["executable_sha256"]:raise RuntimeError("New binary/build mismatch")
    if digest(old)!="3e63591bc1f2b133864da1482f75091a1dd816a762c0cd32203550decbc2ca46":raise RuntimeError("Wrong frozen v3 reference")
    config=ROOT/"configs/cpp_matched_benchmark.json"
    cases=json.loads(config.read_text())["cold"]["cases"]
    metadata={"started_utc":datetime.now(timezone.utc).isoformat(),"status":"running",
              "executable_sha256":sha,"build":build,"legacy_executable_sha256":digest(old),
              "config_sha256":digest(config),"runner_sha256":digest(Path(__file__)),
              "gates":{"full_q_tv":Q_TV_GATE,"absolute_log_q":LOG_Q_GATE},
              "scope":"Numerical implementation equivalence and sample logQ; no physical-target approximation guarantee or performance claim."}
    (output/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    snapshots=output/"source_snapshot";snapshots.mkdir()
    for p in [Path(__file__),ROOT/"experiments/click_sampler.cpp",ROOT/"experiments/build_click_sampler.py"]:
        shutil.copy2(p,snapshots/p.name)
    rows=[];guard_rows=[];extra_rows=[]
    with (output/"records.jsonl").open("w") as journal:
        for index,case in enumerate(cases):
            input_path=ROOT/case["input"]
            if digest(input_path)!=case["input_sha256"]:raise RuntimeError("Input changed")
            with np.load(input_path) as data:V=data["covariance"]
            uniforms=np.random.default_rng(941000+index).random((32,len(V)//2))
            with np.load(ROOT/"results/cpp_cold_v2"/f"laws_{case['case_id']}.npz") as data:
                frozen={k:data[k] for k in ["cpp_pairwise","cpp_click_influence"]}
            for selector in ["pairwise","click_influence"]:
                outputs={}
                for kernel in ["log","product"]:
                    outputs[kernel]=invoke(executable,V,uniforms,selector=selector,k=4,kernel=kernel,
                                           logging=True,enumerate_q=True)
                no_log=invoke(executable,V,uniforms,selector=selector,k=4,kernel="log")
                old_samples=invoke(old,V,uniforms,selector=selector,k=4)
                row={"case_id":case["case_id"],"selector":selector,"input_sha256":case["input_sha256"],
                     "log_feature":assert_logq(outputs["log"]),"product_log_feature":assert_logq(outputs["product"]),
                     "native_results":{k:payload_without_arrays(v) for k,v in outputs.items()}}
                arrays={"covariance":V,"uniforms":uniforms}
                if outputs["log"]["status"]=="ok":
                    log=outputs["log"];qlog=np.asarray(log["distribution"])
                    row["log_tv_to_v3"]=float(.5*np.abs(qlog-frozen["cpp_"+selector]).sum())
                    row["logging_preserves_samples"]=no_log["status"]=="ok" and no_log["raw_samples"]==log["raw_samples"]
                    row["log_samples_match_v3"]=old_samples["status"]=="ok" and old_samples["raw_samples"]==log["raw_samples"]
                    arrays.update(q_log=qlog,samples_log=np.asarray(log["raw_samples"],dtype=np.uint8),
                                  sample_log_probabilities=log["sample_log_probabilities"],q_v3=frozen["cpp_"+selector])
                    row["log_baseline_passed"]=row["log_feature"]["passed"] and row["logging_preserves_samples"] and row["log_samples_match_v3"] and row["log_tv_to_v3"]<=Q_TV_GATE
                    if outputs["product"]["status"]=="ok":
                        product=outputs["product"];qproduct=np.asarray(product["distribution"])
                        row["product_tv_to_log"]=float(.5*np.abs(qproduct-qlog).sum())
                        row["product_samples_match_log"]=product["raw_samples"]==log["raw_samples"]
                        arrays.update(q_product=qproduct,samples_product=np.asarray(product["raw_samples"],dtype=np.uint8),
                                      product_sample_log_probabilities=product["sample_log_probabilities"])
                        row["product_equivalence_passed"]=row["product_log_feature"]["passed"] and row["product_samples_match_log"] and row["product_tv_to_log"]<=Q_TV_GATE
                row.setdefault("log_baseline_passed",False);row.setdefault("product_equivalence_passed",False)
                artifact=output/f"{case['case_id']}_{selector}.npz"
                np.savez_compressed(artifact,**arrays);row["artifact_sha256"]=digest(artifact)
                rows.append(row);journal.write(json.dumps(row)+"\n");journal.flush()
            print(f"fixed small cases {index+1}/{len(cases)}",flush=True)
        # Reuse existing strict-support/weak-event inputs exactly, not newly
        # selected examples that merely happen to pass a changed kernel.
        old_validation=json.loads((ROOT/"results/cpp_validation/native_stable_v3.json").read_text())
        for index,case in enumerate(old_validation["cases"]):
            if not case["stress_case"]:continue
            artifact=ROOT/"results/cpp_validation"/case["artifact_relative_path"]
            if digest(artifact)!=case["artifact_sha256"]:raise RuntimeError("Stress artifact changed")
            with np.load(artifact) as data:
                V=data["covariance"];uniforms=data["uniforms"]
            for kernel in ["log","product"]:
                out=invoke(executable,V,uniforms,selector=case["selection"],k=case["k"],order=case["order"],
                           kernel=kernel,logging=True,enumerate_q=case["enumerated_q"])
                row={"fixture":case["fixture"],"selector":case["selection"],"k":case["k"],"kernel":kernel,
                     "source_artifact":str(artifact.relative_to(ROOT)),"source_artifact_sha256":digest(artifact),
                     "numerical_guard_observed":out["status"]=="error" and out.get("error_type")=="numerical_probability",
                     "native_result":payload_without_arrays(out)}
                guard_rows.append(row);journal.write(json.dumps(row)+"\n");journal.flush()
        # Deterministic zero branches and high-magnitude product fallback.
        for name,V,k in [("vacuum65",np.eye(130),4),("thermal_product_overflow",np.eye(4)*1e200,2)]:
            m=len(V)//2;uniforms=np.random.default_rng(942000+m).random((8,m))
            uniforms[0]=0;uniforms[1]=np.nextafter(1.,0.)
            for kernel in ["log","product"]:
                out=invoke(executable,V,uniforms,selector="click_influence",k=k,order=list(range(m-1,-1,-1)),
                           kernel=kernel,logging=True,enumerate_q=m<=14,trace=True)
                row={"fixture":name,"kernel":kernel,"log_feature":assert_logq(out),
                     "native_result":payload_without_arrays(out)}
                if out["status"]=="ok":
                    row["logq_exact_zero"]=all(x==0 for x in out["sample_log_probabilities"])
                    row["deterministic_samples_correct"]=all(all(bit==(0 if name=="vacuum65" else 1) for bit in sample) for sample in out["raw_samples"])
                    row["fallback_exercised"]=kernel!="product" or name!="thermal_product_overflow" or out["sampling_diagnostics"]["vacuum_product_fallbacks"]>0
                    row["passed"]=row["log_feature"]["passed"] and row["logq_exact_zero"] and row["deterministic_samples_correct"] and row["fallback_exercised"]
                    np.savez_compressed(output/f"{name}_{kernel}.npz",covariance=V,uniforms=uniforms,
                                        samples=np.asarray(out["raw_samples"],dtype=np.uint8),logq=out["sample_log_probabilities"])
                else:row["passed"]=False
                extra_rows.append(row);journal.write(json.dumps(row)+"\n");journal.flush()
    if digest(executable)!=sha:raise RuntimeError("Binary changed during validation")
    log_pass=all(r["log_baseline_passed"] for r in rows)
    guard_log=all(r["numerical_guard_observed"] for r in guard_rows if r["kernel"]=="log")
    guard_product=all(r["numerical_guard_observed"] for r in guard_rows if r["kernel"]=="product")
    metadata.update(status="complete",completed_utc=datetime.now(timezone.utc).isoformat(),
                    passed=log_pass and guard_log and all(r["passed"] for r in extra_rows if r["kernel"]=="log"),
                    log_case_count=len(rows),log_passed_cases=sum(r["log_baseline_passed"] for r in rows),
                    product_passed_cases=sum(r["product_equivalence_passed"] for r in rows),
                    product_equivalence_passed=all(r["product_equivalence_passed"] for r in rows) and guard_product and all(r["passed"] for r in extra_rows if r["kernel"]=="product"),
                    log_guard_cases=sum(r["numerical_guard_observed"] for r in guard_rows if r["kernel"]=="log"),
                    product_guard_cases=sum(r["numerical_guard_observed"] for r in guard_rows if r["kernel"]=="product"),
                    maximum_log_tv_to_v3=max(r.get("log_tv_to_v3",float("inf")) for r in rows),
                    maximum_product_tv_to_log=max(r.get("product_tv_to_log",float("inf")) for r in rows),
                    maximum_logq_residual=max(r["log_feature"].get("maximum_absolute_logq_vs_fullq_error",float("inf")) for r in rows),
                    cases=rows,stress_cases=guard_rows,extra_cases=extra_rows)
    (output/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps({k:metadata[k] for k in ["passed","executable_sha256","log_case_count","log_passed_cases","product_passed_cases","product_equivalence_passed","maximum_log_tv_to_v3","maximum_product_tv_to_log","maximum_logq_residual","log_guard_cases","product_guard_cases"]},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--executable",type=Path,default=ROOT/".cache/click-iteration4-native/click_sampler")
    p.add_argument("--output",type=Path,default=ROOT/"results/iteration4_kernel_validation/first_candidate")
    main(p.parse_args())
