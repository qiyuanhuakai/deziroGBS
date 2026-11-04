"""Replay existing strict-pair samples without the separate full-law sweep."""
from pathlib import Path
import importlib.util
import json
import shutil

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location("kernel_checks",HERE/"check_native_kernels.py")
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
OUT=HERE/"support_replay"
if OUT.exists() and any(OUT.iterdir()):raise FileExistsError("Use fresh support output")
OUT.mkdir(parents=True,exist_ok=True)
binary=ROOT/".cache/click-iteration4-native/click_sampler"
old=ROOT/".cache/click-native/click_sampler"
sha=helpers.digest(binary);oldsha=helpers.digest(old)
source=ROOT/"results/cpp_validation/native_stable_v3.json"
validation=json.loads(source.read_text())
rows=[]
for index,case in enumerate(validation["cases"]):
    if case["fixture"]!="strict_tmsv_4pairs":continue
    path=ROOT/"results/cpp_validation"/case["artifact_relative_path"]
    if helpers.digest(path)!=case["artifact_sha256"]:raise RuntimeError("Reference input changed")
    with np.load(path) as data:V=data["covariance"];uniforms=data["uniforms"]
    legacy=helpers.invoke(old,V,uniforms,selector=case["selection"],k=case["k"],order=case["order"])
    if legacy["status"]!="ok":raise RuntimeError("Frozen successful support replay changed")
    for kernel in ["log","product"]:
        result=helpers.invoke(binary,V,uniforms,selector=case["selection"],k=case["k"],order=case["order"],kernel=kernel,logging=True)
        row={"selector":case["selection"],"k":case["k"],"kernel":kernel,
             "source_artifact":str(path.relative_to(ROOT)),"source_artifact_sha256":helpers.digest(path),
             "native_result":helpers.payload_without_arrays(result)}
        if result["status"]=="ok":
            samples=np.asarray(result["raw_samples"],dtype=np.uint8)
            logq=np.asarray(result["sample_log_probabilities"])
            row.update(samples_match_frozen=result["raw_samples"]==legacy["raw_samples"],
                       forbidden_patterns=int(np.any(samples[:,:4]!=samples[:,4:],axis=1).sum()),
                       maximum_logq_error=float(np.max(np.abs(logq+4*np.log(2)))))
            row["passed"]=row["samples_match_frozen"] and row["forbidden_patterns"]==0 and row["maximum_logq_error"]<2e-10
            artifact=OUT/f"case_{index}_{kernel}.npz"
            np.savez_compressed(artifact,covariance=V,uniforms=uniforms,samples=samples,logq=logq,
                                frozen_samples=np.asarray(legacy["raw_samples"],dtype=np.uint8))
            row["artifact_sha256"]=helpers.digest(artifact)
        else:row["passed"]=False
        rows.append(row)
if helpers.digest(binary)!=sha or helpers.digest(old)!=oldsha:raise RuntimeError("Binary changed during replay")
summary={"passed":all(r["passed"] for r in rows),"executable_sha256":sha,"frozen_executable_sha256":oldsha,
         "source_validation_sha256":helpers.digest(source),"script_sha256":helpers.digest(Path(__file__)),
         "cases":rows,"qualification":"Finite injected-uniform support checks on an analytically easy family; not a universal support or GBS-quality certificate."}
for p in [Path(__file__),HERE/"check_native_kernels.py"]:shutil.copy2(p,OUT/p.name)
(OUT/"results.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps({"passed":summary["passed"],"batches":len(rows),"maximum_logq_error":max(r.get("maximum_logq_error",float("inf")) for r in rows)},indent=2))
