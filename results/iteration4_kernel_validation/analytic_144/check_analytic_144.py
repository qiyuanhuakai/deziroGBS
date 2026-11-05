"""Bounded 144-mode independent-thermal check of the native log-Q channel.

The reference law is a product of analytic Bernoulli click probabilities.
This is an easy analytic implementation test, not a large-GBS advantage test.
"""
from pathlib import Path
import hashlib
import json
import importlib.util
import shutil

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location("kernel_checks",HERE/"check_native_kernels.py")
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
OUT=HERE/"analytic_144"
if OUT.exists() and any(OUT.iterdir()):raise FileExistsError("Use a fresh output directory")
OUT.mkdir(parents=True,exist_ok=True)
EXECUTABLE=ROOT/".cache/click-iteration4-native/click_sampler"
binary_sha=helpers.digest(EXECUTABLE)
m=144;n=8
nbar=np.linspace(.2,.6,m)
V=np.diag(np.tile(1+2*nbar,2))
probability=nbar/(1+nbar)
uniforms=np.random.default_rng(943144).random((n,m))
order=list(range(m-1,-1,-1))
results=[];arrays={"covariance":V,"uniforms":uniforms,"photon_mean":nbar,"click_probabilities":probability,"order":order}
for selector in ["pairwise","click_influence"]:
    for kernel in ["log","product"]:
        result=helpers.invoke(EXECUTABLE,V,uniforms,selector=selector,k=4,order=order,
                              kernel=kernel,logging=True)
        row={"selector":selector,"kernel":kernel,"native_result":helpers.payload_without_arrays(result)}
        if result["status"]=="ok":
            samples=np.asarray(result["raw_samples"],dtype=np.uint8)
            logs=np.asarray(result["sample_log_probabilities"])
            expected=(samples*np.log(probability)+(1-samples)*np.log1p(-probability)).sum(axis=1,dtype=np.longdouble)
            residual=float(np.max(np.abs(logs-expected)))
            independent_samples=np.empty_like(samples)
            for position,mode in enumerate(order):independent_samples[:,mode]=uniforms[:,position]<probability[mode]
            row.update(maximum_logq_error=residual,samples_match_analytic=bool(np.array_equal(samples,independent_samples)),
                       logq_finite=bool(np.all(np.isfinite(logs))),output_modes=samples.shape[1])
            row["passed"]=row["samples_match_analytic"] and row["logq_finite"] and residual<2e-8
            arrays[selector+"_"+kernel+"_samples"]=samples
            arrays[selector+"_"+kernel+"_logq"]=logs
            arrays[selector+"_"+kernel+"_analytic_logq"]=np.asarray(expected,dtype=float)
        else:row["passed"]=False
        results.append(row);print(selector,kernel,row["passed"],flush=True)
if helpers.digest(EXECUTABLE)!=binary_sha:raise RuntimeError("Executable changed during analytic check")
np.savez_compressed(OUT/"fixture.npz",**arrays)
for path in [Path(__file__),HERE/"check_native_kernels.py"]:shutil.copy2(path,OUT/path.name)
summary={"passed":all(r["passed"] for r in results),"executable_sha256":binary_sha,
         "fixture_sha256":helpers.digest(OUT/"fixture.npz"),"script_sha256":helpers.digest(Path(__file__)),
         "source_helper_sha256":helpers.digest(HERE/"check_native_kernels.py"),
         "modes":m,"samples_per_run":n,"cases":results,
         "qualification":"Independent thermal states are classically trivial; this checks coordinate order and logQ at144 modes, not generic GBS accuracy or a Jiuzhang simulation."}
(OUT/"results.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps({"passed":summary["passed"],"maximum_logq_error":max(r.get("maximum_logq_error",float("inf")) for r in results)},indent=2))
