"""Validate the new ISA-optimized official baseline independently of timing runs."""
from pathlib import Path
from datetime import datetime,timezone
import json
import hashlib
import shutil

import numpy as np

from gbslab.frugal import frugal_sample
from gbslab.frugal_scaling import frugal_native_sample
from gbslab.gaussian import covariance_from_interferometer,haar_unitary
from gbslab.exact import ThresholdModel

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/cpp_frugal_validation"
OUT.mkdir(parents=True,exist_ok=True)
if (OUT/"validation.json").exists():raise FileExistsError("Validation archive already exists")
build=ROOT/".cache/frugal-native"
b=json.loads((build/"build.json").read_text())
sha=hashlib.sha256((build/"frugal_sampling_adapter").read_bytes()).hexdigest()
assert b["executable_sha256"]==sha
shutil.copy2(build/"build.json",OUT/"build.json")
for p in [Path(__file__),ROOT/"experiments/frugal_sampling_adapter.cpp",ROOT/"experiments/build_frugal_sampling.py",ROOT/"src/gbslab/frugal_scaling.py"]:
    shutil.copy2(p,OUT/p.name)
fixtures=[]
for m in [8,16,32,64]:fixtures.append((f"vacuum_m{m}",np.eye(2*m),2,128,m<=14))
nbar=np.linspace(.1,1.3,8)
fixtures.append(("thermal_nonuniform8",np.diag(np.tile(1+2*nbar,2)),5,50000,True))
V=covariance_from_interferometer(haar_unitary(8,np.random.default_rng(20032)),.8,.7)
fixtures.append(("haar8_K4",V,4,50000,True));fixtures.append(("haar8_K5",V,5,50000,True))
V=covariance_from_interferometer(haar_unitary(8,np.random.default_rng(4001)),1.0,1.0)
fixtures.append(("clipping8",V,5,50000,True))
V=covariance_from_interferometer(haar_unitary(10,np.random.default_rng(20034)),np.linspace(.35,1.1,10),np.linspace(.2,.9,10))
fixtures.append(("heterogeneous10",V,5,50000,True))
V=covariance_from_interferometer(haar_unitary(12,np.random.default_rng(6101)),.8,.8)
fixtures.append(("bright12",V,5,50000,True))
rows=[]
for index,(name,V,K,n,enumerate_q) in enumerate(fixtures):
    result=frugal_native_sample(V,K,n,820000+index,enumerate_distribution=enumerate_q)
    row={"fixture":name,"modes":len(V)//2,"K":K,"samples":n,"seed":820000+index,
         "simd_lanes":result["simd_lanes"],"trace":result["trace"],"peak_rss_bytes_linux":result["peak_rss_bytes_linux"]}
    assert sum(result["per_mode_click_counts"])==sum(i*v for i,v in enumerate(result["click_count_histogram"]))
    arrays={"covariance":V}
    if name.startswith("vacuum"):assert result["click_count_histogram"][0]==n
    if enumerate_q:
        q=np.array(result["distribution"]);counts=np.array(result["histogram_counts"])
        old=frugal_sample(V,K,0,1,enumerate_distribution=True)
        oldq=np.asarray(old["distribution"])
        delta=1e-6
        empirical=.5*np.abs(counts/n-q).sum()
        limit=.5*np.sqrt(q*(1-q)/n).sum()+np.sqrt(np.log(1/delta)/(2*n))
        row.update(tv_native_to_previous_trace=float(.5*np.abs(q-oldq).sum()),
                   maximum_trace_difference=float(np.max(np.abs(q-oldq))),
                   empirical_tv_to_native_trace=float(empirical),histogram_upper=float(min(1,limit)),
                   histogram_failure_probability=delta,off_support_samples=int(counts[q==0].sum()))
        assert row["tv_native_to_previous_trace"]<1e-10
        assert empirical<=limit and row["off_support_samples"]==0
        marginal=np.bincount(np.arange(q.size)&3,weights=q,minlength=4)
        prefix_error=float(np.max(np.abs(marginal-ThresholdModel(V).marginal((0,1)))))
        row["first_two_modes_error"]=prefix_error;assert prefix_error<1e-10
        if name=="clipping8":assert result["trace"]["saturated_threshold_path_steps"]>0
        arrays.update(q_native=q,q_previous=oldq,histogram_counts=counts)
    np.savez_compressed(OUT/f"{name}.npz",**arrays)
    rows.append(row);print(name,"passed",flush=True)
record={"passed":True,"created_utc":datetime.now(timezone.utc).isoformat(),"executable_sha256":sha,
        "build_cache_key":b["cache_key"],"flags":b["flags"],"cases":rows,
        "notes":["New native binary validated; previous SSE validation not simply reused.",
                 "Large-mode checks are vacuum and output-shape checks, not correlated global quality claims.",
                 "Sampling remains the unmodified official API; small traced laws use continuous uniform thresholds."]}
(OUT/"validation.json").write_text(json.dumps(record,indent=2)+"\n")
(build/"validation.json").write_text(json.dumps(record,indent=2)+"\n")
print(json.dumps({"cases":len(rows),"passed":True,"binary":sha}))
