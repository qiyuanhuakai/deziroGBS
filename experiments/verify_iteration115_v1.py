"""Recompute release metrics from the saved laws and test both native CPU readouts."""
import argparse
import ctypes as ct
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from iteration115_visible_v1 import DATA, ROOT, P, I, U, J, cov, optical, prepare
from iteration115_displacement_v1 import core
from verify_iteration113_v1 import reference


def read(area, name="summary"):
    return json.loads((DATA / f"iteration115_{area}_v1" / f"{name}.json").read_bytes())


def check_native():
    runtime = os.add_dll_directory(str(Path(shutil.which("g++")).parent)) if os.name == "nt" else None
    suffix = ".dll" if runtime else ".so"
    visible = ct.CDLL(str(DATA / "iteration115_visible_v1" / ("visible" + suffix))).visible115
    visible.argtypes = [I]*5 + [U, J, I] + [P]*10
    t, s = optical(3, 2)
    args, law = prepare(t, s, 6)
    covariance = cov(t, s)
    shots = 98304
    rows = np.empty((shots, 3), np.uint8)
    counter = np.zeros(1, np.uint64)
    checked = {}
    for n in (3, 4, 6):
        args[0] = (np.arange(7) >= n).astype(float)
        states = [(a, b, n-a-b) for a in range(n+1) for b in range(n-a+1)]
        truth = np.array([reference(covariance, v) for v in states])
        assert abs(truth.sum()-law["p"][n]) < 2e-12
        rc = visible(shots, 3, len(args[5]), 6, 8, 115810000+n, 0, 0,
                     *[a.ctypes.data for a in args], rows.ctypes.data, counter.ctypes.data)
        assert rc == 0 and np.all(rows.sum(1) == n)
        empirical = np.array([np.all(rows == v, axis=1).mean() for v in states])
        distance = abs(empirical - truth / truth.sum()).sum()/2
        assert distance < .02
        checked[f"mixed_fixed_{n}_empirical_tv"] = float(distance)

    displaced = ct.CDLL(str(DATA / "iteration115_displacement_v1" / ("displaced" + suffix))).displaced115
    displaced.argtypes = [I]*5 + [U, J, I] + [P]*9
    args, _, _ = core(t,s)
    pair = args[1] @ args[0].T
    alpha = np.array([.6+.2j, -.3+.5j, .7-.2j])
    fields = np.tile(alpha, (shots,1))
    cut = 18
    n = np.arange(cut)
    f0 = np.exp(-abs(alpha[:,None])**2/2)*alpha[:,None]**n / np.sqrt([float(math.factorial(i)) for i in n])
    def raise_one(f):
        return np.pad(f[:,:-1], ((0,0),(1,0))) * np.sqrt(n)
    f1 = raise_one(f0) - alpha[:,None].conj()*f0
    f2 = (raise_one(f1) - alpha[:,None].conj()*f1)/np.sqrt(2)
    def outer(v):
        return np.einsum("i,j,k->ijk", *v)
    wave = outer(f0)
    for i in range(3):
        terms = list(f0)
        terms[i] = f2[i]
        wave += pair[i,i]/np.sqrt(2)*outer(terms)
        for j in range(i):
            terms = list(f0)
            terms[i], terms[j] = f1[i], f1[j]
            wave += pair[i,j]*outer(terms)
    target = abs(wave)**2/(1+args[-1][0])
    assert abs(target.sum()-1) < 2e-12
    bits = (np.indices(target.shape)>0).transpose(1,2,3,0) @ (1<<np.arange(3))
    for clicks in (0,1):
        error = np.zeros(1)
        rc = displaced(shots,3,len(s),clicks,8,115820000+clicks,0,0,
                       *[a.ctypes.data for a in args],fields.ctypes.data,rows.ctypes.data,error.ctypes.data)
        assert rc == 0 and error[0] < 1e-10 and rows.max() < cut
        if clicks:
            empirical = np.bincount(rows @ (1<<np.arange(3)),minlength=8)/shots
            truth = np.bincount(bits.ravel(),weights=target.ravel(),minlength=8)
        else:
            code = np.ravel_multi_index(rows.T, target.shape)
            empirical = np.bincount(code,minlength=target.size)/shots
            truth = target.ravel()
        distance = abs(empirical-truth).sum()/2
        assert distance < .02
        checked[("displaced_click" if clicks else "displaced_pnr")+"_empirical_tv"] = float(distance)
    return checked


def release(out):
    out.mkdir(exist_ok=True, parents=True)
    core_checks = check_native()
    data = np.load(DATA/"iteration115_tilt_v1/model.npz")
    fits = read("robustness")
    result = read("tilt")
    target, model = data["truth"][:441], data["total"]
    upper = (abs(target-model).sum()+max(0,1-target.sum())+max(0,1-model.sum()))/2
    assert abs(upper-result["total_tv_upper"]) < 1e-14
    weights = data["weights"]
    assert min(weights)>0 and abs(sum(weights)-1)<1e-13
    robust = np.load(DATA/"iteration115_robustness_v1/totals.npz")
    for row in fits:
        truth, fitted = [robust[row["case"]+"_"+k] for k in ("truth","model")]
        tv = (abs(truth-fitted).sum()+max(0,1-truth.sum())+max(0,1-fitted.sum()))/2
        assert abs(tv-row["heldout_fit_full_tv_upper"]) < 2e-13
    pairs = np.load(DATA/"iteration115_pair_banks_v1/counts.npz")
    pstats = read("pair_banks")
    for i in (0,1):
        h = pairs[f"hardware{10+i}"]
        counts = h.sum((1,2),keepdims=True)
        assert counts.min()==counts.max()
        assert np.isclose((abs(pairs["model"]-h/counts).sum((1,2))/2).mean(),pstats["model_vs_hardware"][i]["avg"])
    measures = []
    for i, row in enumerate(result["measurements"]):
        assert data[f"count{i}"].sum()==row["shots"]==131072
        measures.append(dict(pnr=row["pnr"],shots=row["shots"],batch_16384_median_us=float(np.median(row["batch_times"]["16384"])),
                             click_total_empirical_tv_hardware=row["total_click_tv_hardware"],fp32_fp64_differing_rows=row["fp32_disagreements"],
                             precision_check_rows=row["fp_compared"]))
    report = dict(model="Jiuzhang 3.0 optical transfer, 1152 outputs, original high power, two spectra",
                  tv_units="probabilities (multiply by 100 for percent); total photon TV is a model marginal",
                  base_total_photon_tv_upper=result["base_total_tv_upper"],tilted_total_photon_tv_upper=float(upper),
                  improvement=result["base_total_tv_upper"]/upper,setup_seconds=result["setup_s"],measurements=measures,
                  mean_eight_bin_word_tv=result["bank_word_tv_mean"],pair_reference_checks=pstats["independent_65536_word_checks"],
                  mean_pair_tv_hardware=[x["avg"] for x in pstats["model_vs_hardware"]],
                  mean_hardware_pair_tv=pstats["capture_vs_capture"]["avg"],
                  full_record_numeric_upper=read("fidelity")["mixture_trace_upper"],native_checks=core_checks,
                  held_out=[{k:v for k,v in row.items() if k not in ("weights","log_tilts")} for row in fits])
    panel = read("panel_pilot")
    report["matched_16_channels"] = dict(full_click_word_model_tv=panel["full_model_tv"],
        optical_cache_us=panel["optical_cached"]["us"], enumeration_s=panel["exact_enumeration_s"],
        benchmark=[{k:v for k,v in row.items() if k!="times"} for row in panel["rows"]])
    (out/"validation.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
    fig, (ax,bx) = plt.subplots(1,2,figsize=(10,4.4),layout="constrained")
    ax.plot(target*100,color="#176083",label="Optical reference")
    ax.plot(data["base"]*100,ls=":",color="#c76928",label="Base model")
    ax.plot(model*100,ls="--",color="#318256",label="Seven tilts")
    ax.set(xlim=(30,215),xlabel="Total detected photons",ylabel="Probability (%)",title="Original power · 1,152 outputs")
    ax.legend(frameon=False)
    x = np.arange(len(fits))
    for key,label,marker in [("base_total_tv_upper","Base","o"),("fixed_grid_full_tv_upper","Fixed grid","s"),("heldout_fit_full_tv_upper","Adaptive grid","^")]:
        bx.scatter(x,[100*r[key] for r in fits],label=label,marker=marker)
    bx.set(yscale="log",xticks=x,xticklabels=["0.35×","0.65×","1×","1.2×","576","288"],ylabel="Total photon TV upper bound (%)",
           xlabel="Source amplitude scale | retained outputs",title="Fit on even counts · evaluate all counts")
    bx.grid(axis="y",alpha=.2)
    bx.legend(frameon=False)
    fig.savefig(out/"comparison.png",dpi=170)
    plt.close(fig)
    directories = [DATA / ("iteration115_" + stage + "_v1") for stage in
                   ("visible", "gaps", "displacement", "banks", "tilt", "pair_banks", "fidelity", "robustness", "panel_pilot")]
    files = sorted(p for folder in directories + [out.resolve()] for p in folder.iterdir()
                   if p.is_file() and p.name != "manifest.json")
    manifest = {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf8")
    print(report["improvement"],core_checks,len(files),flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    release(parser.parse_args().output)
