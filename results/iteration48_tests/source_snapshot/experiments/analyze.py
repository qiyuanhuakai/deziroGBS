"""Generate descriptive summaries and figures from committed raw results.

No p values: configurations sharing a seed also share optical structure and
are not independent draws from a claimed application population.
"""
from pathlib import Path
import csv
import json
from collections import defaultdict
from statistics import median

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)
SUITES = ("survey_v2", "adaptive_v2", "heldout_v1")
LABELS = {
    "pairwise": "Pairwise parents",
    "greedy_cmi": "Conditional-MI parents",
    "calibrated_greedy_cmi": "First-click calibrated",
    "calibrated_pairwise": "Calibrated pairwise",
    "zero_memory": "All zeros + recent clicks",
    "zero_pairwise": "All zeros + MI-selected clicks",
    "independent": "Independent modes",
    "previous": "Previous modes",
}


def load(suite):
    return list(csv.DictReader((ROOT / "results" / suite / "metrics.csv").open()))


def stats(values):
    a = np.asarray(values, dtype=float)
    return {"n": len(a), "median": float(np.median(a)),
        "q25": float(np.quantile(a, .25)), "q75": float(np.quantile(a, .75)),
        "min": float(a.min()), "max": float(a.max())}


def compare(rows, method, baseline, k=2):
    indexed = {(r["case_id"], r["strategy"], int(r["k"])): r for r in rows}
    pairs = []
    for r in rows:
        if r["strategy"] != method or int(r["k"]) != k:
            continue
        b = indexed[(r["case_id"], baseline, k)]
        pairs.append({"case_id": r["case_id"], "topology": r["topology"],
            "modes": int(r["modes"]), "squeezing": float(r["squeezing"]),
            "transmissivity": float(r["transmissivity"]), "seed": int(r["seed"]),
            "method_tv": float(r["tv"]), "baseline_tv": float(b["tv"]),
            "improvement": float(b["tv"])-float(r["tv"])})
    return {"n": len(pairs), "improved":sum(p["improvement"]>1e-10 for p in pairs),
        "worsened":sum(p["improvement"] < -1e-10 for p in pairs),
        "tied":sum(abs(p["improvement"])<=1e-10 for p in pairs),
        "median_absolute_tv_improvement": median(p["improvement"] for p in pairs),
        "worst_regressions": sorted(pairs,key=lambda p:p["improvement"])[:5]}


summary = {"suites":{}, "notes":[
    "TV/KL are dense-distribution calculations, subject to floating-point diagnostics.",
    "Quantiles describe this parameter grid, not sampling confidence intervals.",
    "k parents and k clicks plus all zeros are unequal information/cost budgets.",
    "No experimental quantum device or state-of-the-art emulator was benchmarked."]}
allcases = set()
for suite in SUITES:
    rows = load(suite)
    groups = defaultdict(list)
    for row in rows:
        allcases.add(row["case_id"])
        groups[(row["topology"],row["strategy"],int(row["k"]))].append(row)
    groupstats=[]
    for (topology,method,k), rr in sorted(groups.items()):
        groupstats.append({"topology":topology,"strategy":method,"k":k,
            "tv":stats([float(r["tv"]) for r in rr]),
            "kl":stats([float(r["kl"]) for r in rr]),
            "cold_batch_seconds":stats([float(r["fit_seconds"])+float(r["sample_seconds"]) for r in rr])})
    item={"rows":len(rows),"cases":len(set(r["case_id"] for r in rows)),"groups":groupstats}
    if suite != "survey_v2":
        item["comparisons"]={}
        for a,b in [("zero_memory","greedy_cmi"),("zero_pairwise","greedy_cmi"),
                    ("zero_pairwise","zero_memory"),("calibrated_greedy_cmi","greedy_cmi")]:
            item["comparisons"][a+"_vs_"+b]=compare(rows,a,b)
        zr=[r for r in rows if r["strategy"]=="zero_pairwise" and int(r["k"])==2]
        item["zero_pairwise_k2_bound_diagnostics"]={
            "actual_tv":stats([float(r["tv"]) for r in zr]),
            "click_tail_bound":stats([float(r["click_tail_coupling_bound"]) for r in zr]),
            "photon_tail_bound":stats([float(r["photon_tail_scalable_bound"]) for r in zr]),
            "tail_conditioned_tv_from_exact_low_sector_identity":stats([
                float(r["tv"])/float(r["click_tail_coupling_bound"]) for r in zr]),
            "max_low_click_probability_error":max(float(r["low_click_max_abs_error"]) for r in zr)}
    summary["suites"][suite]=item
summary["unique_instances"]=len(allcases)
summary["model_evaluations"]=sum(summary["suites"][s]["rows"] for s in SUITES)

heldout=load("heldout_v1")
base={r["case_id"]:r for r in heldout if r["strategy"]=="greedy_cmi" and r["k"]=="2"}
selected=[r for r in heldout if r["strategy"]=="zero_pairwise" and r["k"]=="2"]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
                     "axes.spines.right":False,"figure.facecolor":"white"})
colors={"haar":"#2563eb","local2":"#00856b","local4":"#d97706"}
names={"haar":"Haar mixing","local2":"Local, 2 layers","local4":"Local, 4 layers"}
fig,axs=plt.subplots(1,2,figsize=(12,4.7),layout="constrained")
for topology in colors:
    rr=[r for r in selected if r["topology"]==topology]
    axs[0].scatter([float(base[r["case_id"]]["tv"]) for r in rr],
        [float(r["tv"]) for r in rr],label=names[topology],c=colors[topology],s=35,alpha=.8,
        edgecolors="white",linewidths=.4)
lo,hi=1e-7,0.5
axs[0].plot([lo,hi],[lo,hi],color="#737373",ls="--",lw=1,label="Equal error")
axs[0].set(xscale="log",yscale="log",xlim=(lo,hi),ylim=(lo,hi),
    xlabel="Conditional-MI parent model: TV",ylabel="All zeros + MI-selected clicks: TV",
    title="Paired held-out cases (72 instances)")
axs[0].legend(fontsize=8,loc="lower right")
axs[0].grid(alpha=.15,which="both")
methods=["greedy_cmi","calibrated_greedy_cmi","zero_memory","zero_pairwise"]
for offset,topology in enumerate(colors):
    meds=[median(float(r["tv"]) for r in heldout if r["strategy"]==method and
                  r["k"]=="2" and r["topology"]==topology) for method in methods]
    axs[1].barh(np.arange(len(methods))+(offset-1)*.24,meds,height=.22,
        color=colors[topology],label=names[topology])
axs[1].set_yticks(np.arange(len(methods)),[LABELS[m] for m in methods],fontsize=9)
axs[1].invert_yaxis()
axs[1].set(xscale="log",xlabel="Median TV across 24 configurations per topology",
           title="Error by optical structure")
axs[1].grid(axis="x",alpha=.15)
fig.suptitle("Threshold GBS prototype | 10 modes | k = 2 | exact small-system evaluation",fontsize=12)
fig.savefig(OUT/"heldout_comparison.png",dpi=180)
fig.savefig(OUT/"heldout_comparison.svg")
plt.close(fig)

scaling_path=ROOT/"results"/"scaling_v1"/"metrics.csv"
if scaling_path.exists():
    rows=load("scaling_v1")
    summary["scaling"]={"rows":len(rows),"max_modes":max(int(r["modes"]) for r in rows),
        "k2_photon_bound_at_64":stats([float(r["photon_tail_bound"]) for r in rows
            if r["modes"]=="64" and r["strategy"]=="zero_pairwise"])}
    fig,ax=plt.subplots(figsize=(8,4.7),layout="constrained")
    for method in ["pairwise","greedy_cmi","calibrated_greedy_cmi","zero_memory","zero_pairwise"]:
        mvals=sorted(set(int(r["modes"]) for r in rows if r["strategy"]==method))
        vals=[median(float(r["total_seconds"]) for r in rows if r["strategy"]==method and int(r["modes"])==m) for m in mvals]
        ax.plot(mvals,vals,marker="o",label=LABELS[method])
    ax.set(xlabel="Optical modes",ylabel="Seconds: fit + first 128 draws (median)",yscale="log",
        title="Larger-mode cost probe; global distribution error NOT evaluated")
    ax.set_xticks([16,32,64]);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(OUT/"scaling_cost.png",dpi=180)
    fig.savefig(OUT/"scaling_cost.svg")
    plt.close(fig)

(OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
lines=["# Descriptive results","","No statistical significance or quantum speedup is claimed.","",
    "## Held-out median TV, k=2","",
    "| Method | Haar | Local 2 | Local 4 |","|---|---:|---:|---:|"]
for method in methods:
    med=[median(float(r["tv"]) for r in heldout if r["strategy"]==method and r["k"]=="2" and r["topology"]==top) for top in ["haar","local2","local4"]]
    lines.append("| "+LABELS[method]+" | "+" | ".join(f"{x:.8f}" for x in med)+" |")
lines += ["","## Paired outcomes on held-out cases","",
          "| Comparison | Lower TV | Higher TV | Tied |","|---|---:|---:|---:|"]
for name,comparison in summary["suites"]["heldout_v1"]["comparisons"].items():
    lines.append(f"| {name} | {comparison['improved']} | {comparison['worsened']} | {comparison['tied']} |")
lines += ["","The 72 held-out configurations use four reserved seeds; squeezing and loss variants share their interferometers.",
    "An equal k is not an equal information or computational budget.",""]
(OUT/"summary_tables.md").write_text("\n".join(lines))
print(json.dumps({"unique_instances":summary["unique_instances"],"model_evaluations":summary["model_evaluations"],
    "output":str(OUT)}))
