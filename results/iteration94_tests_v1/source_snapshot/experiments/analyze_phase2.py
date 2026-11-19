"""Descriptive second-round tables and scientific figures from archived data.

Parameter variants share interferometers. No significance test or population
confidence interval is inferred from paired win counts or grid medians.
"""
from pathlib import Path
from collections import defaultdict
import csv
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/phase2_analysis"
OUT.mkdir(parents=True, exist_ok=True)
TOPOLOGIES = ["haar", "local2", "local4"]
COLORS = {"haar":"#2563eb", "local2":"#00856b", "local4":"#d97706"}
NAMES = {"haar":"Haar mixing", "local2":"Local, 2 layers", "local4":"Local, 4 layers"}


def load(name):
    return list(csv.DictReader((ROOT / "results" / name / "metrics.csv").open()))


def candidate_rows(suite):
    rows = load(suite)
    for suffix in ["_k3_control", "_k4_extension"]:
        if (ROOT / "results" / (suite+suffix) / "metrics.csv").exists():
            rows += load(suite+suffix)
    return rows


def stat(values):
    a=np.asarray(values,dtype=float)
    return {"n":int(a.size), "median":float(np.median(a)), "min":float(a.min()),
            "max":float(a.max()), "q25":float(np.quantile(a,.25)), "q75":float(np.quantile(a,.75))}


def compare(a,b):
    reference={r["case_id"]:r for r in b}
    pairs=[{"case_id":r["case_id"],"topology":r["topology"],
            "a_tv":float(r["tv"]),"b_tv":float(reference[r["case_id"]]["tv"]),
            "a_cold_seconds":float(r["cold_batch_seconds"]),
            "b_cold_seconds":float(reference[r["case_id"]]["cold_batch_seconds"])} for r in a]
    for pair in pairs:
        pair["tv_ratio_a_over_b"]=pair["a_tv"]/pair["b_tv"]
        pair["cold_ratio_a_over_b"]=pair["a_cold_seconds"]/pair["b_cold_seconds"]
    regressions=sorted([r for r in pairs if r["a_tv"]>r["b_tv"]+1e-12],
                       key=lambda r:r["a_tv"]-r["b_tv"],reverse=True)
    return {"n":len(pairs), "a_lower_tv":sum(r["a_tv"]<r["b_tv"]-1e-12 for r in pairs),
            "a_higher_tv":sum(r["a_tv"]>r["b_tv"]+1e-12 for r in pairs),
            "tied":sum(abs(r["a_tv"]-r["b_tv"])<=1e-12 for r in pairs),
            "tv_ratio":stat([r["tv_ratio_a_over_b"] for r in pairs]),
            "cold_ratio":stat([r["cold_ratio_a_over_b"] for r in pairs]),
            "all_regressions":regressions,"worst_regressions":regressions[:5]}


summary={"notes":[
    "Dense TV uses complete small-system laws; Frugal Q is instrumented continuous-uniform threshold law.",
    "The k4 extension was added after the first baseline comparison and is exploratory.",
    "The 24 held-out configurations use six interferometers, with parameter/profile reuse.",
    "Grid medians and win counts are descriptive, not population inference.",
    "k retained clicks and K cumulant order are different resource budgets.",
    "128-draw core cold timings are short-batch latency; Python versus optimized C++ is a material confound."],
    "suites":{}}
frugal=load("phase2_frugal")
all_cases=set(); total_rows=len(frugal)
for suite in ["phase2_discovery","phase2_holdout","phase2_bright"]:
    rows=candidate_rows(suite); total_rows+=len(rows)
    all_cases.update(r["case_id"] for r in rows)
    grouped=defaultdict(list)
    for row in rows:
        grouped[(row["topology"],row["strategy"],row["k"])].append(row)
    for row in frugal:
        if row["suite"]==suite: grouped[(row["topology"],"frugal",row["K"])].append(row)
    groups=[]
    for (topology,method,budget),rr in sorted(grouped.items()):
        groups.append({"topology":topology,"method":method,"budget":int(budget),
                       "tv":stat([r["tv"] for r in rr]),
                       "cold_batch_seconds":stat([r["cold_batch_seconds"] for r in rr])})
    comparisons={}
    for method in ["conditional_mi","click_influence"]:
        for k in ["2","3","4"]:
            a=[r for r in rows if r["strategy"]==method and r["k"]==k]
            b=[r for r in rows if r["strategy"]=="schur_pairwise" and r["k"]==k]
            if a and b: comparisons[f"{method}_k{k}_vs_pairwise_k{k}"]=compare(a,b)
    for k in ["2","3","4"]:
        a=[r for r in rows if r["strategy"]=="click_influence" and r["k"]==k]
        for K in ["2","3","4","5"]:
            b=[r for r in frugal if r["suite"]==suite and r["K"]==K]
            if a and b: comparisons[f"influence_k{k}_vs_frugal_K{K}"]=compare(a,b)
    summary["suites"][suite]={"candidate_rows":len(rows),"cases":len(set(r["case_id"] for r in rows)),
                             "groups":groups,"comparisons":comparisons}
summary["full_law_evaluations"]=total_rows
summary["distinct_main_inputs"]=len(all_cases)

heldout=candidate_rows("phase2_holdout")
lines=["# Second-round descriptive results", "", "## Median full-distribution TV on 24 held-out input configurations", "",
       "k4 is a post-baseline exploratory extension. Six interferometers are reused across parameters/profiles.", "",
       "| Method | Haar | Local 2 | Local 4 | Median core cold 128-draw seconds |", "|---|---:|---:|---:|---:|"]
plot_methods=[("schur_pairwise","2","Pairwise k=2"), ("click_influence","2","Influence k=2"),
              ("schur_pairwise","3","Pairwise k=3"), ("click_influence","3","Influence k=3"),
              ("schur_pairwise","4","Pairwise k=4 (exploratory)"), ("click_influence","4","Influence k=4 (exploratory)"),
              ("frugal","2","Frugal K=2"),("frugal","3","Frugal K=3"),
              ("frugal","4","Frugal K=4"),("frugal","5","Frugal K=5")]
method_groups=[]
for method,budget,label in plot_methods:
    rr=([r for r in frugal if r["suite"]=="phase2_holdout" and r["K"]==budget] if method=="frugal"
        else [r for r in heldout if r["strategy"]==method and r["k"]==budget])
    if not rr: continue
    values=[np.median([float(r["tv"]) for r in rr if r["topology"]==top]) for top in TOPOLOGIES]
    cost=float(np.median([float(r["cold_batch_seconds"]) for r in rr]))
    lines.append("| "+label+" | "+" | ".join(f"{v:.8g}" for v in values)+f" | {cost:.6g} |")
    method_groups.append((method,budget,label,rr))

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                     "axes.spines.top":False,"axes.spines.right":False,"figure.facecolor":"white"})
fig,axs=plt.subplots(1,2,figsize=(13,5.1),layout="constrained")
reference={r["case_id"]:r for r in frugal if r["suite"]=="phase2_holdout" and r["K"]=="5"}
shown_values=[float(r["tv"]) for r in heldout if r["strategy"]=="click_influence" and r["k"] in ("3","4")]
shown_values += [float(r["tv"]) for r in reference.values()]
plot_low=10.0**np.floor(np.log10(min(shown_values)))
plot_high=10.0**np.ceil(np.log10(max(shown_values)))
for topology in TOPOLOGIES:
    for k,marker in [("3","o"),("4","^")]:
        rr=[r for r in heldout if r["strategy"]=="click_influence" and r["k"]==k and r["topology"]==topology]
        if rr: axs[0].scatter([float(reference[r["case_id"]]["tv"]) for r in rr],
                             [float(r["tv"]) for r in rr],color=COLORS[topology],marker=marker,s=40,
                             label=f"{NAMES[topology]}, k={k}",alpha=.83,edgecolors="white",linewidths=.4)
axs[0].plot([plot_low,plot_high],[plot_low,plot_high],"--",color="#777777",lw=1)
axs[0].set(xscale="log",yscale="log",xlim=(plot_low,plot_high),ylim=(plot_low,plot_high),
           xlabel="FrugalGBS K=5: full-law TV",ylabel="Click-influence rule: full-law TV",
           title="Paired accuracy: lower than dashed line is better")
axs[0].legend(fontsize=8,loc="lower right");axs[0].grid(alpha=.15,which="both")
for method,budget,label,rr in method_groups:
    if method=="schur_pairwise":continue
    color="#7c3aed" if method=="frugal" else "#00856b"
    marker="s" if method=="frugal" else "o"
    xx=float(np.median([float(r["cold_batch_seconds"]) for r in rr]))
    yy=float(np.median([float(r["tv"]) for r in rr]))
    axs[1].scatter([xx],[yy],color=color,marker=marker,s=65)
    axs[1].annotate(("K=" if method=="frugal" else "k=")+budget,(xx,yy),xytext=(6,4),textcoords="offset points",fontsize=9,color=color)
axs[1].scatter([],[],color="#7c3aed",marker="s",label="Official FrugalGBS, C++")
axs[1].scatter([],[],color="#00856b",marker="o",label="Click-influence prototype, Python")
axs[1].set(xscale="log",yscale="log",xlabel="Median setup + first 128 draws (seconds)",
           ylabel="Median full-law TV across the grid",title="Accuracy / short-batch cost; implementation confound")
axs[1].margins(x=.12,y=.12)
axs[1].grid(alpha=.15,which="both");axs[1].legend(loc="upper right",fontsize=8)
fig.suptitle("Threshold GBS | 10 modes | 24 configurations | k=4 is exploratory",fontsize=12)
fig.savefig(OUT/"accuracy_frontier.png",dpi=180);fig.savefig(OUT/"accuracy_frontier.svg");plt.close(fig)

stress=json.loads((ROOT/"results/phase2_paired/results.json").read_text())
summary["paired_stress"]={"analytic_rows":len(stress["analytic_series"]),"crosschecks":len(stress["crosschecks"]),
                          "guard_failures":stress["guard_failures"],"analytic_series":stress["analytic_series"]}
validation=load("phase2_tv_validation")
summary["tv_validation"]={"bounds":len(validation),"six_click_rows":[r for r in validation if r["max_exact_clicks"]=="6"]}
fig,axs=plt.subplots(1,3,figsize=(15,4.6),layout="constrained")
for k,color in [(1,"#c2410c"),(2,"#d97706"),(3,"#7c3aed")]:
    rr=[r for r in stress["analytic_series"] if r["k"]==k]
    axs[0].plot([r["modes"] for r in rr],[r["blocked_recent_tv"] for r in rr],marker="o",color=color,label=f"Recent k={k}")
axs[0].axhline(0,color="#00856b",ls="--",label="Correct partner / pair order")
axs[0].set(xscale="log",xlabel="Modes (independent TMSV pairs)",ylabel="Analytic full-distribution TV",
           title="A rigorous failure family, p=0.5",ylim=(-.04,1.04))
axs[0].legend(fontsize=8);axs[0].grid(alpha=.2)
for L,color in [("4","#d97706"),("6","#2563eb")]:
    for seed,marker in [("7101","o"),("7102","s")]:
        rr=[r for r in validation if r["max_exact_clicks"]==L and r["seed"]==seed]
        axs[1].plot([int(r["modes"]) for r in rr],[float(r["upper"]) for r in rr],
                    marker=marker,color=color,ls="-" if seed=="7101" else "--",label=f"L={L}, seed {seed}")
axs[1].scatter([10,10],[float(r["dense_true_tv"]) for r in validation if r["modes"]=="10" and r["max_exact_clicks"]=="6"],
               marker="x",color="black",label="Dense TV at m=10",zorder=5)
axs[1].set(xlabel="Modes",ylabel="TV upper bound",title="512 draws; assumes accurate point probabilities",ylim=(0,.32),xticks=[10,16,32])
axs[1].legend(fontsize=7);axs[1].grid(alpha=.2)
cost_path=ROOT/"results/phase2_kernel_cost/metrics.csv"
if cost_path.exists():
    costs=load("phase2_kernel_cost");summary["kernel_cost"]={}
    for method,label,color in [("old_pairwise","Original determinants","#777777"),("schur_pairwise","Schur-conditioned kernel","#00856b")]:
        vals=[]
        for m in [16,32,64]:
            rr=[r for r in costs if r["method"]==method and r["modes"]==str(m) and r["status"]=="ok"]
            summary["kernel_cost"][f"{method}_m{m}"]=stat([r["total_seconds"] for r in rr])
            vals.append(float(np.median([float(r["total_seconds"]) for r in rr])))
        axs[2].plot([16,32,64],vals,marker="o",color=color,label=label)
    pair_data=json.loads((ROOT/"results/phase2_kernel_cost/pairs.json").read_text())
    summary["kernel_cost"]["identical_sample_pairs"]=sum(r["paired_samples_equal"] for r in pair_data)
    summary["kernel_cost"]["total_pairs"]=len(pair_data)
    axs[2].set(xlabel="Modes",ylabel="Median setup + first 128 draws (seconds)",yscale="log",xticks=[16,32,64],
               title="Same approximation law, k=2; 3 repeats / input")
    axs[2].legend(fontsize=8);axs[2].grid(alpha=.2)
else:
    axs[2].text(.5,.5,"Kernel cost run pending",transform=axs[2].transAxes,ha="center");axs[2].set_axis_off()
fig.suptitle("Scope matters: easy analytic family, low-click validation, and implementation cost",fontsize=12)
fig.savefig(OUT/"scope_and_scaling.png",dpi=180);fig.savefig(OUT/"scope_and_scaling.svg");plt.close(fig)

(OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
lines.extend(["", "## Bound interpretation", "",
    "Twelve fixed bounds allocate alpha=0.05/12; the simultaneous statistical coverage is conditional on correct floating point point probabilities.",
    "No outward-rounded certificate is claimed. Cutoff and numerical failures receive score 1 and are never dropped.", "",
    "## Main evaluation count", "", f"{total_rows} full-law method evaluations on {len(all_cases)} distinct main inputs; full-law references remain at 10 or 12 modes.", ""])
(OUT/"summary_tables.md").write_text("\n".join(lines))
print(json.dumps({"full_law_evaluations":total_rows,"distinct_main_inputs":len(all_cases),"output":str(OUT)}))
