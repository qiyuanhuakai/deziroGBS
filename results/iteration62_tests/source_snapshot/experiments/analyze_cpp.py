"""Descriptive native benchmarks; no population or quantum-speedup inference.

Repetition medians are computed inside each input before paired comparisons.
Cold wrapper repeat 0 includes a verification enumeration and is not included
in sampling-only wrapper latency. Larger cases are cost probes only.
"""
from collections import defaultdict
from pathlib import Path
import argparse
import csv
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
METHODS=["cpp_pairwise","cpp_click_influence","frugal_native"]
LABELS={"cpp_pairwise":"Pairwise k=4", "cpp_click_influence":"Click influence k=4", "frugal_native":"Official FrugalGBS K=5"}
COLORS={"cpp_pairwise":"#64748b", "cpp_click_influence":"#00856b", "frugal_native":"#7c3aed"}


def stats(values):
    a=np.asarray(values,dtype=float)
    return {"n":int(a.size),"median":float(np.median(a)),"min":float(a.min()),"max":float(a.max())}


def load_phase(path):
    metadata=json.loads((path/"metadata.json").read_text())
    if metadata["status"]!="complete" or metadata["failed_rows"]:
        raise RuntimeError(f"Unsuccessful formal phase: {path}")
    rows=list(csv.DictReader((path/"metrics.csv").open()))
    grouped=defaultdict(list)
    for row in rows:grouped[(row["case_id"],row["method"])].append(row)
    summaries=[]
    for (_,method),rr in grouped.items():
        item={k:rr[0][k] for k in ["case_id","method","suite","topology","modes","samples","seed","input_sha256"]}
        item.update(modes=int(item["modes"]),samples=int(item["samples"]),repetitions=len(rr))
        for field in ["setup_seconds","sampling_seconds","cold_seconds","draws_per_second"]:
            item[field]=float(np.median([float(r[field]) for r in rr]))
            item[field+"_min"]=min(float(r[field]) for r in rr)
            item[field+"_max"]=max(float(r[field]) for r in rr)
        clean=[r for r in rr if r.get("wrapper_includes_enumeration","False")=="False"]
        item["sampling_harness_seconds"]=float(np.median([float(r["harness_call_seconds"]) for r in clean])) if clean else None
        if rr[0].get("tv"):
            item["tv"]=float(rr[0]["tv"])
            if method!="frugal_native":item["tv_to_python_same_rule"]=float(rr[0]["tv_to_python_same_rule"])
        summaries.append(item)
    return metadata,summaries


def paired(rows,method,reference="frugal_native"):
    aa={r["case_id"]:r for r in rows if r["method"]==method}
    bb={r["case_id"]:r for r in rows if r["method"]==reference}
    if set(aa)!=set(bb):raise ValueError("Unpaired comparison")
    pairs=[]
    for case,a in aa.items():
        b=bb[case]
        p={"case_id":case,"method":method,"reference":reference}
        for field in ["cold_seconds","sampling_seconds","setup_seconds","sampling_harness_seconds"]:
            if a[field] is not None and b[field] is not None:p[field+"_ratio"]=a[field]/b[field]
        if "tv" in a and "tv" in b:
            p.update(tv_ratio=a["tv"]/b["tv"],lower_tv=a["tv"]<b["tv"]-1e-12,
                     tv=a["tv"],reference_tv=b["tv"])
        pairs.append(p)
    result={"n":len(pairs),"pairs":pairs}
    for field in ["cold_seconds_ratio","sampling_seconds_ratio","setup_seconds_ratio","sampling_harness_seconds_ratio","tv_ratio"]:
        values=[p[field] for p in pairs if field in p]
        if values:result[field]=stats(values)
    result["faster_cold"]=sum(p["cold_seconds_ratio"]<1 for p in pairs)
    result["faster_sampling"]=sum(p["sampling_seconds_ratio"]<1 for p in pairs)
    result["cold_speedup_reference_over_candidate"]=stats([1/p["cold_seconds_ratio"] for p in pairs])
    if "tv_ratio" in result:
        result["lower_tv"]=sum(p["lower_tv"] for p in pairs)
        result["lower_tv_and_faster_cold"]=sum(p["lower_tv"] and p["cold_seconds_ratio"]<1 for p in pairs)
    return result


def main(args):
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    summary={"scope":[
        "Native C++20 on the same CPU, single thread, O3, march=native, no fast-math.",
        "Medians across input configurations are descriptive, not estimates from independent random interferometers.",
        "Small circuits and k=4 were explored previously; native timings do not create a new statistical holdout.",
        "Complete distribution quality only at 10/12 modes; 16/32/64 modes are cost-only.",
        "Pairwise references use a numerical-stability repair; historical pairwise laws are preserved separately.",
        "Cold core time includes setup+draw; throughput is draws only. Harness time uses a shared outer timer and excludes cold repeat0 enumeration, but validation instrumentation differs across interfaces.",
        "Frugal K and retained-click k are different approximation budgets, not equal resource parameters."],"phases":{}}
    phases={}
    for phase,path in [("cold",args.cold),("steady",args.steady),("large",args.large)]:
        meta,rows=load_phase(path)
        if phase=="steady":
            quality={(r["case_id"],r["method"]):r for r in phases["cold"]}
            for row in rows:
                prior=quality[(row["case_id"],row["method"])]
                if prior["input_sha256"]!=row["input_sha256"]:raise RuntimeError("Steady quality input mismatch")
                row["tv"]=prior["tv"]
                row["quality_source"]="Identical input/method/binary full Q from the cold phase; not recomputed during throughput timing"
        phases[phase]=rows
        result={"path":str(path.relative_to(ROOT)),"rows":meta["rows"],"case_summaries":rows,
                "cpp_binary_sha256":meta["cpp_binary"]["sha256"],"frugal_binary_sha256":meta["frugal_binary"]["sha256"],"groups":{}}
        if phase=="cold":
            result["all_inputs_comparisons"]={m+"_vs_frugal":paired(rows,m) for m in METHODS[:2]}
        groupkeys=sorted({r["suite"] if phase!="large" else str(r["modes"]) for r in rows})
        for key in groupkeys:
            rr=[r for r in rows if (r["suite"] if phase!="large" else str(r["modes"]))==key]
            group={"comparisons":{m+"_vs_frugal":paired(rr,m) for m in METHODS[:2]},"method_medians":{}}
            group["comparisons"]["influence_vs_pairwise"]=paired(rr,"cpp_click_influence","cpp_pairwise")
            for method in METHODS:
                mm=[r for r in rr if r["method"]==method]
                group["method_medians"][method]={field:stats([r[field] for r in mm]) for field in
                    ["setup_seconds","sampling_seconds","cold_seconds","draws_per_second"]}
                if "tv" in mm[0]:group["method_medians"][method]["tv"]=stats([r["tv"] for r in mm])
            result["groups"][key]=group
        summary["phases"][phase]=result
    cpp_hashes={v["cpp_binary_sha256"] for v in summary["phases"].values()}
    frugal_hashes={v["frugal_binary_sha256"] for v in summary["phases"].values()}
    if len(cpp_hashes)!=1 or len(frugal_hashes)!=1:raise RuntimeError("Phases used different binaries")

    # An explicitly approximate setup/amortization crossover, not a measured sweep.
    crossover=[]
    for case in sorted({r["case_id"] for r in phases["steady"]}):
        rr={r["method"]:r for r in phases["steady"] if r["case_id"]==case}
        a,b=rr["cpp_click_influence"],rr["frugal_native"]
        delta_per_draw=a["sampling_seconds"]/a["samples"]-b["sampling_seconds"]/b["samples"]
        delta_setup=b["setup_seconds"]-a["setup_seconds"]
        crossover.append({"case_id":case,"modes":a["modes"],"topology":a["topology"],
                          "linear_model_crossover_samples":delta_setup/delta_per_draw if delta_per_draw>0 and delta_setup>0 else None,
                          "assumption":"Linear per-draw cost estimated from8192draw batches; not an experimentally established threshold"})
    summary["estimated_amortization_crossover"]=crossover
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    lines=["# Matched native C++ results", "", "All time ratios are candidate / official FrugalGBS K=5, using medians inside each input first.",
           "Full-TV comparisons are numerical full-law evaluations of 10/12-mode inputs only. 16/32/64-mode entries measure cost only.","",
           "| Workload | Inputs | Method | Median cold-time ratio | Faster cold | Median draw-time ratio | Median TV ratio | Lower TV |",
           "|---|---:|---|---:|---:|---:|---:|---:|"]
    for phase,result in summary["phases"].items():
        for group,gg in result["groups"].items():
            for method in METHODS[:2]:
                c=gg["comparisons"][method+"_vs_frugal"]
                tv=f"{c['tv_ratio']['median']:.6g}" if "tv_ratio" in c else "not evaluated"
                wins=f"{c['lower_tv']}/{c['n']}" if "lower_tv" in c else "not evaluated"
                lines.append(f"| {phase}: {group} | {c['n']} | {LABELS[method]} | {c['cold_seconds_ratio']['median']:.6g} | {c['faster_cold']}/{c['n']} | {c['sampling_seconds_ratio']['median']:.6g} | {tv} | {wins} |")
    lines += ["", "## Important boundaries", ""]+["- "+x for x in summary["scope"]]
    (output/"summary_tables.md").write_text("\n".join(lines)+"\n")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
                         "axes.spines.right":False,"figure.facecolor":"white"})
    fig,axs=plt.subplots(1,3,figsize=(15.7,5.0),layout="constrained")
    cold=phases["cold"]
    topology_colors={"haar":"#2563eb","local2":"#00856b","local4":"#d97706"}
    for suite,marker in [("phase2_holdout","o"),("phase2_bright","^")]:
        rows=[r for r in cold if r["suite"]==suite]
        aa={r["case_id"]:r for r in rows if r["method"]=="cpp_click_influence"}
        bb={r["case_id"]:r for r in rows if r["method"]=="frugal_native"}
        for top,color in topology_colors.items():
            ids=[c for c in aa if aa[c]["topology"]==top]
            if not ids:continue
            label=("12-mode bright" if suite=="phase2_bright" else "10-mode "+top)
            axs[0].scatter([aa[c]["cold_seconds"]/bb[c]["cold_seconds"] for c in ids],
                           [aa[c]["tv"]/bb[c]["tv"] for c in ids],s=48,marker=marker,color=color,label=label,edgecolors="white",linewidths=.4)
    axs[0].axvline(1,color="#777",ls="--");axs[0].axhline(1,color="#777",ls="--")
    axs[0].set(xscale="log",yscale="log",xlabel="128-draw setup + sample time ratio",ylabel="Full-law TV ratio",
               title="Short batches: paired quality and cost")
    axs[0].legend(fontsize=8);axs[0].grid(alpha=.15,which="both")
    steady=phases["steady"]
    bycase={r["case_id"]:r for r in cold if r["method"]=="cpp_click_influence"}
    base={r["case_id"]:r for r in steady if r["method"]=="frugal_native"}
    for mode,marker in [(10,"o"),(12,"^")]:
        rr=[r for r in steady if r["method"]=="cpp_click_influence" and r["modes"]==mode]
        ids=[r["case_id"] for r in rr]
        cold_base={r["case_id"]:r for r in cold if r["method"]=="frugal_native"}
        axs[1].scatter([r["sampling_seconds"]/base[r["case_id"]]["sampling_seconds"] for r in rr],
                       [bycase[c]["tv"]/cold_base[c]["tv"] for c in ids],marker=marker,s=55,
                       color="#00856b" if mode==10 else "#2563eb",label=f"{mode} modes")
    axs[1].axvline(1,color="#777",ls="--");axs[1].axhline(1,color="#777",ls="--")
    axs[1].set(xscale="log",yscale="log",xlabel="8192-draw sampling time ratio (no setup)",ylabel="Full-law TV ratio",
               title="Sustained sampling: setup amortized")
    axs[1].legend(fontsize=8);axs[1].grid(alpha=.15,which="both")
    large=phases["large"]
    for method in METHODS:
        rr=[r for r in large if r["method"]==method]
        xx=[16,32,64];yy=[np.median([r["cold_seconds"] for r in rr if r["modes"]==m]) for m in xx]
        axs[2].plot(xx,yy,"o-",color=COLORS[method],label=LABELS[method],lw=1.5)
    axs[2].set(yscale="log",xlabel="Modes",ylabel="Setup + 512 draws (seconds)",xticks=[16,32,64],
               title="Larger circuits: TV unmeasured")
    axs[2].legend(fontsize=8);axs[2].grid(alpha=.15,which="both")
    fig.suptitle("Native C++, one CPU thread | First two panels: click influence k=4 / official FrugalGBS K=5",fontsize=13)
    fig.savefig(output/"native_frontier.png",dpi=190);fig.savefig(output/"native_frontier.svg");plt.close(fig)
    print(json.dumps({"output":str(output),"phases":{k:v["rows"] for k,v in summary["phases"].items()}},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cold",type=Path,default=ROOT/"results/cpp_cold_v2")
    p.add_argument("--steady",type=Path,default=ROOT/"results/cpp_steady")
    p.add_argument("--large",type=Path,default=ROOT/"results/cpp_large")
    p.add_argument("--output",type=Path,default=ROOT/"results/cpp_analysis")
    main(p.parse_args())
