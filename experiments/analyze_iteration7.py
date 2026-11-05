"""Recompute complete-law comparisons from the frozen iteration-seven arrays."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,default=ROOT/"results/iteration7_forest")
    p.add_argument("--output",type=Path,default=ROOT/"results/iteration7_analysis")
    args=p.parse_args()
    meta=json.loads((args.input/"metadata.json").read_text())
    if meta.get("status")!="complete" or not meta.get("sources_and_binary_stable"):
        raise ValueError("Complete frozen experiment required")
    rows=json.loads((args.input/"metrics.json").read_text())
    inputs=json.loads((args.input/"inputs.json").read_text())
    cfg=json.loads((args.input/"config.json").read_text())
    cases=[r["case_id"] for r in inputs]
    indexed={(r["case_id"],r["method"]):r for r in rows}
    for case in inputs:
        directory=args.input/case["case_id"]
        if sha(directory/"target.npz")!=case["input_sha256"]:
            raise ValueError("Target binding mismatch")
        with np.load(directory/"target.npz",allow_pickle=False) as f:
            target=f["target"]
        for spec in cfg["methods"]:
            row=indexed[case["case_id"],spec["id"]]
            path=directory/(spec["id"]+".npz")
            if sha(path)!=row["law_sha256"]:
                raise ValueError("Law binding mismatch")
            with np.load(path,allow_pickle=False) as f:
                q,fail=f["distribution"],float(f["failure_mass"])
            tv=float(.5*(np.abs(q-target).sum()+fail))
            if abs(tv-row["extended_tv"])>1e-13 or abs(q.sum()+fail-1)>2e-11:
                raise ValueError("Saved metric or normalization mismatch")
    methods=[s for s in cfg["methods"] if s["completion"] in ("chain", "static_mi_tree")]
    comparisons=[]
    for method in methods:
        for kind,baseline in [("pair_same_k",f"pair_k{method['k']}"),
                              ("two_more_retained",f"retained_k{method['k']+2}")]:
            pairs=[]
            for case in cases:
                new,old=indexed[case,method["id"]],indexed[case,baseline]
                ratio=new["extended_tv"]/old["extended_tv"] if old["extended_tv"]>1e-14 else None
                pairs.append({"case_id":case,"new_tv":new["extended_tv"],"baseline_tv":old["extended_tv"],
                    "tv_ratio":ratio,"tv_difference":new["extended_tv"]-old["extended_tv"],
                    "failure_mass":new["failure_mass"],"baseline_failure_mass":old["failure_mass"],
                    "enumeration_time_ratio":new["enumeration_seconds"]/old["enumeration_seconds"],
                    "setup_plus_enumeration_time_ratio":
                        (new["setup_seconds"]+new["enumeration_seconds"])/
                        (old["setup_seconds"]+old["enumeration_seconds"])})
            ratios=[r["tv_ratio"] for r in pairs if r["tv_ratio"] is not None]
            comparisons.append({"method":method["id"],"baseline":baseline,"comparison":kind,
                "improved":sum(r["tv_difference"] < -1e-10 for r in pairs),
                "worsened":sum(r["tv_difference"] > 1e-10 for r in pairs),
                "numerically_tied":sum(abs(r["tv_difference"])<=1e-10 for r in pairs),
                "ratio_cases":len(ratios),"median_tv_ratio":float(np.median(ratios)),
                "max_tv_ratio":max(ratios),"max_absolute_worsening":max(0,max(r["tv_difference"] for r in pairs)),
                "max_failure_mass":max(r["failure_mass"] for r in pairs),
                "median_python_enumeration_time_ratio":float(np.median([r["enumeration_time_ratio"] for r in pairs])),
                "median_python_setup_plus_enumeration_time_ratio":float(np.median(
                    [r["setup_plus_enumeration_time_ratio"] for r in pairs])),
                "pairs":pairs})
    args.output.mkdir(parents=True,exist_ok=True)
    summary={"created_utc":datetime.now(timezone.utc).isoformat(),"analysis_source_sha256":sha(Path(__file__)),
        "metadata_sha256":sha(args.input/"metadata.json"),"metrics_sha256":sha(args.input/"metrics.json"),
        "instances":len(inputs),"method_evaluations":len(rows),
        "distinct_haar_unitaries":len(cfg["modes"])*len(cfg["seeds_per_mode"]),
        "crosschecks_passed":meta["crosschecks_passed"],"comparisons":comparisons,
        "claim_scope":"Complete floating extended TV on these fixed small instances; Python full enumeration time is not sampling throughput or native cost. No novelty or universal accuracy claim."}
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    lines=["# 第七轮完整分布比较","",summary["claim_scope"],"",
        "所有TV均重读保存的P/Q数组计算，包含未解决分支的失败质量。数值相等采用绝对差1e-10，近零基线不计算比值。", "",
        "| 新法 | 比较基线 | TV改善/恶化/数值相等 | TV比值中位数 | 最坏绝对恶化 | 最大失败质量 | Python准备+枚举时间中位比 |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for c in comparisons:
        lines.append(f"| {c['method']} | {c['baseline']} | {c['improved']}/{c['worsened']}/{c['numerically_tied']} | {c['median_tv_ratio']:.6g} | {c['max_absolute_worsening']:.6g} | {c['max_failure_mass']:.6g} | {c['median_python_setup_plus_enumeration_time_ratio']:.4g} |")
    lines += ["", "## 全部固定输入的完整TV", "",
        "| 输入 | " + " | ".join(s["id"] for s in cfg["methods"]) + " |",
        "|---|"+"---:|"*len(cfg["methods"])]
    for case in cases:
        lines.append("| "+case+" | "+" | ".join(f"{indexed[case,s['id']]['extended_tv']:.8g}" for s in cfg["methods"])+" |")
    lines += ["", "Python时间包含准备、静态MI预计算（如适用）、完整指数规模Q枚举、选择器、建树和当前直接事件求值，单次描述性记录；未与C++、Frugal或九章时间对赛。失败没有补抽，成功子律没有重归一化。", "",
        "![完整TV相对分块与强保留点击基线](forest_tv_ratios.png)", ""]
    (args.output/"summary_tables.md").write_text("\n".join(lines))
    matrices=[]
    for kind in ("pair_same_k","two_more_retained"):
        matrices.append(np.asarray([[next(c for c in comparisons if c["method"]==s["id"] and c["comparison"]==kind)["pairs"][i]["tv_ratio"] for i in range(len(cases))] for s in methods],dtype=float))
    logged=[np.log10(np.maximum(a,1e-15)) for a in matrices]
    limit=max(0.1,max(float(np.nanmax(np.abs(a))) for a in logged))
    plt.rcParams.update({"font.family":"DejaVu Sans","svg.fonttype":"none"})
    fig,axes=plt.subplots(2,1,figsize=(12.5,7.4))
    labels=[f"{c['modes']}-{c['profile']['name'][0].upper()}" for c in inputs]
    for ax,values,logs,title in zip(axes,matrices,logged,["A  Relative to disjoint pair completion at the same k", "B  Relative to retaining two more clicks (k + 2)"]):
        im=ax.imshow(logs,cmap="RdBu_r",norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),aspect="auto")
        ax.set_yticks(range(len(methods)),[s["id"].replace("_"," ") for s in methods],fontsize=10)
        ax.set_xticks(range(len(labels)),labels,fontsize=9)
        ax.set_title(title,loc="left",fontsize=13,fontweight="bold",pad=9)
        for y in range(values.shape[0]):
            for x in range(values.shape[1]):
                v=values[y,x]
                if np.isfinite(v):
                    text=f"{v:.2g}"
                    ax.text(x,y,text,ha="center",va="center",fontsize=8,
                            color="white" if abs(logs[y,x])>limit*.56 else "#16212d")
        ax.set_xticks(np.arange(-.5,len(labels),1),minor=True)
        ax.set_yticks(np.arange(-.5,len(methods),1),minor=True)
        ax.grid(which="minor",color="white",linewidth=.5)
        ax.tick_params(which="minor",bottom=False,left=False)
    fig.suptitle("Conditional forests: complete distribution error",fontsize=19,fontweight="bold",y=.97)
    fig.text(.09,.91,"Cell = TV(new) / TV(baseline); blue < 1 improves error; failure-symbol mass is included.",fontsize=11)
    fig.subplots_adjust(left=.13,right=.91,top=.83,bottom=.18,hspace=.55)
    bar=fig.colorbar(im,ax=axes,location="right",fraction=.025,pad=.025)
    bar.set_label("log10 TV ratio",fontsize=10)
    fig.text(.09,.095,"Labels: modes-profile; L = lossy, B = bright, S = stress. All 9 predefined inputs retained.",fontsize=10)
    fig.text(.09,.060,"Floating-point small-system laws, not a universal error guarantee. Gaussian support theorem assumes exact event probabilities.",fontsize=10)
    for ext in ("png","svg"):
        fig.savefig(args.output/f"forest_tv_ratios.{ext}",dpi=180,facecolor="white")
    plt.close(fig)
    print(json.dumps({"instances":len(inputs),"method_evaluations":len(rows),"crosschecks_passed":meta["crosschecks_passed"],
        "comparisons":[{k:c[k] for k in ("method","baseline","improved","worsened","median_tv_ratio","max_failure_mass")} for c in comparisons]},indent=2))


if __name__=="__main__":
    main()
