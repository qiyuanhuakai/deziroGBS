"""Summarize frozen iteration4 evidence; no sampling or target fitting."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmt(value, digits=4):
    if value is None:
        return "未建立"
    return f"{value:.{digits}g}" if isinstance(value, (float, int)) else str(value)


def run(args):
    ppath = ROOT/"results/iteration4_performance/performance_summary.json"
    qpath = ROOT/"results/iteration4_quality/summary.json"
    cpath = ROOT/"results/iteration4_jiuzhang1_control/summary.json"
    rpath = ROOT/"results/iteration4_final_review/summary.json"
    performance, quality, control, review = map(read, (ppath, qpath, cpath, rpath))
    if not review["all_quantum_speed_ratios_absent"]:
        raise RuntimeError("Unexpected newly unlocked quantum ratio; inspect before publication")
    args.output.mkdir(parents=True, exist_ok=True)
    stage_audit = []
    for row in control["rows"]:
        native_path = ROOT/"results/iteration4_jiuzhang1_control"/row["method"]/"worker_native.json"
        native = read(native_path)
        payload = native.get("payload") or native
        stages = payload.get("stage_diagnostics", [])
        completed = {s["phase"]: s["seconds"] for s in stages
                     if s.get("status") == "completed" and isinstance(s.get("seconds"), (int, float))}
        required = ("bitstring_layout", "cumulant_layout", "target")
        setup = sum(completed[x] for x in required) if all(x in completed for x in required) else row["timings"].get("setup_seconds")
        if row["sampling_status"] != "ok" and not all(x in completed for x in required):
            setup = None
        stage_audit.append({"method": row["method"], "source_path": str(native_path.relative_to(ROOT)),
            "source_sha256": sha(native_path), "pipeline_status": native["status"],
            "covariance_sha256": payload.get("covariance_sha256", native.get("covariance_sha256")),
            "generation_phase_completed": native["status"] == "ok" or "sampling" in completed,
            "sample_records_returned": native.get("raw_samples") is not None,
            "last_reported_phase": stages[-1].get("phase") if stages else None,
            "completed_stage_seconds": completed, "setup_seconds_from_completed_phases": setup,
            "sampling_seconds_from_completed_phase": completed.get("sampling", row["timings"].get("sampling_seconds")),
            "wrapper_attempt_wall_seconds": payload.get("wrapper_wall_seconds", row["timings"].get("wall_seconds")),
            "density_completed": "density" in completed or row["method"] == "candidate" and native["status"] == "ok",
            "interpretation": "A completed generation timer is not a recovered sample batch or quality certificate; audit timeout is separate from generator accuracy."})
    (args.output/"actual_control_stage_audit.json").write_text(json.dumps(stage_audit, indent=2)+"\n")
    modes = sorted({r["modes"] for r in quality["rows"]})
    by_mode = []
    for m in modes:
        pairs = [r for r in performance["cases"] if r["modes"] == m]
        qrows = [r for r in quality["rows"] if r["modes"] == m]
        by_mode.append({"modes": m, "paired_cases": len(pairs),
            "median_cold_speedup": float(np.median([r["cold_speedup"] for r in pairs])),
            "min_cold_speedup": min(r["cold_speedup"] for r in pairs),
            "max_cold_speedup": max(r["cold_speedup"] for r in pairs),
            "conditional_accepts": sum(r["decision"].startswith("accepted") for r in qrows),
            "insufficient": sum(r["decision"] == "insufficient_evidence" for r in qrows),
            "rejected": sum(r["decision"] == "rejected" for r in qrows)})
    large_bias = [{"case_id": r["case_id"], "modes": r["modes"], "profile": r["profile"],
                   "target_mean_clicks": r["target_mean_clicks"], "observed_mean_clicks": r["observed_mean_clicks"],
                   "relative_observed_mean_difference": r["observed_mean_clicks"]/r["target_mean_clicks"]-1,
                   "is_predeclared_hypothesis_test": False}
                  for r in quality["rows"] if r["modes"] >= 64 and r["sampling_status"] == "ok"]
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "by_mode": by_mode,
        "performance": {k: v for k, v in performance.items() if k != "cases"},
        "identity_decisions": quality["decisions"], "actual_model_control": control,
        "actual_control_stage_audit": stage_audit,
        "descriptive_large_mode_bias": large_bias,
        "final_gap_review": {k: v for k, v in review.items() if k != "rows"},
        "inputs_sha256": {str(path.relative_to(ROOT)): sha(path) for path in (ppath, qpath, cpath, rpath)},
        "scope": "Synthetic quality/cost and a separate actual Jiuzhang1 Gaussian-model control. No global large-Jiuzhang reproduction established."}
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    table = ["# 第四轮结果表", "", "## 全模式优化与质量", "",
        "加速比=旧 log 核耗时/新 product 核耗时；每个输入先取3次原生 setup+draw 中位数，再对配对比值汇总。每个模式数有2个电路×2个亮度。", "",
        "| 模式数 | 配对中位加速 | 配对范围 | D1有条件接受 /4 | D1证据不足 /4 |",
        "|---:|---:|---:|---:|---:|"]
    for row in by_mode:
        table.append(f"| {row['modes']} | {row['median_cold_speedup']:.3f}× | {row['min_cold_speedup']:.3f}–{row['max_cold_speedup']:.3f}× | {row['conditional_accepts']} | {row['insufficient']} |")
    table += ["", "质量容差固定为TV≤0.1。8/12模式采用完整数值分布；更大模式用失败质量包含在内的统计上界。32比较族α=.05。不能把该结果外推到未测试电路或九章。", "",
        "## 九章一号作者模型：相同输入、固定样本数", "",
        "| 方法 | 整个证据流程 | 准备秒 | 抽样秒 | 额外logQ秒 | 包装器总秒 | 子进程峰值RSS GiB | 目标/观测平均点击 | D1 |",
        "|---|---|---:|---:|---:|---:|---:|---|---|"]
    for row in control["rows"]:
        t, res = row["timings"], row["resources"]
        stages = next(s for s in stage_audit if s["method"] == row["method"])
        peak = res.get("peak_memory_bytes")
        extra = "已含在抽样内" if t.get("density_within_sampling") else fmt(t.get("density_seconds"))
        if not stages["density_completed"] and stages["last_reported_phase"] == "density":
            extra = "超时，未完成"
        setup = stages["setup_seconds_from_completed_phases"]
        target_mean = row.get("target_mean_clicks")
        if target_mean is None:
            for other in control["rows"]:
                other_stage = next(s for s in stage_audit if s["method"] == other["method"])
                if stages["covariance_sha256"] and stages["covariance_sha256"] == other_stage["covariance_sha256"]:
                    target_mean = other.get("target_mean_clicks")
                    if target_mean is not None:
                        break
        table.append(f"| {row['method']} | {row['sampling_status']} | {fmt(setup)} | {fmt(stages['sampling_seconds_from_completed_phase'])} | {extra} | {fmt(stages['wrapper_attempt_wall_seconds'])} | {fmt(peak/2**30 if peak else None)} | {fmt(target_mean)}/{fmt(row.get('observed_mean_clicks'))} | {row['decision']} |")
    table += ["", "每种方法只有一次固定1024样本批次。CPU单线程、Frugal K5双精度；不代表论文8GPU、1000万样本实验。Frugal生成阶段完成，但附加logQ审计超时，完整样本尚未输出；`failed`表示证据流程失败，并非Frugal质量被拒绝。阶段计时从超时前已刷出的原生日志恢复，原失败记录不修改。没有用较低K或重试批次替代。抽样时间边界不同，已明确标记。总时长包含logQ审计，质量不足时不能称为同质量的速度胜利。", "",
        "## 审计结论", "",
        f"对既有证据重做D2审查：{review['reports']}份报告，新增采样0次，判决变化{review['verdict_changes']}份。量子/经典可比速度比全部为空。", "",
        "Frugal的主文对标是九章二号144位阈值任务及九章三号144位粗粒化任务；本地没有导入这些论文实例的完整原始数据，所以没有声称复现其图表。详见 research/iteration4_frugal_claims.md。"]
    (args.output/"summary_tables.md").write_text("\n".join(table)+"\n")
    make_plot(args.output, performance, quality)
    print(json.dumps({"output": str(args.output), "modes": modes, "quality_decisions": quality["decisions"],
                      "actual_control_status": [r["sampling_status"] for r in control["rows"]]}, indent=2))


def make_plot(output, performance, quality):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                        "axes.spines.top": False, "axes.spines.right": False,
                        "axes.titleweight": "bold", "axes.labelcolor": "#243247",
                        "text.color": "#243247", "xtick.color": "#43546a", "ytick.color": "#43546a"})
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5), constrained_layout=True)
    fig.suptitle("GBS classical lab · full-mode iteration", fontsize=21, fontweight="bold")
    colors = {"lossy": "#087f8c", "bright": "#c96b24"}
    modes = sorted({r["modes"] for r in quality["rows"]})
    index = {m: j for j, m in enumerate(modes)}
    ax = axes[0, 0]
    for j, m in enumerate(modes):
        values = [r["cold_speedup"] for r in performance["cases"] if r["modes"] == m]
        ax.scatter(j+np.linspace(-.14, .14, len(values)), values, c="#8a9bad", s=30, zorder=3)
        ax.plot([j-.22, j+.22], [np.median(values)]*2, color="#145a7a", lw=3)
    ax.axhline(1, color="#9b4b4b", lw=1.2, linestyle="--")
    ax.set(xticks=range(len(modes)), xticklabels=modes, xlabel="Output modes", ylabel="Log / product native time ratio",
           title="A  Same-rule implementation speed", ylim=(.79, 1.44))
    ax.text(.97, .05, "30/32 case medians faster\nOverall paired median: 1.095×", transform=ax.transAxes, va="bottom", ha="right", fontsize=11,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#edf5f8", "edgecolor": "none"})
    ax.grid(axis="y", alpha=.18)
    ax = axes[0, 1]
    for row in quality["rows"]:
        j = index[row["modes"]]
        offset = -.16 if row["profile"] == "lossy" else .16
        offset += -.035 if row["case_id"].endswith("95101") else .035
        full = row["full_tv"] is not None
        value = row["full_tv"] if full else row["global_tv_upper"]
        if value is not None:
            ax.scatter(j+offset, max(value, 1e-6), marker="o" if full else "^",
                       color=colors[row["profile"]], s=48, edgecolors="white", linewidths=.5, zorder=3)
    ax.axhline(.1, color="#9b4b4b", lw=1.3, linestyle="--")
    ax.set(xticks=range(len(modes)), xticklabels=modes, xlabel="Output modes", ylabel="TV or conservative TV upper bound",
           yscale="log", ylim=(1e-5, 1.5), title="B  Distribution evidence")
    legend = [Line2D([0], [0], marker="o", color="none", markerfacecolor="#5a6b7b", label="Complete numerical TV"),
              Line2D([0], [0], marker="^", color="none", markerfacecolor="#5a6b7b", label="Statistical upper bound"),
              Line2D([0], [0], color="#9b4b4b", linestyle="--", label="Preset tolerance 0.1")]
    ax.legend(handles=legend, loc="lower right", frameon=False, fontsize=10)
    ax.grid(axis="y", alpha=.18)
    ax = axes[1, 0]
    for profile in ("lossy", "bright"):
        for seed in ("95101", "95102"):
            selected = [r for r in quality["rows"] if r["profile"] == profile and r["case_id"].endswith(seed)]
            xs = [index[r["modes"]] for r in selected]
            ys = [100*(r["observed_mean_clicks"]/r["target_mean_clicks"]-1) for r in selected]
            ax.plot(xs, ys, ".-", color=colors[profile], alpha=.8 if seed == "95101" else .45,
                    label=profile if seed == "95101" else None, linewidth=1.2)
    ax.axhline(0, color="#5b697b", lw=1, linestyle="--")
    ax.set(xticks=range(len(modes)), xticklabels=modes, xlabel="Output modes",
           ylabel="Observed mean / target mean − 1 (%)", title="C  Bright large-mode samples drift low")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.18)
    ax.text(.02, .03, "Descriptive fixed-batch means; not a TV certificate.", transform=ax.transAxes, fontsize=9.5)
    ax = axes[1, 1]
    ax.axis("off")
    ax.text(0, .98, "D  What this iteration establishes", fontsize=16, fontweight="bold", va="top")
    blocks = [
        (.81, "192 performance runs · 32 synthetic inputs", "8–256 modes; 2 circuits × 2 brightness settings per size.\nSame binary, paired kernels, 3 repetitions; no log-Q timing overhead."),
        (.58, "10 conditional accepts · 22 inconclusive", "8 complete-law comparisons at 8/12 modes;\n2 statistical bounds at 16 modes. No large-mode quality acceptance."),
        (.34, "One target, one detector law, one tolerance", "Zero-mean Gaussian, threshold clicks, TV ≤ 0.1.\n32-comparison confidence family: α = 0.05; floating-point qualified."),
        (.10, "Jiuzhang equivalence remains unestablished", "Synthetic mode count is not experimental-instance reproduction.\nA separate actual Jiuzhang1-model control is archived.")]
    for y, heading, detail in blocks:
        ax.text(0, y, heading, va="top", fontsize=12.4, fontweight="bold", color="#145a7a")
        ax.text(0, y-.055, detail, va="top", fontsize=10.5, linespacing=1.5)
    fig.savefig(output/"full_mode_validation.png", dpi=180, facecolor="white")
    fig.savefig(output/"full_mode_validation.svg", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"results/iteration4_analysis")
    run(parser.parse_args())
