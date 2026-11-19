"""Recompute round-eight error/cost comparisons from bound saved artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from iteration8_common import ROOT, save, sha, total_variation, utc, validate_law


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", type=Path, default=ROOT / "results/iteration8_quality")
    parser.add_argument("--timing", type=Path, default=ROOT / "results/iteration8_timing")
    parser.add_argument("--regression", type=Path, default=ROOT / "results/iteration8_regression")
    parser.add_argument("--output", type=Path, default=ROOT / "results/iteration8_analysis")
    args = parser.parse_args()
    qmeta = json.loads((args.quality / "metadata.json").read_text())
    tmeta = json.loads((args.timing / "metadata.json").read_text())
    rmeta = json.loads((args.regression / "metadata.json").read_text())
    if any(m.get("status") != "complete" or not m.get("sources_and_binaries_stable") for m in (qmeta, rmeta)):
        raise ValueError("Complete quality and numerical-regression panels are required")
    if not tmeta.get("complete_matrix_keys_verified") or not tmeta.get("sources_and_binaries_stable"):
        raise ValueError("Timing must finish every declared job, retaining all failures")
    config = json.loads((args.quality / "config.json").read_text())
    if len({m["config_sha256"] for m in (qmeta, tmeta, rmeta)}) != 1:
        raise ValueError("Panel configurations do not match")
    inputs = json.loads((args.quality / "inputs.json").read_text())
    rows = json.loads((args.quality / "metrics.json").read_text())
    timings = json.loads((args.timing / "metrics.json").read_text())
    regressions = json.loads((args.regression / "metrics.json").read_text())
    index = {(r["case_id"], r["method"]): r for r in rows}
    cases = [r["case_id"] for r in inputs]
    for case in inputs:
        directory = args.quality / case["case_id"]
        if sha(directory / "target.npz") != case["input_sha256"]:
            raise ValueError("Target file differs from input binding")
        with np.load(directory / "target.npz", allow_pickle=False) as f:
            P = f["target"]
        for spec in config["methods"]:
            row = index[case["case_id"], spec["id"]]
            lawpath = directory / (spec["id"] + ".npz")
            if sha(lawpath) != row["law_sha256"]:
                raise ValueError("Saved native law hash mismatch")
            with np.load(lawpath, allow_pickle=False) as f:
                q, failure = f["distribution"], float(f["failure_mass"])
            validate_law(q, failure, case["modes"])
            if abs(total_variation(q, failure, P) - row["extended_tv"]) > 1e-13:
                raise ValueError("Saved complete TV disagrees with P/Q arrays")
    candidates = [s for s in config["methods"] if s["completion"] != "none"]
    tolerance = config["numerical_equality_absolute_tv"]
    quality_comparisons = []
    for candidate in candidates:
        method, baseline = candidate["id"], f"retained_k{candidate['k'] + 2}"
        pairs = []
        for case in cases:
            new, old = index[case, method], index[case, baseline]
            pairs.append({"case_id": case, "new_tv": new["extended_tv"], "baseline_tv": old["extended_tv"],
                "tv_ratio": new["extended_tv"] / old["extended_tv"] if old["extended_tv"] > 1e-14 else None,
                "tv_difference": new["extended_tv"] - old["extended_tv"],
                "failure_mass": new["failure_mass"], "baseline_failure_mass": old["failure_mass"]})
        ratios = [r["tv_ratio"] for r in pairs if r["tv_ratio"] is not None]
        quality_comparisons.append({"method": method, "baseline": baseline,
            "improved": sum(r["tv_difference"] < -tolerance for r in pairs),
            "worsened": sum(r["tv_difference"] > tolerance for r in pairs),
            "numerically_tied": sum(abs(r["tv_difference"]) <= tolerance for r in pairs),
            "median_tv_ratio": float(np.median(ratios)), "ratio_cases": len(ratios),
            "maximum_failure_mass": max(r["failure_mass"] for r in pairs), "pairs": pairs})
    groups = {}
    for row in timings:
        key = (row["case_id"], row["method"], row["requested_samples"])
        groups.setdefault(key, []).append(row)
    aggregated = {}
    for key, group in groups.items():
        repetitions = {r["repetition"] for r in group}
        if len(group) != 3 or repetitions != {0, 1, 2}:
            raise ValueError("Timing repetitions are incomplete or duplicated")
        values = [r["timings_seconds"]["cold_batch"] for r in group if r["status"] == "complete"]
        aggregated[key] = {"case_id": key[0], "method": key[1], "samples": key[2],
            "successful_repetitions": len(values), "minimum": min(values) if values else None,
            "maximum": max(values) if values else None,
            "median": float(np.median(values)) if len(values) == 3 else None,
            "wrapper_median": float(np.median([r["process_wrapper_seconds"] for r in group]))}
    cost_comparisons = []
    for candidate in candidates:
        method, baseline = candidate["id"], f"retained_k{candidate['k'] + 2}"
        pairs = []
        for profile in config["timing"]["profiles"]:
            case = f"m12_{profile}_s{config['timing']['seed_base']}"
            for n in config["timing"]["sample_counts"]:
                a, b = aggregated[case, method, n], aggregated[case, baseline, n]
                ratio = a["median"] / b["median"] if a["median"] is not None and b["median"] is not None else None
                tvdifference = index[case, method]["extended_tv"] - index[case, baseline]["extended_tv"]
                pairs.append({"case_id": case, "profile": profile, "samples": n,
                    "new_cold_seconds": a["median"], "baseline_cold_seconds": b["median"],
                    "time_ratio": ratio, "tv_difference": tvdifference,
                    "pareto_improvement": ratio is not None and ratio < 1 and tvdifference <= tolerance,
                    "new_range_seconds": [a["minimum"], a["maximum"]],
                    "baseline_range_seconds": [b["minimum"], b["maximum"]]})
        ratios = [r["time_ratio"] for r in pairs if r["time_ratio"] is not None]
        cost_comparisons.append({"method": method, "baseline": baseline,
            "completed_cells": len(ratios), "faster_cells": sum(x < 1 for x in ratios),
            "pareto_improved_cells": sum(p["pareto_improvement"] for p in pairs),
            "median_time_ratio": float(np.median(ratios)) if ratios else None, "pairs": pairs})
    diagnostics = [r.get("kernel_diagnostics") for r in timings if r["status"] == "complete" and r.get("kernel_diagnostics")]
    precision_work = {name: sum(d[name] for d in diagnostics)
                      for name in ("probability_calls", "auto_refinements", "auto_refinement_successes", "auto_refinement_failures")}
    summary = {"created_utc": utc(), "analysis_source_sha256": sha(Path(__file__)),
        "inputs": len(inputs), "distinct_haar_unitaries": len(config["modes"]) * len(config["seeds_per_mode"]),
        "quality_method_evaluations": len(rows), "timing_runs": len(timings),
        "successful_timing_runs": sum(r["status"] == "complete" for r in timings),
        "quality_failure_rows": sum(r["failure_mass"] > 0 for r in rows),
        "maximum_quality_failure_mass": max(r["failure_mass"] for r in rows),
        "quality_comparisons": quality_comparisons, "cost_comparisons": cost_comparisons,
        "timing_aggregates": list(aggregated.values()), "timing_precision_work": precision_work,
        "historical_regressions": regressions,
        "historical_failed_rows": sum(r["old_failure_mass"] > 0 for r in regressions),
        "historical_repaired_rows": sum(r["old_failure_mass"] > 0 and r["failure_mass"] == 0 for r in regressions),
        "maximum_historical_table_delta": max(r["table_law_tv_difference"] for r in regressions),
        "quality_maximum_python_delta": max(c["law_tv_difference"] for c in json.loads((args.quality / "crosschecks.json").read_text()) if c["kind"] == "python_orchestration"),
        "quality_maximum_table_delta": max(c["law_tv_difference"] for c in json.loads((args.quality / "crosschecks.json").read_text()) if c["kind"] == "independent_table_same_native_R_E"),
        "source_bindings": {str(p): sha(p) for p in (args.quality / "metadata.json", args.quality / "metrics.json",
            args.timing / "metadata.json", args.timing / "metrics.json", args.regression / "metrics.json")},
        "scope": "These fixed small synthetic instances and this shared host; heuristic floating indicators, complete failure-aware TV, no universal TV or Jiuzhang/novelty claim."}
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "summary.json", summary)
    lines = ["# 第八轮完整分布与原生总成本", "",
        "TV包含失败符号；质量判定绝对差容差1e-10；每个计时单元取预定3次完整冷批次的中位数。", "",
        "| 新法 | 基线 | TV改善/恶化/数值相等 | TV比值中位数 | 完整成本比中位数 | 更快单元 | 质量不差且更快 |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for a, b in zip(quality_comparisons, cost_comparisons):
        lines.append(f"| {a['method']} | {a['baseline']} | {a['improved']}/{a['worsened']}/{a['numerically_tied']} | {a['median_tv_ratio']:.6g} | {b['median_time_ratio']:.6g} | {b['faster_cells']}/6 | {b['pareto_improved_cells']}/6 |")
    lines += ["", "## 全部新输入的完整TV", "", "| 输入 | " + " | ".join(s["id"] for s in config["methods"]) + " |",
              "|---|" + "---:|" * len(config["methods"])]
    for case in cases:
        lines.append("| " + case + " | " + " | ".join(f"{index[case, s['id']]['extended_tv']:.9g}" for s in config["methods"]) + " |")
    lines += ["", "## 原生完整批次成本（秒，中位数）", "",
              "| 输入 | 样本数 | " + " | ".join(s["id"] for s in config["methods"]) + " |", "|---|---:|" + "---:|" * len(config["methods"])]
    for profile in config["timing"]["profiles"]:
        case = f"m12_{profile}_s{config['timing']['seed_base']}"
        for n in config["timing"]["sample_counts"]:
            lines.append(f"| {case} | {n} | " + " | ".join(f"{aggregated[case, s['id'], n]['median']:.8g}" if aggregated[case, s['id'], n]["median"] is not None else "失败/不完整" for s in config["methods"]) + " |")
    lines += ["", "## 历史数值回归（不混入新质量面板）", "",
              "| 历史输入 | 方法 | 原TV | 新TV | 原失败质量 | 新失败质量 | 与表参考的TV差 |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in regressions:
        lines.append(f"| {r['case_id']} | {r['method']} | {r['old_extended_tv']:.9g} | {r['extended_tv']:.9g} | {r['old_failure_mass']:.9g} | {r['failure_mass']:.9g} | {r['table_law_tv_difference']:.4g} |")
    (args.output / "summary_tables.md").write_text("\n".join(lines) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    quality_matrix = np.array([[p["tv_ratio"] for p in c["pairs"]] for c in quality_comparisons])
    cost_matrix = np.array([[p["time_ratio"] if p["time_ratio"] is not None else np.nan for p in c["pairs"]] for c in cost_comparisons])
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), gridspec_kw={"height_ratios": [1.2, 1]})
    for ax, matrix, title, labels in zip(axes, (quality_matrix, cost_matrix),
        ("A  Complete TV ratio: native forest / retained baseline", "B  Complete native batch time ratio: forest / retained baseline"),
        ([f"{c['modes']}{c['profile']['name'][0].upper()}-{str(c['seed_base'])[-1]}" for c in inputs],
         [f"{p['profile'][0].upper()} / N={p['samples']}" for p in cost_comparisons[0]["pairs"]])):
        logs = np.log10(matrix)
        limit = max(.1, float(np.nanmax(np.abs(logs))))
        im = ax.imshow(logs, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto")
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=10)
        ax.set_xticks(range(len(labels)), labels, fontsize=9)
        ax.set_yticks(range(len(candidates)), [s["id"].replace("_", " ") for s in candidates], fontsize=10)
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                value = matrix[y, x]
                ax.text(x, y, f"{value:.3g}" if np.isfinite(value) else "failed", ha="center", va="center", fontsize=9,
                        color="white" if np.isfinite(value) and abs(logs[y, x]) > .55 * limit else "#102030")
        bar = fig.colorbar(im, ax=ax, fraction=.022, pad=.02)
        bar.set_label("log10 ratio", fontsize=9)
    fig.suptitle("Shared Schur events: accuracy, reliability, and full sampler cost", fontsize=17, fontweight="bold", y=.98)
    fig.subplots_adjust(left=.12, right=.92, top=.87, bottom=.14, hspace=.52)
    fig.text(.08, .065, "Blue < 1 favors forest; red > 1 favors retained. k=2 vs retained k=4; k=4 vs retained k=6.", fontsize=10)
    fig.text(.08, .032, "All fixed cells retained. L/B/S = lossy/bright/stress; time = setup + sampling, median of 3. Shared host; no large-mode TV claim.", fontsize=9)
    for extension in ("png", "svg"):
        fig.savefig(args.output / ("quality_cost." + extension), dpi=180, facecolor="white")
    plt.close(fig)
    print(json.dumps({key: summary[key] for key in ("inputs", "quality_method_evaluations", "timing_runs", "quality_failure_rows", "historical_repaired_rows", "timing_precision_work")}, indent=2))


if __name__ == "__main__":
    main()
