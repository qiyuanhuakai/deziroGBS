#!/usr/bin/env python3
"""Recompute iteration-5 summaries and figures from frozen artifacts only.

No sampling, target fitting, resampling, or timing occurs here.  Performance
ratios are ratios of within-case/within-batch method medians, as specified by
the benchmark runner, not medians of repetition ratios or pooled timings.
Quality plots keep complete numerical TV separate from statistical LOWER
bounds.  A lower bound is never substituted for the true TV or an upper bound.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROFILES = {"lossy": "Low brightness / higher loss", "bright": "Bright"}
PROFILE_COLORS = {"lossy": "#137B80", "bright": "#C36C28"}
BATCH_COLORS = {"short": "#087F8C", "long": "#7757A7"}
METHODS = ("uncached", "prefix_cached", "official_frugal")
TIMING_FIELDS = ("setup_seconds", "sampling_seconds", "cold_seconds", "harness_seconds")
INVALID_DECISIONS = {"invalid_model", "invalid_evidence", "evaluation_failed", "out_of_scope"}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def relative(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(path).resolve())


def integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def finite(value, name, *, minimum=0, maximum=None, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < minimum or positive and value <= 0
            or maximum is not None and value > maximum):
        raise ValueError(f"Invalid {name}: {value!r}")
    return float(value)


def maybe_probability(value, name):
    return None if value is None else finite(value, name, maximum=1)


def ratio_stats(values, *, planned=None):
    values = [finite(v, "time ratio", positive=True) for v in values]
    return {
        "planned_pairs": len(values) if planned is None else planned,
        "complete_pairs": len(values),
        "unavailable_pairs": (len(values) if planned is None else planned)-len(values),
        "faster_pairs": sum(v > 1 for v in values),
        "slower_pairs": sum(v < 1 for v in values),
        "equal_pairs": sum(v == 1 for v in values),
        "median_ratio": float(np.median(values)) if values else None,
        "minimum_ratio": min(values) if values else None,
        "maximum_ratio": max(values) if values else None,
        "interpretation": "Ratios >1 favor the cached candidate; counts are descriptive, not significance tests.",
    }


def expected_performance(config):
    expected = []
    repetitions = integer(config["repetitions"], "repetitions", minimum=1)
    for modes in config["modes"]:
        integer(modes, "modes", minimum=1)
        counts = config["sample_counts_by_modes"][str(modes)]
        if (len(counts) != 2 or any(type(n) is not int or n <= 0 for n in counts)
                or counts[0] >= counts[1]):
            raise ValueError("The frozen benchmark must declare distinct increasing short/long counts")
        for profile in config["profiles"]:
            for batch, n in zip(("short", "long"), counts):
                methods = list(METHODS[:2])
                if modes in config["official_frugal_modes"]:
                    methods.append("official_frugal")
                expected.append({"case_id": f"m{modes}_{profile['name']}",
                    "modes": modes, "profile": profile["name"], "samples": n,
                    "batch": batch, "methods": methods, "repetitions": repetitions})
    if len({(r["case_id"], r["samples"]) for r in expected}) != len(expected):
        raise ValueError("Duplicate planned performance case/batch")
    return expected


def recompute_performance(directory):
    config = read(directory/"config.json")
    metrics, archived = read(directory/"metrics.json"), read(directory/"summary.json")
    expected = expected_performance(config)
    expected_index = {(r["case_id"], r["samples"]): r for r in expected}
    if not isinstance(metrics, list):
        raise ValueError("Performance metrics must contain the raw run list")
    groups, seen = defaultdict(list), set()
    for row in metrics:
        key = (row["case_id"], row["samples"])
        if key not in expected_index:
            raise ValueError(f"Unplanned performance case/batch: {key}")
        spec = expected_index[key]
        if (row["method"] not in spec["methods"] or row["modes"] != spec["modes"]
                or row["profile"] != spec["profile"]):
            raise ValueError(f"Performance method/instance metadata changed: {key}")
        repetition = integer(row["repetition"], "repetition")
        run_key = (*key, row["method"], repetition)
        if repetition >= spec["repetitions"] or run_key in seen:
            raise ValueError(f"Repeated or unplanned timed attempt: {run_key}")
        seen.add(run_key)
        if row["status"] == "ok":
            for field in TIMING_FIELDS:
                finite(row[field], field, positive=field != "setup_seconds")
            if not math.isclose(row["cold_seconds"], row["setup_seconds"]+row["sampling_seconds"],
                                rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("A cold timer is not its own setup+sampling sum")
        elif row["status"] != "failed":
            raise ValueError("Unrecognized timed-attempt status")
        groups[(*key, row["method"])].append(row)
    expected_runs = sum(len(r["methods"])*r["repetitions"] for r in expected)
    if len(metrics) != expected_runs:
        raise ValueError("Timing archive is incomplete: every planned attempt, including failures, must be recorded")
    failures = [r for r in metrics if r["status"] != "ok"]
    if archived.get("rows") != len(metrics) or archived.get("failures") != len(failures):
        raise ValueError("Timing summary counts disagree with the raw attempts")
    if archived.get("all_same_seed_summary_checks_passed") is not True:
        raise ValueError("A same-seed summary equivalence check failed; review before plotting speed claims")
    saved_pairs = {(r["case_id"], r["samples"]): r for r in archived["paired"]}
    if len(saved_pairs) != len(archived["paired"]) or set(saved_pairs) != set(expected_index):
        raise ValueError("Timing summary does not identify exactly the planned case/batch pairs")

    paired, frugal_rows = [], []
    for spec in expected:
        key = (spec["case_id"], spec["samples"])
        medians, incomplete = {}, []
        for method in spec["methods"]:
            rows = groups[(*key, method)]
            if len(rows) != spec["repetitions"]:
                raise ValueError(f"Incomplete raw method group: {key}, {method}")
            if any(r["status"] != "ok" for r in rows):
                incomplete.append(method)
                continue
            medians[method] = {field: float(np.median([r[field] for r in rows]))
                               for field in TIMING_FIELDS}
        current = {**spec, "medians": medians, "incomplete_methods": incomplete}
        if all(method in medians for method in METHODS[:2]):
            current["cold_speedup"] = medians["uncached"]["cold_seconds"]/medians["prefix_cached"]["cold_seconds"]
            current["sampling_speedup"] = medians["uncached"]["sampling_seconds"]/medians["prefix_cached"]["sampling_seconds"]
        if "official_frugal" in medians and "prefix_cached" in medians:
            current["frugal_over_cached_cold"] = medians["official_frugal"]["cold_seconds"]/medians["prefix_cached"]["cold_seconds"]
            current["frugal_over_cached_sampling"] = medians["official_frugal"]["sampling_seconds"]/medians["prefix_cached"]["sampling_seconds"]
        saved = saved_pairs[key]
        if set(saved["medians"]) != set(medians):
            raise ValueError(f"Archived method-median availability differs: {key}")
        for method, times in medians.items():
            for field, value in times.items():
                if not math.isclose(saved["medians"][method][field], value, rel_tol=1e-12, abs_tol=1e-14):
                    raise ValueError(f"Archived median differs from raw attempts: {key}/{method}/{field}")
        for name in ("cold_speedup", "sampling_speedup", "frugal_over_cached_cold", "frugal_over_cached_sampling"):
            if (name in saved) != (name in current):
                raise ValueError(f"Archived ratio availability differs: {key}/{name}")
            if name in current and not math.isclose(saved[name], current[name], rel_tol=1e-12, abs_tol=1e-14):
                raise ValueError(f"Archived ratio differs from raw attempts: {key}/{name}")
        paired.append(current)
        if "official_frugal" in spec["methods"]:
            frugal_rows.append(current)

    def stats(rows, field):
        return ratio_stats([r[field] for r in rows if field in r], planned=len(rows))

    return {
        "planned_inputs": len({r["case_id"] for r in expected}),
        "planned_case_batch_pairs": len(expected), "planned_run_rows": expected_runs,
        "observed_run_rows": len(metrics), "failed_run_rows": len(failures),
        "failure_records": failures, "recomputed_summary_matches_archive": True,
        "all_same_seed_summary_checks_passed": True,
        "median_definition": f"For each input and batch size, take the median of each method's {config['repetitions']} recorded repetition times, then divide method medians. A cold repetition time is setup+sampling; its median need not equal median(setup)+median(sampling). Aggregate statistics are unweighted medians/minima/maxima of these per-case/per-batch ratios.",
        "ratios_direction": "uncached/cached for implementation speed; official Frugal/cached for the separate cost-only baseline comparison",
        "overall_cold": stats(paired, "cold_speedup"),
        "overall_sampling": stats(paired, "sampling_speedup"),
        "cold_by_batch": {b: stats([r for r in paired if r["batch"] == b], "cold_speedup") for b in ("short", "long")},
        "cold_by_profile": {p["name"]: stats([r for r in paired if r["profile"] == p["name"]], "cold_speedup") for p in config["profiles"]},
        "cold_by_modes": {str(m): stats([r for r in paired if r["modes"] == m], "cold_speedup") for m in config["modes"]},
        "paired": paired, "frugal_small_mode": frugal_rows,
        "claim_scope": archived["claim_scope"],
        "timing_scope": config["primary_metric"],
        "repetitions": config["repetitions"],
        "configuration": config,
    }


def analyze_quality(directory):
    archived, config = read(directory/"summary.json"), read(directory/"config.json")
    plan = read(directory/"comparison_plan.json")
    planned = {r["id"]: r for r in plan}
    if len(planned) != len(plan) or len(plan) != config["planned_comparisons"]:
        raise ValueError("Quality comparison plan is inconsistent")
    if archived.get("epsilon") != config["epsilon"] or archived.get("familywise_alpha") != config["familywise_alpha"]:
        raise ValueError("Quality summary differs from its frozen error/tolerance plan")
    source_rows = {r["case_id"]: r for r in archived["rows"]}
    if len(source_rows) != len(archived["rows"]) or set(source_rows) != set(planned):
        raise ValueError("Quality summary must record every planned comparison, including not-run cases")
    rows = []
    for case_id, case in planned.items():
        raw = source_rows[case_id]
        row = {**raw, "modes": case["modes"], "method": case["method"],
               "synthetic": case["synthetic"], "samples": case["count"],
               "pilot_size": case["pilot"], "heldout_size": case["count"]-case["pilot"],
               "profile": (case.get("profile") or {}).get("name"),
               "sampling_status": raw.get("sampling_status", raw.get("status", "unknown")),
               "decision": raw.get("decision", raw.get("status", "unknown"))}
        for key in ("modes", "method", "synthetic"):
            if key in raw and raw[key] != case[key]:
                raise ValueError(f"Quality case metadata differs from its plan: {case_id}/{key}")
        for field in ("full_tv", "global_tv_upper", "original_witness_lower", "marginal_tv_lower"):
            row[field] = maybe_probability(raw.get(field), field)
        row["evidence_plottable"] = row["sampling_status"] == "ok" and row["decision"] not in INVALID_DECISIONS
        rows.append(row)
    eps = config["epsilon"]
    return {
        "planned_comparisons": len(plan), "familywise_alpha": archived["familywise_alpha"],
        "alpha_allocation": {"original_D1": config["old_identity_suite_alpha"], "heldout_marginal": config["marginal_suite_alpha"]},
        "epsilon": eps, "decisions": dict(Counter(r["decision"] for r in rows)),
        "original_decisions": dict(Counter(r.get("original_decision", r["decision"]) for r in rows)),
        "synthetic_decisions": dict(Counter(r["decision"] for r in rows if r["synthetic"])),
        "complete_numerical_tv_cases": [r["case_id"] for r in rows if r["full_tv"] is not None and r["evidence_plottable"]],
        "marginal_rejections": [r["case_id"] for r in rows if r["evidence_plottable"] and r["marginal_tv_lower"] is not None and r["marginal_tv_lower"] > eps],
        "additional_rejections_after_marginal_channel": [r["case_id"] for r in rows if r["decision"] == "rejected" and r.get("original_decision") != "rejected" and r["marginal_tv_lower"] is not None and r["marginal_tv_lower"] > eps],
        "unavailable_or_invalid_evidence": [r["case_id"] for r in rows if not r["evidence_plottable"]],
        "rows": rows, "scope": archived["quality_scope"],
        "bounds_interpretation": "Witness values are LOWER bounds on full-law TV, not TV estimates or acceptance certificates. Complete numerical TV and conservative upper bounds are separate fields. Click means are descriptive and never share an error axis.",
    }


def analyze_small_laws(directory, performance_directory, performance):
    """Read the optional deterministic review, without evaluating P or Q anew."""
    if not (directory/"summary.json").exists():
        return None, []
    metadata, summary = read(directory/"metadata.json"), read(directory/"summary.json")
    if (metadata.get("status") not in ("complete", "complete_with_failures")
            or metadata.get("summary_sha256") != sha(directory/"summary.json")
            or summary.get("performance_binding_unchanged") is not True
            or summary.get("new_samples") != 0 or summary.get("new_performance_trials") != 0
            or summary.get("statistical_alpha_used") is not None
            or summary.get("preregistered_independent_statistical_confirmation") is not False):
        raise ValueError("Supplementary complete-law metadata is inconsistent")
    bindings = summary["inputs_sha256"]
    for name, expected in bindings.items():
        path = Path(name)
        path = path if path.is_absolute() else ROOT/path
        if sha(path) != expected:
            raise ValueError(f"Supplementary audit binding changed: {name}")
    for name in ("metadata.json", "config.json", "metrics.json", "summary.json",
                 "candidate_build_and_validation.json", "frugal_build_and_validation.json"):
        path = performance_directory/name
        if bindings.get(relative(path)) != sha(path):
            raise ValueError(f"Supplementary law refers to another performance archive: {name}")
    expected_cases = {r["case_id"] for r in performance["frugal_small_mode"]}
    rows = {r["case_id"]: r for r in summary["rows"]}
    if (len(rows) != len(summary["rows"]) or set(rows) != expected_cases
            or summary.get("planned_inputs") != len(expected_cases)):
        raise ValueError("Supplementary law review omitted or added a performance comparison input")
    inputs = [directory/"metadata.json", directory/"summary.json",
              directory/"binary_bindings.json", directory/"case_plan.json"]
    for case_id, row in rows.items():
        input_path = performance_directory/f"input_{case_id}.npz"
        if row["input_sha256"] != sha(input_path) or bindings.get(relative(input_path)) != sha(input_path):
            raise ValueError(f"Supplementary law input differs from the timed input: {case_id}")
        path = directory/row["laws_path"]
        if sha(path) != row["laws_sha256"]:
            raise ValueError(f"Supplementary probability archive changed: {case_id}")
        inputs.append(path)
        with np.load(path, allow_pickle=False) as archive:
            for role in ("candidate", "frugal"):
                declared = maybe_probability(row[f"{role}_tv"], f"{role}_tv")
                if declared is None:
                    continue
                p, q = archive["target_p"], archive[f"{role}_q"]
                if (p.shape != (1 << row["modes"],) or q.shape != p.shape
                        or not np.all(np.isfinite(p)) or not np.all(np.isfinite(q))
                        or np.any(p < 0) or np.any(q < 0)
                        or abs(float(p.sum())-1) > 1e-10 or abs(float(q.sum())-1) > 1e-10):
                    raise ValueError(f"Supplementary numerical probability vector is invalid: {case_id}/{role}")
                value = .5*float(np.sum(np.abs(p-q), dtype=np.longdouble))
                if not math.isclose(value, declared, rel_tol=1e-12, abs_tol=1e-15):
                    raise ValueError(f"Supplementary TV differs from its saved laws: {case_id}/{role}")
                diagnostic = row["diagnostics"][role]
                native_path = directory/diagnostic["native_record"]
                if sha(native_path) != diagnostic["native_record_sha256"]:
                    raise ValueError(f"Supplementary native record changed: {case_id}/{role}")
                native = read(native_path)
                if (native["sample_count"] != 0
                        or not np.array_equal(native["distribution"], archive[f"{role}_q_raw"])):
                    raise ValueError(f"Supplementary native law/draw count differs: {case_id}/{role}")
                inputs.append(native_path)
        complete = row["candidate_tv"] is not None and row["frugal_tv"] is not None
        if (row["status"] == "ok") != complete:
            raise ValueError("Supplementary per-input completion flag is inconsistent")
        if complete and row["candidate_lower_tv"] is not (row["candidate_tv"] < row["frugal_tv"]):
            raise ValueError("Supplementary relative-TV flag is inconsistent")
    complete = [r for r in rows.values() if r["status"] == "ok"]
    if summary.get("complete_inputs") != len(complete):
        raise ValueError("Supplementary completed-input count is inconsistent")
    joint = []
    for timed in performance["frugal_small_mode"]:
        law = rows[timed["case_id"]]
        if law["status"] != "ok" or "frugal_over_cached_cold" not in timed:
            continue
        joint.append({"case_id": timed["case_id"], "batch": timed["batch"],
            "samples": timed["samples"], "candidate_lower_numerical_tv": law["candidate_lower_tv"],
            "candidate_faster_cold": timed["frugal_over_cached_cold"] > 1,
            "candidate_faster_sampling": timed["frugal_over_cached_sampling"] > 1})
    result = {"status": metadata["status"], "directory": relative(directory),
        "design": summary["design"], "preregistered_independent_statistical_confirmation": False,
        "statistical_alpha_used": None, "new_samples": 0, "new_performance_trials": 0,
        "scope": summary["scope"], "law_semantics": summary["law_semantics"],
        "strict_floating_point_certificate": False,
        "planned_inputs": len(expected_cases), "complete_inputs": len(complete),
        "candidate_lower_numerical_tv_inputs": sum(r["candidate_lower_tv"] for r in complete),
        "rows": list(rows.values()), "joint_cost_quality_pairs": joint,
        "candidate_faster_cold_and_lower_tv_pairs": sum(r["candidate_faster_cold"] and r["candidate_lower_numerical_tv"] for r in joint),
        "candidate_faster_sampling_and_lower_tv_pairs": sum(r["candidate_faster_sampling"] and r["candidate_lower_numerical_tv"] for r in joint),
        "joint_comparison_scope": "Descriptive local comparisons on the four frozen performance inputs and their two batch sizes; no independent statistical confirmation, equal-TV constraint, large-mode extrapolation, or quantum-hardware comparison.",
        "saved_probability_vectors_verified": True, "performance_bindings_verified": True}
    return result, inputs


def fmt(value, digits=4):
    if value is None:
        return "—"
    return f"{value:.{digits}g}" if type(value) in (int, float) else str(value)


def make_tables(output, performance, quality, small_laws=None):
    cold = performance["overall_cold"]
    lines = ["# 第五轮：可复现汇总", "", "## 同一候选采样律的缓存速度", "",
        "每个输入、批量和方法分别取预定重复运行的中位数。冷启动时间先在每次运行中计算 setup+sampling，再取中位数；倍率是未缓存中位数/缓存中位数。跨案例汇总再次对这些配对倍率取中位数，没有合并原始耗时或改取逐重复倍率。", "",
        f"计划 {performance['planned_inputs']} 个输入、{performance['planned_case_batch_pairs']} 个输入×批量配对、{performance['planned_run_rows']} 次方法运行；记录 {performance['observed_run_rows']} 次，其中失败 {performance['failed_run_rows']} 次。", "",
        f"冷启动完整配对 {cold['complete_pairs']}/{cold['planned_pairs']}：缓存更快 {cold['faster_pairs']}，更慢 {cold['slower_pairs']}，相同 {cold['equal_pairs']}；中位倍率 {fmt(cold['median_ratio'])}×，范围 {fmt(cold['minimum_ratio'])}–{fmt(cold['maximum_ratio'])}×。", "",
        "| 汇总范围 | 完整/计划配对 | 更快 | 中位倍率 | 范围 |",
        "|---|---:|---:|---:|---:|"]
    for label, s in [("全部冷启动", cold), ("全部纯采样", performance["overall_sampling"])] + [
            (f"{b} batch 冷启动", s) for b, s in performance["cold_by_batch"].items()] + [
            (f"{p} 冷启动", s) for p, s in performance["cold_by_profile"].items()]:
        lines.append(f"| {label} | {s['complete_pairs']}/{s['planned_pairs']} | {s['faster_pairs']} | {fmt(s['median_ratio'])}× | {fmt(s['minimum_ratio'])}–{fmt(s['maximum_ratio'])}× |")
    lines += ["", "### 全部输入×批量配对", "",
        "| 输入 | 批量类别 | N | 未缓存冷启动秒 | 缓存冷启动秒 | 冷启动倍率 | 纯采样倍率 |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for r in performance["paired"]:
        a, b = r["medians"].get("uncached", {}), r["medians"].get("prefix_cached", {})
        lines.append(f"| {r['case_id']} | {r['batch']} | {r['samples']} | {fmt(a.get('cold_seconds'))} | {fmt(b.get('cold_seconds'))} | {fmt(r.get('cold_speedup'))} | {fmt(r.get('sampling_speedup'))} |")
    lines += ["", "每个计时进程的缓存从空开始，计入分配、填充、查询和全部 miss；没有由验证或枚举预热。原生准备加采样计时不含包装器输入/输出等开销，后者单独保存在原始记录。3 次重复的中位数是描述性比较，不能据此宣称普适显著加速。", "",
        "## 官方 Frugal 小模式成本对照", "",
        "这些倍率统一为官方 Frugal/缓存候选，>1 表示缓存候选更快。冷启动与纯采样分开报告。K5 Frugal 与 k4 retained-click 是不同采样律；计时本身不建立相同误差或量子优势结论。", ""]
    laws_by_case = {r["case_id"]: r for r in small_laws["rows"]} if small_laws else {}
    suffix = " 候选完整TV | Frugal完整TV |" if small_laws else ""
    lines += ["| 输入 | 批量/N | 候选冷启动秒 | Frugal冷启动秒 | 冷启动倍率 | 候选采样秒 | Frugal采样秒 | 采样倍率 |"+suffix,
              "|---|---|---:|---:|---:|---:|---:|---:|"+("---:|---:|" if small_laws else "")]
    for r in performance["frugal_small_mode"]:
        c, f = r["medians"].get("prefix_cached", {}), r["medians"].get("official_frugal", {})
        law = laws_by_case.get(r["case_id"], {})
        suffix = f" {fmt(law.get('candidate_tv'), 6)} | {fmt(law.get('frugal_tv'), 6)} |" if small_laws else ""
        lines.append(f"| {r['case_id']} | {r['batch']}/{r['samples']} | {fmt(c.get('cold_seconds'))} | {fmt(f.get('cold_seconds'))} | {fmt(r.get('frugal_over_cached_cold'))} | {fmt(c.get('sampling_seconds'))} | {fmt(f.get('sampling_seconds'))} | {fmt(r.get('frugal_over_cached_sampling'))} |"+suffix)
    if small_laws:
        lines += ["", "### 性能输入的补充完整分布质量审查", "",
            "上述 TV 使用正式计时保存的同一协方差文件、同一二进制与冻结参数，枚举全部 P、候选 Q、Frugal Q；每个输入的采样律对两种批量相同，因此同一个 TV 列于两行。两个接口均请求 0 个样本；枚举发生在正式计时完成后，任何枚举开销均未加入或替换性能记录。", "",
            f"4 个预定小模式输入完成 {small_laws['complete_inputs']} 个；候选完整数值 TV 较小 {small_laws['candidate_lower_numerical_tv_inputs']} 个。可合并核对的 {len(small_laws['joint_cost_quality_pairs'])} 个输入×批量配对中，候选冷启动更快且 TV 较小 {small_laws['candidate_faster_cold_and_lower_tv_pairs']} 个；纯采样更快且 TV 较小 {small_laws['candidate_faster_sampling_and_lower_tv_pairs']} 个。", "",
            "这项确定性审查在性能配置冻结后补充，覆盖全部四个预定 Frugal 对照输入；不是预注册独立统计确认，不新增统计 alpha，也没有新采样、重测时间或挑选有利输入。数值 TV 来自全分布，并非统计下界；保留浮点限定和可查的归一化修正。Frugal Q 来自已验证的 upstream SIMD 强制分支 trace，该枚举功能属于适配器审计接口。局部比较不能外推到大模式、Frugal 论文 GPU 配置或九章硬件。"]
    lines += ["", "## 独立固定质量批次", "",
        f"容差 TV≤{quality['epsilon']}；{quality['planned_comparisons']} 个比较的总族错误预算为 {quality['familywise_alpha']}，原 D1 与新边缘见证各分配 {quality['alpha_allocation']['original_D1']} / {quality['alpha_allocation']['heldout_marginal']}。完整数值 TV、统计上界与统计下界分别列出，不能互换。", "",
        "| 合成输入 | N / pilot | 采样状态 | 完整数值TV | 原见证TV下界 | 新边缘TV下界 | 全局TV上界 | 合并决定 |",
        "|---|---:|---|---:|---:|---:|---:|---|"]
    for r in quality["rows"]:
        if r["synthetic"]:
            lines.append(f"| {r['case_id']} | {r['samples']} / {r['pilot_size']} | {r['sampling_status']} | {fmt(r['full_tv'])} | {fmt(r['original_witness_lower'])} | {fmt(r['marginal_tv_lower'])} | {fmt(r['global_tv_upper'])} | {r['decision']} |")
    lines += ["", "新见证只用 pilot 选事件，在独立留出后缀上作单侧检验。下界超过容差可以拒绝；未超过只能是该见证未拒绝，不能接受全分布接近。数值结果均保留浮点限定，失败与缺失没有记作零误差。", "",
        "### 九章一号作者 Gaussian 模型：单独对照", "",
        "| 方法 | N / pilot | 采样状态 | 原见证TV下界 | 新边缘TV下界 | 全局TV上界 | 目标/观测平均点击 | 合并决定 |",
        "|---|---:|---|---:|---:|---:|---|---|"]
    for r in quality["rows"]:
        if not r["synthetic"]:
            lines.append(f"| {r['method']} | {r['samples']} / {r['pilot_size']} | {r['sampling_status']} | {fmt(r['original_witness_lower'])} | {fmt(r['marginal_tv_lower'])} | {fmt(r['global_tv_upper'])} | {fmt(r.get('target_mean_clicks'))} / {fmt(r.get('sample_mean_clicks'))} | {r['decision']} |")
    lines += ["", "这是已绑定的 JZ1 作者数学协方差模型，不是硬件实测分布，也不是 Frugal 论文的 JZ2/JZ3 GPU 实验复现。点击均值是描述性诊断，不是 TV。Frugal 本批为 generation-only，没有额外 logQ 回放，因此没有从这条缺失通道推导全局 TV 接受。候选采样包含 logQ；两者这些质量批次的时间边界不同，不从它们生成配对速度胜利。", "",
        "## 图与源记录", "",
        "![同采样律缓存速度](prefix_cache_speed.png)", "",
        "![完整TV与下界分别显示](quality_evidence.png)", "",
        "性能由 `../iteration5_performance/metrics.json` 重新计算并与其 `summary.json` 核对；质量来自 `../iteration5_quality/summary.json`，未重新抽样或重算目标。精确输入散列、完整配对及失败记录见本目录 `summary.json`。",
        "", "图的 SVG 版本与 PNG 同名；图中 lower bound 始终表示下界，不代表实际 TV。"]
    (output/"summary_tables.md").write_text("\n".join(lines)+"\n")


def plot_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": "#243247", "axes.titleweight": "bold",
        "text.color": "#243247", "xtick.color": "#43546A", "ytick.color": "#43546A",
        "svg.fonttype": "none", "savefig.facecolor": "white"})
    return plt


def save_figure(fig, output, stem, description):
    fig.savefig(output/f"{stem}.png", dpi=220, metadata={"Description": description})
    fig.savefig(output/f"{stem}.svg", metadata={"Description": description, "Creator": "deziroGBS analyze_iteration5.py"})


def plot_performance(output, performance):
    plt = plot_style()
    from matplotlib.ticker import FuncFormatter
    config = performance["configuration"]
    modes = config["modes"]
    profiles = [p["name"] for p in config["profiles"]]
    fig, axes = plt.subplots(1, len(profiles), figsize=(14, 6.4), squeeze=False)
    fig.subplots_adjust(left=.075, right=.98, bottom=.26, top=.78, wspace=.25)
    all_ratios = [r["cold_speedup"] for r in performance["paired"] if "cold_speedup" in r]
    low = min([1]+all_ratios)*.74
    high = max([1]+all_ratios)*1.40
    for i, profile in enumerate(profiles):
        ax = axes[0, i]
        selected = [r for r in performance["paired"] if r["profile"] == profile]
        for batch, marker in (("short", "o"), ("long", "s")):
            by_mode = {r["modes"]: r for r in selected if r["batch"] == batch}
            ys = [by_mode[m].get("cold_speedup", np.nan) for m in modes]
            ax.plot(range(len(modes)), ys, marker=marker, markersize=5.5, lw=1.6,
                    color=BATCH_COLORS[batch], label=f"{batch.capitalize()} batch")
        ax.axhline(1, linestyle="--", lw=1.1, color="#9B4B4B")
        spec = next(p for p in config["profiles"] if p["name"] == profile)
        ax.set(title=f"{'AB'[i]}  {PROFILES.get(profile, profile)}\nr={spec['squeezing']}, transmission={spec['transmissivity']}",
               xticks=range(len(modes)), xticklabels=modes, xlabel="Output modes",
               ylabel="Uncached / cached native time", yscale="log", ylim=(low, high))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}×"))
        ax.grid(axis="y", which="both", alpha=.15)
        ax.legend(frameon=False, loc="upper right")
        stat = performance["cold_by_profile"][profile]
        ax.text(.025, .025, f"Faster: {stat['faster_pairs']}/{stat['complete_pairs']} complete pairs\n"
                f"Median ratio: {fmt(stat['median_ratio'], 3)}×",
                transform=ax.transAxes, fontsize=9.4, va="bottom",
                bbox={"facecolor": "white", "alpha": .93, "edgecolor": "none", "pad": 3})
    fig.suptitle("Bounded prefix-cache speed at unchanged candidate law", fontsize=17, fontweight="bold", y=.96)
    fig.text(.5, .875, "Setup + complete sample batch; ratio of method medians; each process starts with an empty cache",
             ha="center", fontsize=10.5)
    counts = "; ".join(f"{m}: {config['sample_counts_by_modes'][str(m)][0]}/{config['sample_counts_by_modes'][str(m)][1]}" for m in modes)
    fig.text(.075, .13, "Modes: short/long sample counts  |  "+counts, fontsize=9.1)
    fig.text(.075, .075, f"{config['repetitions']} repetitions per method/input/batch; allocation, insertion, lookup and cold misses included.\n"
             "Ratios >1 favor caching. These descriptive comparisons do not establish equal-quality superiority to Frugal or quantum hardware.", fontsize=9.2, linespacing=1.5)
    save_figure(fig, output, "prefix_cache_speed", "Frozen native setup+sampling ratios; medians within each case/batch; no prewarming or quality equivalence to Frugal claimed.")
    plt.close(fig)


def plot_quality(output, quality):
    plt = plot_style()
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 6.5),
                             gridspec_kw={"width_ratios": [1.05, 2.1, 1.25]})
    fig.subplots_adjust(left=.06, right=.985, bottom=.25, top=.77, wspace=.30)
    rows, eps = quality["rows"], quality["epsilon"]
    full = [r for r in rows if r["synthetic"] and r["full_tv"] is not None and r["evidence_plottable"]]
    large = [r for r in rows if r["synthetic"] and r["modes"] > 12]
    jz = [r for r in rows if not r["synthetic"]]
    values = [r[f] for r in large+jz if r["evidence_plottable"]
              for f in ("original_witness_lower", "marginal_tv_lower") if r[f] is not None]
    bound_top = min(1.05, max([eps*1.5]+[v*1.22+.015 for v in values]))

    ax = axes[0]
    for i, r in enumerate(full):
        ax.scatter(i, r["full_tv"], marker="D", s=58, color=PROFILE_COLORS.get(r["profile"], "#475569"), zorder=4, clip_on=False)
        ax.annotate(fmt(r["full_tv"]), (i, r["full_tv"]), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9)
    ax.set(xticks=range(len(full)), xticklabels=[f"{r['modes']} modes\n{r['profile']}" for r in full],
           ylabel="Complete numerical TV", title="A  Full distribution enumerated",
           ylim=(-.004, max([eps*1.4]+[r["full_tv"]*1.4 for r in full])))
    ax.set_xlim(-.5, max(.5, len(full)-.5))
    if not full:
        ax.text(.5, .5, "No complete-law evidence available", transform=ax.transAxes, ha="center", wrap=True)

    ax = axes[1]
    modes = sorted({r["modes"] for r in large})
    index = {m: i for i, m in enumerate(modes)}
    for r in large:
        if not r["evidence_plottable"]:
            continue
        base = index[r["modes"]]+(-.17 if r["profile"] == "lossy" else .17)
        color = PROFILE_COLORS.get(r["profile"], "#475569")
        for field, shift, marker, filled in (("original_witness_lower", -.055, "o", False),
                                             ("marginal_tv_lower", .055, "^", True)):
            if r[field] is not None:
                ax.scatter(base+shift, r[field], marker=marker, s=49,
                           facecolors=color if filled else "none", edgecolors=color,
                           linewidths=1.1, zorder=4, clip_on=False)
    ax.set(xticks=range(len(modes)), xticklabels=modes, xlabel="Synthetic output modes",
           ylabel="TV lower bound — NOT actual TV", title="B  Large-system rejection witnesses", ylim=(-.004, bound_top))
    ax.set_xlim(-.5, max(.5, len(modes)-.5))
    legend = [Line2D([0], [0], color=PROFILE_COLORS[p], lw=2, label=p) for p in ("lossy", "bright")]
    legend += [Line2D([0], [0], marker="o", markerfacecolor="none", markeredgecolor="#475569", color="none", label="Original witness lower bound"),
               Line2D([0], [0], marker="^", markerfacecolor="#475569", color="none", label="Held-out marginal lower bound")]
    ax.legend(handles=legend, frameon=False, fontsize=8.8, loc="upper left", bbox_to_anchor=(0, -.15), ncol=2)

    ax = axes[2]
    for i, r in enumerate(jz):
        if not r["evidence_plottable"]:
            ax.text(i, bound_top*.45, r["sampling_status"], ha="center", fontsize=9)
            continue
        for field, shift, marker, filled in (("original_witness_lower", -.08, "o", False),
                                             ("marginal_tv_lower", .08, "^", True)):
            if r[field] is not None:
                color = "#7757A7"
                ax.scatter(i+shift, r[field], marker=marker, s=60,
                           facecolors=color if filled else "none", edgecolors=color,
                           linewidths=1.2, clip_on=False, zorder=4)
        label = "rejected" if r["decision"] == "rejected" else "inconclusive" if r["decision"] == "insufficient_evidence" else r["decision"]
        ax.text(i, bound_top*.94, label, ha="center", va="top", fontsize=8.8)
    ax.set(xticks=range(len(jz)), xticklabels=["Candidate" if r["method"] == "candidate" else "Official Frugal" for r in jz],
           ylabel="TV lower bound — NOT actual TV", title="C  Published JZ1 Gaussian model\n100 modes; author covariance", ylim=(-.004, bound_top))
    ax.set_xlim(-.5, max(.5, len(jz)-.5))
    for ax in axes:
        ax.axhline(eps, color="#9B4B4B", lw=1.1, linestyle="--", zorder=2)
        ax.grid(axis="y", alpha=.17)
    axes[0].annotate(f"Tolerance {eps:g}", xy=(.97, eps), xycoords=("axes fraction", "data"),
                     xytext=(0, 4), textcoords="offset points", ha="right", fontsize=8.7, color="#9B4B4B")
    fig.suptitle("Primary (Hoeffding) analysis: complete TV and rejection lower bounds", fontsize=17, fontweight="bold", y=.965)
    fig.text(.5, .88, f"Primary fixed quality batches · {quality['planned_comparisons']} planned comparisons · familywise α={quality['familywise_alpha']:g}",
             ha="center", fontsize=10.7)
    fig.text(.06, .04, f"B/C: primary Hoeffding lower bounds above TV={eps:g} reject; smaller bounds are inconclusive. Panel A is complete numerical TV.\n"
             "Secondary Clopper–Pearson (CP) refinement is reported separately; its decisions are not plotted here.\n"
             "Declared iid first-batch/split protocol and accurate floating-point targets are assumed; missing evidence is omitted, never set to zero.",
             fontsize=9.0, linespacing=1.5)
    save_figure(fig, output, "quality_evidence", "Primary (Hoeffding) analysis. Panel A: complete numerical TV. Panels B/C: one-sided statistical TV LOWER bounds only. Secondary Clopper-Pearson refinement is reported separately and its decisions are not plotted here. JZ1 author model is separate from synthetic targets and experimental hardware data.")
    plt.close(fig)


def run(args):
    performance = recompute_performance(args.performance)
    quality = analyze_quality(args.quality)
    small_laws, law_inputs = analyze_small_laws(args.small_laws, args.performance, performance)
    inputs = [args.performance/name for name in ("summary.json", "metrics.json", "config.json")]
    inputs += [args.quality/name for name in ("summary.json", "config.json", "comparison_plan.json")]
    for path in (args.performance/"metadata.json", args.quality/"evaluation_manifest.json"):
        if path.exists():
            inputs.append(path)
    inputs += law_inputs
    result = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_source_sha256": sha(Path(__file__)),
        "inputs_sha256": {relative(p): sha(p) for p in inputs},
        "performance": performance, "quality": quality,
        "supplementary_small_laws": small_laws,
        "scope": "Same-law prefix memoization cost, small-mode Frugal costs with optional supplementary deterministic complete-law review, and independently sampled primary target-quality evidence. No quantum speed ratio or full large-Jiuzhang acceptance is inferred.",
        "sampling_runs_performed_by_analysis": 0,
        "figures": {"prefix_cache_speed": ["prefix_cache_speed.png", "prefix_cache_speed.svg"],
                    "quality_evidence": ["quality_evidence.png", "quality_evidence.svg"]},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    make_tables(args.output, performance, quality, small_laws)
    plot_performance(args.output, performance)
    plot_quality(args.output, quality)
    write_json(args.output/"summary.json", result)
    print(json.dumps({"output": relative(args.output), "cold": performance["overall_cold"],
        "quality_decisions": quality["decisions"],
        "additional_rejections": quality["additional_rejections_after_marginal_channel"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--performance", type=Path, default=ROOT/"results/iteration5_performance")
    parser.add_argument("--quality", type=Path, default=ROOT/"results/iteration5_quality")
    parser.add_argument("--small-laws", type=Path, default=ROOT/"results/iteration5_small_laws")
    parser.add_argument("--output", type=Path, default=ROOT/"results/iteration5_analysis")
    run(parser.parse_args())
