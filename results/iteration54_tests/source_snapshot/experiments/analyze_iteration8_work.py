"""Post-panel descriptive work attribution; no reruns or parameter changes."""
import json
from pathlib import Path
from iteration8_common import ROOT, save, sha, utc

def main():
    result = {"created_utc": utc(), "posthoc_descriptive_only": True,
              "source_sha256": sha(Path(__file__)), "panels": {}, "input_bindings": {}}
    for panel in ("quality", "regression", "timing"):
        root = ROOT / ("results/iteration8_" + panel)
        path = root / "metrics.json"
        rows = json.loads(path.read_text())
        result["input_bindings"][str(path.relative_to(ROOT))] = sha(path)
        records = []
        for row in rows:
            if row["method"].startswith("retained"):
                continue
            if panel == "timing":
                records.append((row["kernel_diagnostics"], row["timings_seconds"]["cold_batch"]))
            else:
                native_path = root / row["case_id"] / (row["method"] + "_native.json")
                native = json.loads(native_path.read_text())
                result["input_bindings"][str(native_path.relative_to(ROOT))] = sha(native_path)
                records.append((native["enumeration_kernel_diagnostics"], None))
        total = {k: sum(d[k] for d, _ in records) for k in
                 ("probability_calls", "auto_refinements", "auto_refinement_successes", "auto_refinement_failures")}
        if panel == "timing":
            total["sum_cold_seconds"] = sum(t for _, t in records)
            for k in ("query_seconds", "state_seconds", "event_seconds", "output_seconds", "tracked_add_sub"):
                total["sum_longdouble_" + k] = sum(d["longdouble"][k] for d, _ in records)
            total["raw_batch_query_fraction_of_cold"] = total["sum_longdouble_query_seconds"] / total["sum_cold_seconds"]
            total["attribution_scope"] = "Raw event query includes state and event/output work; shared log scoring is outside this timer. Counters are not all machine instructions. No controlled ablation isolates the cost of indicators, LD arithmetic, or reuse."
        result["panels"][panel] = total
    save(ROOT / "results/iteration8_analysis/work_audit.json", result)
    print(json.dumps(result["panels"], indent=2))

if __name__ == "__main__":
    main()
