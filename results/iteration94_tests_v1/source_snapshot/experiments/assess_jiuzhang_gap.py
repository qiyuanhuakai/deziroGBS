#!/usr/bin/env python3
"""Audit a normalized run manifest, or demonstrate with archived synthetic m64."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gbslab.jiuzhang_gap import assess_jiuzhang_gap, format_gap_markdown, load_reference_catalog


def archived_m64_manifest():
    path = ROOT / "results/cpp_large/records.jsonl"
    for line in path.read_text().splitlines():
        record = json.loads(line)
        metrics = record["metrics"]
        if (metrics.get("case_id") == "m64_seed7201"
                and metrics.get("method") == "cpp_click_influence" and metrics.get("repeat") == 0):
            native = record["native_result"]
            histogram = native["click_count_histogram"]
            count = metrics["samples"]
            return {
                "run_id": "archived_synthetic_m64_seed7201_cpp_click_influence_k4_repeat0",
                "status": "ok", "synthetic": True, "instance_id": None,
                "source_archive": str(path.relative_to(ROOT)),
                "source_archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "modes": 64, "output_bins": 64, "source_count": 64,
                "detector_model": "threshold", "output_encoding": "binary_threshold_bins",
                "covariance_sha256": native["covariance_sha256"],
                "noise_features": ["uniform_loss"], "topology": "haar",
                "sampling": {"count_requested": count, "count_observed": count,
                             "iid": True, "no_retries": True, "postselection": "none"},
                "click_statistics": {"observed_mean": sum(i*n for i,n in enumerate(histogram))/count,
                                     "observed_maximum": max(i for i,n in enumerate(histogram) if n),
                                     "sample_size": count, "source": "archived observed histogram; not target-law mean"},
                "timings": {"setup_seconds": metrics["setup_seconds"],
                            "sampling_seconds": metrics["sampling_seconds"],
                            "wall_seconds": metrics["harness_call_seconds"],
                            "scope": "complete wrapper call excluding compiler and external reference-quality computation"},
                "resources": {"cpu": "see archived cpp_large metadata", "threads": native["threads"],
                              "gpus": 0, "peak_memory_bytes": None},
                "implementation": {"maximum_modes": 256},
                "native_provenance": {key: native.get(key) for key in
                    ["executable_sha256", "selector", "k", "order", "covariance_sha256"]}
            }
    raise ValueError("Required archived m64 record was not found")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="Normalized JSON run manifest; omit for archived m64 demo")
    parser.add_argument("--evidence", type=Path, help="JSON from the GBS identity discriminator")
    parser.add_argument("--reference", default="jiuzhang4_l1024")
    parser.add_argument("--target-kind", default="published_gaussian_model",
                        choices=["published_gaussian_model", "empirical_hardware_samples"])
    parser.add_argument("--epsilon", type=float, default=.1)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT/"results/iteration4_jiuzhang")
    args = parser.parse_args()
    run = json.loads(args.run.read_text()) if args.run else archived_m64_manifest()
    evidence = json.loads(args.evidence.read_text()) if args.evidence else None
    catalog = load_reference_catalog(args.catalog)
    report = assess_jiuzhang_gap(run, evidence, reference_id=args.reference,
                                target_kind=args.target_kind, epsilon=args.epsilon, catalog=catalog)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"run_manifest.json").write_text(json.dumps(run, indent=2, allow_nan=False)+"\n")
    (args.output/"gap_report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    (args.output/"gap_report.md").write_text(format_gap_markdown(report))
    print(json.dumps({"verdict": report["verdict"], "reference_id": report["reference_id"],
                      "failed": [k for k,g in report["gates"].items() if g["status"] == "fail"],
                      "unknown": [k for k,g in report["gates"].items() if g["status"] == "unknown"],
                      "speed_ratio": report["timing"]["classical_wall_over_reference_wall"],
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
