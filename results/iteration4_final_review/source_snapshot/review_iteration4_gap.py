"""Re-assess saved evidence with the final D2 ingestion fixes; never resample."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import shutil

from gbslab.jiuzhang_gap import assess_jiuzhang_gap, load_reference_catalog

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def run(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a new audit directory")
    for source in (args.synthetic, args.actual):
        if json.loads((source/"metadata.json").read_text()).get("status") != "complete":
            raise RuntimeError("Only completed, saved experiment families can be reviewed")
    args.output.mkdir(parents=True)
    snapshot = args.output/"source_snapshot"
    snapshot.mkdir()
    for src in [Path(__file__), ROOT/"src/gbslab/jiuzhang_gap.py", ROOT/"configs/jiuzhang_reference.json"]:
        shutil.copy2(src, snapshot/src.name)
    catalog = load_reference_catalog()
    records = []
    schedules = [(args.synthetic, "synthetic", ("jiuzhang1", "jiuzhang4_l1024")),
                 (args.actual, "actual_jiuzhang1", ("jiuzhang1", "jiuzhang2", "jiuzhang3", "jiuzhang4_l1024"))]
    for source, label, references in schedules:
        for manifest_path in sorted(source.glob("*/run_manifest.json")):
            folder = manifest_path.parent
            run_record = json.loads(manifest_path.read_text())
            evidence_path = folder/"identity.json"
            evidence = json.loads(evidence_path.read_text())
            out = args.output/label/folder.name
            out.mkdir(parents=True)
            for reference in references:
                previous_name = "gap_jiuzhang4.json" if label == "synthetic" and reference == "jiuzhang4_l1024" else f"gap_{reference}.json"
                previous_path = folder/previous_name
                previous = json.loads(previous_path.read_text())
                current = assess_jiuzhang_gap(run_record, evidence, reference_id=reference,
                    epsilon=evidence["epsilon"], target_kind="published_gaussian_model", catalog=catalog)
                save(out/f"{reference}.json", current)
                changes = {key: {"before": previous["gates"][key]["status"], "after": value["status"]}
                           for key, value in current["gates"].items()
                           if value["status"] != previous["gates"][key]["status"]}
                records.append({"family": label, "case": folder.name, "reference": reference,
                    "run_manifest_sha256": digest(manifest_path), "identity_evidence_sha256": digest(evidence_path),
                    "original_gap_sha256": digest(previous_path),
                    "original_gap_path": str(previous_path.relative_to(ROOT)),
                    "verdict_before": previous["verdict"], "verdict_after": current["verdict"],
                    "local_quality_before": previous.get("local_distribution_quality", {}).get("status"),
                    "local_quality_after": current["local_distribution_quality"]["status"],
                    "changed_gate_statuses": changes,
                    "speed_ratio_before": previous["timing"]["classical_wall_over_reference_wall"],
                    "speed_ratio_after": current["timing"]["classical_wall_over_reference_wall"]})
    summary = {"status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "Independent review fixed five malformed-metadata ingestion gaps; original reports retained",
        "new_sampling_calls": 0, "source_hashes": {p.name: digest(p) for p in snapshot.iterdir()},
        "reports": len(records), "verdict_changes": sum(r["verdict_before"] != r["verdict_after"] for r in records),
        "status_change_reports": sum(bool(r["changed_gate_statuses"]) for r in records),
        "all_quantum_speed_ratios_absent": all(r["speed_ratio_after"] is None for r in records), "rows": records}
    save(args.output/"summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", type=Path, default=ROOT/"results/iteration4_quality")
    parser.add_argument("--actual", type=Path, default=ROOT/"results/iteration4_jiuzhang1_control")
    parser.add_argument("--output", type=Path, default=ROOT/"results/iteration4_final_review")
    run(parser.parse_args())
