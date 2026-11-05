"""Presentation-only QA adjustment; never rerun analysis, integrals or samples.

Keep the first frozen-analyzer figures.  Rerender their unchanged summary data
through the frozen plotting function, suppress cluttered log minor-tick labels,
and identify the unavailable m=32 retained-k6 curve.  Scientific source files
and summary/statistical values are not edited.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

from matplotlib.ticker import NullFormatter


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import iteration10_analyze as frozen


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    record_path = HERE / "layout_adjustment.json"
    if record_path.exists():
        raise FileExistsError("Presentation adjustment was already archived")
    summary_path = HERE / "summary.json"
    summary_hash = digest(summary_path)
    summary = json.loads(summary_path.read_text())
    source_path = ROOT / "experiments/iteration10_analyze.py"
    if digest(source_path) != summary["analysis_source_sha256"]:
        raise ValueError("Frozen plotting source changed")
    config_path = ROOT / "results/iteration10_quality/config.json"
    if digest(config_path) != summary["config_sha256"]:
        raise ValueError("Frozen experiment configuration changed")
    config = json.loads(config_path.read_text())
    backup = HERE / "initial_render"
    backup.mkdir(exist_ok=False)
    originals = {}
    for name in ("quality_cost.png", "quality_cost.svg"):
        originals[name] = digest(HERE / name)
        shutil.copy2(HERE / name, backup / name)
    figures = []
    close = frozen.plt.close
    frozen.plt.close = lambda figure: figures.append(figure)
    try:
        with tempfile.TemporaryDirectory(prefix="iteration10-plot-layout-") as temporary:
            frozen.plot_quality_cost(Path(temporary), summary["quality_rows_detail"],
                                     summary["timing_aggregates"], config)
    finally:
        frozen.plt.close = close
    if len(figures) != 1 or len(figures[0].axes) != 5:
        raise ValueError("Unexpected frozen plot layout")
    figure = figures[0]
    for axis in figure.axes[2:]:
        axis.xaxis.set_minor_formatter(NullFormatter())
    large_axis = figure.axes[-1]
    handles, labels = large_axis.get_legend_handles_labels()
    labels = ["Retained k=6: failed" if label == "Retained k=6" else label for label in labels]
    large_axis.legend(handles, labels, fontsize=8, loc="best", framealpha=.9)
    large_axis.text(.03, .94, "Retained k=6: 0/3 completed at each N",
        transform=large_axis.transAxes, va="top", fontsize=8, color="#9b2f25")
    for extension in ("png", "svg"):
        figure.savefig(HERE / ("quality_cost." + extension), dpi=180, facecolor="white")
    close(figure)
    if digest(summary_path) != summary_hash:
        raise ValueError("Presentation step changed scientific summary")
    record = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "presentation only; suppress minor tick labels and label failed curve",
        "analysis_rerun": False, "native_or_reference_experiments_rerun": False,
        "summary_sha256_unchanged": summary_hash,
        "frozen_analysis_source_sha256": digest(source_path),
        "layout_script_sha256": digest(__file__), "initial_figure_hashes": originals,
        "final_figure_hashes": {name: digest(HERE/name) for name in originals}}
    record_path.write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
