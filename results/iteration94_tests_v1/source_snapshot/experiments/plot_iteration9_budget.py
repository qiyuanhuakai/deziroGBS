"""Plot the frozen analytic core-budget rows; no new numerical experiment."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from iteration8_common import ROOT, save, sha

def main():
    source = ROOT / "results/iteration9_core_budget/budget_rows.json"
    rows = json.loads(source.read_text())
    output = ROOT / "results/iteration9_visualization"
    output.mkdir(exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = {"lossy": "#087f8c", "bright": "#bc6500", "stress": "#a83357"}
    for profile, color in colors.items():
        group = sorted((r for r in rows if r["profile"] == profile and r["epsilon_tv"] == .1), key=lambda r: r["modes"])
        x = [r["modes"] for r in group]
        for field, style, alpha in (("core", "-", 1), ("unattenuated_pure_input_reference", "--", .55)):
            axes[0].plot(x, [r[field]["cutoff_photons"] for r in group], style, color=color, alpha=alpha, marker="o", ms=3,
                         label=profile + (" core" if field == "core" else " pure input"))
            axes[1].plot(x, [r[field]["log10_even_core_dimension"] for r in group], style, color=color, alpha=alpha, marker="o", ms=3)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks([8, 32, 128, 256, 1024], ["8", "32", "128", "256", "1024"])
        ax.set_xlabel("Optical modes m")
        ax.grid(alpha=.2)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Minimum even photon cutoff K")
    axes[0].set_title("A  Photon cutoff decreases after loss decomposition", loc="left", fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].set_ylabel("log10 of dense even-sector dimension D")
    axes[1].set_title("B  The remaining dense representation is still enormous", loc="left", fontsize=11, fontweight="bold")
    fig.suptitle("A smaller quantum core is useful, but a dense Fock table is insufficient", fontsize=16, fontweight="bold", y=.98)
    fig.subplots_adjust(left=.07, right=.98, bottom=.21, top=.84, wspace=.26)
    fig.text(.07, .09, "Solid: pure core after uniform-loss decomposition. Dashed: unattenuated pure input, not the measured mixed output.", fontsize=10)
    fig.text(.07, .043, "TV budget 0.1 uses tail <= 0.01. Analytic resource calculation only; D is neither MPS bond dimension nor a sampling lower bound.", fontsize=9)
    for extension in ("png", "svg"):
        fig.savefig(output / ("core_budget." + extension), dpi=180, facecolor="white")
    plt.close(fig)
    save(output / "provenance.json", {"source": str(source.relative_to(ROOT)), "data_sha256": sha(source),
        "plotter_sha256": sha(Path(__file__)), "input_panel_changed": False})

if __name__ == "__main__":
    main()
