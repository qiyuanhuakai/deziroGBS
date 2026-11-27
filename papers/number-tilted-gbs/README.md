# Photon-number calibration of a dual-readout Gaussian boson sampler

**qiyuanhuakai** · 27 November 2026

[Read the paper (PDF)](manuscript.pdf) · [Download the LaTeX source package](latex-source.zip) · [Matched 4,096-shot measurements](../../results/iteration115_panel_accuracy_v2/validation.json) · [Run protocol and input digest](../../results/iteration115_panel_accuracy_v2/protocol.json)

The paper tests one physical approximation with two readouts. It distinguishes total-photon accuracy on the high-power optical model, full PNR and threshold accuracy on a 16-detector panel, and discrepancies from measured Jiuzhang 3.0 clicks.

## Reproduce the paper

The source package contains the manuscript, two vector figures, and the data-derived table and macros. Extract it and run `pdflatex manuscript.tex` twice. The checked-in PDF uses the same sources.

The following commands run from the archive root with the Windows, CUDA, and upstream-tool environment of the [round-115 checkpoint](../../research/iteration115_checkpoint_v1.md). The scripts expect the fixed optical input and earlier stage outputs from that archive. Run in a separate checkout to preserve the original timings.

```sh
python experiments/iteration115_panel_v1.py results/iteration115_panel_v1 --shots 4096 --block 128
python experiments/iteration115_panel_accuracy_v2.py results/iteration115_panel_accuracy_v2
python experiments/iteration115_panel_accuracy_v2.py results/iteration115_panel_accuracy_v2 --complete
python experiments/iteration115_panel_accuracy_v2.py results/iteration115_panel_accuracy_v2 --audit-clicks
python experiments/iteration115_paper_v2.py
```

The first command measures both GPU readouts and both photonic-engines backends, including rare high-photon trajectories. The next two compute the PNR bounds, verify click and PNR histograms, recover the 4,096 scored surrogate rows by deterministic replay, benchmark FrugalGBS, and estimate finite-sample fluctuations. The last command regenerates both figures and the table. The manuscript describes the role of Codex in the research, analysis and writing.

[Optical probabilities and sampler histograms](../../results/iteration115_panel_v1/panel.npz) · [Saved PNR and direct-click rows](../../results/iteration115_panel_accuracy_v2/scored_readouts.npz) · [Two-bank optical and model laws](../../results/iteration115_panel_accuracy_v2/pnr_laws.npz) · [Supplement checksums](../../results/iteration115_panel_accuracy_v2/manifest.json)
