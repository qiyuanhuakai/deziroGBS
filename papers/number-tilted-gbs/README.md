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

[Optical probabilities and sampler histograms](../../results/iteration115_panel_v1/panel.npz) · [Saved PNR and direct-click rows](../../results/iteration115_panel_accuracy_v2/scored_readouts.npz) · [Two-bank optical and model laws](../../results/iteration115_panel_accuracy_v2/pnr_laws.npz) · [First paper-version checksums](https://github.com/qiyuanhuakai/deziroGBS/blob/25d53393e73ffb3b80d9a0628683b8a8f1428705/results/iteration115_panel_accuracy_v2/manifest.json)

## Corrected positive-P comparison and fixed-number test

Revision 3 compares the trained positive-P adaptation on the same 16-detector panel and separately tests a correlated Gaussian projection across six bank pairs. Its whole-circuit bound applies to the projected state; the fast sampler has its own throughput and error results. Run these commands from the archive root. They use the same optical file as the original study.

```sh
python experiments/iteration115_projection_review_v3.py
python experiments/iteration115_goodman_review_v3.py
python experiments/iteration115_review_certificates_v3.py
```

[Results and numerical bounds](../../results/iteration115_review_v3/validation.json) · [Raw conditional distributions](../../results/iteration115_review_v3) · [Revision checksums](../../results/iteration115_review_v3/manifest.json) · [Goodman et al.](https://arxiv.org/abs/2604.12330)

## Spatial calibration and the limit of total-only reweighting

Revision 4 uses the original high-power transfer and the first four bank pairs in the revision-3 panel list. The full PNR lower bound applies to all total-only reweightings of the stated base. The joint-statistic upper endpoints apply to the eight-bank word paired with the global count. The spatially filtered sampler has its own full-record timings and panel errors in Section 9.

```sh
python experiments/iteration116_sector_floor.py
python experiments/iteration116_word_filter.py
python experiments/iteration116_panel_transfer.py
```

The first command computes the witness and a feasible competing distribution. The second fits unconditional spatial filters and benchmarks both readouts. The third independently reconstructs the complete stress and disjoint-panel click laws and bounds their PNR errors. The archive includes the per-batch timings, optical and model probability arrays, and examples of full records. To reproduce the separate pure-state bond-rank calculation, run `python experiments/iteration116_schmidt.py results/iteration116_schmidt/spectrum.json`. At the central cut the best rank-65,536 approximation to the stronger pure remainder has trace distance 5.9905%. That calculation concerns one chosen pure-state decomposition before measurement.

[Sector bound](../../results/iteration116_sectors/floor.json) · [Filters and timings](../../results/iteration116_sectors/filtered.json) · [Complete panel comparisons](../../results/iteration116_sectors/panel_transfer.json) · [Checksums](../../results/iteration116_sectors/manifest.json)
