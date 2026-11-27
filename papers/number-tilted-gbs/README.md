# Sampling a lossy Gaussian boson model with bounded full-record error

**qiyuanhuakai** · 27 November 2026

[Read the paper (PDF)](manuscript.pdf) · [Download the LaTeX source package](latex-source.zip) · [Matched 4,096-shot measurements](../../results/iteration115_panel_accuracy_v2/validation.json) · [Run protocol and input digest](../../results/iteration115_panel_accuracy_v2/protocol.json)

Revision 6 compares two complete 1,152-port readouts on the original high-power optical calibration. Retaining seven and five pairs in the two undisplaced residuals gives a 3.0914% full-record TV upper bound. Resolving the displacement environment gives a separate 2.330% upper confidence limit at 99%. Independent calculations on the historical 16-port stress panel give 0.0036391% complete click TV and a 0.00368%–0.1490% bracket for complete PNR TV.

[Full-width pilot and timings](../../results/iteration118_auxiliary/cap14_weak10_8.json) · [Total-count law](../../results/iteration118_auxiliary/law_7_5.json) · [Complete panel click law](../../results/iteration118_auxiliary/panel_7_5_28.json) · [Panel PNR bound](../../results/iteration118_auxiliary/panel_distance.json) · [Environment calculation](../../results/iteration118_auxiliary/environment_7_5.json) · [SHA-256 manifest](../../results/iteration118_auxiliary/manifest.json)

## Reproduce the higher-cutoff experiment

Run from the fixed archive root using the original high-power input described below. Python requires NumPy, SciPy, mpmath and threadpoolctl, in addition to earlier input preparation dependencies. The complete click enumeration compiles a C++17/OpenMP kernel with g++. The sampling driver limits BLAS to one thread per worker. The archive includes full output rows, per-record timings, the determinant coefficients recovered as probability tables, and an independent small-system amplitude calculation.

```sh
python experiments/iteration118_check.py
python experiments/iteration118_end_to_end.py
python experiments/iteration118_measure.py --pairs 5 --shots 128 --workers 4
python experiments/iteration118_measure.py --pairs 7 --weak-ten --shots 2 --workers 1
python experiments/iteration118_measure.py --pairs 7 --weak-ten --shots 8 --workers 4
python experiments/iteration118_laws.py 5 5
python experiments/iteration118_laws.py 7 5
python experiments/iteration118_environment.py 7 5 --rows 32768
python experiments/iteration118_environment_check.py
python experiments/iteration118_replay.py
python experiments/iteration118_panel.py
python experiments/iteration118_panel.py --nodes 32 --radius 0.71
python experiments/iteration118_panel_distance.py
```

Run both panel programs with `--banks 71 72` and `--banks 102 143` for the disjoint spatial checks. Run `iteration118_figure.py` to regenerate the full-record figure.

Run `python experiments/iteration118_verify.py` on an unchanged archive to verify all revision 6 digests and stored results.

A record index has independent streams for spectrum, readout, auxiliary proposals, displacement and detector draws. Repeating an index with a different worker count is a deterministic replay. The two-row pilot and the first two rows of the larger run share seeds. The prefix diagnostics were added after the first two runs.

## Correlated four-photon core

Revision 5 samples both readouts on all 1,152 physical detector ports, before any bank compression. The four-photon section gives the sequential contraction, its full-state error bound, and the effect of restoring first-order optical coherence. The original and adjusted models each produced 4,096 scored PNR and 4,096 directly sampled threshold rows at high power. The low and median power calibrations are separate official matrices and spreadsheets.

[Every scored high-power row](../../results/iteration117_quartic/scored_4096.npz) · [Coherence-restored rows](../../results/iteration117_quartic/transport_scored_4096.npz) · [Complete PNR bounds](../../results/iteration117_quartic/resolved_pnr.json) · [All three powers](../../results/iteration117_quartic/transport_laws.json) · [Independent oscillator check](../../results/iteration117_quartic/validation.json) · [Revision 5 checksums](https://github.com/qiyuanhuakai/deziroGBS/blob/adce0699a5fe5f4b6d162317d31165cfaaeae652/results/iteration117_quartic/manifest.json)

Run the following from the fixed archive root. The input script downloads six published files by byte range, checks their sizes and CRCs, and records their SHA-256 digests. It requires network access, Python with NumPy, SciPy, openpyxl, requests and the archived project dependencies, and an extraction tool compatible with the Windows tar command. The readout kernel rebuilds with a C++17/OpenMP compiler.

```sh
python experiments/iteration117_power_inputs.py
python experiments/iteration117_quartic.py --shots 4096 --batch 32
python experiments/iteration117_quartic.py --shots 4096 --batch 32 --transport
python experiments/iteration117_capture.py
python experiments/iteration117_quartic_laws.py
python experiments/iteration117_power_laws.py
python experiments/iteration117_power_laws.py --transport
python experiments/iteration117_quartic_panels.py 34 35
python experiments/iteration117_quartic_panels.py 34 35 --transport
python experiments/iteration117_resolved_pnr.py --cap 6
python experiments/iteration117_resolved_pnr.py --cap 6 --transport
python experiments/iteration117_verify.py
```

For each of `low`, `median` and `high`, pass the power name to `iteration117_bank_sweep.py`, followed by `iteration117_pair_sweep.py`. The first program enumerates 256 words for each of 144 banks. The second evaluates all 10,296 joint empty/occupied laws and connected empty-bank correlations. Use `iteration117_quartic.py --power low --shots 512 --batch 32` and the corresponding median command for the two additional record sets. Specify 71 72 or 102 143 for the two disjoint unmodified-model panel checks. `iteration117_figure.py` rebuilds Figure 3. A preliminary partition test recovered 8.389% of the squared off-diagonal remainder covariance with 72 matched pairs; the corresponding full-record TV upper bound was 99.368%. [Partition calculation](../../results/iteration117_blocks/partition.json)

## Reproduce the paper

The source package contains the manuscript, four vector figures, and the data-derived table and macros. Extract it and run `pdflatex manuscript.tex` twice. The checked-in PDF uses the same sources.

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
