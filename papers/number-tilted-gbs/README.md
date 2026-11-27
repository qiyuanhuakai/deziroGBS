# Full-record error bounds for classical sampling of lossy Gaussian boson models

**qiyuanhuakai** · 27 November 2026

[Read the paper (PDF)](manuscript.pdf) · [Download the LaTeX source package](latex-source.zip) · [Matched 4,096-shot measurements](../../results/iteration115_panel_accuracy_v2/validation.json) · [Run protocol and input digest](../../results/iteration115_panel_accuracy_v2/protocol.json)

## Revision 10

Section 2 proves a full-record TV bound for arbitrary output width and finite squeezing under an intensity transmission ceiling. The 50-source, 15-pair bound is analytical. The timed calibrated sampler in Table 2 uses nested lower-order truncations. Appendix A derives a width-linear sequential implementation at fixed rank and cutoff and a brightness-uniform local number instrument.

[Three-power input certificates and complete runs](../../results/iteration122_families) · [Exact rational family bounds](../../results/iteration122_families/families.json) · [Synthetic optical checks](../../results/iteration122_families/optical_checks.json) · [Sample checks](../../results/iteration122_families/sample_checks.json) · [SHA-256 manifest](../../results/iteration122_families/manifest.json)

Run from the repository root. The archive verifier requires NumPy and its experiment imports. The interval checks additionally use python-flint 0.9.0. The native examples use an OpenMP-enabled C++17 compiler. The clamped generator uses the CUDA and Python environment documented below.

~~~sh
python experiments/iteration122_verify.py
python experiments/iteration122_powers.py verify
python experiments/iteration122_family_certificate.py
~~~

The raw matrices and squeezing arrays for all three public powers are stored with the source file hashes. To regenerate in a separate results directory or scratch checkout, preserve the optical inputs and remove previously generated destination files there. Run the following for each new calibrated power (`low`, `median`) and each readout (`0` PNR, `1` direct clicks). The high-power delivery comes from the archived round-120 run.

~~~sh
python experiments/iteration122_powers.py base low 0
python experiments/iteration122_powers.py score low 0
python experiments/iteration122_dilation.py base 0
python experiments/iteration122_dilation.py score 0
~~~

For the fast four-photon law, an input NPZ contains a complex transfer `T` (visible outputs by occupied inputs) and real nonnegative `squeezing` (spectra by occupied inputs, phases absorbed into columns). Set a rational intensity ceiling at least as large as the squared operator norm. The following commands write fresh samples for the two archived synthetic inputs. The per-record timings cover generation of both spectra, their merger and production of the visible records, after setup. The generic rank-two reference, local CDF checks and explicit Fock tests have separate entry points.

~~~sh
python experiments/iteration122_loss.py sample --input results/iteration122_families/unequal_loss/input.npz --output replica-1152.npz --eta 7/100 --seed 122
python experiments/iteration122_loss.py sample --input results/iteration122_families/bright/input.npz --output replica-bright.npz --eta 7/100 --seed 123
python experiments/iteration122_memory.py
python experiments/iteration122_readout_check.py
python experiments/iteration122_tests.py
python experiments/iteration122_statistics.py
~~~

The following historical instructions refer to the [fixed revision 9](https://github.com/qiyuanhuakai/deziroGBS/tree/d1654c41d24bf417bd4c6200f49fb302520c1086).

## Revision 9

The full-width corrected positive-P comparison uses the same calibrated transfer, powers, detector ordering and two spectra as the clamped sampler. The archive contains the training specifications for both fits, record-generation timings, example full records, optical event checks and all eight confirmatory counts. The fitted maps are specified by implementation, training streams, numeric settings and per-layer hashes. The independent implementations of Goodman et al.'s equations are described in the paper.

[Paper](manuscript.pdf) · [LaTeX source](latex-source.zip) · [Full comparison](../../results/iteration121_positive_p/comparison.json) · [Checksums](../../results/iteration121_positive_p/manifest.json)

These checks use NumPy and SciPy. The optical checkers additionally require python-flint 0.9.0. The third verifies the optical endpoint of the 14-bank PNR witness. Replace `pnr` with `click` for directly generated clicks, or omit `--large` for the 12-bank events.

~~~sh
python experiments/iteration120_interval.py
python experiments/iteration121_verify.py
python experiments/iteration121_interval.py pnr --large
~~~

For full regeneration, use a separate checkout with an empty round-121 results directory. Keep the archived results for comparison. Experiment scripts use the round-120 calibration tables. The archived GPU run used CUDA-enabled PyTorch 2.12.1, Python 3.11, NumPy, SciPy and threadpoolctl. At each line run both routes, replacing ROUTE with `pnr` and then `click`. The local cache stores reproducible FP64 fitted matrices. The confirmatory record scripts use the frozen event files and save their hashes. The interval program recomputes all optical word probabilities for the selected banks.

~~~sh
python experiments/iteration121_goodman.py ROUTE
python experiments/iteration121_witness.py design ROUTE
python experiments/iteration121_witness.py test ROUTE
python experiments/iteration121_records.py ROUTE
python experiments/iteration121_precision.py ROUTE
python experiments/iteration121_transfer.py design ROUTE --cache rerun/model-cache
python experiments/iteration121_transfer.py test ROUTE --cache rerun/model-cache
python experiments/iteration121_interval.py ROUTE
python experiments/iteration121_interval.py ROUTE --large
~~~

After both routes are generated, run `python experiments/iteration121_fields_check.py --cache rerun/model-cache` to compare the CPU and GPU field calculations.

The class-wide PNR test uses a rational score recorded in `poisson_witness.json`. Run `python experiments/iteration121_poisson.py verify` with the research dependencies and python-flint. The result is a strict lower bound for every mixture of conditionally independent Poisson counts on the given optical input. Its magnitude is far smaller than the measured discrepancies of the tested generators.

Historical manifests include the PDF and source package at their respective tags. Use [the revision 8 archive](https://github.com/qiyuanhuakai/deziroGBS/tree/archive/iteration120-certified-readouts-v1) to validate the revision 8 manifest.

## Revision 8

On the archived Jiuzhang 3.0 high-power optical input, the complete PNR law and the complete direct-click law are each within **0.4815% TV** of their optical target in ideal arithmetic. The independent checker starts from the binary transfer and squeezing values. It encloses the photon-pair tails and verifies both four-dimensional matrix certificates using interval arithmetic. The sample generator uses floating point.

[Paper](manuscript.pdf) · [Source package](latex-source.zip) · [Certificate](../../results/iteration120_contractions/operator_certificate.json) · [Interval verification](../../results/iteration120_contractions/interval_optical.json) · [Runtime data](../../results/iteration120_contractions) · [Checksums](../../results/iteration120_contractions/manifest.json)

### Verify the principal claim on a CPU

From the archive root, use Python 3.11 with NumPy and python-flint 0.9.0. The first command recomputes the bound, independently of the optimizer, GPU, compiled sampler and hardware downloads. The second checks all deposited decisions, hashes and source-package contents.

~~~sh
python experiments/iteration120_interval.py
python experiments/iteration120_verify.py
~~~

### Generate samples and repeat the hardware checks

The two routes each used 256 fresh candidates. PNR accepted 70 at 9.18 seconds per delivered row and direct clicks accepted 69 at 11.93 seconds per row. For generation use the C++17/OpenMP compiler, scientific Python dependencies from revision 7, and CUDA-enabled PyTorch (the archived benchmark used a laptop RTX 5080). Use a separate checkout and a fresh output directory for generating timed arrays. The runners reject overwriting an existing sample batch. The raw acquisition data remain in the official release. Predictions and raw file digests are deposited here.

~~~sh
python experiments/iteration120_replay.py --workers 2 --shots 64
python experiments/iteration120_generate.py base 0 --workers 10 --output rerun/iteration120
python experiments/iteration120_generate.py score 0 --workers 2 --output rerun/iteration120
python experiments/iteration120_generate.py base 1 --workers 10 --output rerun/iteration120
python experiments/iteration120_generate.py score 1 --workers 2 --output rerun/iteration120
python experiments/iteration120_calibration.py evaluate
python experiments/iteration120_response.py evaluate
python experiments/iteration120_readout.py
python experiments/iteration120_panel_certificate.py
python experiments/iteration120_panel_laws.py
python experiments/iteration120_hardware_bounds.py
~~~

The calibration and response predictions were separately frozen before captures 14–15 and 16. The gain-only model improved eight-bank occupancy TV and worsened conditional within-bank multiplicities. The Poisson and empty-bank refill models agree on every bank-occupancy event, and differ on detector multiplicities. The three panel confidence bounds refer to the joint multiplicities of two eight-port banks. To reconstruct both frozen prediction tables, use the corresponding freeze commands after restoring training capture 10 with the round-112 downloader. To propose new numerical dual matrices use iteration120_operator.py with Clarabel; the proof checker verifies the deposited matrices directly.

[First frozen test](https://github.com/qiyuanhuakai/deziroGBS/tree/321db37fc23a43e9d1f7f8a037e03fbc68b362ae/results/iteration120_contractions) · [Second frozen test](https://github.com/qiyuanhuakai/deziroGBS/tree/b935454181ef17a42962e78b226d367b8ec07b16/results/iteration120_contractions) · [Capture 16 comparison](../../results/iteration120_contractions/response_holdout.json) · [Panel confidence results](../../results/iteration120_contractions/hardware_panel_bounds.json)

## Reproduce revision 7

Use the [fixed revision 7 archive](https://github.com/qiyuanhuakai/deziroGBS/tree/d467fa25f2e5277e78bfe1cd1a9d2d14f925f343) when checking the historical manifest and its paper.

For either complete readout of the published high-power optical model, the layered rejection sampler gives a numerical TV upper bound of 0.4824% in ideal arithmetic. A separate event fixed using hardware file 10 gives simultaneous 99% TV lower limits of 3.766% and 3.813% between the same calibration and the two held-out averaged conditional click laws. See the paper for the sampling assumptions and measured throughput.

[Accepted rows](../../results/iteration119_weights/clamped_bands_64.npz) · [All acceptance decisions](../../results/iteration119_weights/clamped_bands_64.json) · [Held-out counts](../../results/iteration119_weights/hardware_holdout.json) · [Frozen protocol](https://github.com/qiyuanhuakai/deziroGBS/tree/bfd6320687e6a193fe4d7194c81f79e3d4688d3d/results/iteration119_weights) · [Independent checks](../../results/iteration119_weights/audit.json) · [Revision 7 checksums](https://github.com/qiyuanhuakai/deziroGBS/blob/d467fa25f2e5277e78bfe1cd1a9d2d14f925f343/results/iteration119_weights/manifest.json)

Run from the archive root, with the original round-115 optical input. The verification command checks the deposited artifacts. For new measurements, use a separate checkout with a C++17/OpenMP compiler, Python, NumPy, SciPy, threadpoolctl and Matplotlib. The hardware evaluation downloads the two source members for the byte-level audit.

~~~sh
python experiments/iteration119_verify.py
python experiments/iteration119_check.py
python experiments/iteration119_benchmark.py --start 4096 --shots 4096 --cap 7 --workers 6
python experiments/iteration119_fast.py
python experiments/iteration119_mixture.py --shots 64 --workers 5 --batch 32
python experiments/iteration119_clamps.py --shots 64 --workers 7
python experiments/iteration119_mixture.py --shots 8 --workers 5 --batch 16
python experiments/iteration119_timing.py
python experiments/iteration119_hardware.py evaluate
python experiments/iteration119_audit.py
python experiments/iteration119_frugal.py
python experiments/iteration119_powers.py
python experiments/iteration119_figure.py
~~~

The likelihood comparison used 4,096 four-photon candidates per route. A separate band sampler generated 64 higher-order candidates per route. The final acceptances are a subset of those 64. To reproduce the eight-row sequential pilot in a clean output directory, run the draw and score stages of iteration119_subpercent.py, then iteration119_clamps.py with --pilot. Run iteration119_optical.py for the unclipped nine/six-pair state.

## Reproduce the revision 6 experiment

Revision 6 compares two complete 1,152-port readouts on the original high-power optical calibration. Retaining seven and five pairs in the two undisplaced residuals gives a 3.0914% full-record TV upper bound. Resolving the displacement environment gives a separate 2.330% upper confidence limit at 99%. Independent calculations on the historical 16-port stress panel give 0.0036391% complete click TV and a 0.00368%–0.1490% bracket for complete PNR TV.

[Full-width pilot and timings](../../results/iteration118_auxiliary/cap14_weak10_8.json) · [Total-count law](../../results/iteration118_auxiliary/law_7_5.json) · [Complete panel click law](../../results/iteration118_auxiliary/panel_7_5_28.json) · [Panel PNR bound](../../results/iteration118_auxiliary/panel_distance.json) · [Environment calculation](../../results/iteration118_auxiliary/environment_7_5.json) · [SHA-256 manifest](https://github.com/qiyuanhuakai/deziroGBS/blob/ce56b3677fc1ec90ab2c581240a2f8d6c0dc8d45/results/iteration118_auxiliary/manifest.json)

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
