# Iteration 4: an evidence gate for distance to Jiuzhang

Checked against primary sources on **2026-11-04**. This document accompanies
`src/gbslab/jiuzhang_gap.py`, `configs/jiuzhang_reference.json` and the CLI
`experiments/assess_jiuzhang_gap.py`.

## What the discriminator decides

The default frontier is **Jiuzhang 4.0 L1024**, published in Nature on
13 May 2026. Earlier generations remain useful, explicitly named bridge targets.
The checker does not turn mode count, a maximum-click event, a passed witness,
or a short synthetic timing into an experimental reproduction claim.

Two targets are kept separate:

- `published_gaussian_model`: the distribution of a particular registered
  covariance/calibration and detector map. Agreement with this target is a
  precise mathematical simulation claim, conditional on numerical validation.
- `empirical_hardware_samples`: the underlying distribution produced by a
  hardware run. Its measured samples, calibration, selection rules and drift
  matter. A covariance distributed as the intended ground truth does not
  establish that the hardware follows that covariance exactly.

Each report has independent gates for instance identity, complete output extent,
measurement alphabet, modeled imperfections, implementation capacity,
distribution quality, sample protocol, and timing/resource comparability.
Statuses are `pass`, `fail`, or `unknown`; unknown is preserved as an evidence
gap. There is **no scalar readiness percentage**. The checker is an evidence
auditor, not an additional statistical oracle: it trusts the curated catalog and
the validation records supplied to it.

## Experimental reference definitions

| Reference | Squeezed input states | Main optical output modes | Binary detection bins | Hardware detector channels | Reported maximum events |
|---|---:|---:|---:|---:|---:|
| Jiuzhang 1.0 | 50 | 100 | 100 | 100 | 76 |
| Jiuzhang 2.0 | 50 | 144 | 144 | 144 | 113 |
| Jiuzhang 3.0 | 50, from 25 two-mode pairs | 144 | 1,152 | 144 | 255 |
| Jiuzhang 4.0 S64 | 64 | 4,336 | 4,336 | 16 | not cataloged |
| Jiuzhang 4.0 M256 | 256 | 5,104 | 5,104 | 16 | not cataloged |
| Jiuzhang 4.0 L1024 | 1,024 | 8,176 | 8,176 | 16 | 3,050 |

Sources: [Jiuzhang1 author paper](https://arxiv.org/pdf/2012.01625),
[Jiuzhang2 author paper](https://arxiv.org/pdf/2106.15534),
[Jiuzhang3 author paper](https://arxiv.org/pdf/2304.12240),
[Jiuzhang4 published paper](https://www.nature.com/articles/s41586-026-10523-6),
[Jiuzhang4 author preprint](https://arxiv.org/html/2508.09092v3).

These are detection events, with the measurement rule shown in the catalog;
they are not all true photon-number-resolving counts. Jiuzhang3 fans each main
output into eight binary bins. It can be modeled with the complete 1,152-bit
record followed by explicit grouping, or its corresponding calibrated grouped
count distribution. An ordinary 144-bit threshold output discards this
information. Jiuzhang4 reuses 16 SNSPDs over time; its 8,176 output positions
are spatial/temporal modes. Its circuit has three 16-mode interferometers and
delay arrays, rather than a generic Haar matrix of the same dimension.

The sources give overall efficiency 43% for Jiuzhang3 and 51% for Jiuzhang4.
Their detailed target models include partial distinguishability. These are
setting-dependent calibrated models; an arbitrary uniform-loss input is a
different instance even if its average photon count is similar.

The catalog records mean 43 clicks only for Jiuzhang1's cited high-power
setting. It does not replace missing means with reported maxima or extrapolate
the hardness of a rare event to typical samples. It also keeps pump repetition
rates, selected-event collection rates and quoted sample intervals separate.

## Data access and the usable bridge found in this iteration

The official landing pages are:

- [Jiuzhang1](https://quantum.ustc.edu.cn/web/en/node/915): samples, complex
  transfer matrix and squeezing information. The matrix already includes
  input phases and efficiency; loss must not be applied twice. Its H/V row
  convention must be corrected before reconstruction.
- [Jiuzhang2](https://quantum.ustc.edu.cn/web/node/951): four archive parts and
  an explicit recommendation of low-click instances for accessible classical
  checks. The row-order correction is documented here too.
- [Jiuzhang3](https://quantum.ustc.edu.cn/index.php/en/node/1121): ten archive
  parts and a README for approximately 135 million samples.
- [Jiuzhang4](https://quantum.ustc.edu.cn/node/1227): links to a
  [public shared folder](https://pan.ustc.edu.cn/share/index/eab75b7ad8fb46d8bb4c).
  The landing page was read; the shared-folder contents were not accessible
  through the browsing tool. No covariance/sample filenames, hashes, license,
  or successful download are invented for that folder.

No explicit data license was visible on these observed landing pages. The
large original archives were not imported in this iteration.

A smaller usable artifact was found in the
[polyquantique covariance repository](https://github.com/polyquantique/cov_matrix_optimization).
Its README identifies `cov_0.npy` as the **unoptimized Jiuzhang1 ground-truth
covariance**. The optimization example reads that file and writes a distinct
`opt_cov_0.npy`; the original decomposition script writes `sq_cov.npy`. Thus
the imported file is not silently substituted by the easier optimized pure
component.

The 320,128-byte file was read through the GitHub connector and placed in the
local experiment cache. Repository commit:
`b1bb0949aeb40de71e55e292489ba83b45be1f7b`.

| Identity/check | Result |
|---|---|
| Original file SHA-256 | `892a143db06952cd939bf080dc3321894108630b01a89c74751e85ca5a9531d5` |
| Git blob SHA | `72c4d387461a1dc79da4c7a9504a5de5e31dfb93` |
| Shape and dtype | 200 by 200, float64 |
| Raw array byte SHA-256 | `d674b2651c9c82f1316d5b48443a6090d27d9bec2ba2cc569979183558a65452` |
| Canonical wrapper covariance SHA-256 | `2af0a953bec2d0a42ddd7ef98baeaa256f44525d3a6b94cb0576ec80bebc31ae` |
| Maximum symmetry difference | approximately 1.55e-15 |
| Minimum eigenvalue of V+iOmega | approximately -3.87e-15 in xxpp, hbar=2 |
| Mean photons under that convention | approximately 78.6155 |

The canonical hash uses contiguous float64 bytes after `V*0.5 + V.T*0.5`,
matching the native wrapper. The source scripts use the same symplectic
convention. These numerical checks support input consistency; they do not
reconstruct the experimental optical calibration or certify hardware fidelity.

The catalog registers `instance_id="jiuzhang1_author_cov0"`. Matching this
exact model can now pass the **published-model instance gate**, while the
hardware-data and complete distribution-quality gates remain independent.
Source hashes, access observations and checks are archived in
`results/iteration4_jiuzhang_data/provenance.json`. The source repository's root
listing and README did not expose an explicit license. The linked original
[Zenodo record](https://zenodo.org/records/7992736) could not be read. These are
access observations, not legal conclusions; the binary was not publicly
redistributed by this work.

## API and input contract

```python
from gbslab.jiuzhang_gap import assess_jiuzhang_gap

report = assess_jiuzhang_gap(
    run_manifest,
    identity_evidence,
    reference_id="jiuzhang1",  # default: jiuzhang4_l1024
    target_kind="published_gaussian_model",
    epsilon=0.1,
)
```

A run manifest records `run_id`, completion `status`, `synthetic`,
`instance_id`, `covariance_sha256`, `modes`, `output_bins`, `source_count`,
`detector_model`, `output_encoding`, `noise_features` and
`implementation.maximum_modes`. Nested `sampling` records requested/observed
counts, IID/no-retry attestations and postselection. Nested `timings` records
setup, sampling, complete wall time and its scope. Nested `resources` records
CPU, threads, GPU count and measured peak memory; a missing memory measurement
stays unknown. Setup and sampling are nonoverlapping stages contained in the
complete wall interval. The checker rejects a wall time shorter than their
sum, allowing only a timer tolerance of `max(1e-6, 1e-9 * stage_sum)` seconds.
Timing scope must be a nonempty string, and every sample count must be an
integer rather than a boolean or a numerically equal floating-point value.

Copy `native_provenance` from the actual native output and from the identity
evidence. The gate binds quality to the **same binary, selector, k, order,
numerical kernel and covariance**. Held-out validation seeds and sample counts
may differ from timing samples. They may not silently use another sampler rule.

Local validation is retained in `local_distribution_quality`. The reference
`distribution_quality` gate stays unknown until instance, extent, detector and
imperfection gates bind the same target. A synthetic acceptance therefore
never appears as accepted Jiuzhang quality.

The identity evidence schema is consumed directly: `target`, `model_check`,
`decision`, `evidence_scope`, `full_law`, `global_tv` and `native_provenance`.
For statistical global bounds it also requires
`statistical_protocol_valid=True`, a complete first-batch IID/no-retry protocol,
unconditional path-density attestation, matching requested/observed integer
counts, and matching copies of the bound's protocol. The confidence plan's
unique comparison IDs, comparison count, familywise alpha, per-comparison
allocation, equal upper-bound/witness split and plan digest must agree with
the alpha actually used by the bound. Complete numerical enumeration does not
require an IID statistical protocol.
Only same-target complete numerical TV or a qualified global upper bound can
pass quality. A loose upper bound means insufficient evidence; it is not a
proof of excessive TV. A rejected witness can reject, but passed witnesses
cannot accept. An incomplete batch or failed evaluator cannot be silently
filtered into acceptance.

The empirical-hardware branch currently reports the missing measured-data
binding, and does not infer an underlying-law TV certificate from finite
witness acceptance. It is not a two-sample distribution tester. A future
hardware comparison must add its measured dataset and a separately justified
statistical protocol; pretending this capability already exists would be
misleading.

Even accepted quality does not automatically authorize a speed ratio. The
catalog would also need a matched timing record with target kind, covariance,
sample count, selection rule, timing scope and actual resource measurements.
No current experimental catalog entry has such a manufactured record. Quoted
hardware sample periods are displayed as reference facts only.

## Demonstration and current gap

The CLI's default demonstration reuses one existing native m64 run and does
not resample or rerun a benchmark:

```bash
python experiments/assess_jiuzhang_gap.py
PYTHONPATH=src python -m unittest discover -s tests -p test_jiuzhang_gap.py -v
```

The report in `results/iteration4_jiuzhang/` identifies a synthetic Haar
circuit, 64 binary outputs, 512 samples, uniform loss, and no large-system
full-law quality evidence. Against Jiuzhang4, instance, output extent,
imperfections and implementation capacity fail; quality and resource/timing
comparability remain unknown. The speed ratio is null. The measured local
draw rate is retained only as a local fact.

The implementation's 256-mode cap remains explicit. A dense float64
covariance for 8,176 modes alone occupies `32 * 8176**2` bytes, approximately
1.99 GiB. This is a single-matrix storage calculation, not a prediction of
total simulator memory or runtime. Copies, complex physicality validation,
factorizations, partial distinguishability and sampling work add other costs.

Synthetic 100- or 144-mode runs can test scaling and numerical behavior, but
do not close the instance or quality gates. The author-distributed m100
covariance is more useful progress: it makes a concrete published-model
experiment possible, with explicit failures or inconclusive validation
preserved. Root's separate fresh-run artifacts hold the measured result.

Nineteen focused tests cover alphabet mismatch, bin grouping, identical-size
synthetic circuits, exact covariance binding, kernel changes, witnesses,
inconclusive upper bounds, failed batches, hardware/model separation, capacity
and blocked timing ratios. Independent review added malformed-protocol,
inconsistent-alpha, invalid integer-count and impossible-timing regressions;
these harden evidence ingestion without resampling or changing the declared
statistical plan. The only complete-comparability fixture is labeled
artificial and does not appear as an experimental result.

## Baselines and the next meaningful comparison

Historical claims about a supercomputer taking centuries, or an extrapolated
MPS construction taking over 10^42 years, are algorithm- and assumption-specific
estimates. They are not universal classical lower bounds, nor measured
benchmarks against the present code.

Modern comparison should include the relevant primary classical methods,
including [Oh et al., Nature Physics 2024](https://www.nature.com/articles/s41567-024-02535-8)
and the 2026 paper
[Efficient simulation of low-entanglement bosonic Gaussian states in polynomial time](https://www.nature.com/articles/s41534-026-01254-7).
The latter's reported MPS preparation cost must not be treated as complete
sampling time. FrugalGBS remains a useful implemented baseline; beating it on
a synthetic instance does not settle all contemporary classical competition.

The next experiment should preserve the complete output extent of a specified
real covariance and improve the sampler/error evaluator together. Low-click
Jiuzhang2 calibrations and the registered Jiuzhang1 model are actionable
bridges. Advancing to Jiuzhang3 requires its fanout response; advancing to the
latest Jiuzhang4 requires its actual structured calibration, distinguishability
model and a representation that exceeds the current dense-mode cap.
