# Iteration 39: frozen official Frugal law and global witness panel

This plan is fixed before the confirmation outputs are inspected. The 19 pilot cases and calibration are exploratory engineering evidence only.

## Inputs and implementation

Use the round 35 shifted 144-mode covariance, byte-bound to its published manifest. Reuse the round 35 exact physical reference and measurement transfer, round 36 moment intervals, and round 37 ten-mode physical atom intervals. The extended adapter calls the pinned unchanged public FrugalGBS CPU sampler (fa8b796f2afebd4845a2e335e457735952617ef8), then separately replays its instrumented copy and requires identical bitstrings. One thread; compiler settings and runtime files are fixed by the round 39 build record.

## Global confirmation

Four configurations: cumulant cutoffs 1, 2, 3, 4; 1024 outputs each; seed 391440 for every configuration. Preserve all raw outputs, random numbers, thresholds and internal states. Run configurations sequentially. Independent Python mt19937_64 replay verifies the native stream including padded SIMD tail lanes. A process failure invalidates that configuration's statistical analysis; do not refill, reseed or infer quality from survivors. Nonfinite or saturated thresholds are diagnostics of the official algorithm, not additional failure outcomes with target probability zero.

Predefined bounded observables are X=N/144 and Y=(N-67)^2/5929. The center 67 is fixed from the previous reference mean, not fitted to this cohort. Bound the reference expectations by exact moment arithmetic. Product/Markov mean limits allocate alpha=1/2000 to each side of each observable: 16 tails total. Events N>=40,50,60,70,80,90,100 allocate alpha=1/14000 each: 28 tails total. Familywise alpha is 16/2000+28/14000=0.01. Quantiles and roots use exact 40-bit dyadic endpoint inequalities. Event target bounds use the minimum of the existing moment and integer-support quadratic bounds. The maximum observable/event discrepancy is a lower bound on full-distribution TV under the independent identically distributed random-word model. Subtract the round 35 transfer for the shifted target. Shared streams across configurations do not invalidate the union bound. A fixed pseudorandom seed is not a proof of independence. Zero lower bounds do not provide TV upper bounds.

## Complete ten-mode laws

Blocks starting at modes 0, 54, 134; cutoffs 1 through 5; 257 samples each; seeds 391001 through 391015 in block-major order. Export all 1023 thresholds, including zero-mass prefixes and NaN continuations. Repeat prefix export in reversed traversal order with zero samples. Compare all threshold and internal-state values. Run the old round 38 adapter on identical CSV/seed/sample count and compare output histograms. Reconstruct normalized exact rational laws both for continuous uniforms and for independent uint64 words mapped by the inspected GCC binary64 conversion and upper-endpoint clamp. Independently reconstruct via rounding-cell counts and depth-first traversal. Compare against reused physical atom intervals and check the full-path coupling bound. These statements are conditional on the exported threshold table and stated RNG model, not uniform-seed distributions or other standard libraries.

## Reporting and limits

Preserve failure artifacts and source snapshots. Use exclusive output directories and verify dependencies before and after execution. Report setup, official sampling, and audit costs separately; timings are nonisolated and do not establish a cross-method speed ranking. Cutoff 5 at 144 modes is excluded by the declared 20-million-entry resource budget, not declared algorithmically impossible. Compare with current FrugalGBS, MPS and Jiuzhang literature at the scope supported by their sources; do not infer full-scale superiority from local laws or a one-sided discrepancy witness.
