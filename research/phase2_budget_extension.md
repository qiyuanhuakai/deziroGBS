# Exploratory extension to four retained clicks

**Increasing the click-influence budget from k=3 to k=4 substantially improves
full-law accuracy on these small instances.** At k=4 it has lower TV than
official FrugalGBS K=5 on all 24 reused holdout-suite cases and all three
bright cases. The current Python implementation is much slower than the
official C++ baseline: median paired cold-batch time ratios are 24.5 and
74.7, respectively. These results establish an observed accuracy/cost
tradeoff, not a speed advantage.

## Status and controlled scope

This is a **post-baseline exploratory budget extension; not preregistered
held-out selection**. The decision to try k=4 followed inspection of the
Frugal comparison. The 24 original holdout cases have therefore been reused
for exploration; their win counts are descriptive and should not be called
an independent confirmatory test.

The two extension configs copy the original optical-instance settings and
128-sample count, retaining only `schur_pairwise` k=4 and `click_influence`
k=4:

- [Holdout-suite extension config](../configs/phase2_holdout_k4_extension.json):
  N=10; Haar, local depth 2, and local depth 4; squeezing 0.4 or 0.8;
  nominal transmissivity 0.5; uniform or heterogeneous profiles; seeds
  5101 and 5102. There are 24 optical instances and 48 new method rows.
- [Bright extension config](../configs/phase2_bright_k4_extension.json):
  N=12; Haar; squeezing 0.8; transmissivity 0.8; uniform profiles;
  seeds 6101–6103. There are three instances and six new method rows.

Both runs completed with **zero failures**. All regenerated interferometers,
squeezing vectors, transmission vectors, covariances, and target arrays were
checked against their original archives and match bit-for-bit. All 27 target
requests were cache hits. The exact target remains evaluator-only; candidate
samplers receive the covariance and method parameters.

No sampling algorithms were changed. `conditional_memory.py`,
`schur_memory.py`, `zero_memory.py`, and `exact.py` match the original suite
snapshots byte-for-byte. The current frozen runner includes the previously
made chain-diagnostic normalization/conditional-KL corrections and stronger
output-overwrite guard; these are the differences from the earlier runner
snapshot. They do not change sampling or the TV calculation.

## Full-law accuracy against Frugal K=5

TV uses the archived complete target distribution and each complete induced
sampler law. The 128 random draws are a latency probe and are not used to
estimate these TV values. Frugal's law is the previously validated numerical
trace of its pinned SIMD implementation, including its threshold policy;
see [Frugal integration](frugal_integration.md).

The ratio column is the **median of paired per-instance TV ratios**, not a
ratio of the two marginal medians. A ratio below one favors the listed
event-memory method.

| Reused suite | Method | Median TV | Lower TV than Frugal K=5 | Median paired TV ratio |
|---|---|---:|---:|---:|
| 24 original holdout cases | Frugal K=5 | 0.000417873 | Reference | 1.000 |
| 24 original holdout cases | Click influence k=3 | 0.000430668 | 3/24 | 1.388 |
| 24 original holdout cases | Pairwise k=4 | 0.000122369 | 20/24 | 0.379 |
| 24 original holdout cases | Click influence k=4 | 0.0000523911 | 24/24 | 0.179 |
| 3 bright cases | Frugal K=5 | 0.0417732 | Reference | 1.000 |
| 3 bright cases | Click influence k=3 | 0.0288772 | 3/3 | 0.756 |
| 3 bright cases | Pairwise k=4 | 0.0194202 | 3/3 | 0.465 |
| 3 bright cases | Click influence k=4 | 0.0142352 | 3/3 | 0.342 |

For click influence k=4, the paired TV ratio to Frugal K=5 ranges from
**0.0292 to 0.5767** on the 24 cases, and **0.3408 to 0.3777** on the three
bright cases. Thus the median paired relative TV reductions are 82.1% and
65.8%. The bright k=4 TVs are 0.0163490, 0.0142352, and 0.0127551 for
seeds 6101, 6102, and 6103.

Increasing click influence from k=3 to k=4 reduces TV on **all 27 cases**.
The median paired k4/k3 TV ratios are 0.1247 for the 24-case suite and 0.4883
for the bright suite. The earlier loss against Frugal K=5 at k=3 therefore
does not persist when this extra retained-click budget is allowed. This is
an empirical result; it establishes neither monotonic TV improvement for
arbitrary states nor performance outside these instance families.

The same-budget control also matters. Click influence k=4 improves over
pairwise k=4 on 23/24 original holdout cases and 3/3 bright cases, with median
paired TV ratios 0.605 and 0.761. The exception is local depth 4, squeezing
0.4, uniform profile, seed 5102: click influence TV is `1.8999e-5`, versus
pairwise `1.6004e-5`. Selection contributes beyond the budget increase, but
does not dominate the cheaper selector on every case.

## Descriptive implementation cost

Both suites use one OpenMP thread and one OpenBLAS thread. A cold batch is
construction/preparation plus 128 samples. Python sampling time includes
lazy selection; Frugal preparation includes layout and target generation.
Full-law enumeration is excluded from these costs. Frugal process launch,
file transfer, and JSON parsing are also excluded, matching the existing
benchmark's in-process comparison. These are single-batch latency probes,
not steady-state throughput measurements.

| Reused suite | Method | Median cold batch, ms | Median sampling alone, ms | Median paired cold-time ratio to Frugal K=5 |
|---|---|---:|---:|---:|
| 24 original holdout cases | Frugal K=5 | 4.47 | 0.757 | 1.00 |
| 24 original holdout cases | Click influence k=3 | 122.6 | 122.1 | 27.82 |
| 24 original holdout cases | Pairwise k=4 | 77.6 | 71.9 | 16.44 |
| 24 original holdout cases | Click influence k=4 | 110.8 | 110.5 | 24.52 |
| 3 bright cases | Frugal K=5 | 11.57 | 1.756 | 1.00 |
| 3 bright cases | Click influence k=3 | 765.8 | 765.6 | 66.41 |
| 3 bright cases | Pairwise k=4 | 266.8 | 262.1 | 23.14 |
| 3 bright cases | Click influence k=4 | 865.0 | 864.7 | 74.75 |

For click influence, the paired k4/k3 cold-time medians are 0.963 on the
24-case suite and 1.162 on the bright suite. An increased budget need not
increase measured work monotonically: selection is skipped while every
prior click still fits in memory. In the 24-case suite, median selection
computations per sampled batch fall from 39.5 to 12.5; median small
determinants used by selection fall from 2,903 to 1,971.5, while those used
by the conditional sampler rise from 1,949.5 to 2,601.5. In the bright suite,
selection computations fall from 298 to 163, but larger retained events
increase both determinant counters. These counters explain the competing
costs; they are not a complete flop count and determinant dimensions differ.

The original and extension runs occurred at different times, and independent
TV-validation work may have shared the machine. Small latency differences,
especially the apparent 3.7% median decrease on the 24-case suite, should
not be interpreted as a reproducible speedup. The substantial gap to the
optimized C++/SIMD baseline is directly visible. A Python/C++ comparison
does not isolate algorithmic complexity, and k retained click constraints
and Frugal cumulant cutoff K are different resources.

## Interpretation and archives

This extension identifies a useful next question: how much of the k=4
accuracy can be retained while reducing event-selection and determinant
cost, under a clearly specified compute budget? It does not demonstrate
quantum advantage being overcome, state-of-the-art sampling speed, or an
asymptotic improvement. The small dense evaluations remain exponential
research tools. The three bright cases and the correlated parameter grid
also do not support population-level significance claims.

Raw metrics, all induced laws, unchanged optical inputs, per-branch
diagnostics, configs, and source snapshots are archived in
[holdout k4 extension](../results/phase2_holdout_k4_extension/) and
[bright k4 extension](../results/phase2_bright_k4_extension/). Comparators
come from [original holdout](../results/phase2_holdout/),
[original bright](../results/phase2_bright/), and
[official Frugal benchmark](../results/phase2_frugal/).

Reproduction uses the unchanged runner:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
  python -m gbslab.phase2 \
  --config configs/phase2_holdout_k4_extension.json \
  --output results/phase2_holdout_k4_extension

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
  python -m gbslab.phase2 \
  --config configs/phase2_bright_k4_extension.json \
  --output results/phase2_bright_k4_extension
```

Use fresh output directories for a repeated run; the runner refuses to
overwrite completed or in-progress archives.
