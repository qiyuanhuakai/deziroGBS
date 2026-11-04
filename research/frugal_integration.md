# Pinned official FrugalGBS CPU baseline

This project can now run the **official FrugalGBS CPU sampler** and enumerate
the small-instance probability law induced by its implemented branch rule.
The latter is our instrumentation, not an official API. Its purpose is to
compare full probability distributions at 8–12 modes without confusing a
finite sample histogram with the sampler's law.

## Source and build provenance

The upstream source is [Tom Dodd's FrugalGBS](https://github.com/tomdodd4598/FrugalGBS),
pinned to commit `fa8b796f2afebd4845a2e335e457735952617ef8`. The repository's
`CITATION.cff` requests citation of Dodd, Martínez-Cifuentes, Brown, Quesada,
and García-Patrón, *A fast and frugal Gaussian Boson Sampling emulator*,
[arXiv:2511.14923](https://arxiv.org/abs/2511.14923). This integration evaluates
the pinned implementation, and does not attempt to independently reimplement
the paper's method or reproduce its large-device benchmarks.

`third_party/FrugalGBS` contains the unchanged CPU/core source and its MIT
license, copyright 2026 "tomdodd4598". The generated instrumentation header
includes that notice and permission text in full. xsimd is pinned separately
to `fb285a4a66617693c32ff0d71bd63ff812130e89`, with its BSD license in
`third_party/xsimd`. `third_party/upstream_manifest.json` records source
provenance. The available dependency versions are Eigen 3.4.0 and xsimd 14.3.0.

Build without CMake:

```bash
python experiments/build_frugal.py
```

The builder invokes the C++ compiler with `-O3 -std=c++20 -fopenmp
-DFGBS_GPU_ENABLED=0`, links the official frontend/core and CPU source, and
generates `fgbs/global.hpp` from the upstream template with double precision
(`FGBS_REAL_PRECISION=2`) and maximum cutoff 5. It uses neither fast-math nor
`-march=native`. The initial environment used GCC 13.3.0; its default xsimd
batch holds two doubles. The default compiled mode counts are 2, 4, 8, 10,
and 12. Additional small counts, up to 16, can be selected explicitly:

```bash
python experiments/build_frugal.py --modes 2,4,8,10,12,16
```

All generated files and the executable are written under
`.cache/frugal-build`, outside the upstream tree. `--eigen`, `--xsimd`,
`--compiler`, and `--build-dir` override local paths. `EIGEN3_INCLUDE_DIR`
and `CXX` also supply defaults. `.cache/frugal-build/build.json` records
the compiler identity and binary hash, command, full dependency-tree hashes,
upstream function hash, generated configuration, build duration, and final
executable SHA256. These inputs form the build cache key; cached executable
bytes are also checked before reuse. A replacement executable and metadata
are installed atomically after compilation succeeds.

## Interface and covariance convention

The covariance is zero displacement, **xxpp order, hbar=2, vacuum covariance
I**, matching the rest of this project. This is verified in upstream
[`compute_correlator`](https://github.com/tomdodd4598/FrugalGBS/blob/fa8b796f2afebd4845a2e335e457735952617ef8/include/fgbs/core/correlator_utils.hpp):
mode `j` selects rows/columns `j` and `j+N`, and the vacuum-overlap calculation
uses the covariance plus identity. The adapter performs no covariance
reordering. Uniform or nonuniform loss and nonuniform squeezing are accepted
through their resulting physical covariance. The input should be physical;
the upstream validator checks finite symmetry and positivity of `V+I`, but
does not enforce the full uncertainty principle.

```python
from gbslab.frugal import frugal_sample

result = frugal_sample(
    model,                      # ThresholdModel, or an xxpp covariance array
    cutoff=3,
    sample_count=200_000,
    seed=123,
    enumerate_distribution=True,
    threads=1,
)
q = result["distribution"]      # full induced law, indexed by integer mask
counts = result["histogram_counts"]
timings = result["timings_seconds"]
```

The Python wrapper returns JSON-compatible values. It passes only covariance,
cutoff, sample count, and seed to the adapter. It does not pass the exact GBS
target distribution or use a target-dependent fit. `cutoff` ranges from 1
through `min(5,N)`. A one in bit `i` of an integer mask means a threshold click
in physical mode `i`. Mode zero is the least significant bit.

The executable also accepts a headerless, square CSV covariance directly:

```bash
.cache/frugal-build/frugal_adapter \
  --input covariance.csv --modes 10 --cutoff 3 \
  --samples 200000 --seed 123 --threads 1 --enumerate
```

It emits one JSON object to standard output and errors to standard error.
Omitting `--enumerate` returns `distribution: null`; zero samples are valid
when only the induced law is wanted. The timed path uses the public upstream
functions `generate_bitstring_layout`, `generate_cumulant_layout`,
`generate_sampling_target`, and **unmodified `generate_samples`**.

## What the full-law trace measures

The key distinction is between the sampler's internal approximate prefix
mass and the actual probability with which its random rule reaches a prefix.
In the pinned
[`cpu_sampler.hpp`](https://github.com/tomdodd4598/FrugalGBS/blob/fa8b796f2afebd4845a2e335e457735952617ef8/include/fgbs/cpu/cpu_sampler.hpp),
let `p` be the internal prefix mass and `a` the proposed next zero-prefix
mass **after** the upstream `clamped_prob` clamps it to `[0,1]`. The random
step chooses one if and only if

\[
U>a/p,\qquad U\sim\mathrm{Uniform}[0,1].
\]

For finite thresholds, the actual conditional probability of the zero branch
is therefore `clip(a/p, 0, 1)`. The internal state still updates to `a` after
a zero or `p-a` after a one. In particular, if `a>p`, the zero branch is
certain but the internal prefix mass becomes `a`, not the true prefix
probability. Reading the final internal mass as the probability of the
output string would be incorrect.

Our builder extracts the complete upstream `generate_sample` function and
requires its SHA256 to equal
`9c14a870f2875e808025d9e3535a71ba86144e5639ab14f0b7be5b594dabb3fb`.
It emits a separate `frugal_trace.hpp` with these limited changes:

1. Keep the entire upstream cumulant, partial-marginal, elision, Kahan-sum,
   and proposed-zero arithmetic. Use its same `ProbSimd` type.
2. Give every SIMD lane the same forced output-prefix bits. Read the
   threshold from lane zero; all lane computations are then identical.
3. Replace only the random branch choice with the specified bit and retain
   the upstream internal-state updates. Accumulate the **actual conditional
   branch probabilities** separately in a long-double product.
4. Stop a path once its probability is zero. Enumerate all `2**N` bitstrings
   and verify normalization, without renormalizing the output.

The induction is simple: the upstream working arrays and internal mass are
deterministic functions of a prefix; the trace computes the same state and
the probability of the branch selected by its threshold rule. The product
is consequently the induced output law. Using the actual SIMD operations
avoids a separate scalar approximation to the marginal calculations.

This is a **numerically enumerated full law of the pinned floating-point CPU
implementation under continuous uniform thresholds**, including threshold
saturation. It is not an exact-arithmetic physical GBS distribution. It also
does not enumerate the finite PRNG state space or its finite-precision
uniform-output lattice. Finite-RNG versus ideal continuous-uniform effects
are not part of the claimed full-distribution comparison. The instrumented
path is not used for sample timing.

`trace.saturated_threshold_path_steps` counts threshold values outside
`[0,1]` across replayed paths; shared prefixes are counted repeatedly. It is
neither a unique-prefix count nor a clipping probability. Nonfinite thresholds
are reported by the executable and rejected by the Python wrapper. A
nonpositive or nonfinite internal mass on a reachable path fails explicitly.
No exactness claim is made merely because the cutoff equals the mode count;
the implementation's marginal-elision behavior is included in the trace.

## Validation

The reproducible integration check is:

```bash
PYTHONPATH=src python -m gbslab.frugal --validate --samples 200000 \
  --output .cache/frugal-build/validation.json
```

It checks vacuum at 2, 4, 8, 10, and 12 modes; nonidentical independent
thermal modes; correlated Haar instances at every cutoff 2–5; lossless
squeezing; nonuniform loss and squeezing at 10 modes; and a physical
8-mode, cutoff-5 fixture with active threshold saturation. The two-mode
prefix law is also compared with this project's independent Gaussian
vacuum-overlap calculation. The validation code does not pass this reference
calculation to FrugalGBS.

The completed 15-case validation found:

| Check | Result |
|---|---:|
| All enumerated laws normalized without adjustment | Passed |
| Vacuum law TV from the analytic delta distribution, worst case | `5.80e-14` |
| Nonidentical thermal-law TV from analytic product law, K=2 | `3.68e-16` |
| Nonidentical thermal-law TV from analytic product law, K=5 | `2.86e-15` |
| First-two-mode marginal maximum error | At most `3.3e-16` |
| Nine non-vacuum official-sampling crosschecks | 200,000 samples each |
| Empirical TV between official samples and traced Q | `0.0118`–`0.0207` |
| Corresponding conservative finite-sample upper bounds | `0.0205`–`0.0316` |
| Official samples assigned exactly zero probability by Q | 0 |
| Clipping-active N=8, K=5 fixture | 4 saturated replayed path steps; histogram TV `0.013245` versus bound `0.022488` |

These empirical TV values are **sampling fluctuations around Q**, not
estimates of the physical-target approximation error. The crosscheck uses
an explicit per-case failure probability `delta=1e-6`. For M independent
draws and empirical histogram Qhat,

\[
\mathbb E\operatorname{TV}(\widehat Q,Q)
\leq\frac12\sum_x\sqrt{\frac{Q(x)(1-Q(x))}{M}}.
\]

Changing one sample changes TV by at most `1/M`. A bounded-differences bound
therefore adds `sqrt(log(1/delta)/(2M))` to this expectation upper bound for
the validation threshold. This conservative statistical check supports the
instrumentation equivalence; the direct source transformation and branch
argument are the stronger reasons for its correctness.

The validation JSON records the executable SHA256 and matching build cache
key. Validation fails if the executable changes during the run, so archived
validation can be checked against the binary used in a subsequent benchmark.

Tiny vacuum leakage and saturation counts on extremely rare vacuum prefixes
are visible instead of being discarded. The validation tolerance for analytic
laws is `1e-10`, well above the observed roundoff and well below the errors
of interest in the surrogate comparisons.

## Fair timing and interpretation

`timings_seconds` separately records covariance-file reading, bitstring
layout, cumulant layout, their summed `layout` time, target generation,
official sampling, histogram construction, and full-law enumeration.
`wrapper_wall_seconds` additionally includes process launch, file transfer,
and JSON parsing. Compilation is outside all these timings. One OpenMP
thread is the default, and the selected thread count and SIMD width are
reported in each result.

For an amortized sampling comparison, report layout plus target preparation
separately from samples per second, and disclose the number of samples over
which preparation is amortized. Full-law enumeration is a small-instance
evaluation cost and must not be charged as if it were ordinary sampling.
Conversely, do not include enumeration as a hidden precomputation that warms
the target before the official timing: the adapter samples first and
enumerates afterward.

This baseline is optimized C++/SIMD and the event-memory prototype is Python.
Observed wall time is a comparison of these implementations; it does not by
itself isolate algorithmic complexity. Full-law TV at the same covariance
and output ordering is directly comparable at these small sizes. Any claim
of improvement must state both that accuracy comparison and the measured
preparation/sampling costs. This adapter alone provides no evidence of a
quantum-advantage violation or a large-instance asymptotic improvement.
