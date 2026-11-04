# Iteration 4: official FrugalGBS samples with separately traced density

The main record below describes the frozen v1 adapter and its validation.
The final section records the subsequent v2 checkpoint fix; v1 binaries,
validation artifacts, and the actual failed audit remain unchanged.

## What is available

The new evidence adapter calls the pinned, unchanged official CPU
`generate_samples` API exactly once. It retains each physical-mode bitstring in
the returned order. Optional density tracing then follows each observed path
through a generated copy of the pinned conditional arithmetic. Generation and
density evaluation have separate timers. The adapter does not retry, filter
samples, reduce K, or substitute another method.

The isolated executable is
`.cache/frugal-evidence-native/frugal_evidence_adapter`, SHA256
`8009d86b6f46f99c079c2f9dc391a7df8e94132cd19bb0dbfb8645ba541a75b3`.
It supports compiled mode counts **8, 10, 12, 16, 32, 64, 100, 144**, with runtime
cutoff **K=1 through 5**. This is the CPU float64 build, upstream
`real_precision=2`, eight SIMD lanes on the recorded machine, and
`-O3 -std=c++20 -ffp-contract=off -fopenmp -DFGBS_GPU_ENABLED=0 -march=native`.
It was compiled in 28.67 seconds. Build details and hashes are archived in
`results/iteration4_frugal_adapter/build.json`.

The official source remains pinned at
`fa8b796f2afebd4845a2e335e457735952617ef8`. The instrumented `generate_sample`
body must match SHA256
`9c14a870f2875e808025d9e3535a71ba86144e5639ab14f0b7be5b594dabb3fb`
before the builder proceeds. Its MIT notice remains in the generated header.
Both previous Frugal executables and the frozen retained-click binary were
hashed before and after validation and remained unchanged. No upstream files
were edited.

## What the density means

For a fixed output prefix, the official kernel computes `zero_next_prob` and
`prev_prob` in its existing SIMD double arithmetic. It chooses a click when
the uniform random variate is greater than

\[
t_i = \frac{\texttt{zero\_next\_prob}}{\texttt{prev\_prob}}.
\]

Under ideal independent continuous uniform variates, the zero-branch mass is

\[
z_i = \max(0,\min(1,t_i)),\qquad
Q(x)=\prod_i z_i^{1-x_i}(1-z_i)^{x_i}.
\]

This final threshold clipping matters even though the upstream kernel already
clips `zero_next_prob` before division. A proposed zero mass can exceed the
stored prefix mass, and the actual branch law then becomes deterministic.
The trace records this actual conditional law rather than treating the raw
approximate prefix mass as a normalized output probability.

The trace keeps all marginal/cumulant arithmetic from the original source
body. It substitutes only the forced branch and output accumulation, using
the physical bit requested by the observed sample. `prev_prob` and parity
updates match the selected upstream branch. All SIMD lanes follow the same
forced prefix, so lane zero supplies the density calculation. Long-double
`log`/`log1p` accumulation is converted to float64 at output.

Legal deterministic branches add zero to log Q. An impossible forced branch
has log mass minus infinity; an impossible observed sample or a nonfinite
reachable threshold makes the evidence run fail. A tiny positive product
underflow does not terminate the log-density trace. Nonfinite thresholds are
explicitly counted. NaN comparisons select the zero branch in the upstream
operator, but such traces are not accepted as valid density evidence.

These are densities for the **computed Frugal conditional law Q**. They do not
assert that Q equals the target Gaussian threshold distribution P. They are
also not exact probabilities of the finite PRNG state space. That distinction
is present in the returned metadata and must remain in downstream reports.
Finite-precision arithmetic has not been replaced by interval arithmetic.

## Cross-mode implementation choices

The older full-law tracer used a uint64 mask and was intended only for small
enumerated distributions. The new scorer takes the upstream `Bitstring<N>`
and calls `get_bit<N>` for every physical mode, including indices above 63.
The full-law optional output remains capped at 14 modes. Large runs never
allocate a `2**m` histogram or distribution.

The four tracing workspaces are heap vectors. Their payload is

\[
8w\,[2\binom m2+2\binom m3]\ \text{bytes},
\]

where w is the SIMD lane count (8 for this build). The implementation preserves
the upstream size-one convention for degenerate small arrays. At 100 and 144
modes the payload is 21,331,200 and 63,697,920 bytes respectively. Storing these
arrays on the stack would make a nominal extension of the old small tracer
unsafe.

Scoring is serial and repeats one SIMD path per sample. It therefore leaves
SIMD lanes duplicating the forced prefix, making density tracing materially
more expensive per sample than official batched generation. This is an audit
cost, separately reported. Packing distinct observed paths into SIMD lanes is
a possible subsequent optimization, requiring a fresh equivalence gate; it
has not been introduced here.

## Python API and evidence binding

```python
from gbslab.frugal_evidence import frugal_evidence_sample, assess_frugal_samples
from gbslab.gbs_identity import ConfidencePlan, fixed_batch_protocol

plan = ConfidencePlan(("frugal_fixed_case",), familywise_alpha=.05)
protocol = fixed_batch_protocol(1024, source="experiment declared before sampling")
out = frugal_evidence_sample(
    V, cutoff=5, sample_count=1024, seed=123,
    record_log_probabilities=True, raw_samples=True,
    enumerate_distribution=False, timeout=1800, threads=1,
)
report = assess_frugal_samples(
    V, out, epsilon=.1, plan=plan, comparison_id="frugal_fixed_case",
    protocol=protocol, max_exact_clicks=6,
)
```

The wrapper validates covariance shape, symmetry, finiteness, and Gaussian
uncertainty; native metadata; complete sample counts; raw-sample agreement
with all returned histograms; finite nonpositive log Q; repeated-pattern
density consistency; and log Q/full-Q agreement when full enumeration is
requested. Covariance symmetrization is canonical and its SHA256 matches the
identity evaluator's convention. No automatic compilation occurs.

Normalized provenance includes the executable hash, matching build key,
covariance hash, seed, count, physical order, and the rule identifiers
`selector="official_frugal_double_elision"`, `k=cutoff`,
`vacuum_kernel="upstream"`. It additionally records the pinned upstream
commit, source-body hash, precision, API, and density timing separation. These
are traceability fields, not cryptographic proof of program correctness.

`assess_frugal_samples` calls the generic `assess_gbs_identity` evaluator
directly. It does not route this sampler through retained-click-specific
assumptions. The caller must supply a protocol fixed before sampling and a
confidence plan. A successful sample batch, or agreement on low-order
witnesses, does not constitute a global GBS certificate.

The C++ process writes flushed JSON stage diagnostics to stderr. The wrapper
retains them on timeout or failure in `FrugalEvidenceError.payload` and
`.stderr`. If density scoring fails after generation, the native JSON retains
the entire requested raw batch, marks the evidence as failed, and returns no
valid log-Q array. There is no resampling of failed paths.

## Validation results

The source/hash-bound record is
`results/iteration4_frugal_adapter/validation.json`. Its 11 cases cover every
compiled mode size, retaining covariances, samples, log Q, and small complete
laws in NPZ artifacts. All cases passed. These timings were recorded during
validation, possibly concurrently with other work; they are not formal speed
measurements.

| Gate | Observed result |
|---|---|
| Complete laws at m8, m10, m12 versus frozen official tracer | TV exactly 0 in all six enumerated cases |
| Per-sample log Q versus enumerated Q | Maximum absolute discrepancy 8.8818e-16 |
| Sample histograms versus frozen official generator | Identical in all nine cases where the old binary supports the mode count |
| Ordered samples with density tracing enabled/disabled | Identical on the clipping fixture; also checked at K2 and K5 in unit tests |
| Nonmultiple SIMD batch | Exactly 17 outputs and 17 log densities, including m100 and m144 |
| Clipped-threshold fixture | Saturated paths present; finite normalized Q; no observed zero-mass pattern |
| Analytic thermal tails at m8,16,32,64,100,144 | Maximum log-Q error 3.4195e-14; clicks only in the last three physical modes |
| Threshold boundary probe | 12 thresholds, 24 forced branches, 96 official SIMD branch decisions passed |
| Product-underflow probe | 6,000 positive branch factors retain finite log density after product underflow |

The boundary probe includes zero, one, their neighboring floating-point
values, thresholds outside [0,1], positive/negative infinity, and NaN. It
compares the unchanged upstream branch operator to the trace rule; nonfinite
cases verify explicit flags and are not valid evidence cases.

The separate eight-test unit suite also passed. It covers invalid input before
process launch, nonmultiple batches, unchanged official output, clipping,
physical bits beyond 64, generic-evaluator covariance/batch binding, failure
rejection, retry-protocol refusal, zero draws, and raw-output opt-out. Tests
never invoke a compiler.

Product-state fixtures at 100 and 144 modes establish bit representation,
workspace handling, and density correctness for those soluble fixtures. They
do not establish global approximation quality on a dense correlated target,
nor a comparison with a quantum experiment.

## Memory preflight for the authorized 100-mode K5 run

Source inspection of `core/structs.hpp`, `layout.hpp`, `data.hpp`,
`core/correlator_utils.hpp`, and `cpu/cpu_cumulants.hpp` identifies the dominant
arrays. Let

\[
L(m,K)=\sum_{j=0}^{K}\binom mj.
\]

For all supported mode counts `upto<N>` is one byte. The persistent bitstring
layout payload is

\[
L\,[4\lceil m/32\rceil+1+K].
\]

The float64 cumulant target adds approximately 8L bytes. During target
construction an additional 8L-byte correlator vector remains live while the
cumulant vector is computed. The remaining matrix temporaries are at most
10 by 10 at K5, with recursive depth at most 5 per thread. The cumulant
partition layout depends on K rather than L. No O(L squared) allocation was
found.

| Modes, K | Layout entries | Persistent layout + target + one tracing workspace | Target-stage large-vector payload |
|---|---:|---:|---:|
| 64, 5 | 8,303,633 | 0.175 GiB | 0.232 GiB |
| 100, 5 | 79,375,496 | 2.238 GiB | 2.809 GiB |
| 144, 5 | 498,685,189 | 15.850 GiB | 19.506 GiB |

The exact array calculations and source hashes are in
`results/iteration4_frugal_adapter/memory_preflight.json`. They are payload
calculations, not guarantees on allocator overhead or peak RSS. The inspected
environment had an 8 GiB cgroup memory ceiling and roughly 0.73 GiB resident
usage before the authorized run. This leaves substantial headroom for the
100-mode case; it does not support the 144-mode K5 CPU run. A native address
space ceiling can turn unexpected expansion into a recorded failed run;
failure must not trigger an unreported retry or a lower K.

The adapter validation itself did **not** run 100- or 144-mode K5 sampling. The
authorized real Jiuzhang-1 comparison is a separate parent-coordinated
experiment with its own fixed protocol and budget. Its eventual results must
report generation, target setup, and density tracing separately and keep the
CPU float64 precision explicit when comparing with published implementations.

## Reproduction

```sh
python experiments/build_frugal_evidence.py --native
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m unittest tests.test_frugal_evidence -v
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python results/iteration4_frugal_adapter/validate_adapter.py --compile-boundary-probe
```

Use a separate archival directory or copy the record before rerunning an
experiment. The earlier milestone binaries, data, and results remain intact.

## Checkpoint follow-up: preserve completed generation before density

The actual 100-mode K5 control exposed a data-preservation defect in v1. The
official generator completed all 1,024 requested outputs in 130.2207 seconds
after layout and target setup. Our separate density audit subsequently reached
the 1,800-second process deadline. Because v1 emitted its only sample-bearing
JSON after density tracing, neither the completed sample array nor log Q was
available in the parent process. The archived failure is
`results/iteration4_jiuzhang1_control/frugal/native.json`. This is a failure of
the audit pipeline to finish and retain evidence, not a rejection of Frugal's
distribution quality. The original batch was not retried.

V2 adds the opt-in Python argument `sample_checkpoint_path` and native
`--sample-checkpoint`. The complete requested sample array is published
immediately after official generation and before any density work. The
checkpoint contains physical-mode samples, counts, modes, K, seed, thread
count, precision, upstream function identity, and completed-stage timings.
The wrapper supplies and verifies the expected covariance and executable
hashes, checks all samples and counts, and hashes the resulting checkpoint.

The destination must be fresh. The native process reserves a temporary file
before expensive setup, writes and fsyncs the complete checkpoint, and
atomically publishes it without replacing an existing file. Partial files
are never treated as completed generation. Checkpoint serialization and sync
are timed separately in `timings_seconds["sample_checkpoint"]`; the unchanged
official `generate_samples` call and its sampling timer are unaffected.

On a density timeout, `FrugalEvidenceError.payload` now includes
`sample_checkpoint_path`, `sample_checkpoint_sha256`, `completed_generation`,
`completed_generation_sample_count`, and the verified checkpoint metadata.
The overall status remains an error. Checkpoints have
`status="generation_complete"`, `density_complete=false`, and no log-Q array;
passing one directly to the complete-density assessment helper yields
`evaluation_failed`. A malformed or incomplete file is explicitly reported
as invalid. No retry, postselection, or new quality acceptance is introduced.

The separate v2 binary is
`.cache/frugal-evidence-checkpoint-native/frugal_evidence_adapter`, SHA256
`8986aafa436744a65ea72663e77d28b90acf22497ca3d5f6bc50bead674751dd`.
The current builder defaults to this directory and refuses to overwrite the
frozen v1 cache. Calls without the new option retain the historical default
executable. Calls that request a checkpoint and omit `executable` select the
checkpoint binary directly, without attempting a fallback or another batch.
Future controlled runs can make that choice explicit:

```python
from gbslab.frugal_evidence import CHECKPOINT_EXECUTABLE, frugal_evidence_sample

out = frugal_evidence_sample(
    V, cutoff=5, sample_count=1024, seed=123,
    executable=CHECKPOINT_EXECUTABLE,
    sample_checkpoint_path="results/new_fixed_run/sample_checkpoint.json",
    timeout=1800, threads=1,
)
```

The fix is gated by four focused m8 tests in
`results/iteration4_frugal_checkpoint_validation/validation.json`: exact
sample/log-Q/full-Q agreement with v1, checkpoint-before-density phase order,
verified timeout-path recovery with one process call, refusal to overwrite
prior evidence, and refusal to promote a malformed checkpoint. Timeout
recovery is tested through an explicitly injected wrapper timeout after a
small native run; no new 100-mode batch was generated. The v1 cross-mode
numerical tests were not repeated for this storage-only change.

```sh
python experiments/build_frugal_evidence.py --native \
  --build-dir .cache/frugal-evidence-checkpoint-native
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m unittest tests.test_frugal_evidence.FrugalCheckpointTests -v
```
