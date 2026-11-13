# Native sampling kernels

## Iteration75: integer photon prefixes and direct clicks

`NativePrefixPNRSampler` in `src/gbslab/native_prefix_pnr.py` accepts an output-by-input complex isometry and real pure squeezing parameters `t=tanh(r)`. The C++ Hermite core retains up to 31 photon factors, then uses a Gaussian approximation; returned photon counts are not truncated. The ideal coupling error is bounded by the probability of reaching the first compression. Polynomial term/resource checks explicitly reject expensive high-input/high-depth requests. The bound is not a complete floating-point or PRNG certificate.

```python
from gbslab.native_prefix_pnr import NativePrefixPNRSampler
with NativePrefixPNRSampler(v, t, prefix_library, retained=31,
                            workers=16, runtime_dirs=runtime) as model:
    counts = model.sample(4096, 759101)['counts']
    clicks = (counts > 0).astype('uint8')
```

The recorded Windows binary is `results/iteration75_prefix_pilot_v1/prefix.dll`.
Direct click users continue to use `NativePhiloxClickSampler` with the iteration74
fused FP64 CUDA library. Deep-Wick and repeated-pure-decomposition PNR variants
are exposed separately; resplitting did not improve full PNR TV in the strong case.

Reproduce into **new output directories**, preserving the first runs:

```powershell
python -X utf8 experiments/iteration75_prefix_pilot_v1.py --output results/reproduce75_prefix
python -X utf8 experiments/iteration75_pnr_quality_v2.py --output results/reproduce75_quality
python -X utf8 experiments/iteration75_task_comparison_v1.py --output results/reproduce75_tasks
python -X utf8 experiments/iteration75_lowrank_v2.py --output results/reproduce75_lowrank
python -X utf8 experiments/iteration75_tests_v1.py --output results/reproduce75_tests
python -X utf8 experiments/verify_iteration75.py --output results/reproduce75_verification
```

Quality/timing drivers use explicitly hashed recorded binaries; a new pilot output
does not silently replace those paths. Run only one timing process at a time.
The verifier checks the published original cohorts, not newly named reproductions.
See the [iteration75 report](../RESEARCH_REPORT_ITERATION75.md) for failed builds,
budget-interrupted GPU panels, exact scopes and task-specific Frugal comparisons.

## Iteration72: photon counts and direct CUDA clicks

`NativeNewtonPNRSampler` uses safeguarded Newton survival inversion for the
iteration71 finite-tap signed-branch model. It returns integer photon counts;
the finite-tap and moment-compression biases remain. `NativeCudaBranchClickSampler`
performs bounded-branch threshold trajectories directly in FP64 CUDA. The two
outputs serve different measurement tasks. See the [report](../RESEARCH_REPORT_ITERATION72.md)
and [methods](../research/iteration72_methods.md) for measured accuracy and limits.

Existing recorded binaries on the tested Windows/CUDA machine can be used with:

```python
from pathlib import Path
import shutil
import numpy as np
from gbslab.native_pnr_newton import NativeNewtonPNRSampler
from gbslab.native_cuda_click_branches import NativeCudaBranchClickSampler

root = Path.cwd()  # repository root; install the package or add src to PYTHONPATH
runtime = [Path(shutil.which('g++')).parent, Path(shutil.which('nvcc')).parent]
data = np.load(root / 'results/iteration71_pnr_branch_pilot_v2/input_3.npz')
with NativeNewtonPNRSampler(
    data['v'], data['x'],
    root / 'results/iteration72_newton_pilot_v1/newton.dll',
    noise_variance=data['variance'], workers=16, runtime_dirs=runtime,
) as pnr:
    result = pnr.sample(1024, 123)
    photon_counts = result['counts']
    clicks_from_counts = (photon_counts > 0).astype(np.uint8)

with NativeCudaBranchClickSampler(
    data['v'], data['x'],
    root / 'results/iteration72_cuda_pilot_v1/cpu.dll',
    gpu_library=root / 'results/iteration72_cuda_pilot_v1/gpu.dll',
    noise_variance=data['variance'], branch_budget=4, runtime_dirs=runtime,
) as threshold:
    result = threshold.sample(1024, 123)
    direct_clicks = result['clicks']
    # Conditional along each realized displacement/path, not a marginal prefix law.
    conditional_probabilities = result['conditional_click_probabilities']
```

The libraries are loaded explicitly; import never builds code. CUDA limits are
128 inputs, 65536 outputs, branch budget128, batch1048576, and estimated work
memory below75% of free device memory. Only50 inputs/144 outputs were tested in
this round. Empty batches allocate no CUDA workspace. Arithmetic failures raise
an exception, including literal vacuum-probability underflow; no partial batch is
silently returned. The model owns copies of its isometry/covariance. The inherited
CPU worker setting does not change CUDA launch geometry or parallelize RNG.

`wall_seconds` for CUDA sampling includes native CPU RNG, array checks and
allocation, transfers, kernel and output copies. Constructor cost is reported
separately by the comparison drivers. CUDA timing components and per-trajectory
compression/weight diagnostics are returned. PNR statistics include root
iterations, fallback count and maximum survival residual; this residual is not
a TV error certificate.

Fresh output paths are required; never overwrite recorded experiment results:

```powershell
python experiments/iteration72_cuda_pilot_v1.py --output results/cuda72_new_run
python experiments/iteration72_newton_pilot_v1.py --output results/newton72_new_run
python experiments/iteration72_frugal_gpu_build_v5.py --output results/frugal72_new_build
python experiments/iteration72_frugal_gpu_comparison_v1.py --output results/frugal72_new_comparison --frugal-build results/frugal72_new_build
python experiments/iteration72_small_gpu_comparison_v1.py --output results/small_gpu72_new_comparison
python experiments/iteration72_tests_v1.py --output results/tests72_new_run
python experiments/verify_iteration72.py --output results/verify72_new_run
```

The frozen drivers use their recorded reference locations, and the small timing,
test and verification drivers load the published iteration72 binaries. They are
evidence-replay tools, not an automatic compiler selection system. Builds target
the tested `sm_120` GPU, CUDA13.5, MSVC2022 and GCC15.1; Eigen3.4.0 is expected in
the sibling `external_sources` folder. A different architecture requires a new
versioned build driver and separate outputs. Frugal v5 needs the MSVC LLVM OpenMP
runtime; its driver locates the installed runtime. Upstream CUDA sources remain
unchanged, while a documented temporary host layout compatibility patch is
applied; see [exact diff](../results/iteration72_frugal_gpu_build_v5/host_layout_compatibility.diff).

## Gaussian reset PNR sampling

`gaussian_reset_cpu.cpp` implements the approximate centered and displaced Gaussian reset models of Lim, arXiv:2610.06526v1, with full detector scans, diagonal rejection and coherent-reference rejection. It returns actual integer PNR counts. These are approximate Gaussian-target samples; thresholding counts does not certify the full PNR distribution. See [methods](../research/iteration69_methods.md) and [results](../RESEARCH_REPORT_ITERATION69.md).

The current implementation targets Windows x86-64 GNU C++ and Eigen 3.4.0. It guards/restores x87 precision, uses no fast math, and has no photon/proposal cutoff. Sampling uses one native thread; optional NumPy ILP64 OpenBLAS is explicitly supplied. Rebuild `-march=native` binaries on the destination machine. Eigen is an explicit local dependency, available from the [Eigen project](https://eigen.tuxfamily.org/).

```powershell
python experiments/iteration69_build.py --output results/reset_new_build --eigen ../external_sources/eigen_3_4_0_iteration38/eigen-3.4.0
python experiments/iteration69_benchmark.py --output results/reset_new_benchmark --build results/reset_new_build
python experiments/iteration69_validate_public_v2.py --output results/reset_new_validation --library results/reset_new_build/gaussian_reset_cpu.dll --reference results/iteration69_archive/iteration69_anchor_v9_20261110 --diagonal-reference results/iteration69_archive/iteration69_displaced_v6_20261110
```

Every output directory must be new. Build and validation record source/dependency hashes. Historical pilot sources and unsuccessful runs are kept in `results/iteration69_archive`; embedded `.git` path components are stored as `git_source_snapshot` with a byte-identity mapping. The public benchmark reruns the final single-thread comparison; full historical quality protocols remain in the archived drivers.

```python
import numpy as np
from gbslab.native_gaussian_reset import (
    NativeGaussianResetSampler, lossy_squeezed_components,
)

x, noise = lossy_squeezed_components(2, 0.7, 0.65)
with NativeGaussianResetSampler(
    np.eye(2, dtype=np.complex128), x,
    "results/reset_new_build/gaussian_reset_cpu.dll",
    noise_variance=noise,
    # Supply your GNU C++ runtime directory if it is not on the loader path.
    runtime_dirs=[],
) as sampler:
    result = sampler.sample(1000, seed=69, selector="diagonal")
    pnr_counts = result["counts"]
    threshold_clicks = pnr_counts > 0
```

Selectors are `scalar`, `eigen`, `leverage`, `diagonal`, `openblas`, and `anchor`. `openblas` requires a library exposing `scipy_cblas_zgemm64_` and its thread setter/getter. The wrapper restores the prior process-wide BLAS thread count, including on error; concurrent callers changing it are unsupported. `anchor` can reduce proposals but adds per-trajectory preparation, so it is not the default. Numerical envelope/physicality failures raise an exception rather than silently clipping a probability or returning partial output.

## Finite MPS displaced-threshold trajectories

The CPU and CUDA sources implement the same finite-input MPS measurement law as `gbslab.displaced_sampling`. They do not construct a Gaussian MPS or certify its approximation error. Hidden refined labels are internal measurement outcomes, not photon counts.

## Build

The recorded build uses Windows, GCC 15.1/OpenMP, Visual Studio 2022 BuildTools, CUDA 13.5, and an NVIDIA compute-capability 12.0 device. The current build driver fixes `sm_120`; other GPUs/platforms need a separately validated build. CPU `-march=native` binaries should be rebuilt on their destination machine. Python requires NumPy and SciPy.

```powershell
python experiments/iteration68_build.py --output results/my_native_build
python experiments/iteration68_validate.py --output results/my_native_validation --build results/my_native_build
```

Output directories must not exist. `--cpu-only` can build just the CPU library; the full validation driver requires both libraries. Build flags disable fast math and fused contraction. The source and compiler fingerprints, commands, stdout/stderr and compiled-library fingerprints are recorded. CUDA build setup retains its uniquely named temporary directory for inspection.

## Use

```python
from pathlib import Path
import shutil
import numpy as np
from gbslab.native_displaced_sampling import NativeDisplacedThresholdSampler

cuda_bin = Path(shutil.which("nvcc")).parent
with NativeDisplacedThresholdSampler(
    tensors, "results/my_native_build/native_cuda.dll", backend="cuda",
    runtime_dirs=[cuda_bin, cuda_bin / "x64"],
) as sampler:
    result = sampler.sample(beta, uniforms)
    clicks = result["clicks"]
```

`tensors` is a nonempty list of arrays with shapes `(left_bond, physical_dimension, right_bond)`. Boundaries are scalar. `beta` and `uniforms` have shape `(samples, modes)`, and uniforms must be real in `[0,1)`. Default construction applies exact QR normalization without truncation. `canonical=True` explicitly trusts an already normalized, right-canonical input; it is intended for pipelines that have performed that step themselves. Per-trajectory norm checks do not independently prove this precondition.

For CPU, set `backend="cpu"`, point to `native_cpu.dll`, and put the GCC runtime directory in `runtime_dirs`. An optional `blas_library` must expose NumPy's **ILP64** `scipy_cblas_zgemm64_` ABI. Omitting it uses the explicit scalar/SIMD contraction. `threads` controls OpenMP; with that BLAS library it also temporarily sets and then restores the process-wide OpenBLAS thread count. Concurrent callers changing this shared setting are unsupported. No work runs on import, no library is built implicitly, and failed numerical checks do not silently switch algorithms.

CUDA currently supports local physical dimension at most 32, requires the whole model in device memory, and does not stream arbitrarily large bonds. Host interfaces use 32-bit matrix dimensions. It rejects nonfinite inputs, unrepresentable positive probabilities, unresolved exact cancellation, invalid coordinates and conditional norm drift. This is FP64 numerical protection, not a mathematical finite-precision certificate.

## Reproduce the measured panel

Restore the archived historical state and noise files before running:

```powershell
python experiments/iteration68_benchmark.py --output results/my_native_benchmark --build results/my_native_build --state results/iteration25_panel_retry/run_19.state.npz --noise results/iteration28_panel/case_14.npz
$env:GBSLAB_WIRE_FOREST_LIBRARY = Join-Path (Get-Location) 'results/iteration31_build/libwire_forest.dll'
python experiments/iteration68_tests.py --output results/my_native_tests --build results/my_native_build
```

The older state is already distributed as chunks in the repository's historical large-artifact archive. The new pilot archive also stores its derived normalized cache in byte-preserving chunks. See the archive manifests for hashes and part order; concatenating listed parts restores the original file. The benchmark's generation seed, preparation costs, warmups, balanced run order, per-chunk elapsed time, output probabilities and source snapshots are retained in its output directory.

The scientific report is [iteration 68](../RESEARCH_REPORT_ITERATION68.md). The current 15.778x result is versus the improved NumPy implementation on this hardware and input panel, not versus every classical GBS method.

## Iteration 70 two independent sampling routes

`gaussian_reset_parallel.cpp` generates integer PNR records, using safeguarded Newton waiting times and independent C++ workers. `gaussian_direct_click.cpp` produces binary clicks directly, using per-port Gaussian conditional moments; it never generates photon counts. Both are approximate, and their errors must be measured separately. `gaussian_pgf_quad.cpp` supplies a small-system IEEE binary128 PNR reference, with combinatorial scaling rather than an arbitrary-mode speed claim.

Build the two sampler libraries explicitly on Windows with GNU C++ and local Eigen headers:

```powershell
python experiments/iteration70_build.py --output results/my_dual_build --eigen ../external_sources/eigen_3_4_0_iteration38/eigen-3.4.0
```

The Python interfaces only validate and organize native batches:

```python
from gbslab.native_gaussian_reset import lossy_squeezed_components
from gbslab.native_gaussian_reset_parallel import NativeParallelGaussianResetSampler
from gbslab.native_direct_click import NativeDirectClickSampler

# V is a complex M-by-N isometry; gcc_runtime is the local GCC bin directory.
x, noise = lossy_squeezed_components(V.shape[1], 0.7, 0.65)
with NativeParallelGaussianResetSampler(
    V, x, "results/my_dual_build/gaussian_reset_parallel.dll",
    noise_variance=noise, workers=16, runtime_dirs=[gcc_runtime]
) as sampler:
    photon_counts = sampler.sample(1024, 1234)["counts"]

with NativeDirectClickSampler(
    V, x, "results/my_dual_build/gaussian_direct_click.dll",
    noise_variance=noise, workers=16, runtime_dirs=[gcc_runtime]
) as sampler:
    clicks = sampler.sample(1024, 1234)["clicks"]
```

PNR selectors retain the iteration69 choices; `flow="newton"` and `backend="eigen"` are defaults. An explicit NumPy ILP64 OpenBLAS library enables `selector="openblas"` and optional `backend="lapack"`. BLAS uses one internal thread; `workers` controls independent trajectories. The library restores process-wide BLAS settings, so concurrent callers changing those settings are unsupported. Summed worker component times are not batch elapsed time; use `wall_seconds`.

Replay the public assemblies and run the full suite with historical native backends:

```powershell
python experiments/iteration70_validate_public.py --output results/my_dual_replay --build results/my_dual_build --reference results/iteration70_archive
python experiments/iteration70_tests.py --output results/my_dual_tests --build results/iteration68_build --reset-build results/iteration69_build --dual-build results/my_dual_build
python experiments/iteration70_pgf_reference.py --output results/my_pnr_reference
python experiments/iteration70_benchmark.py --output results/my_dual_benchmark
```

The benchmark deliberately uses the archived generic-CPU PNR and official Frugal binaries and rebuilds direct-click with the same main compiler settings. This reproduces the reported comparison, rather than silently replacing its baselines. Source snapshots, build commands and binary hashes are retained. The regular sampler build above uses `-march=native`; its replay is verified separately.

The PNR reference requires GCC libquadmath. Large quality extensions are explicit commands in `iteration70_quality_extension.py` and `iteration70_quality_extension_v2.py`; they generate 20 million and 80 million records respectively and preserve every record in batches. See [the report](../RESEARCH_REPORT_ITERATION70.md), [method and numerical scope](../research/iteration70_methods.md), and [next experiments](../research/iteration71_plan.md).

## Iteration71 signed branches: distinct click and PNR APIs

`gaussian_click_branches_v1.cpp` keeps bounded signed Gaussian threshold branches and compresses excess branches to their first two moments. Budget one replays iteration70. Sufficient uncompressed branches reproduce the known exact threshold identity in real arithmetic, at exponential worst-case cost. A transient update may allocate twice the branch budget.

`gaussian_pnr_branches_v2.cpp` keeps signed states between PNR waiting times. Finite-tap threshold conditioning approximates photon subtraction; finite tap strength and moment compression both cause bias. `condition_trigger` is checked before constructing a jump expansion, not a global cancellation bound. Version1 is retained as failed research evidence and is not the recommended PNR interface. Neither version applies a photon-count cutoff; numerical or storage failure aborts the batch.

```python
from gbslab.native_click_branches import NativeBranchClickSampler
from gbslab.native_pnr_branches import NativeBranchPNRSampler

# V, x, noise and gcc_runtime follow the iteration70 example above.
with NativeBranchClickSampler(
    V, x, "results/iteration71_branch_pilot_v1/branches.dll",
    noise_variance=noise, workers=16, branch_budget=4,
    runtime_dirs=[gcc_runtime]
) as sampler:
    bits = sampler.sample(1024, 1234)["clicks"]

with NativeBranchPNRSampler(
    V, x, "results/iteration71_pnr_branch_pilot_v2/pnr_branches.dll",
    noise_variance=noise, workers=16, branch_budget=2,
    tap_strength=0.001, condition_trigger=1e6,
    runtime_dirs=[gcc_runtime]
) as sampler:
    counts = sampler.sample(1024, 1234)["counts"]
    clicks_from_counts = counts > 0
```

The click API also provides `prefix_probability(bits, mean=realized_mean)`. With input displacement noise this evaluates the explicit fixed-mean conditional model, not its noise-averaged law; omitting the realized mean is rejected. All inputs use the earlier grouped quadrature conventions.

The frozen experiment drivers build with local GCC15 and Eigen3.4, generic CPU instructions, `-O3 -std=c++20 -shared -ffp-contract=off -fno-fast-math -DEIGEN_DONT_PARALLELIZE`. Dependencies include the preceding iteration's source components and archived inputs. No compilation happens on import. Fresh output directories are required:

```powershell
python experiments/iteration71_branch_pilot_v1.py --output results/my_click_branches
python experiments/iteration71_pnr_branch_pilot_v2.py --output results/my_pnr_branches
python experiments/iteration71_pnr_holdout.py --output results/my_pnr_confirmation
python experiments/iteration71_frugal_comparison.py --output results/my_branch_comparison
python experiments/iteration71_tests.py --output results/my_branch_tests
```

The holdout, comparison and tests deliberately reference the published frozen binaries and inputs. The first two drivers build fresh DLLs at their requested outputs and record compiler commands, sources and hashes. `iteration71_make_pnr_*.py` files document the one-time v1-to-v2 construction; they refuse to overwrite the existing source and are not required to rebuild it.

See [iteration71 evidence](../RESEARCH_REPORT_ITERATION71.md), [mathematical scope](../research/iteration71_methods.md), and [next steps](../research/iteration72_plan.md). The native timing advantage over the tested generic Frugal build is not a large-system equal-error claim.

## Iteration73 analytic PNR and device Philox clicks

The computation runs in C++/CUDA. Python only validates inputs, calls the native libraries and organizes experiments. `gaussian_pnr_polynomial73_v1.cpp` represents photon subtraction by polynomial factors and removes the finite-tap approximation. `retained=0..3` compresses exact first/second moments after `retained+1` jumps. Compression still approximates the state. There is no photon-count cutoff or silent rejection of long trajectories.

**Experimental numerical boundary:** for an undisplaced pure squeezed one-mode input with `X=diag(-0.25,0.5)`, retained depths 1/2/3 fail explicitly when a theoretically zero vacuum probability rounds negative. Retained depth 0 completes but loses parity. This interface does not yet correctly support that case. The API raises `ArithmeticError` and does not return a partial batch; the original evidence is preserved in iteration73 parity results.

```python
from gbslab.native_polynomial_pnr import NativePolynomialPNRSampler
from gbslab.native_philox_click import NativePhiloxClickSampler

# V, x, noise and gcc_runtime use the grouped quadrature convention above.
# cuda_runtime is the CUDA bin directory. Archived binaries target Windows;
# the CUDA binary targets sm_120 and needs compatible hardware/runtime.
with NativePolynomialPNRSampler(
    V, x, "results/iteration73_polynomial_pilot_v2/polynomial.dll",
    noise_variance=noise, retained=3, workers=16,
    runtime_dirs=[gcc_runtime],
) as sampler:
    counts = sampler.sample(1024, 1234)["counts"]
    clicks_from_counts = counts > 0

with NativePhiloxClickSampler(
    V, x, "results/iteration72_cuda_pilot_v1/cpu.dll",
    gpu_library="results/iteration73_philox_pilot_v1/philox.dll",
    noise_variance=noise, branch_budget=4,
    runtime_dirs=[gcc_runtime, cuda_runtime],
) as sampler:
    first = sampler.sample(512, 1234, row_offset=0)["clicks"]
    following = sampler.sample(512, 1234, row_offset=512)["clicks"]
```

Philox stream subsequences are keyed by `row_offset+row`; splitting a batch preserves each record. A repeated seed with offset zero repeats the same stream. The old CPU random engine uses a different seed mapping. `export_random=True` returns realized means and uniform draws for CPU replay and includes that extra transfer cost; production benchmarks disable it. The bounded-branch physical approximation is unchanged from iteration72. Workspace checks and explicit failure remain in force.

Build drivers require GCC15, Eigen3.4, and (for GPU) CUDA13.5 plus MSVC Build Tools at the locations recorded in the driver. Native sources include frozen earlier components; archived inputs and binaries used by subsequent validation must also be available. First-executed outputs are immutable: always choose fresh output directories.

```powershell
python experiments/iteration73_polynomial_pilot_v2.py --output results/my_polynomial_build
python experiments/iteration73_philox_pilot_v1.py --output results/my_philox_build
python experiments/iteration73_frugal_cpu_build_v5.py --output results/my_frugal_avx2_build
python experiments/verify_iteration73.py --output results/my_iteration73_verification
```

Build outputs above do not silently replace archived binaries. The verifier deliberately checks and replays the published iteration73 evidence. The holdout, cost and test drivers likewise name frozen dependency paths; those references are part of the experiment specification. The standalone Frugal build requires the documented identical-body inline ABI patch, and its separate C ABI timing library is built by `iteration73_frugal_cpu_library_v1.py` against the archived v5 include copy.

See [iteration73 results](../RESEARCH_REPORT_ITERATION73.md), [mathematics and compatibility changes](../research/iteration73_methods.md), and [iteration74 plan](../research/iteration74_plan.md). Small-system Frugal advantages, the pure-state failure and large-system quality limitations are part of the published results.

## Iteration74 structured pure cores and fused GPU branches

`NativeStructuredPNRSampler` takes an explicit vector `tanh_squeezing` in (-1,1), defining the product-pure covariance before the passive isometry. It preserves that parameterization until the first Gaussian moment compression. Photon-factor normalizations use a positive Bargmann polynomial norm. The previously failing undisplaced pure endpoint now evaluates to exact zero for odd factors. After compression, the algorithm still approximates the state and can produce odd total counts from an initially even pure state; the strong four-mode example still has a substantial full-distribution error.

```python
from gbslab.native_structured_pnr import NativeStructuredPNRSampler

with NativeStructuredPNRSampler(
    V, tanh_squeezing,
    "results/iteration74_structured_pilot_v1/structured.dll",
    mean=mean, noise_variance=noise, retained=3, workers=16,
    runtime_dirs=[gcc_runtime],
) as sampler:
    counts = sampler.sample(1024, 1234)["counts"]
    clicks_from_counts = counts > 0
```

The input is explicit physical squeezing, not automatic rounding or projection of a supplied covariance to a pure state. Noise variances and grouped quadratures retain the earlier conventions. The tested product-pure form plus passive complex V does not support arbitrary mixed conditional covariance in the positive-norm branch; the frozen general covariance routine handles states after compression.

`NativeClassicalTailPNRSampler` in `gbslab.native_classical_tail_pnr` is a separate experimental alternative using `results/iteration74_classical_pilot_v1/classical.dll`. After compression, if the entire Gaussian excess covariance is positive semidefinite, it samples an actual Gaussian displacement mixture and finishes with independent Poisson output counts. It never clips negative eigenvalues. Many pilot inputs never enter that branch, so this is not the default or a universal speedup. Its three additional statistics are `classical_completions`, `classical_photons`, and `classicality_checks`.

For direct clicks, keep the `NativePhiloxClickSampler` API and set `gpu_library` to either `results/iteration74_fused_pilot_v1/philox.dll` or `results/iteration74_fused_pilot_v2/philox.dll`. Both halve matrix/mean slot storage by merging overflowing branches without writing all children. Version1 preserves tested arithmetic bitwise; version2 adds parallel reduction, changes rounding and is faster on the measured large panel. It preserves tested click records, with small conditional-probability differences. The approximation budget and Philox stream offsets are unchanged.

```powershell
python experiments/iteration74_structured_pilot_v1.py --output results/my_structured_build
python experiments/iteration74_classical_pilot_v1.py --output results/my_classical_tail_build
python experiments/iteration74_fused_pilot_v2.py --output results/my_fused_gpu_build
python experiments/verify_iteration74.py --output results/my_iteration74_verification
```

These drivers use the same compiler/toolchain locations and archived dependencies described above. New output directories are mandatory. The verifier targets published frozen artifacts, not the newly named build directories. Full tests use `iteration74_tests_v3.py`, which closes the actual DLL-directory registrations created inside each test after that test ends, avoiding Windows' aggregate directory-registration limit without changing numerical implementations or assertions. Earlier failed test runs are retained.

See [iteration74 mathematics and failures](../research/iteration74_methods.md), [frontier scope](../research/iteration74_frontier_comparison.md), and [iteration75 plan](../research/iteration75_plan.md).
