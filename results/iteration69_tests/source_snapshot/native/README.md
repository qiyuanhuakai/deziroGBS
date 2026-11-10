# Native sampling kernels

## Gaussian reset PNR sampling

`gaussian_reset_cpu.cpp` implements the approximate centered and displaced Gaussian reset models of Lim, arXiv:2610.06526v1, with full detector scans, diagonal rejection and coherent-reference rejection. It returns actual integer PNR counts. These are approximate Gaussian-target samples; thresholding counts does not certify the full PNR distribution. See [methods](../research/iteration69_methods.md) and [results](../RESEARCH_REPORT_ITERATION69.md).

The current implementation targets Windows x86-64 GNU C++ and Eigen 3.4.0. It guards/restores x87 precision, uses no fast math, and has no photon/proposal cutoff. Sampling uses one native thread; optional NumPy ILP64 OpenBLAS is explicitly supplied. Rebuild `-march=native` binaries on the destination machine. Eigen is an explicit local dependency, available from the [Eigen project](https://eigen.tuxfamily.org/).

```powershell
python experiments/iteration69_build.py --output results/reset_new_build --eigen ../external_sources/eigen_3_4_0_iteration38/eigen-3.4.0
python experiments/iteration69_benchmark.py --output results/reset_new_benchmark --build results/reset_new_build
python experiments/iteration69_validate_public.py --output results/reset_new_validation --library results/reset_new_build/gaussian_reset_cpu.dll --reference results/iteration69_archive/iteration69_anchor_v9_20261110 --diagonal-reference results/iteration69_archive/iteration69_displaced_v6_20261110
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
