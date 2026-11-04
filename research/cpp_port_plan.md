# Native C++ continuation requested by the user

The user explicitly requested that performance be investigated in C++, and
suggested handwritten assembly for suitable parts. This continuation began
after commit `63736954511ce7b7f8af1be83c458968692b5f9e`; the earlier Python/Frugal
comparison remains archived as a prototype comparison, not an algorithmic
speed verdict.

## Correctness contract

- Preserve the retained-click mathematical model and both selection rules.
- Keep all past clicked modes in the Gaussian zero-conditioned state, because
  selectors may reintroduce previously omitted click constraints.
- Match injected uniforms using `u < p_one`, physical output bit order, and
  arbitrary measurement-order permutations.
- Compare full induced laws at small size and preserve explicit failure
  behavior on unresolved conditioning events. No fast-math, silent retry,
  low-probability dropping, or tolerance relaxation for performance.
- Sampling and full-law enumeration share the conditional kernel; enumeration
  follows timed sampling and is excluded from performance metrics.

## Fair comparison

Both new native builds use `-O3 -march=native -ffp-contract=off`, double
precision, and one CPU thread. Eigen/xsimd may still contain explicit SIMD
or FMA instructions; this flag is not a proof of scalar bitwise equivalence.
The official Frugal sampling API remains unchanged. Its new adapter avoids
any `2**m` histogram above 14 modes, enabling larger cost probes.

The earlier SSE Frugal binary has its own validation. The new native Frugal
binary receives separate analytic, sample/trace, saturation, and large-output
checks; its old validation is not simply rebound to different executable code.
Every benchmark checks binary hashes against build and validation artifacts.

`configs/cpp_matched_benchmark.json` was fixed before viewing native candidate
timing. It defines three methods: pairwise k4, click-influence k4, and official
Frugal K5. k and K remain different accuracy/resources. Planned phases:

| Phase | Inputs | Draws per method run | Repetitions | Quality scope |
|---|---:|---:|---:|---|
| Cold | 24 prior holdout + 3 bright | 128 | 3 | Complete Q/P evaluated on first repetition |
| Steady | 6 high-r uniform cases + 3 bright | 8192 | 3 | Same small input families, no enumeration during run |
| Larger cost | 16/32/64 modes, two seeds each | 512 | 3 at16/32;1 at64 | Cost only; no complete target or matched-error claim |

Runs use fresh processes/models, rotate method order, record setup and draws
separately, and retain all failures. Python wrapper validation, file I/O,
hashing, process launch, and serialization are reported separately from C++
core times. Input uncertainty checks in the candidate wrapper are not hidden
inside a claim about end-to-end latency.

## Optimization decision

First validate a straightforward fixed-size Eigen implementation. Then profile
state updates, selection, and event kernels with non-overlapping accounting.
Profiling instrumentation is off in benchmark runs. Reuse of subset vacuum
probabilities and the final selected conditional are candidate optimizations,
but only after a measured bottleneck is identified and the unchanged law is
revalidated.

Compiler-generated SIMD, intrinsics, or handwritten assembly should be judged
against the same stable numerical contract. Handwritten assembly is not a
deliverable to add merely for appearance; a measured hotspot and a beneficial
validated kernel replacement are the acceptance conditions. The host exposes
AVX2/FMA/AVX-512 and is an AMD EPYC 9V74. Hardware `perf` is unavailable in this
environment; component clocks and generated-code inspection remain possible.
