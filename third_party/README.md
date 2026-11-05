# Pinned upstream source for reproducible classical baselines

This directory contains source subsets necessary for the CPU FrugalGBS adapter.
No paper text or experimental data is redistributed here.

| Project | Source | Pinned commit | License | Included files |
|---|---|---|---|---:|
| FrugalGBS | https://github.com/tomdodd4598/FrugalGBS | fa8b796f2afebd4845a2e335e457735952617ef8 | MIT, LICENSE.md | 55 |
| xsimd | https://github.com/xtensor-stack/xsimd | fb285a4a66617693c32ff0d71bd63ff812130e89 | BSD-3-Clause, LICENSE | 122 |

Each included file was checked against its pinned Git blob SHA. The FrugalGBS
license's final-newline bytes were restored to match upstream exactly before
archival. This changed generated comment text and the build cache key, but a
rebuild produced the **same executable SHA256** as the original benchmark.
The identity proof and original/new keys are archived in
`results/phase2_frugal/rebuild_identity.json`; the 15-case validation was reused
for the byte-identical executable, not described as a new test run.

`upstream_manifest.json` records the included paths and provenance. The official
sampling implementation is unchanged. Generated configuration and forced-branch
instrumentation live outside these directories under `.cache/frugal-build/`.
The instrumented full-law trace is a local evaluation tool, not an official
FrugalGBS API. See `research/frugal_integration.md` for validation and limits.

Eigen is a build dependency detected on the host and is not copied into this
archive. Its headers and the compiler are content-hashed in the build record.
The existing binary is environment-specific; rebuilding is the portable path.
