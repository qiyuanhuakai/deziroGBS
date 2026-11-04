# Matched native C++ results

All time ratios are candidate / official FrugalGBS K=5, using medians inside each input first.
Full-TV comparisons are numerical full-law evaluations of 10/12-mode inputs only. 16/32/64-mode entries measure cost only.

| Workload | Inputs | Method | Median cold-time ratio | Faster cold | Median draw-time ratio | Median TV ratio | Lower TV |
|---|---:|---|---:|---:|---:|---:|---:|
| cold: phase2_bright | 3 | Pairwise k=4 | 0.330603 | 3/3 | 4.66817 | 0.464896 | 3/3 |
| cold: phase2_bright | 3 | Click influence k=4 | 0.525918 | 3/3 | 7.9924 | 0.342091 | 3/3 |
| cold: phase2_holdout | 24 | Pairwise k=4 | 0.31732 | 24/24 | 4.06257 | 0.423803 | 19/24 |
| cold: phase2_holdout | 24 | Click influence k=4 | 0.3724 | 23/24 | 4.84127 | 0.178562 | 24/24 |
| steady: phase2_bright | 3 | Pairwise k=4 | 4.03381 | 0/3 | 4.93826 | 0.464896 | 3/3 |
| steady: phase2_bright | 3 | Click influence k=4 | 7.17658 | 0/3 | 8.78844 | 0.342091 | 3/3 |
| steady: phase2_holdout | 6 | Pairwise k=4 | 4.68167 | 0/6 | 5.68154 | 0.705225 | 5/6 |
| steady: phase2_holdout | 6 | Click influence k=4 | 5.77 | 0/6 | 6.97084 | 0.277549 | 6/6 |
| large: 16 | 2 | Pairwise k=4 | 0.183939 | 2/2 | 1.03258 | not evaluated | not evaluated |
| large: 16 | 2 | Click influence k=4 | 0.195198 | 2/2 | 1.10803 | not evaluated | not evaluated |
| large: 32 | 2 | Pairwise k=4 | 0.0233216 | 2/2 | 0.184395 | not evaluated | not evaluated |
| large: 32 | 2 | Click influence k=4 | 0.034087 | 2/2 | 0.271644 | not evaluated | not evaluated |
| large: 64 | 2 | Pairwise k=4 | 0.00338218 | 2/2 | 0.036268 | not evaluated | not evaluated |
| large: 64 | 2 | Click influence k=4 | 0.00655253 | 2/2 | 0.0707585 | not evaluated | not evaluated |

## Important boundaries

- Native C++20 on the same CPU, single thread, O3, march=native, no fast-math.
- Medians across input configurations are descriptive, not estimates from independent random interferometers.
- Small circuits and k=4 were explored previously; native timings do not create a new statistical holdout.
- Complete distribution quality only at 10/12 modes; 16/32/64 modes are cost-only.
- Pairwise references use a numerical-stability repair; historical pairwise laws are preserved separately.
- Cold core time includes setup+draw; throughput is draws only. Harness time uses a shared outer timer and excludes cold repeat0 enumeration, but validation instrumentation differs across interfaces.
- Frugal K and retained-click k are different approximation budgets, not equal resource parameters.
