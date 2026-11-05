# Pure-core photon-budget feasibility

This is a predeclared analytic resource calculation, not a sampling implementation.
No target P/Q, forest samples, or fitted parameters are used. The exact trace-distance
argument is described in `research/iteration9_core_budget_theory.md`; the numerical
tails below are ordinary floating-point evaluations, not outward-rounded certificates.

For uniform loss, write a=1-eta+eta exp(-2r), b=1-eta+eta exp(2r), and
s=-log(a)/2. The exact covariance decomposition is V=W+C, where the pure core W
has squeeze s and C is positive semidefinite because ab=1+4 eta(1-eta)sinh(r)^2.
For identical core modes, N_core=2J with J negative-binomial(shape=m/2, p=sech(s)^2).
The minimal even K is chosen with Pr(N_core>K)<=epsilon_TV^2; normalized pure-core
truncation then has exact trace distance sqrt(tail) before numerical evaluation.

The same random-displacement channel and measurement contract that exact distance.
In hbar=2 conventions use real quadrature shift d~N(0,C) and alpha=(d_q+i*d_p)/2.
C can be singular; a Gaussian pushforward defines the distribution without a density determinant.
After displacement the output can have odd photon number and unbounded photon support.
The even-sector dimension below belongs only to the truncated pure-core vector.
It is not the cost or support of a completed displaced-threshold sampling algorithm,
and the error budget is not a certificate for any existing forest sampler.

The raw comparison uses squeeze r for the unattenuated pure input. It does not describe
the mixed target's measured photon number. No speed or novelty claim is made.

## Large-system resource check

`log10 D` is the logarithm of sum_even_n<=K binomial(m+n-1,n). A single dense
complex128 core vector would require 16D bytes, before any workspace or measurement cost.

| Profile | m | TV budget | Core mean N | Core K | Core log10 D | Raw-input K | Raw-input log10 D |
|---|---:|---:|---:|---:|---:|---:|---:|
| lossy | 256 | 0.1 | 5.06806 | 14 | 22.928 | 146 | 112.876 |
| lossy | 256 | 0.01 | 5.06806 | 20 | 30.095 | 174 | 124.472 |
| lossy | 1024 | 0.1 | 20.2722 | 36 | 67.065 | 498 | 416.083 |
| lossy | 1024 | 0.01 | 20.2722 | 48 | 83.873 | 550 | 440.533 |
| bright | 256 | 0.1 | 41.1842 | 66 | 69.590 | 352 | 178.038 |
| bright | 256 | 0.01 | 41.1842 | 82 | 79.941 | 406 | 190.132 |
| bright | 1024 | 0.1 | 164.737 | 212 | 244.414 | 1238 | 674.564 |
| bright | 1024 | 0.01 | 164.737 | 244 | 268.062 | 1340 | 700.459 |
| stress | 256 | 0.1 | 168.337 | 226 | 143.085 | 736 | 244.228 |
| stress | 256 | 0.01 | 168.337 | 266 | 155.460 | 838 | 256.705 |
| stress | 1024 | 0.1 | 673.346 | 786 | 536.169 | 2630 | 939.257 |
| stress | 1024 | 0.01 | 673.346 | 858 | 561.458 | 2818 | 965.313 |

At m=1024 the core dimensions in this fixed panel range from 10^67.065
to 10^561.458. The corresponding minimum single-vector storage is about
10^68.269 bytes. The reduced cutoff does not make dense enumeration practical.
For any fixed nonzero core squeezing, mean core photons grows linearly with m;
this computation establishes no polynomial scaling in m for the dense core representation.

## Validation and provenance

The 42 predeclared comparisons contain 84 cutoff calculations.
The accepted cutoff and previous even cutoff tails are both stored. An independent
positive-term long-double PMF recurrence checks the SciPy survival probabilities.
The recurrence geometric remainder is a floating-point termination diagnostic, not an interval certificate.
Validation passed: True; checks: 20;
maximum recurrence/SciPy absolute tail difference: 1.21431e-16.
The one six-mode Haar covariance check, eta=0/1 and vacuum limits, m=2 geometric
tails, and stable log-dimension against exact small integer sums are archived separately.
Plan frozen: 2026-11-05T01:53:49.134434+00:00; evaluation started: 2026-11-05T01:57:18.528569+00:00.
Configuration SHA256: `b2f159fa8a947df67fb0b3e71c93d803c8dcd2406c3e9072b82c073f07bdff6c`.

Files: `budget_rows.json` contains every size/profile/epsilon and both full budget records;
`validation.json` contains checks; `covariance_check.npz` contains U, W, C and target V;
`plan.json`, `config.json`, `source_snapshot/`, `metadata.json` and `manifest.json` bind provenance.

Reproduce in a new directory with:

```sh
PYTHONPATH=src python experiments/iteration9_core_budget.py freeze --output results/new_core_budget
PYTHONPATH=src python experiments/iteration9_core_budget.py run --output results/new_core_budget
```
