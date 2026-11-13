# Iteration 82 methods and reproduction

Executed scientific files are frozen. Use a new source version and a fresh output
directory for corrections; never overwrite a previous run, including failures.

## Factorized integer PNR

The calibrated oriented table is P_l(i,j)=A_li K_ij B_lj. The count vector does
not distinguish (i,j) from (j,i), so no transpose coin is needed. An integer
alias proposal approximates q_i(j)=K_ij/sum_j K_ij. For label l, acceptance
uses ceil((B_lj/max_j B_lj)*2^53) against the top 53 random bits. Construction
computes Z_li from the actual quantized proposal and acceptance probabilities,
then uses root weights proportional to A_li sum_j K_ij max_j B_lj Z_li.
Latent and root aliases are also integer tables. Exact rational audits export
all these tables on small plans; they bound implementation error relative to
the normalized binary64 factor input, separately from quantum approximation.

`gaussian_pnr_factor82_v1.cpp` includes the frozen iteration80/77 low-total
sampler. A record shares one latent label and draws all pairs conditionally.
The plan validates physical pair/cross constraints and owner identity. Zero
rows have unreachable delta proposals. Small reachable acceptance probabilities
and excessive rejection work fail explicitly. Storage is O(m²+Lm), while
preparation still includes O(Lm²) matrix multiplication. Limits include m<=1024,
L<=131072 and L*m<=64 million, so not all combinations within individual limits
are supported. The wrapper restricts equal nonnegative squeezing, zero mean
and no added noise/loss. Native constraints validate supplied factors.

v2 includes v1 unchanged and adds an actual integer-plan vacuum evaluator.
For each label it reconstructs the normalized pair law, then computes no-hit
subset probabilities for at most eight physical ports. The negative-binomial
generating function sums all total pair counts, replacing J<=4 with exact
pure-state coefficients for B8. This audits the theoretical random-draw law,
not independence of the PRNG. A 20-billion-cell work ceiling remains explicit.

## Stable zero-mean direct clicks

The new kernel uses long double matrices (64 significant binary bits on this
build) and __float128 weights (113 bits). It computes click probabilities with
expm1q, retains the finite-budget Gaussian moment compression and tightens
normalization checks. It does not clip conditional probabilities. Displacement
and a wrapper noise-mixture option are rejected; supplied zero-mean mixed
Gaussian covariances remain supported. The rare B128 failures are preserved.
Mixture failure probabilities are computed from disjoint minimal prefix
cylinders and are separate from reference mass on those cylinders.

## Execution order and costs

Drivers run from the repository root with `python -X utf8`, each with
`--output results/<fresh-name>`. The completed frozen outputs are:

1. `iteration82_factor_pilot_v1`: build factor v1; small and large integer records.
2. `iteration82_stable_pilot_v1`: build stable kernel; all three fixed groups,
   two sectors and budgets 1,2,4,8,16,32,128; keep every failing leaf.
3. `iteration82_factor_accuracy_v1`: build factor v2; actual-plan vacuum laws,
   Frugal CPU and direct CPU comparisons on three fixed synthetic inputs.
4. `iteration82_timing_v1`: serial complete timings, three rotations, 16 CPU
   workers, initial CUDA context warmed explicitly. No other heavy job ran
   alongside these panels. PNR includes fresh native design, calibration,
   plan, sampling, counts-to-bits and destruction. The preparation also charges
   the experimental archived-factor relative-difference diagnostic. No I/O,
   DLL loading or initial CUDA context is included. Reuse setup is separate.
5. `iteration82_gpu_accuracy_v1`: force all 256 first-eight paths on the actual
   GPU and compare their conditional products with independent CPU laws.
6. `iteration82_integer_audit_v1`: exact Python Fraction reconstruction of all
   exported labels on eleven small plans; fixed-total N34 TV and guard bounds.
7. `iteration82_tests_v1`: configure the explicitly built optional libraries,
   run the full suite, preserve skips, hash executed sources.
8. `verify_iteration82_v1`: independent verifier without loading native DLLs;
   80-decimal-digit Cholesky references, singular-value low-total coefficients,
   independent factor contractions, record accounting and timing/quality joins.

See each output's plan/build/manifest files for precise commands, dependency
hashes and compiler flags. GCC15.1/Eigen3.4/OpenMP builds disable fast math and
FP contraction. The stable build links quadmath. Runtime DLL directories are
explicit. CUDA is the previously frozen iteration74 fused FP64 implementation.

The first-eight marginal TV is only a lower bound on full TV. Target checks
are high-precision numerical comparisons, not directed-rounding certificates.
The Jiuzhang reference remains the explicitly specified two independent common
spectral sectors, combined by OR, from iteration81. It is not a fitted assertion
that all experimental imperfections are captured. Speed covers only eight
detectors there. Official Frugal cutoff and our branch budget are distinct
parameters; equal numeric budget does not mean equal accuracy or work.
