# Iteration 10 independent reference and protocol audit

Date: 2026-11-05. Owner: independent reference/test reviewer. This document
records the algebra, numerical-reference limits, development checks, and
independent implementation gates. It does not replace the measurement proof in
[iteration10_coherent_povm_theory.md](iteration10_coherent_povm_theory.md) or the
core-state error budget in
[iteration9_core_budget_theory.md](iteration9_core_budget_theory.md).

Status at this revision: all **19 independent tests pass**, comprising 11
reference-only checks and 8 native integration checks, with no skipped tests.
The full 298-test project gate and completed formal evidence have also been
reviewed: all 51 quality laws are available; 60 of 66 timing processes completed
and six retained-k6 processes failed explicitly. Section 9 records that audit.
No formal panel was run to choose these tests, parameters, reference cutoffs,
or rejection thresholds.

## 1. Scope and conclusion

Coherently projecting the pure Gaussian core onto **total** input photon
number at most K, normalizing, and then applying the original correlated
Gaussian displacement channel defines a specific approximate quantum state
and binary threshold law. The ideal-arithmetic guarantee

\[
  \operatorname{TV}(P_V,Q_K)\leq\sqrt{\tau_K}
\]

belongs to that constructed law. Here tau_K is the discarded pure-core photon
probability. This is not a guarantee for retained-click, forest, Frugal, or
other previously implemented laws. A finite-arithmetic sampler with failed
paths needs a separate implementation-error/failure analysis; passing a
floating reference comparison is not an outward-rounded TV certificate.

The local refined measurement is mathematically sound. The independent
validation must nevertheless check the complex bra, coherent number sectors,
all hidden channels, the mixed state obtained after forgetting those channels,
and the full joint displacement distribution. A histogram alone would not
establish any of these properties.

The representation is dense. For r remaining modes its full total-budget
dimension is binomial(r+K,K). An initially even state becomes generally
even-and-odd after a coherent local contraction. An implementation that keeps
only even occupations after the first measurement changes the law. A hidden
channel label is a basis label, not a measured input photon number, so it cannot
be subtracted from the remaining photon budget.

## 2. Independent coefficient construction

The reference lives in
[experiments/iteration10_reference.py](../experiments/iteration10_reference.py).
It does not import the native wrapper, invoke the executable, reuse native
coefficients, or call its measurement kernel.

In hbar=2 and xxpp order, put

\[
 a=1-\eta+\eta e^{-2r},\quad
 b=1-\eta+\eta e^{2r},\quad s=-\tfrac12\log a,
 \qquad B=-\tanh(s)UU^T.
\]

The untruncated vacuum amplitude is sech(s)^(m/2). Starting from relative
vacuum coefficient d_0=1, choose any occupied i in n and use

\[
 d_n={1\over\sqrt{n_i}}
       \sum_{j:n_j-\delta_{ij}>0}
       B_{ij}\sqrt{n_j-\delta_{ij}}\,d_{n-e_i-e_j}.
\]

The reference enumerates occupations by total degree, retains all sectors
through K, and normalizes the entire coherent vector once. It does not draw
a number sector or discard cross-sector phases. The odd-total initial
coefficients are zero. A separate direct pairing-sum hafnian verifies small
coefficients, including repeated-mode multiplicities.

The coefficient norm times the squared raw vacuum amplitude is checked
against the independent negative-binomial retained weight. For identical
squeezes, N_core=2J with J negative-binomial of shape m/2 and success probability
sech(s)^2. K odd, if used, means the same initial projection as K-1; it does not
change the all-parity space needed during measurement.

Useful phase checks include d_(2_i)=B_ii/sqrt(2), d_(1_i,1_j)=B_ij, and the
two-mode interferometer U=1/sqrt(2)[[1,i],[i,1]], whose only nonzero initial
coefficients through K=4 are proportional to 1, -i tanh(s), and -tanh(s)^2
on |00>, |11>, and |22>.

## 3. Independent fixed-displacement law

For a fixed displacement, the finite input-space vacuum vector is

\[
 u_n(\alpha)=e^{-|\alpha|^2/2}{(-\alpha)^n\over\sqrt{n!}}.
\]

For every subset Z, the reference groups Fock occupations by their identical
outside occupation and evaluates

\[
 q_Z=\sum_{n_{\bar Z}}
       \left|\sum_{n_Z}c_{n_Z,n_{\bar Z}}
                    \prod_{j\in Z}u_{n_j}(\alpha_j)^*\right|^2.
\]

A superset Möbius transform converts these all-no-click marginals to the
complete threshold law. Physical mode zero is bit zero. A second small-fixture
calculation directly multiplies the local matrices E0=uu^dagger and E1=I-E0
between every Fock amplitude pair. It shares neither hidden projective channels
nor the Möbius inversion with the native refined-law calculation.

The phase-sensitive fixture

\[
 |\psi\rangle={2\sqrt2\over3}|0\rangle+{i\over3}|2\rangle,
 \qquad \alpha={1+i\over\sqrt2}
\]

has no-click probability 25/(18e). An unconjugated local contraction gives
1/(2e); number-sector dephasing gives 17/(18e). This is a physical truncated
core, not an arbitrary impossible covariance example.

For the channel-budget fixture K=2, input |0,2>, and first-mode alpha=1,
the w-channel has probability 2/5 and the other channels have combined
probability 3/5. Every nonzero channel leaves |2> on the remaining mode, and
the binary no-click probability is 1/e. Restricting channel+remaining_N<=K
would incorrectly discard 3/5 of the probability. Individual orthogonal
channel probabilities depend on the basis completion; the binary law and
coarse-grained remaining density operators do not.

## 4. Analytic Gaussian displacement average

The real residual covariance is

\[
 C=O\operatorname{diag}(0,\delta I)O^T,\qquad
 \delta={4\eta(1-\eta)\sinh^2r\over a},\quad
 O=\begin{pmatrix}\Re U&-\Im U\\\Im U&\Re U\end{pmatrix}.
\]

The positive identity for delta avoids a subtractive negative variance at the
vacuum and lossless boundaries. A whole displacement vector is drawn once:
alpha=i sqrt(delta) U z/2 for z~N(0,I). Although E[alpha alpha^dagger]=delta I/4,
the pseudo covariance E[alpha alpha^T]=-delta UU^T/4 is generally nonzero.
Independent circular complex noise would be a different channel.

For selected modes Z, let y=(Re alpha_Z, Im alpha_Z) and Sigma=C_Z/4. The
Gaussian tilt by exp(-y^T y) has normalization and covariance

\[
 \gamma_Z=\det(I+2\Sigma)^{-1/2},\qquad
 T_Z=(I+2\Sigma)^{-1}\Sigma.
\]

This formula permits singular Sigma; no inverse of Sigma or nonsingular
Gaussian density is needed. For alpha=x+iy under the tilted distribution,

\[
 A=T_{qq}-T_{pp}+i(T_{qp}+T_{pq}),\qquad
 H=T_{qq}+T_{pp}+i(T_{pq}-T_{qp}),\qquad
 G=\begin{pmatrix}A&H\\H^T&A^*\end{pmatrix}.
\]

G is the symmetric second-moment matrix of the complex linear forms
(alpha,alpha*). It is not a Hermitian covariance to be inserted into a
conjugated Wick rule. For occupation pairs with identical outside occupations,

\[
 q_Z=\gamma_Z\sum_{n,n'}c_n^*c_{n'}
 \delta_{n_{\bar Z},n'_{\bar Z}}
 {(-1)^{|n_Z|+|n'_Z|}\,
    E_{T_Z}[\alpha_Z^{n_Z}(\alpha_Z^*)^{n'_Z}]
  \over\sqrt{n_Z!n'_Z!}}.
\]

All moments have degree at most 2K. The independent Isserlis recurrence pairs
one repeated variable with each possible partner, counting multiplicities.
Degree 0, 2, and 4 formulas are evaluated in vectorized form; higher degrees
use a per-subset memoized recurrence and complex-conjugate key symmetry.
There is no Monte Carlo law estimate and no finite integration grid in the
primary average. This exact analytic formula is evaluated in ordinary floating
arithmetic, so its implementation remains a numerical reference.

### 4.1 Small independent integration checks

The one-mode residual has alpha=i y. With v=Var(y), u=v/(1+2v), and t=tanh(s),
the K=2 coherent average has

\[
 q_0={(1+2v)^{-1/2}\,[1+t u+(3t^2/4)u^2]\over1+t^2/2}.
\]

The positive coherence contribution over number-sector dephasing is
(1+2v)^(-1/2) t u/(1+t^2/2). The K=4 check uses moments through degree eight,
with E[y^(2j)]=(2j-1)!! u^j under the tilt. These checks do not reuse the
general Wick recurrence.

A separate deterministic Gauss-Hermite check uses two-mode Haar seed 101105,
r=.4, eta=.4, and K=2 and 4. It directly integrates raw tensor-POVM pattern
expectations, with z=sqrt(2) times the Hermite nodes and product weights divided
by pi. It does not repair or normalize the law at each quadrature node.

The first development check, K=4 at orders 8 and 14, failed its 2e-11
convergence threshold: the maximum difference was
**2.5127122604828855e-11**. The failure was reported; the tolerance was not
relaxed. Before any formal panel, orders 12, 20, and 28 were fixed, with a
1e-12 gate on both 20-vs-28 and 28-vs-analytic-reference differences.

| K | max difference, 12 vs 20 | 20 vs 28 | 28 vs analytic Wick |
|---:|---:|---:|---:|
| 2 | 2.220446049250313e-16 | 4.440892098500626e-16 | 3.3306690738754696e-16 |
| 4 | 3.3306690738754696e-16 | 5.551115123125783e-16 | 4.440892098500626e-16 |

All orders and the earlier failure are also printed in the test diagnostics.
Finite-order quadrature agreement is an independent numerical check, not a
rigorous enclosure of the integral. No monotone quadrature-convergence claim
is made.

## 5. Floating-reference acceptance and resource limits

The reference returns both the raw Möbius law and a clearly identified
corrected probability vector. It rejects nonfinite values, any negative atom
below -1e-11, a raw mass error above 1e-11, or an L1 correction above 1e-9.
Within those fixed tolerances, it sets negative atoms to zero and divides by
the remaining mass. It records the negative mass, minimum, mass residual,
L1/max correction, and all tolerances. These corrections must remain visible
in the archived result and cannot be treated as an exact probability theorem.
The empty-subset marginal is evaluated from the supplied amplitudes' numerical
norm rather than forcibly set to one. Internal long-double corrections,
exported-array corrections, and conversion L1 differences are recorded
separately, so exporting the reference arrays does not conceal a repair.

The Gaussian tilt rejects a nonpositive/nonfinite determinant, an
unrepresentable normalization, or covariance asymmetry exceeding 1e-12.
A material imaginary part of a diagonal expectation exceeding 1e-11 is
also rejected. Accepted covariance symmetry/eigenvalue residuals and all
these diagnostics are saved. The state budget check tolerance in the formal
configuration is 1e-8, explicitly a numerical comparison tolerance.

The fixed reference caps are 12 modes, 2,500,000 compatible unordered
amplitude pairs over all nonempty subsets, and 100,000 cached Wick moments
per subset. Pair counts include all even-basis occupations, including zero
coefficients; exact zeros may be omitted from arithmetic but do not permit
a larger selected panel. Exceeding a cap returns an explicit resource
rejection, with no replacement input.

| Reference | Initial even basis | Full total-budget basis | Nonempty-subset compatible pairs | Largest single-subset pair count |
|---|---:|---:|---:|---:|
| m=4, K=4 | 46 | 70 | 3,633 | 1,081 |
| m=6, K=4 | 148 | 210 | 89,430 | 11,026 |
| m=8, K=4 | 367 | 495 | 1,501,508 | 67,528 |
| m=12, K=2 | 79 | 91 | 2,012,337 | 3,160 |

These are combinatorial counts computed without probability calculations on
formal inputs. Compatible pairs n,n' occur in exactly
2^(m-|{j:n_j!=n'_j}|) vacuum subsets, with the empty subset removed from
diagonal pairs. Work also includes 2^m tilt solves/determinants, grouping,
Wick arithmetic, and Möbius inversion. The count is not a sampling-runtime
model or a scalable full-Q algorithm.

## 6. Frozen experiment design review

The current configuration selects eight fresh primary inputs:
m=4,8; two Haar seed bases 1050801 and 1050802, with seed=base+1000m;
and lossy (r=.6, eta=.35) or bright (r=.9, eta=.65) profiles. K=0,2,4 gives
24 core laws; retained-click budgets 2,4,6 give 24 comparison laws. The
separate m=12 lossy supplement uses the first seed base, K=2, and retained
budgets 4,6. Thus the planned primary and supplement total is 51 laws.

Every formal core row should bind the independent and native occupation
coefficient maps, normalized mass, tail, covariance, input hash, binary/build
identity, and reference diagnostics. Agreement is checked by occupation,
not by an assumed common array order.

**The two comparison laws have deliberately different implementation scopes.**
The core primary value is labelled `construction_reference_tv`: export the
native coefficients, normalize that vector explicitly, and evaluate its law
under the specified **ideal** correlated Gaussian displacement and threshold
POVM by the analytic floating reference. This establishes a complete
construction-level Q. It does not integrate every native long-double rounding,
finite-PRNG outcome, and numerical-failure path over continuous displacement.
The fixed-alpha refined native enumeration is a kernel gate on its explicitly
listed fixtures. A finite formal batch with no failures does not prove a
global native failure probability of zero. The retained-click comparison is
its enumerated extended native law, including the failure symbol. Tables
must identify these scopes instead of presenting both distances as equivalent
native implementation-error guarantees.

Strictly, the sqrt(tau_K) projection theorem concerns exact core coefficients.
For a normalized rounded native vector with pure-state trace distance delta
from the exact truncated state, channel contraction and the triangle
inequality give sqrt(tau_K)+delta. This round supplies a floating coefficient
agreement gate and a 1e-8 nominal bound check, not an outward certificate of
delta or of the total finite-arithmetic sampler error.

The timing plan has 66 fresh native processes: three repetitions at two batch
sizes for five methods at m=8, three methods at m=12, and three methods at
m=32. Native setup plus sampling includes basis/state construction, the joint
displacement draws, all contractions/normalizations, RNG, and allocation.
Wrapper/input/output time is reported separately. It uses a common host and
single-thread settings, but the host is not claimed to be dedicated.

Quality-cost conclusions require measured full laws for both methods and
must disclose unequal quality. The m=32,K=4 dense basis has 58,905 states;
its short timing batches carry only the ideal core tail budget, without a
computed full-Q TV certificate or a demonstrated finite-arithmetic TV
budget. The m=64,K=6 request is a predeclared resource rejection, not an
alternative timing input and not a numerical-precision failure.

## 7. Validation state and remaining review

The 11 independent reference tests passed in the first successful development
run in 4.162 s, and again in 4.549 s after the explicit input-norm and array
conversion diagnostics were added. The table above records the latter run.
They cover direct pairing coefficients, two-mode complex interference,
phase/coherence counterexamples, tensor-POVM versus marginal inversion,
singular Gaussian averaging through degree eight, independent quadrature,
K=0 covariance marginals, vacuum/lossless boundaries, the small-system state
budget, resource caps, and visible repair/rejection rules.

The completed native integration tests used the already built executable via
GBSLAB_COHERENT_CORE_EXECUTABLE and did not compile. The eight checks compare
occupation-indexed coefficients at m=1,2,4,6,8; the phase-sensitive binary
probability; all hidden channels in the budget fixture; both coarse-grained
remaining density matrices including odd support; complete fixed-alpha native
laws (refined and vacuum at m=2,4; vacuum at m=6,8); jointly injected Gaussian
noise against fixed displacement and identical quantum uniforms; a positive
Poisson tail below binary64 epsilon; and stopped batches/failed-mass
enumeration/resource rejection. In the last fixture, one complete sample is
retained before the next displacement fails, with no retry or replacement.

The combined independent run passed **19/19, zero failures, errors, or skips,
in 4.923 s**. The executable digest was checked before and after all native
tests. Its unchanged digest is
`f6a9f7c8447a73d07485314c9c1ebd8ea0b1a09383e5dcbfde6d2eb4c794aac2`.

The archive is
[results/iteration10_independent_validation/validation.json](../results/iteration10_independent_validation/validation.json),
with the complete test log and source snapshot beside it. The validation file
SHA-256 is `5198bb04a0f332d7128900f806272b1b19cc93ae6896a88fae018eb0fe56060a`;
the test log is `4e3d9c919f990c923dc11d15766da47ed183ca979ce8b4133e41d5d55d99e993`.
The earlier low-order quadrature failure is preserved in both records.

The independent scientific source and tests are frozen at:

| File | SHA-256 |
|---|---|
| experiments/iteration10_reference.py | `fa90d4edb1426de93ee75ff5ca11f9f0161ff4a5afae49032ead609be79d6f29` |
| tests/test_coherent_core_independent.py | `257fecdf606f4f0fbc406847d8f5559233b817af3b247b988cfcf2a0afc0f7dd` |

These are fixture results, not a substitute for the project's final full-suite
gate or for the formal construction-quality and sampling-cost panel. No hash is
invented for an unreviewed or not-yet-created runner. The completed formal-result
review is recorded separately in Section 9.

## 8. Analysis protocol frozen before the formal panel

The independent analyzer is
[experiments/iteration10_analyze.py](../experiments/iteration10_analyze.py),
frozen at SHA-256
`ce633b718646063d3dd06975d5c8dbc6edf5f1e0916bc2094f4eb93ca0b1d566`.
The metadata helper is frozen at
`a82b67f59e23ce288aecf9c7088da45c07c606d3c277e76d0d10a361ebfdd4bc`.
Both were completed before the full project gate and formal calculations.
Their plan keys were independently compared with the runner: 9 quality
inputs, 25 core construction laws and 26 retained laws, plus 66 timing jobs.
The complete seeded timing permutation also agreed, not merely its set of jobs.

The metadata helper fixes a concrete NumPy extended-precision problem:
longdouble.item() can still be a longdouble, so recursive generic conversion
did not terminate. Ordinary finite floating values now become JSON numbers.
A finite nonzero value that would become binary64 zero, or a finite value that
would become infinity, remains a decimal string. The fixed development probes
included ordinary values, 1e-400, 1e400, complex arrays, zero, integer/boolean
scalars, and explicit nonfinite rejection. Full native coefficient precision
is separately preserved in NPZ; the metadata conversion does not claim to
preserve every low-order mantissa bit of ordinary long-double values.

The analyzer requires the full fixed key sets and verifies archived gate,
build, source snapshot, artifact manifest, input, law, coefficient, and raw
timing hashes. It re-reads target P and saved Q arrays and recomputes each TV.
It does not recompute targets from the covariance or rerun the analytic Wick
integration. For core rows it independently checks occupation alignment,
the native JSON/NPZ coefficient relation, normalization into the reference
input, covariance splitting, the tail/budget binding, raw versus corrected
reference probabilities, and saved-marginal Möbius consistency. Complete
core rows retain `failure_mass=None`.

The successful full panel is expected to have 76 explicit scientific
crosschecks: two for each of 25 core rows and one for each of 26 retained
rows. The analyzer also handles a complete matrix with explicitly unavailable
rows. It does not turn that status into an all-success panel. An unknown
completion count after timeout remains unknown, not zero.

Each timing median requires all three predetermined repetitions. A failed
repetition leaves the median unavailable; a median of the remaining successful
runs is not substituted. The principal cost remains setup plus sampling.
Input+setup+sampling, wrapper validation, process wall, and total wrapper time
are preserved separately. Argument parsing and output serialization are not
silently included in a narrower native component timer. Continuous TV
differences and time ratios are reported without adding a data-dependent or
invented tie rule. Any “construction reference not worse and faster” flag
explicitly compares the two different evidence scopes and is not a native
Pareto/error certificate. The m=32 rows have no such flag.

This section records the pre-panel analysis design and code identities only.
The formal P/Q and timing artifacts had not yet been read at this point.

## 9. Completed formal evidence audit

The frozen analyzer was run **once**, after the quality and timing processes
finished. It exited successfully and wrote
[summary.json](../results/iteration10_analysis/summary.json),
[summary_tables.md](../results/iteration10_analysis/summary_tables.md), and
the scientific PNG/SVG figure. No target law, analytic displacement integral,
sampling trajectory, or formal experiment was rerun by this audit.

The audit re-read and hash-checked **444 distinct saved evidence paths**,
including all source snapshots and artifact manifests. It loaded all 51 saved
P/Q pairs, 25 native/reference coefficient archives and the associated native
JSON outputs, and all 66 raw timing results. It independently recomputed TV
values, coefficient consistency, reference corrections, each timing median,
the full planned key sets, and the exact seeded job permutation. Current
scientific source hashes still matched the frozen full-suite gate.

### 9.1 Verified completion and scope

| Evidence | Verified result |
|---|---|
| Complete project gate | 298 tests; zero failures, errors, or skips; 37.06618682199951 s |
| Quality panel | 9 covariance inputs; 51/51 complete laws; 76/76 checks; 67.29428299000028 s |
| Primary quality | 8 inputs and 48 laws: 24 core, 24 retained |
| Separate supplement | m=12 lossy input; 1 core and 2 retained laws |
| Timing | All 66 planned processes recorded; 60 complete and 6 numerical failures; 49.444044014999236 s |
| Complete median cells | 20 of 22 method/input/batch cells have all three repetitions |
| Largest native/independent coefficient difference | 1.7564075194265172e-16 |
| Largest reference-law repair L1 | 5.335665940244547e-16 |

The 8 primary covariance inputs use **4 distinct Haar unitaries**: two seeds
at each of m=4 and m=8, with each unitary reused for the two declared optical
profiles. The m=12 supplement adds one unitary. Thus quality has 5 distinct
unitaries and 9 covariance inputs, not 9 independent Haar matrices. Timing
reuses the m=8/m=12 physical inputs and adds one m=32 unitary/input.

All 25 core rows retain the correct construction-reference scope and
`failure_mass=None`. All 26 retained quality enumerations completed, with
saved failure mass zero for those enumerated laws. These statements do not
transfer to the m=32 sampling jobs or to the global finite-native core law.

### 9.2 Quality and cost outcome

The following recorded values were independently recomputed from saved P/Q
arrays and raw three-run timing groups. TV columns deliberately retain their
different scopes.

| Input, first seed | Method | Core construction-reference TV | Retained extended TV | Cold batch, N=1024, median seconds |
|---|---|---:|---:|---:|
| m=8 lossy | core K=2 | 0.005628950913783894 | — | 0.034565003999999996 |
| m=8 lossy | core K=4 | 0.0002854520424967267 | — | 0.285666601 |
| m=8 lossy | retained k=4 | — | 0.000028998009169812824 | 0.004727193 |
| m=8 lossy | retained k=6 | — | 5.992683003614223e-8 | 0.00471783 |
| m=12 lossy | core K=2 | 0.009634317179383565 | — | 0.104240987 |
| m=12 lossy | retained k=4 | — | 0.0002466201965042413 | 0.010524688 |
| m=12 lossy | retained k=6 | — | 5.488472047085281e-6 | 0.012069363999999999 |

There are 16 predeclared paired cost cells with complete quality values for
both compared methods. **None** meets the descriptive condition
“construction-reference TV no larger and cold batch time smaller.” No new
numerical tie threshold was introduced. This is already an unfavorable
outcome for the present dense prototype before attempting a stronger global
finite-native error comparison.

For example, at m=8 lossy and N=1024, core K=4 takes about 60.43 times the
cold-batch time of retained k=4 while its construction-reference TV is about
9.84 times the retained extended-law TV. These are measurements on the fixed
instance and shared host, not a general lower bound on coherent methods.
Increasing K improves these fixed construction laws, but the resulting dense
state work is substantial; there is no reported classical-frontier win.

### 9.3 Larger-input observations and explicit failures

For m=32 lossy, core K=4 has a 58,905-dimensional full basis. Its cold-batch
medians are 1.830535952 s at N=16 and 13.942826520999999 s at N=128.
Its nominal ideal projection budget is 0.06926937701926986. No complete
constructed Q, retained Q, or target P was computed at this size. The nominal
budget and successful finite batches do not supply a finite-native TV bound.

Retained k=4 completed the same two m=32 batch sizes, with medians
0.0023034659999999997 s and 0.023501053 s. Those are cost-only observations;
the absence of full laws prevents an equal-quality comparison.

All six failed timing processes were m=32 retained k=6. For each of N=16
and N=128, repetitions 0,1,2 reported 0,2,0 completed samples before the
unchanged cancellation-indicator guard rejected the next transition.
The three reported indicators were approximately 1.026770079e-6,
1.005839373e-6, and 1.761685136e-6, above the fixed 1e-6 threshold.
The two batch sizes reuse the same three seeds, so the duplicated failure
prefixes are not six independent probability trials. The old retained
backend records completion counts but does not preserve those partial raw
sample bits. The failed attempts were retained, not retried or replaced;
both retained-k6 median cells remain unavailable. The timing metadata's
`investigation_required` status is preserved.

The separate m=64,K=6 resource request was rejected before state allocation
or backend invocation. Its full basis is 131,115,985, the sum of stage bases
is 1,329,890,705, and the planned major-container payload is 87,072,514,697
bytes. This is a combinatorial planning estimate, not a measured RSS or a
numerical failure. The preflight rejection is outside the 66 timing jobs and
does not replace any of them.

### 9.4 Figure QA and immutable numerical outputs

The first PNG/SVG render had crowded minor-log-tick labels in the m=32 panel.
The frozen analyzer and its summary were not edited or rerun. A separate
presentation-only script in the analysis output directory rerendered the
unchanged summary through the frozen plotting function, suppressed those
minor labels, and explicitly labelled the missing retained-k6 curve as failed.
The first images remain in `initial_render/`. The final figure was visually
inspected: its labels are readable, all missing quality cells remain visible,
and the m=32 failure/no-full-Q restrictions are explicit.

[layout_adjustment.json](../results/iteration10_analysis/layout_adjustment.json)
records both image versions and verifies that the summary SHA-256 was
unchanged. This was a layout adjustment only, with no new experiment or
statistical postprocessing rule.

| Frozen or final artifact | SHA-256 |
|---|---|
| Configuration | `4fff4925ccaeea2c618eb8eafa829abad2ae8bdaa201d5d2ccfea9a8afc4c844` |
| Full-suite validation | `4a05d2d2ea367e8e48a05fb5432524f87e63929c8aa0a285f3905092bb231e6e` |
| Quality metadata | `2305d288bf2a0e7359e20db08919528c50696b103da8d168c1ed617fe98ca8bd` |
| Timing metadata | `929f7db3e45e6a9c47466384017d44eb9e1166e2c220dc200bc33f1ccf36001f` |
| Summary | `c35a9cb7f88f4c04e795bc73ede83079ed53c281968f4ba31a1250053123f9c0` |
| Summary tables | `cc2c39ecfc739503f19434774395d7e2669793f12abee72c3d8609aa2aaacd71` |
| Final PNG | `b7d37069eaf288a1298725bc136df680ea6c8ec508830cd9430c6ff4b2a46743` |
| Final SVG | `fc4d306d8f873f6b725e9b792ce0846f9ff7618006534906000b476e933bf931` |
| Layout-only adjustment record | `8437276d3734f4629a89de7268cbcbf2c25be757be4e4a2531ab5a70a8431d9e` |
