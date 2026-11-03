# First-experiment methodology audit

Date: 2026-11-03. This document describes implementation scope and claim limits,
not an independent literature novelty certificate.

## 1. What this experiment computes

The current benchmark concerns zero-mean Gaussian optical states, passive
interferometers, independent output attenuation, and ideal **threshold**
detectors. A bit records whether a mode contains at least one detected photon.
It does not reproduce photon-number-resolving outputs, unresolved detector
multiplexing, displacement, partial distinguishability, dark counts, or a
particular large experimental data set.

The covariance convention is hbar = 2, vacuum covariance I, with all x
quadratures followed by all p quadratures. Probability vector index x uses
little-endian bits: bit i is the click in mode i.

Threshold GBS and its exact probability calculations are established subjects;
see Quesada, Arrazola and Killoran, *Gaussian Boson Sampling using threshold
detectors*, Phys. Rev. A 98, 062322 (2018),
<https://arxiv.org/abs/1807.01639>,
<https://doi.org/10.1103/PhysRevA.98.062322>.
The exponential reference implementation here is not a new exact-sampling
complexity result.

## 2. Two mathematically distinct reference calculations

The main reference uses Gaussian no-click probabilities,

\[
q(S)=P(X_i=0\text{ for every }i\in S)
    =\det[(V_S+I)/2]^{-1/2},
\]

and inclusion-exclusion to recover the full threshold distribution on a small
mode subset. Enumerating the full m-mode distribution requires 2^m entries;
small-m validation does not make that step scalable.

The second reference uses the pure state

\[
|\psi\rangle=a_0\exp\!\left(\frac12\sum_{ij}B_{ij}
a_i^\dagger a_j^\dagger\right)|0\rangle,
\qquad B=-U\operatorname{diag}(\tanh r)U^T,
\quad a_0=\prod_i\cosh(r_i)^{-1/2}.
\]

The annihilation identity
\(a_i|\psi\rangle=\sum_j B_{ij}a_j^\dagger|\psi\rangle\)
gives the implemented recurrence, for any i with n_i > 0,

\[
a_{\boldsymbol n}=\frac{1}{\sqrt{n_i}}
\sum_j B_{ij}\sqrt{n_j-\delta_{ij}}
a_{\boldsymbol n-\boldsymbol e_i-\boldsymbol e_j}.
\]

It uses no Gaussian determinant probabilities and no threshold-engine code.
The independent implementation is for validation, with exponentially growing
Fock occupations. General Gaussian Fock representations are documented by
The Walrus: <https://the-walrus.readthedocs.io/en/latest/gbs.html>.

For each pre-loss occupation n, a mode with transmissivity eta has no-click
probability (1-eta)^n. This permits a separate probability-level loss and
threshold calculation. The measurement pulled back through independent loss
is diagonal in the pre-loss Fock basis, so off-diagonal density-matrix
elements do not contribute to these probabilities.

### Photon cutoff and a measurement-specific error bound

The Fock reference retains every occupation with total pre-loss photon number
at most K, records its total mass w, and reports both unnormalized probabilities
and their normalization. Write the full measured distribution as

\[
P=w P_{\le K}+(1-w)P_{>K}.
\]

Therefore

\[
\operatorname{TV}(P,P_{\le K})
=(1-w)\operatorname{TV}(P_{>K},P_{\le K})\le 1-w.
\]

This bound relies on the diagonal loss-plus-threshold measurement. It must
not be described as a general trace-distance bound between the full and
normalized truncated **pure states**; that trace distance is generally
sqrt(1-w). In finite precision an explicit numerical tolerance is also needed.

The tests check analytic single-mode squeezing, two-mode squeezed pair
occupations, vacuum, complete loss, each total-photon sector's invariance
under passive mixing, and the covariance/Fock agreement for complex mixing
with nonuniform squeezing and attenuation. These comparisons test independent
calculations rather than only replaying the same formula.

## 3. Surrogate interpretation and what is established

The candidate sampler is a normalized bounded-parent Bayesian network,

\[
Q(x)=\prod_i P(x_i\mid x_{\mathrm{pa}(i)}),
\qquad |\mathrm{pa}(i)|\le k,
\]

whose conditional tables are calculated from small Gaussian threshold
marginals. The directed factorization and mutual-information parent selection
are established statistical ideas. Their implementation, or a favorable
small-m result, does not by itself establish a new GBS algorithm. Any novelty
claim needs a more focused comparison with existing GBS approximations.

The full target distribution is reserved for small-instance **evaluation**.
It must not be used to fit parent sets or then omitted from the resource
accounting. Pairwise and greedy conditional-mutual-information selection
should query only the low-order marginals reported in their diagnostics.

For the exact conditionals above and a fixed sampling order, the identity

\[
D_{\rm KL}(P\|Q)=\sum_i\left[
H_P(X_i\mid X_{\mathrm{pa}(i)})
-H_P(X_i\mid X_{<i})\right]
\]

follows by expanding the logarithm and using the entropy chain rule.
Pinsker's inequality then gives TV <= sqrt(KL/2) when KL uses natural logs.
Computing full-prefix entropy by full enumeration at small m is an evaluation
diagnostic, **not** a scalable certificate for a large instance.

Greedy conditional information reduces the immediate conditional-entropy
objective, but does not guarantee a globally optimal parent set. A synthetic
even-parity distribution provides a useful test that pairwise independence
does not imply joint independence; that synthetic test is not evidence that
every GBS instance has the same structure. In particular, even **photon**
number from a lossless squeezed input does not imply even **click** count:
collisions can give odd numbers of threshold clicks.

## 4. Numerical accuracy and resource accounting

Inclusion-exclusion subtracts many similar positive numbers. Float64
determinants remain the accuracy bottleneck even if the subtraction uses
extended precision. Small negative roundoff can be diagnosed and corrected;
large or aggregate corrections must trigger failure rather than be hidden.
The exact engine reports its correction diagnostics. At higher modes,
squeezing or extreme attenuation, independent high-precision checking may
be necessary before interpreting tiny differences between algorithms.

A fair timed comparison needs fresh model caches per cold fit. Running a
full exact enumeration first on the same cached object supplies the later
surrogate with an exponentially expensive cache for free. Warm-cache timing
can also be reported, but it must be explicitly labeled and preprocessing
included. Parent selection, covariance preparation, conditional-table
construction, numerical stabilization, storage and actual sampling are
distinct costs worth recording.

Exact sampling from Q can be fast after preprocessing while Q remains a poor
approximation to P. Do not compare that sample rate directly with a quantum
device without matching distribution quality, task, outputs, input
information, batch size and the treatment of preprocessing. No optical
throughput comparison is supplied by the first small-scale experiment.

## 5. Claims permitted by this first stage

Appropriate claims include:

- A reproducible implementation and an independent small-instance reference.
- Measured TV/KL for the specified synthetic instances and detector model.
- A demonstrated improvement or failure within the tested model family.
- A hypothesis for a larger, separately validated next experiment.

The first stage cannot establish a general polynomial-time GBS sampler,
refute boson-sampling complexity conjectures, reproduce all measured outputs
of a large quantum-advantage experiment, or show that quantum hardware has
been beaten at equal quality and cost. It also cannot establish that a
promising construction is novel without checking the relevant literature and
released implementations.
