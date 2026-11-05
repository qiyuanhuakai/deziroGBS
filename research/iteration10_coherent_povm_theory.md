# Iteration 10: coherent finite-core sampling by refined threshold measurements

Date: 2026-11-05. Scope: independent exact-arithmetic derivation and
primary-literature review. This note does not modify an earlier implementation
or certify a floating-point implementation. The iteration-9 foundation and
resource limits remain those in
[iteration9_core_budget_theory.md](iteration9_core_budget_theory.md).

## 1. Mathematical conclusion

The proposed local measurement refinement is valid. For every normalized pure
state with total input photon number at most \(K\), every finite displacement
vector, and every order in which distinct output modes are measured, it
generates exactly the displaced threshold **joint binary law** in exact
arithmetic. The sampler may retain a single pure state conditioned on its
internal measurement labels. Marginalizing those labels recovers the mixed
state conditioned only on the reported binary history.

This removes the need to store a growing mixed-state decomposition along one
sample trajectory. It does not make the dense pure-state representation small,
nor does a trajectory provide the marginal likelihood of its binary output.
The method is an instance of established POVM refinement and Gaussian Fock
recurrences, not a claimed new general measurement principle or a scalable
solution to large GBS instances [Pellonpaa14, Yao24, Oh24].

The two most consequential implementation requirements are:

- A local contraction uses the complex-conjugated measurement vector.
- The unmeasured state remains supported at total photon number at most \(K\),
  but can have both parities. A hidden channel index is not a photon count
  to subtract from that budget.

## 2. The finite input space and exact displaced threshold POVM

For \(r\) unmeasured modes, use

\[
 \mathcal H_{r,K}
 =\operatorname{span}\{|\boldsymbol n\rangle:
      n_j\geq0,\ |\boldsymbol n|\leq K\},
 \qquad
 D_r=\dim\mathcal H_{r,K}={r+K\choose K}.
 \tag{1}
\]

It is a subspace of \(\mathcal L_K^{\otimes r}\), where
\(\mathcal L_K=\operatorname{span}\{|0\rangle,\ldots,|K\rangle\}\).
The local dimension \(K+1\) therefore suffices for the **undisplaced input**
of every measurement in the procedure.

Use \(D(\alpha)=\exp(\alpha a^\dagger-\alpha^*a)\), with
\(q=a+a^\dagger,\ p=(a-a^\dagger)/i\), so a displacement changes quadrature
means by \((2\Re\alpha,2\Im\alpha)\). Pulling the vacuum effect back through
the displacement gives

\[
 D(\alpha)^\dagger|0\rangle\langle0|D(\alpha)
 =|-\alpha\rangle\langle-\alpha|.
\]

Let \(P_K^{\rm loc}\) project onto \(\mathcal L_K\), and set

\[
 u=P_K^{\rm loc}|-\alpha\rangle,\qquad
 u_n=e^{-|\alpha|^2/2}\frac{(-\alpha)^n}{\sqrt{n!}},
 \quad 0\leq n\leq K.
 \tag{2}
\]

The exact restricted effects are

\[
 E_0=uu^\dagger,\qquad E_1=I_{K+1}-uu^\dagger.
 \tag{3}
\]

They are not the effects obtained by truncating the output displacement
matrix and discarding high output photon numbers. Equations (2)–(3) already
include the infinite displaced output space in the click effect.

Writing \(x=|\alpha|^2\),

\[
 t=u^\dagger u=e^{-x}\sum_{n=0}^K\frac{x^n}{n!}
    =\Pr\{\operatorname{Poisson}(x)\leq K\},\qquad
 q=1-t=\Pr\{\operatorname{Poisson}(x)>K\}.
 \tag{4}
\]

For every finite \(K,\alpha\), \(t>0\). If \(\alpha\neq0\), then \(0<t<1\)
and \(q>0\); if \(\alpha=0\), then \(t=1,q=0,u=|0\rangle\).
Thus \(E_0,E_1\succeq0\) and \(E_0+E_1=I_{K+1}\). A numerical zero caused
by underflow in \(t\) or cancellation in \(q\) is not an analytic zero.

## 3. Refined outcomes and pure conditional states

Define \(w=u/\sqrt t\), and complete it to an orthonormal basis
\(\{h_0=w,h_1,\ldots,h_K\}\) of \(\mathcal L_K\). For a normalized current
state \(|\Psi\rangle\in\mathcal H_{r,K}\), define vectors on the other modes

\[
 |b_c\rangle=(\langle h_c|\otimes I)|\Psi\rangle,\qquad
 \pi_c=\langle b_c|b_c\rangle,\qquad
 \sum_{c=0}^K\pi_c=1.
 \tag{5}
\]

One measurement step can be performed as follows:

1. Draw a hidden channel \(c\) with probability \(\pi_c\).
2. For \(c=0\), report no-click with probability \(t\), and click with
   probability \(q\). For \(c>0\), report click.
3. Retain the normalized pure state
   \(|b_c\rangle/\sqrt{\pi_c}\) on the unmeasured modes.

A zero-probability channel is never selected. Conditional on channel zero,
the extra binary coin does not change its remaining state: its two Kraus
operators differ only by a scalar.

### 3.1 Exact coarse-grained effects and conditional state

The corresponding destructive, rank-one Kraus maps are

\[
 L_{0,0}=\sqrt t\,\langle w|,\qquad
 L_{1,0}=\sqrt q\,\langle w|,\qquad
 L_{1,c}=\langle h_c|\quad(1\leq c\leq K).
 \tag{6}
\]

They obey

\[
 L_{0,0}^\dagger L_{0,0}=E_0,\qquad
 \sum_{c=0}^K L_{1,c}^\dagger L_{1,c}
 =q|w\rangle\langle w|+\sum_{c=1}^K|h_c\rangle\langle h_c|
 =E_1.
 \tag{7}
\]

The unnormalized conditional remaining states, after discarding the hidden
label, are therefore

\[
 \sigma_0=t|b_0\rangle\langle b_0|,\qquad
 \sigma_1=q|b_0\rangle\langle b_0|
                  +\sum_{c=1}^K|b_c\rangle\langle b_c|.
 \tag{8}
\]

These are precisely
\(\operatorname{tr}_{\rm measured}[(E_x\otimes I)
|\Psi\rangle\langle\Psi|]\), equivalently the reduced states obtained by the
physical local displacement, threshold measurement, and destruction of
that mode. The partial-trace identity follows by expanding the measured
index in any orthonormal basis.

The qualification “destructive” matters. The refined procedure need not
reproduce the post-measurement state of a measured optical mode retained
for subsequent coherent interaction. GBS measures each output once and
does not reuse it, so the reduced-state identity (8) is sufficient.

### 3.2 Exact joint binary law and arbitrary measurement order

For fixed \(\boldsymbol\alpha\), summing a hidden path probability over all
labels compatible with a binary pattern gives

\[
 Q_K(\boldsymbol x\mid\boldsymbol\alpha)
 =\langle\psi_K|
       \bigotimes_{j=1}^m E_{x_j}(\alpha_j)
   |\psi_K\rangle.
 \tag{9}
\]

To see this directly, multiply the unnormalized Kraus contractions along a
path. Intermediate normalizations cancel from its probability. Summing the
outer products of the Kraus factors at each mode uses (7), yielding (9).
Operators on distinct modes commute, so every permutation of the measured
modes gives the same binary law, provided physical mode labels and
displacement components are permuted consistently.

Equivalently, induction using (8) shows that the ensemble of stored pure
states is the correct conditional mixed state at each binary prefix.
Continuing with one sampled ensemble member is valid. Replacing that
ensemble by a sum of its amplitudes, or always choosing its channel-zero
member after a click, is not valid.

The contractions in (5) act on the coherent state itself. They do not first
measure total input photon number. Coherence between number sectors is
retained wherever it contributes to the remaining conditional amplitudes.

## 4. Stable orthogonal completion and complex phases

Because \(u_0=e^{-x/2}>0\), \(w_0\) is real and positive. Let

\[
 z=e_0+w,\qquad d=1+w_0,\qquad
 H=\frac{zz^\dagger}{d}-I_{K+1}.
 \tag{10}
\]

Since \(z^\dagger z=2d\), \(H=H^\dagger\), \(H^2=I\), and \(He_0=w\).
Thus the columns \(h_c=He_c\) give the required basis. Here \(d\geq1\);
this avoids normalizing \(e_0-w\), which is small near vacuum.
At \(\alpha=0\), it gives \(H=\operatorname{diag}(1,-1,\ldots,-1)\).
For \(K=0\), it gives \(H=(1)\).

The local contraction is

\[
 b_c(\boldsymbol r)=
 \sum_{n=0}^{K-|\boldsymbol r|}
       H_{n,c}^{*}\,\Psi(n,\boldsymbol r).
 \tag{11}
\]

It uses a bra, not an unconjugated transpose. In particular, the no-click
amplitude uses \((-\alpha^*)^n\), not \((-\alpha)^n\).

An exact phase-sensitive check is

\[
 |\psi\rangle=\frac{2\sqrt2}{3}|0\rangle+\frac{i}{3}|2\rangle,\qquad
 \alpha=\frac{1+i}{\sqrt2}.
\]

The correct vacuum probability is \(25/(18e)\); using \(u^T\) instead of
\(u^\dagger\) produces \(1/(2e)\). The state is a valid normalized \(K=2\)
projection of a core with \(\tanh s=1/2\) and a one-mode interferometer
\(U=e^{-i\pi/4}\). This is an algebraic fixture, not a sampled observation.

For stable evaluation, the common exponential may be removed when forming
the direction,

\[
 w_n=\frac{(-\alpha)^n/\sqrt{n!}}
           {\sqrt{\sum_{\ell=0}^K x^\ell/\ell!}}.
 \tag{12}
\]

Scaled or logarithmic evaluation can prevent simultaneous underflow of all
entries of \(u\). The weights \(t,q\) still have to be evaluated. A direct
positive Poisson-tail evaluation can resolve a small \(q\) that subtraction
from a rounded \(t\) loses. A near-unit float is not a proof that its
complement vanishes. These observations are numerical design principles,
not interval error bounds for an implementation.

## 5. The support bookkeeping that the refinement requires

Equation (11) shows that every retained internal state lies in
\(\mathcal H_{r-1,K}\). Its occupation indices still satisfy
\(|\boldsymbol r|\leq K\), because a bra on another mode cannot introduce
new photons into those indices.

However, even if the initial core has only even total number, a displaced
local bra can combine terms with different measured-mode occupation.
The remaining state can then contain both even and odd total number.
Only the initial core coefficients can be indexed by the even sector
alone. Subsequent pure states need the full space (1).

For a fixed remaining index \(\boldsymbol r\), the original local fiber has
length \(K-|\boldsymbol r|+1\). It must be padded with zero entries to length
\(K+1\) before applying \(H^\dagger\). The hidden channel can take every
value \(0,\ldots,K\), including \(c>K-|\boldsymbol r|\). It is a basis label,
not a measured number, and must not reduce the remaining photon budget.

For example, take \(K=2\), input \(|0,2\rangle\), and measure the first mode
with \(\alpha=1\). The Householder basis (10) gives

\[
 w=\frac{(1,-1,1/\sqrt2)^T}{\sqrt{5/2}},\qquad
 (\pi_0,\pi_1,\pi_2)=(2/5,2/5,1/5).
\]

All three normalized remaining states are \(|2\rangle\), up to phase.
Enforcing \(c+|\boldsymbol r|\leq K\) would discard probability \(3/5\).
The correct binary no-click probability is
\((2/5)\,[5/(2e)]=1/e\).
The individual nonzero-channel weights depend on the chosen orthogonal
completion; their sum and the binary law do not.

The finite-support statement concerns the internal **undisplaced**
representation. The actual displaced remaining physical state can have
unbounded photon support. Future displacements are deferred into their
local effects (3); no physical output cutoff is being silently imposed.

## 6. Building the coherent Gaussian core

Use the same q-squeezed convention as the iteration-9 note. For equal core
squeeze \(s\) and passive unitary \(U\), define the complex symmetric matrix

\[
 B=-\tanh(s)\,UU^T.
 \tag{13}
\]

The transpose is not an adjoint: replacing \(UU^T\) by \(UU^\dagger\) would
remove the intended interference structure. The pure core has the expansion

\[
 |\psi_p\rangle=c_{\boldsymbol0}
    \exp\!\left(\frac12\sum_{j,\ell}B_{j\ell}
                    a_j^\dagger a_\ell^\dagger\right)|\boldsymbol0\rangle,
 \qquad
 c_{\boldsymbol0}=\operatorname{sech}^{m/2}s.
 \tag{14}
\]

Equivalently, \(c_{\boldsymbol0}=\det(I-BB^\dagger)^{1/4}\), with a positive
vacuum phase. Equations (13)–(14) and the following recurrence are standard
Gaussian Fock representations [Yao24, Eq. (48), Eq. (70), Sec. III.2].

The commutator with the quadratic exponential gives
\(a_j|\psi_p\rangle=\sum_\ell B_{j\ell}a_\ell^\dagger|\psi_p\rangle\).
For any multi-index with \(n_j>0\), matching the coefficient of
\(|\boldsymbol n-e_j\rangle\) gives

\[
 \sqrt{n_j}\,c(\boldsymbol n)
 =\sum_{\ell:\,n_\ell-\delta_{j\ell}>0}
       B_{j\ell}\sqrt{n_\ell-\delta_{j\ell}}\,
       c(\boldsymbol n-e_j-e_\ell).
 \tag{15}
\]

The excluded terms are exactly those with invalid negative indices.
This is triangular in total number. Starting with the vacuum, all odd
coefficients vanish. Two useful checks are
\(c(2e_j)=B_{jj}c_{\boldsymbol0}/\sqrt2\) and
\(c(e_j+e_\ell)=B_{j\ell}c_{\boldsymbol0}\) for \(j\neq\ell\).
Any valid choice of \(j\) gives the same exact amplitude; an implementation
should fix a deterministic choice for reproducibility of floating arithmetic.

Computing coefficients only through \(K\) does not approximate coefficients
inside that support: (15) depends only on lower-number coefficients.
Normalizing their coherent vector produces precisely
\(\Pi_K|\psi_p\rangle/\sqrt{1-\tau_K}\). A common nonzero initial scale,
such as replacing \(c_{\boldsymbol0}\) by 1, is permissible if the final
vector is normalized; it must not be mistaken for physical unnormalized
probability mass when computing the tail.

### 6.1 Sampling the correct correlated displacement

For uniform loss, put

\[
 \delta=b-a^{-1}
        =\frac{4\eta(1-\eta)\sinh^2r}{a}.
\]

The nonnegative product form is preferable to subtracting two nearly equal
variances near a boundary. With \(z_{\rm real}\sim\mathcal N(0,I_m)\), the
full displacement vector can be generated as

\[
 \boldsymbol\alpha=\frac{i\sqrt\delta}{2}\,U z_{\rm real}.
 \tag{16}
\]

Indeed, its real quadrature shift is
\(\sqrt\delta\,O_{:,m:2m}z_{\rm real}\), with covariance \(C\) from
iteration 9. Although

\[
 \mathbb E[\boldsymbol\alpha\boldsymbol\alpha^\dagger]
       =\frac{\delta}{4}I,\qquad
 \mathbb E[\boldsymbol\alpha\boldsymbol\alpha^T]
       =-\frac{\delta}{4}UU^T,
 \tag{17}
\]

the second matrix is generally nonzero and can couple modes. The vector
is generally not circular complex Gaussian. Replacing it by independent
circular noise on the basis of the first covariance would change the
target. Draw the joint vector once for each attempted sample; no
unconditional resampling of future components after observing a binary
prefix is justified.

## 7. Costs and the meaning of avoiding mixed-state rank growth

A direct calculation of all \(K+1\) channel contractions in (11) takes at
most \(O((K+1)D_r)\) arithmetic operations at one mode. The rank-one form
of \(H+I\) allows its action to be evaluated without building a dense
transformation matrix. With straightforward positive norm accumulation
over all padded fibers, the conservative complete-trajectory bound remains

\[
 O\!\left(\sum_{r=1}^m(K+1)D_r\right)
 \subseteq O\!\left(m(K+1){m+K\choose K}\right).
 \tag{18}
\]

This is a finite-core arithmetic bound, not a bit-complexity guarantee.
It excludes construction of \(B\), index maps, allocation, random-number
generation, and numerical fallback work unless those are separately counted.
Forming \(UU^T\) is ordinarily cubic setup work, and applying (16) to each
real random vector ordinarily costs \(O(m^2)\).

One pure state costs \(O(D_r)\) complex amplitudes, plus index and work
storage. Storing every channel vector simultaneously can use
\(O((K+1)D_{r-1})\) extra space; accumulating channel norms and then
recomputing only the selected vector trades that storage for work.
The initial even-core dimension remains the smaller sum reported in
iteration 9, but it is not a valid general later-state allocation.

Selecting one hidden trajectory avoids keeping all its alternative
conditional density-matrix terms. Strongly enumerating all hidden paths,
or integrating an exact binary likelihood over them, can still require
exponential work. The fixed-error cutoff can itself grow linearly with
\(m\), so the binomial in (18) is still exponential in that regime.
No MPS compression is supplied by the hidden-channel construction alone.

## 8. Likelihood, approximation errors, and failure

### 8.1 A hidden trajectory is not a binary likelihood

Given \(\boldsymbol\alpha\), the product of selected \(\pi_c\) values and
the applicable \(t/q\) coin probabilities is the joint probability of
that hidden-channel/binary trajectory. It is not
\(Q_K(\boldsymbol x\mid\boldsymbol\alpha)\).

Even the binary probability \(p_0=t\pi_0\) computed in a stored pure state
is conditioned on its **hidden history**, rather than only on the binary
prefix. Multiplying those conditional binary values along a sampled hidden
trajectory therefore does not produce the marginal binary probability.
The output law requires both sums and an integral:

\[
 Q_K(\boldsymbol x)=
 \int\sum_{\boldsymbol c\ {\rm compatible\ with}\ \boldsymbol x}
       \Pr(\boldsymbol x,\boldsymbol c\mid\boldsymbol\alpha)\,
       \mu_C(d\boldsymbol\alpha).
 \tag{19}
\]

A diagnostic field such as **log_joint_hidden_given_alpha** can record a
defined path weight. It must not inherit the old samplers' **logQ** meaning.
An exact or independently evaluated small-system binary law can be reported
separately, together with any integration error of its numerical evaluation.

### 8.2 What follows from the iteration-9 bound

In exact arithmetic, the present measurement procedure samples the same
\(Q_K\) used in the earlier theorem. Therefore

\[
 \operatorname{TV}(P_V,Q_K)\leq\sqrt{\tau_K}.
 \tag{20}
\]

If a stored normalized core has a proved trace-distance error
\(\delta_{\rm state}\), the common displacement and measurement channels
contract that error. If the actual sampling procedure additionally has a
proved output-law error \(\delta_{\rm sample}\), then

\[
 \operatorname{TV}(P_V,Q_{\rm implemented})
 \leq\sqrt{\tau_K}+\delta_{\rm state}+\delta_{\rm sample}.
 \tag{21}
\]

Model, displacement, and measurement approximations must be included in
valid terms of that budget. Finite-precision moment agreement, compensated
sums, normalization tolerances, and agreement of two tail calculations are
not proofs of such a bound. In particular, a Poisson-tail evaluation or a
rounded core coefficient does not acquire a strict interval certificate
merely because the mathematical expression is positive.

If a sampler can fail on a displacement or hidden state, its failure
symbol must be retained in the output law. Retrying until success
generally changes the sampled displacement/trajectory distribution.
The ideal state theorem alone does not cover that change or the failure
mass. A numerical implementation and its empirical checks must retain
this distinction.

## 9. Primary literature and interpretation

| Primary source | Directly relevant established method | Consequence for the present work |
|---|---|---|
| Pellonpää [Pellonpaa14], Sec. III | Rank-one refinement of a POVM and recovery of the original outcomes by relabeling; distinction between effects and instruments. | Hidden labels and coarse graining are standard measurement constructions. The destructive reduced-state identity is derived here for this threshold instance. |
| Yao, Miatto, and Quesada [Yao24], Eq. (48), Eq. (70), Sec. III.2 | Gaussian Fock amplitudes obey a linear recurrence; the squeezed-input matrix is \(-U\operatorname{diag}(\tanh r_j)U^T\). | Equation (15) is the zero-mean specialization, with the correct sign and transpose. |
| Miatto and Quesada [Miatto20] | Direct recursive Gaussian operator matrix elements, with attention to Fock-cutoff errors. | Recurrence-based finite Fock construction is established; truncating an intermediate displacement matrix is not equivalent to the exact effect (3). |
| Oh et al. [Oh24], Sec. III | Pure Gaussian core plus correlated displacement, MPS representation, and sequential sampling after local displacement. | The overall pure-core/displacement route is prior art. This note provides no priority claim for combining these ingredients. |
| Liu et al. [Liu26], Sec. II.3 and III | Low-entanglement Gaussian-to-MPS construction via projected creation operators, with explicit dependence on retained bond dimensions. | A possible compressed representation must be studied separately; the dense sampler's small-core correctness does not establish that compression or its global error. |

The cited papers suffice to rule out presenting generic POVM refinement,
the pure-core displacement decomposition, or the Gaussian recurrence as
new principles. This focused review is not an exhaustive novelty search,
and makes no claim that the exact Householder implementation or its
combination has never appeared before.

## 10. Analytic identities available to independent implementation tests

The mathematical review supplies the following independent checks, not
claims about tests already executed:

- \(K=0\): only channel zero exists and the binary probability is
  \(p_0=e^{-|\alpha|^2}\), with scalar remaining state.
- \(\alpha=0\): \(t=1,q=0\), and the channels reduce to number-basis
  refinement of an ordinary threshold measurement.
- The complex one-mode fixture in Section 4 detects a missing conjugate.
- The \(|0,2\rangle\) fixture in Section 5 detects incorrect channel
  truncation and verifies probability completeness.
- A direct small-system contraction of
  \(\bigotimes_j\{uu^\dagger,I-uu^\dagger\}\), independent of the
  Householder construction, must agree with the summed refined binary law.
- Coherent superpositions across total number must agree with that
  contraction and generally disagree with a pre-dephased sector mixture.
- Gaussian averaging must retain the full real covariance or equivalently
  both complex covariances in (17). Deterministic integration or moment
  identities need their own numerical error assessment; finite Monte Carlo
  agreement is not exact full-law verification.

No structural obstruction to the proposed sampler was found. The known
resource limit is its state representation, and implementation acceptance
still depends on numerical, support, likelihood-label, and failure checks
against these exact identities.

## References

- **[Pellonpaa14]** Juha-Pekka Pellonpää, *Complete measurements of quantum
  observables*, Foundations of Physics **44**, 71–90 (2014).
  [Author manuscript](https://arxiv.org/pdf/1206.2506);
  [DOI](https://doi.org/10.1007/s10701-013-9764-y).
- **[Yao24]** Yuan Yao, Filippo Miatto, and Nicolás Quesada,
  *Riemannian optimization of photonic quantum circuits in phase and Fock
  space*, SciPost Physics **17**, 082 (2024).
  [Author manuscript, v5](https://arxiv.org/html/2209.06069v5);
  [Published paper](https://scipost.org/SciPostPhys.17.3.082/pdf).
- **[Miatto20]** Filippo M. Miatto and Nicolás Quesada,
  *Fast optimization of parametrized quantum optical circuits*,
  Quantum **4**, 366 (2020).
  [Published paper](https://quantum-journal.org/papers/q-2020-11-30-366/).
- **[Oh24]** Changhun Oh, Minzhao Liu, Yuri Alexeev, Bill Fefferman, and
  Liang Jiang, *Classical algorithm for simulating experimental Gaussian
  boson sampling*, Nature Physics **20**, 1461–1468 (2024).
  [Author manuscript, v2](https://arxiv.org/html/2306.03709v2);
  [DOI](https://doi.org/10.1038/s41567-024-02535-8).
- **[Liu26]** Tong Liu, Hui-Ke Jin, Tao Xiang, and Hong-Hao Tu,
  *Efficient simulation of low-entanglement bosonic Gaussian states in
  polynomial time*, npj Quantum Information **12**, 110 (2026).
  [Author manuscript, v2](https://arxiv.org/html/2512.10643v2);
  [DOI](https://doi.org/10.1038/s41534-026-01254-7).
