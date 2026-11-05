# Iteration 9: pure-core photon truncation, its certificate, and its implementation limits

Date: 2026-11-05. Status: independent derivation and primary-literature review;
no iteration-8 source, test, target law, sampling law, or timing result is changed.

## 1. Conclusion and scope

For identical, aligned input squeezing and uniform pure loss, the proposed
decomposition into a weaker pure Gaussian state followed by correlated Gaussian
displacements is exact. It is established prior art: Oh et al. give the same
effective squeezing formula and use the pure-state/displacement decomposition in
their MPS sampler [Oh24, Sec. III.1, Eqs. (4)–(5), and Appendix B]. There is no
novel-decomposition claim here.

An independent, explicit error budget follows by projecting that pure core onto
**total** photon number at most \(K\), retaining the coherent projected state,
and normalizing. If the discarded probability is \(\tau_K\), the exact pure-state
trace distance is \(\sqrt{\tau_K}\). Applying the same displacement channel and
threshold measurement to the original and projected core gives

\[
 \operatorname{TV}(P_V,Q_K)\leq\sqrt{\tau_K}.
 \tag{1}
\]

Here \(Q_K\) is specifically the measurement law of that constructed state.
It is **not** the retained-click, product, pair, chain, static-tree, or Frugal law.
No identity or distance comparison connecting those laws to \(Q_K\) has been
proved. The present repository has a scalar feasibility calculation, not an
implementation that constructs and samples \(Q_K\). The bound is also not a
polynomial-time theorem: the core representation and its displaced measurement
remain substantial algorithmic tasks.

The derivations below establish the claim and its conventions directly. The
literature review in Section 8 distinguishes this mathematical budget from
existing state approximations and probability approximations.

## 2. Covariance conventions and the exact decomposition

Use \(\hbar=2\), vacuum covariance \(I_{2m}\), and the quadrature order
\((q_1,\ldots,q_m,p_1,\ldots,p_m)\), with
\(q_j=a_j+a_j^\dagger\), \(p_j=(a_j-a_j^\dagger)/i\).
For a complex unitary interferometer \(U\), let

\[
 O=\begin{pmatrix}\Re U&-\Im U\\\Im U&\Re U\end{pmatrix}.
\]

This real matrix is both orthogonal and symplectic. The input consists of \(m\)
zero-mean squeezed vacua, all with the same finite \(r\geq0\) and the same
q-squeezed orientation. Uniform loss has transmissivity \(0\leq\eta\leq1\),
mixes with vacuum, and is independent of the input. These conventions agree
with [`covariance_from_interferometer`](../src/gbslab/gaussian.py).

Uniform loss commutes with passive interference. Thus

\[
 V=\eta O\operatorname{diag}(e^{-2r}I_m,e^{2r}I_m)O^T+(1-\eta)I_{2m}
  =O\operatorname{diag}(aI_m,bI_m)O^T,
 \tag{2}
\]

where

\[
 a=1-\eta+\eta e^{-2r},\qquad b=1-\eta+\eta e^{2r}.
\]

No assumption that \(U\) is real is needed. Discarding its complex phases would
generally change the state.

The elementary identity

\[
 ab=1+4\eta(1-\eta)\sinh^2r\geq1
 \tag{3}
\]

implies that

\[
 \begin{split}
 V_p&=O\operatorname{diag}(aI_m,a^{-1}I_m)O^T,\\
 C&=V-V_p=O\operatorname{diag}(0,\delta I_m)O^T,\\
 \delta&=b-a^{-1}=\frac{ab-1}{a}\geq0.
 \end{split}
 \tag{4}
\]

The covariance \(V_p\) is pure: \(V_p\Omega V_p=\Omega\), where
\(\Omega=\left(\begin{smallmatrix}0&I\\-I&0\end{smallmatrix}\right)\).
Its input squeeze is

\[
 s=-\tfrac12\log a,\qquad 0\leq s\leq r.
 \tag{5}
\]

Consequently \(|\psi_p\rangle=\widehat U\bigotimes_j S_j(s)|0\rangle\),
where the squeeze convention is \(S(s)=\exp[\tfrac{s}{2}(a^2-a^{\dagger2})]\).
Oh et al. use the opposite quadrature orientation in their displayed
single-mode covariance; exchanging q and p makes their formula identical
[Oh24]. This \(s\) should not be confused with the squeeze
\(\tfrac14\log(b/a)\) in a squeezed-thermal Williamson representation.

### 2.1 The displacement distribution, including singular residuals

Let \(z\sim\mathcal N(0,I_m)\),
\(B=\sqrt\delta\,O_{:,m:2m}\), and \(d=Bz\). Then
\(\mathbb E[dd^T]=C\). For each mode define

\[
 \alpha_j(d)=\frac{d_{q_j}+i d_{p_j}}2,\qquad
 D(\alpha)=\exp\!\left[\sum_j(\alpha_j a_j^\dagger-\alpha_j^*a_j)\right].
\]

The Gaussian state of covariance \(V\) is exactly

\[
 \rho_V=\Phi_C(|\psi_p\rangle\langle\psi_p|),\qquad
 \Phi_C(\sigma)=\mathbb E_d[D(\alpha(d))\sigma D(\alpha(d))^\dagger].
 \tag{6}
\]

For example, this follows by convolving the Gaussian Wigner function of
\(V_p\) with the classical Gaussian shift: the covariance becomes \(V_p+C\)
and the mean remains zero. A Gaussian state is fixed by these moments.
For nontrivial loss and squeezing, \(C\) has rank \(m\), rather than \(2m\).
Equation (6) uses its Gaussian pushforward measure and does not require an
inverse, determinant, or nonsingular density for \(C\).

The boundary cases are consistent: \(\eta=0\) or \(r=0\) gives a vacuum core;
\(\eta=1\) gives \(s=r\) and \(C=0\). Finite \(r\) ensures \(a>0\).
Known input squeeze phases can be incorporated into the passive input
rotations and hence the effective \(U\). Nonuniform output loss generally
does not commute with \(U\); (2)–(5) cannot be used unchanged in that case.
Unequal input squeezes with uniform loss admit a modewise version of (4),
but their photon-pair sum is not the single negative-binomial law below.

### 2.2 Energy accounting does not count simulation cost

The means obey

\[
 \begin{split}
 \mu_p&=m\sinh^2s=\frac m4(a+a^{-1}-2),\\
 \mu_V&=m\eta\sinh^2r=\frac m4(a+b-2),\\
 \frac14\operatorname{tr}C&=\frac m4(b-a^{-1}),\qquad
 \mu_p+\tfrac14\operatorname{tr}C=\mu_V.
 \end{split}
 \tag{7}
\]

For \(r>0,\eta>0\), the core/target mean ratio is
\(\eta/[\eta+(1-\eta)e^{2r}]\). At fixed \(r\) and small loss transmission,
\(\mu_p=O(m\eta^2)\), whereas \(\mu_V=O(m\eta)\).
At fixed \(\eta<1\), increasing \(r\) without bound gives
\(s\to-\tfrac12\log(1-\eta)\) and
\(\mu_p/m\to\eta^2/[4(1-\eta)]\).
These are reductions of a quantum-state parameter, not runtime or hardness
claims. We do not claim that minimizing mean photons minimizes a tail
quantile, a tensor bond dimension, or complete sampling cost.

## 3. Exact projection and the trace-distance/TV budget

Let \(\widehat N=\sum_j a_j^\dagger a_j\),
\(\Pi_K=\mathbf1\{\widehat N\leq K\}\), and

\[
 \tau_K=\langle\psi_p|(I-\Pi_K)|\psi_p\rangle,\qquad
 |\psi_K\rangle=\frac{\Pi_K|\psi_p\rangle}{\sqrt{1-\tau_K}}.
 \tag{8}
\]

For finite squeezing and \(K\geq0\), the nonzero vacuum component ensures
that this normalization exists. Specify the conventions

\[
 T(\rho,\sigma)=\tfrac12\|\rho-\sigma\|_1,\qquad
 \operatorname{TV}(P,Q)=\tfrac12\sum_x|P(x)-Q(x)|.
\]

The kept and discarded components of \(|\psi_p\rangle\) are orthogonal, so
\(|\langle\psi_p|\psi_K\rangle|^2=1-\tau_K\). Diagonalizing the difference
of the two pure-state projectors in their two-dimensional span gives

\[
 T(|\psi_p\rangle\langle\psi_p|,|\psi_K\rangle\langle\psi_K|)
 =\sqrt{\tau_K}.
 \tag{9}
\]

Define \(\rho_K=\Phi_C(|\psi_K\rangle\langle\psi_K|)\). Convexity and
unitary invariance of the trace norm give

\[
 T(\rho_V,\rho_K)
 \leq\mathbb E_d\,T(D_d|\psi_p\rangle\langle\psi_p|D_d^\dagger,
                   D_d|\psi_K\rangle\langle\psi_K|D_d^\dagger)
 =\sqrt{\tau_K}.
 \tag{10}
\]

For ideal threshold effects
\(M_x=\bigotimes_j[|0\rangle\langle0|\text{ if }x_j=0;
 I-|0\rangle\langle0|\text{ if }x_j=1]\), define
\(P_V(x)=\operatorname{tr}(M_x\rho_V)\) and
\(Q_K(x)=\operatorname{tr}(M_x\rho_K)\). Measurement contraction then proves
(1). Thus the sufficient tail condition is

\[
 \boxed{\tau_K\leq\varepsilon^2\quad\Longrightarrow\quad
        \operatorname{TV}(P_V,Q_K)\leq\varepsilon.}
 \tag{11}
\]

The same contraction holds for any common measurement channel. No factor of
two is omitted: both distances use the half-norm conventions above.

There is a sharper special case when \(C=0\): the number projection commutes
with passive \(U\), and photon counting sees the original PNR distribution
conditioned on \(N\leq K\). Its TV is exactly \(\tau_K\); thresholding can
only reduce that TV. The general square-root bound remains valid. The
sharper argument must not be carried over to nonzero displacement, which
mixes photon-number sectors coherently.

## 4. Exact photon-pair law, parity, and the resource dimension

For one squeezed vacuum, the total photon probabilities are

\[
 \Pr(N=2j)=\frac{(2j)!}{2^{2j}(j!)^2}\,\operatorname{sech}s\,
                  \tanh^{2j}s,\qquad \Pr(N=2j+1)=0.
 \tag{12}
\]

Its pair-number generating function is
\(\operatorname{sech}s(1-z\tanh^2s)^{-1/2}\).
Independence of the \(m\) inputs and \([\widehat U,\widehat N]=0\) imply

\[
 J=N_{\rm core}/2\sim\operatorname{NB}(\gamma=m/2,p=\operatorname{sech}^2s),
 \qquad
 \Pr(J=j)=\frac{\Gamma(j+\gamma)}{\Gamma(\gamma)\Gamma(j+1)}
                    p^\gamma(1-p)^j.
 \tag{13}
\]

Here \(p\) is the success probability and \(j\) counts failures in the
standard negative-binomial convention; \(\gamma\) need not be an integer.
In particular,

\[
 \tau_K=\Pr\!\left(J>\lfloor K/2\rfloor\right),\quad
 \mathbb E N_{\rm core}=m\sinh^2s,\quad
 \operatorname{Var}N_{\rm core}=2m\sinh^2s\cosh^2s.
 \tag{14}
\]

Odd \(K\) gives exactly the same projector on the core as the preceding
even \(K\), so the budget reports the smallest nonnegative **even** cutoff.
At \(s=0\), \(p=1\), \(J=0\) deterministically, and \(K=0\) suffices.

Passive interference preserves total parity but not each mode's parity.
The dense physical Fock subspace for the truncated core therefore has
dimension

\[
 D_{\rm even}(m,K)=\sum_{j=0}^{\lfloor K/2\rfloor}
                  {m+2j-1\choose 2j}.
 \tag{15}
\]

Counting only even occupation in every output mode would be incorrect.
This is a dimension for one dense representation of the **core**; it is
neither the displaced output's support nor a lower bound for every possible
representation. Displacement can produce odd total counts and arbitrarily
large counts even from a core with \(K=0\).

For fixed \(K\), (15) is \(O(m^K)\). For fixed nonzero \(s\) and fixed
\(0<\varepsilon<1\), let \(K_m\) be the exact minimal cutoff in (11).
The law of large numbers for the independent input photon counts gives

\[
 K_m/m\longrightarrow\rho=\sinh^2s>0.
\]

The largest summand bounds (15) from below, and multiplying that summand by
\(1+K_m/2\) bounds it from above. Stirling's formula then gives

\[
 \lim_{m\to\infty}\frac1m\log D_{\rm even}(m,K_m)
  =(1+\rho)\log(1+\rho)-\rho\log\rho>0.
 \tag{16}
\]

The even-only cutoff does not change this exponential rate. This rules out
calling a dense enumeration scalable at fixed nonzero core squeeze; it
does not rule out MPS, structured contractions, or other representations.
An untruncated Gaussian covariance is compact despite an infinite Fock
expansion. Conversely, the normalized finite-\(K\) core is generally
non-Gaussian; passing its covariance alone to a Gaussian threshold oracle
would not implement (8).

## 5. Why drawing a photon-number sector first changes the algorithm

The projected state in (8) retains coherence between its even-\(N\) sectors.
Replacing it by a classical mixture of those sectors is an additional
dephasing channel. After displacement, its measurement law can differ.

Take one q-squeezed core with \(\tanh s=1/2\), and keep \(N\leq2\).
Equation (12), with the signed squeezed-vacuum amplitudes, gives

\[
 |\psi_K\rangle=\frac{2\sqrt2}{3}|0\rangle-\frac13|2\rangle,
 \qquad
 \rho_{\rm deph}=\frac89|0\rangle\langle0|+\frac19|2\rangle\langle2|.
 \tag{17}
\]

For deterministic real displacement \(\alpha=1\),

\[
 \langle0|D(1)|0\rangle=e^{-1/2},\qquad
 \langle0|D(1)|2\rangle=e^{-1/2}/\sqrt2.
\]

Consequently the coherent and dephased no-click probabilities are

\[
 p_0^{\rm coh}=\frac1{2e},\qquad
 p_0^{\rm deph}=\frac{17}{18e},\qquad
 |p_0^{\rm coh}-p_0^{\rm deph}|=\frac4{9e}\approx0.16350.
 \tag{18}
\]

In (18) the first probability is \(1/(2e)\), not \(e^{-1/2}\).
This is an analytic displacement witness, not a Gaussian-noise sampling
benchmark and not a claimed counterexample to (11): the dephased state is
simply not the state to which (11) was proved.

The cross term need not disappear under the actual anisotropic Gaussian
noise of (4). For example, set \(r=\log3\) and \(\eta=3/4\). Then
\(a=1/3,b=7,\delta=4\), so the same core has \(\tanh s=1/2\), and the
residual displacement can be written \(\alpha=iY\),
\(Y\sim\mathcal N(0,1)\). Directly from (17),

\[
 p_0^{\rm coh}(Y)-p_0^{\rm deph}(Y)=\frac49Y^2e^{-Y^2},\qquad
 \mathbb E[p_0^{\rm coh}-p_0^{\rm deph}]=\frac4{27\sqrt3}>0.
 \tag{19}
\]

More generally the expectation is
\(4v/[9(1+2v)^{3/2}]\) for variance \(v>0\). Thus even a valid physical
Gaussian displacement mixture does not justify independent sector draws.
Adding “classical photon counts” to core photon counts is likewise not an
implementation of displacement, which acts on amplitudes.

## 6. A finite measurement formulation, without an efficiency claim

For a fixed sampled displacement, the no-click effect acting on the
undisplaced core is

\[
 D(\alpha)^\dagger|0\rangle\langle0|D(\alpha)=|{-\alpha}\rangle\langle{-\alpha}|.
\]

Restrict the local **input** basis to \(|0\rangle,\ldots,|K\rangle\), and let

\[
 v_n=e^{-|\alpha|^2/2}\frac{(-\alpha)^n}{\sqrt{n!}},\qquad
 E_0=vv^\dagger,\quad E_1=I_{K+1}-vv^\dagger.
 \tag{20}
\]

Both effects are positive because
\(\|v\|^2=e^{-|\alpha|^2}\sum_{n=0}^K|\alpha|^{2n}/n!\leq1\), and they
sum to the identity. Since a global core with \(N\leq K\) has each local
occupation at most \(K\), its exact displaced-threshold probability is

\[
 Q_K(x\mid\alpha)=\langle\psi_K|\bigotimes_jE_{x_j}(\alpha_j)|\psi_K\rangle.
 \tag{21}
\]

This standard pulled-back POVM identity already includes all displaced
output photon numbers; ideal threshold detection need not impose a second
output photon cutoff. It does not make constructing the coherent core or
contracting (21) efficient. A sequential method must handle conditional
states correctly; coarse-graining a click can leave a mixed residual state.
No such implementation or complexity bound is provided by this note.

## 7. The fixed numerical feasibility calculation and its interpretation

The companion calculation is
[`experiments/iteration9_core_budget.py`](../experiments/iteration9_core_budget.py),
with predeclared configuration
[`configs/iteration9_core_budget.json`](../configs/iteration9_core_budget.json).
Its complete archived output is
[`results/iteration9_core_budget/`](../results/iteration9_core_budget/README.md).
It uses no target law, sampled law, or fitted parameter.

The panel contains \(m=8,12,32,64,128,256,1024\), budgets \(0.1,0.01\), and
the three fixed \((r,\eta)\) profiles lossy \((0.6,0.35)\), bright
\((0.9,0.65)\), and stress \((1.2,0.85)\). Each core cutoff is compared with
the cutoff for the **unattenuated pure input** at squeeze \(r\). That
reference is not the mixed target's measured-photon distribution.

The archived \(m=1024\) core results are:

| Profile | TV budget | Even core cutoff \(K\) | Tail at \(K\) | Tail at \(K-2\) | \(\log_{10}D_{\rm even}\) |
|---|---:|---:|---:|---:|---:|
| lossy | 0.1 | 36 | 0.0088281746173 | 0.017057688855 | 67.064909543 |
| lossy | 0.01 | 48 | 0.000070986809948 | 0.00017460637994 | 83.872517065 |
| bright | 0.1 | 212 | 0.0089718730757 | 0.011481989080 | 244.414109825 |
| bright | 0.01 | 244 | 0.000070409055431 | 0.00010000079136 | 268.061939923 |
| stress | 0.1 | 786 | 0.0098093197334 | 0.010889266828 | 536.169115653 |
| stress | 0.01 | 858 | 0.000099677818343 | 0.00011566293935 | 561.457984555 |

These are archived floating evaluations, not interval upper bounds.
The 42 comparisons contain 84 cutoff calculations. The archive reports
20 checks passed, including 168 accepted/previous-tail comparisons against
an independent positive-term long-double PMF recurrence; the maximum
absolute tail difference is \(1.214306433183765\times10^{-16}\).
An independent read-only review confirmed the survival convention
\(\operatorname{sf}(j)=\Pr(J>j)\), the minimal-even-cutoff logic, the
recurrence ratio and geometric remainder for the panel's shape \(\gamma\geq4\),
and the dimension sum. Its nine-file archive manifest had no hash mismatch.

The recurrence initializes \(p^\gamma\), then uses
\(t_{j+1}/t_j=(1-p)(j+\gamma)/(j+1)\). For \(\gamma\geq1\) these ratios
decrease. Once a ratio is below one, the remaining positive tail is bounded
in exact arithmetic by the corresponding geometric series. The code's
ordinary floating evaluation is a termination diagnostic, not a rigorously
outward-rounded bound; agreement of two numerical methods does not change
that status. Both the accepted and previous-even-cutoff tails are saved.

Even the smallest displayed core space would take approximately
\(10^{68.269}\) bytes for one dense complex128 vector. This is a reason to
reject dense enumeration as the large-system implementation proposal for
this panel. It is not an MPS bond-dimension lower bound or a complexity
barrier for every structured algorithm.

Provenance of the calculation reviewed here:

| Item | SHA-256 |
|---|---|
| Frozen config | `b2f159fa8a947df67fb0b3e71c93d803c8dcd2406c3e9072b82c073f07bdff6c` |
| Budget script | `55024644840713bda71d834946818d09aa4e47b3888f4102c23f4639e260da0b` |
| Budget rows | `108a9ca8bbfdce3a83d2d72ce160c4e2cb41cf981cca6bf2b34c40cd8fcbc355` |
| Validation | `ecf00e54f1a16fe38aca48a5133382ad47988baa858f57ef9f16432b78b12b46` |
| Result manifest | `6838c9aea4d03b630963ce49e8f56906837f08a698a22dc63ee052e0b60a47de` |

## 8. Primary literature and what its guarantees do not transfer

| Work | Relevant established result | Boundary for this project |
|---|---|---|
| Qi et al. [Qi20] | A sufficient noisy-GBS simulation criterion from proximity to efficiently samplable classical Gaussian states, incorporating squeeze, loss, and detector parameters. | Failure of their sufficient test is not a hardness proof. Their squeezed-input count is not this note's photon cutoff. |
| Oh et al. [Oh24] | Pure Gaussian core plus PSD classical displacement noise; the same single-mode effective squeeze; MPS construction and displaced sampling. | The decomposition and broad algorithmic route are prior art. A useful core does not remove representation, displacement, or measurement costs. |
| Liu et al. [Liu26] | Gaussian singular-value decomposition and projected creation operators construct MPS tensors without per-entry hafnian evaluation. | The cost depends on retained bond and local dimensions. Their conversion benchmark and maximum bond discarded weight are not automatically a certified full-law TV budget. |
| Dodd et al. [Dodd26] | Frugal uses low-order cumulants and approximate marginals to generate threshold samples; the paper chooses cumulant order heuristically. | Its cumulant/XEB evidence and truncation parameter do not give the pure-state tail certificate (11). |

In [Liu26], Sec. II.3 and Eq. (37) explicitly retain the dependence on bond
dimension and the entanglement spectrum; Sec. III identifies the numerical
task as MPS conversion. A statement polynomial in a retained tensor size
does not prove that size is polynomial in \(m\) at a fixed global error.
Their covariance normalization also differs: convert the vacuum convention
before comparing a displayed covariance to the present \(\hbar=2\) matrices.

For any MPS continuation, one must derive a global state-error budget for
the actual sequence of local/bond approximations. A maximum individual
discarded weight is not itself the global trace distance. Conversely, the
global photon projection used here need not be the best MPS truncation or
the best runtime/accuracy tradeoff. This note does not establish a new
algorithm, a priority claim for its elementary bound, or superiority over
the cited methods.

## 9. Conditions for turning the bound into a sampling result

A concrete continuation would need to specify and validate all of the
following parts of one algorithm:

1. Construct a representation of the normalized, **coherent** state (8), or
   prove a separate state-distance bound for another representation.
2. Preserve the correlated Gaussian displacement measure of (6), including
   its quadrature convention and singular support.
3. Sample the resulting threshold POVM law, for example through (21), with
   valid conditional probabilities and correctly updated conditional states.
4. Account separately for representation and arithmetic errors, input-model
   mismatch, and implementation failure. A valid comparison to the ideal
   constructed law can be added to \(\sqrt{\tau_K}\) by the triangle
   inequality; those additional bounds have not been established here.
5. Bind a numerical tail certificate to the actually used model and cutoff.
   The current binary64/SciPy and long-double agreement is feasibility
   evidence, not a directed-rounding certificate.
6. Report setup, memory, per-sample cost, and any failure mass for that
   completed sampler. If there is a failure symbol, use the extended output
   law instead of conditioning on success.

In particular, no total-variation claim for a finite-precision implementation
follows merely from closeness of Gaussian random-number moments. A discrete
floating generator and an ideal continuous distribution need not be close
in TV as measures on real coordinates; a usable analysis must bound the
induced state or output law. No such numerical implementation bound is
asserted here.

The photon cutoff \(K\), the existing samplers' retained-click budget \(k\),
Frugal's cumulant order, and an MPS bond dimension are different parameters.
They must remain distinct in future configurations and error statements.

## References

- **[Qi20]** Haoyu Qi, Daniel J. Brod, Nicolás Quesada, and Raúl García-Patrón,
  *Regimes of Classical Simulability for Noisy Gaussian Boson Sampling*,
  Physical Review Letters **124**, 100502 (2020).
  [Author manuscript](https://arxiv.org/pdf/1905.12075);
  [DOI](https://doi.org/10.1103/PhysRevLett.124.100502).
  See the main sufficient criterion and its state-distance/data-processing
  argument, and the supplemental squeezed-thermal conventions.
- **[Oh24]** Changhun Oh, Minzhao Liu, Yuri Alexeev, Bill Fefferman, and
  Liang Jiang, *Classical algorithm for simulating experimental Gaussian
  boson sampling*, Nature Physics **20**, 1461–1468 (2024).
  [Author manuscript, v2](https://arxiv.org/html/2306.03709v2);
  [DOI](https://doi.org/10.1038/s41567-024-02535-8).
  Section III.1, Eqs. (4)–(5), Appendix B, and Section III.2 are the directly
  relevant decomposition and MPS/displacement construction.
- **[Liu26]** Tong Liu, Hui-Ke Jin, Tao Xiang, and Hong-Hao Tu,
  *Efficient simulation of low-entanglement bosonic Gaussian states in
  polynomial time*, npj Quantum Information **12**, 110 (2026).
  [Author manuscript, v2](https://arxiv.org/html/2512.10643v2);
  [DOI](https://doi.org/10.1038/s41534-026-01254-7).
  See Sections II.3 and III, Eq. (37), and Appendix C.
- **[Dodd26]** Tom Dodd, Javier Martínez-Cifuentes, Oliver Thomson Brown,
  Nicolás Quesada, and Raúl García-Patrón, *A fast and frugal Gaussian boson
  sampling emulator*, Science Advances **12**, eaef8431 (2026).
  [Published full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC13665930/);
  [DOI](https://doi.org/10.1126/sciadv.aef8431).
  See “Emulator description,” the cumulant-order choice, and the discussion
  of benchmark metrics and total variation distance.

## Final primary-report review

The completed [RESEARCH_REPORT_ITERATION9.md](../RESEARCH_REPORT_ITERATION9.md)
was independently reviewed after the derivation and resource archive were
available. The reviewed report SHA-256 is:

    823d8e8eb152644b721ff71652122b47d7ed5d982299eeac2e7ba99b5626fb34

No blocking discrepancy was found. The report uses the half-trace-distance
convention, retains the coherent normalized projector, and assigns the
additional \(\delta_{\rm state}\) to a separately proved state trace-distance
bound and \(\delta_{\rm sample}\) to a separately proved output TV bound.
The stated sufficient budget
\(\sqrt{\tau_K}+\delta_{\rm state}+\delta_{\rm sample}\) is consistent with
channel contraction and the triangle inequality. The values \(1/(2e)\),
\(4/(9e)\), and \(4/(27\sqrt3)\) in the two coherence witnesses agree with
Section 5 above.

All 42 appendix rows match the frozen
[budget_rows.json](../results/iteration9_core_budget/budget_rows.json) exactly
at the report's displayed rounding, including both cutoffs, both logarithmic
dimensions, and the core tail. The narrative keeps the floating-budget,
dense-representation, prior-art, unimplemented-sampler, and no-transfer-to-
existing-forest limitations. This was a read-only text, algebra, and archived
value review; no sampling, tail recomputation, test suite, or benchmark was
run, and the primary report was not edited.
