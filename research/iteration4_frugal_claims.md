# What FrugalGBS has demonstrated, and how to compare with it

Audit date: **2026-11-04**. The machine-readable ledger is
`configs/frugal_claims.json`. This is a primary-source scope audit; local
same-evaluator experiments are separate artifacts produced by the experiment
harness.

## Publication and exact scope

The latest verified publication is Dodd et al., *Science Advances* 12(42),
eaef8431, published online **14 October 2026**,
[DOI 10.1126/sciadv.aef8431](https://doi.org/10.1126/sciadv.aef8431).
The [author preprint](https://arxiv.org/html/2511.14923v1) remains v1 dated
18 November 2025. The [published abstract and captions](https://pubmed.ncbi.nlm.nih.gov/42873857/)
and [author institution record](https://www.research.ed.ac.uk/en/publications/a-fast-and-frugal-gaussian-boson-sampling-emulator/)
confirm publication. Indexed passages from the
[final full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC13665930/) also verify
the experimental comparison scope, TV limitations and implementation resources.
Direct later opens returned a browser challenge. The ledger marks the remaining
preprint-only details, including the explicit XEB subsample count and auxiliary
index memory, rather than silently assuming every version is identical.

The paper's result is a substantial improvement in practical **classical
emulation costs and experimental benchmark scores**. A claim that it already
produces every Jiuzhang distribution faster than its quantum device would be
stronger than the evidence audited here.

| Audited target | Output used by Frugal | Quality evidence | D2 implication |
|---|---|---|---|
| J2-P65-2 | 144 threshold bits | Click cumulants; XEB in click sectors 5 through 20 | Meaningful real-instance witness comparison; no global-TV acceptance |
| J2-P65-5 | 144 threshold bits | Click cumulants; total-click distribution | Joint-law quality remains unproved by these witnesses |
| J3-high | 144 coarse-grained threshold bits | Click cumulants; total-click distribution | A reduced task; not the full 1,152-bin pseudo-PNR readout |
| Jiuzhang4 | No demonstrated full instance in audited results | Proposed extension | No current full-JZ4 defeat established |

The table summarizes the author paper's comparison sections and discussion.
The final paper explicitly omits Jiuzhang1 benchmarking, citing prior
squashed-state explanations of much of its quantum-advantage evidence. Our
Jiuzhang1 control therefore tests a separate real-model target; it does not
replicate one of Frugal's reported experimental benchmark figures.
The most consequential distinction is the Jiuzhang3 alphabet. Combining a
group of eight detection bins into one click bit is a many-to-one map. For
distributions P,Q and a deterministic grouping f,

\[
\operatorname{TV}(f_*P,f_*Q)\leq \operatorname{TV}(P,Q).
\]

This inequality follows by collecting signed probability differences within
each group and applying the triangle inequality. Therefore, good agreement
after grouping is useful but cannot certify agreement before grouping. Nor
can one infer the detailed pseudo-PNR response from 144 binary outcomes.

## Why the first discriminator must also test Frugal

Our error standard should be symmetric. The proposed sampler and official
Frugal implementation should receive the same physical covariance, detector
alphabet, requested epsilon, comparison allocation, sample count and failure
policy. A direct quality certificate obtained on a smaller or different input
cannot be transferred to the paper's experimental instances.

Low-order cumulants and XEB are scientifically useful witnesses. Their
acceptance does not exhaust the joint distribution. For example, binary laws
uniform on even parity versus odd parity agree on every proper subset yet
have disjoint support and TV equal to one. This mathematical counterexample
explains the logical gap; it does not assert that either Frugal or the hardware
has that failure mode. A total-click histogram supplies another marginal,
still leaving many individual bitstring probabilities unresolved.

The D1 global bound uses actual sampler path probabilities when available.
Those probabilities must include every normalization, clipping decision and
failure policy used by the implementation. An auxiliary approximate marginal
table is not automatically the normalized law of the sampling code. That is
why the control adapter must expose official draws and evaluate their actual
conditional-chain probabilities with the same numerical path. Instrumentation
and point-probability evaluation have their own overhead and should be timed
separately from plain sampling.

D2 retains `local_distribution_quality` as local context. Reference quality
remains unknown until the exact instance, complete extent, measurement and
imperfection gates bind the declared Jiuzhang target. This applies equally to
our approximation and to Frugal. A missing or rejected global certificate is
not disguised as a successful benchmark; conversely, an inconclusive bound
is not a proof that the algorithm is inaccurate.

## Runtime and memory need consistent boundaries

The final paper's Fig. 1 reports **10 million samples in 115 minutes on eight
GPUs**, versus approximately **10 seconds** for its schematic quantum
collection. The final implementation section confirms H100 SXM 80 GB GPUs. The
reported generation rates therefore do not show a quantum sampling-rate
victory. The nominal ratio is about 690, but the figure alone does not specify
enough matched protocol detail for a precise end-to-end experimental comparison.

The paper separates reusable indexing work for a chosen M,K from
instance-dependent cumulant construction and sampling. A one-shot comparison
must include the appropriate setup; a steady-state comparison must state its
amortization. The headline generation time does not by itself settle those
boundaries.

The final implementation confirms the reported 2 GB of cumulants per GPU and
approximately 2 MB of tables per sampling thread. Additional auxiliary indices
are described in the preprint. These figures are not total resident memory. The public
[FrugalGBS library](https://github.com/tomdodd4598/FrugalGBS), pinned here at
`fa8b796f2afebd4845a2e335e457735952617ef8`, defaults to single precision in
`CMakeLists.txt`; our control builder explicitly selects double precision.
The audited paper text does not explicitly identify its arithmetic precision,
so the library default is not treated as proof of the publication's setting.
An apparent factor-of-two storage difference must be explained before making
a resource comparison.

## Data that can actually be identified

The published data-availability statement distinguishes three resources:
generated samples/benchmark data (reference 34), the original results code
(reference 35), and the derived public library (reference 36). The first two
destinations could not be resolved through the available retrievals. They
remain explicit unknowns in the ledger. The pinned public library tree was
inspected; it does not contain the J2/J3 experimental covariance matrices or
sample archives. It would be misleading to call our build of that library a
replication of the paper's data analysis.

The official [Jiuzhang2](https://quantum.ustc.edu.cn/web/node/951) and
[Jiuzhang3](https://quantum.ustc.edu.cn/index.php/en/node/1121) data pages remain
the experimental source route. The
[Chicago author implementation](https://github.com/sss441803/BosonSupremacy),
commit `ff3d49e2cb6aaa594869f3ba568566db44c4fcc8`, supplies a concrete importer:

- J2-P65-2 maps to `waist=65um/0.3`; J2-P65-5 to `waist=65um/1.65`.
  The README gives the four official archive URLs. `make_cov.py` reconstructs
  a covariance from the setting and transfer matrix. Its sample loader also
  specifies detector-port selection and bit-order reversal.
- J3-high reconstruction reads `sq_parameter/high_power.xlsx` and
  `matrix/high power.mtx`. The sample loader unpacks 1,152-bit records. The
  reduced covariance reconstruction combines two squeezing components into
  an effective value; it must be labeled as that reduced model.

The raw J2/J3 archives were not imported here, and no small author covariance
mirror with independently verified provenance was found. File hashes and
complete calibration mappings therefore remain unknown. Code licenses in the
author repository do not automatically establish the licensing of external
experimental datasets.

The Jiuzhang1 author covariance already acquired by this project is a useful
real-model control for both samplers. It is a separate target from Frugal's
main J2/J3 figures. A successful JZ1 comparison can be reported as such; it
does not retroactively reproduce those figures.

## Practical research consequence

The most defensible next claim is narrow and falsifiable: on a specified
complete covariance and detector law, compare setup cost, draw cost, explicit
failure behavior and D1 error evidence for both implementations. If the new
method improves that frontier, preserve the result with input hashes and a
fixed validation protocol. Then import an actual J2 calibration and repeat.

Matching a witness suite already matched by Frugal would be a useful
engineering result, but a stronger research contribution would explain and
control full-law error, improve complete-mode cost at fixed validated quality,
or extend a method to the actual unresolved readout/noise model. The ledger
prevents those distinct accomplishments from being collapsed into a generic
claim of defeating quantum supremacy.
