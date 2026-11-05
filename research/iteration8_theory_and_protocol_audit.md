# Iteration 8: shared-event theory and protocol audit

Status: pre-formal implementation and protocol audit, 2026-11-05. The CLI,
wrapper, and their integration tests have been frozen after the checks in
section 7; the root's full-suite gate and formal panel remain separate. The working
repository was restored from iteration 7 commit
`108b732443e273ee1b898dff21b351a14b2a6ce8`. Historical iteration 7 inputs
remain historical inputs. No public upload is authorized by this audit.

The mathematical starting point is
`research/iteration7_batched_event_kernel.md` (SHA256
`2a6444ec4b5e1ed39354afcf88e961e2a9526e6e9141c8a4c58db0c42035fc28`).
Iteration 7 established a useful local reuse opportunity, not a complexity
breakthrough or a numerical certificate. This iteration tests whether that
opportunity survives a complete native sampler and explicit numerical guards.

## 1. Fixed event model and common scale

At a measured prefix write `Z` for observed zeros, `R` for retained clicks,
`F` for omitted clicks, `i` for the current mode, and `E` for the prescribed
complete spanning tree on `F`. Let `r=|R|<=k`, `f=|F|`, and `e=|E|`.
For a node, edge, or empty set `G`, the exact event identity is

\[
 M_{a,G}=P(Z=0,R=1,X_i=a,X_G=1)
 =\sum_{S\subseteq R}(-1)^{|S|}P(Z\cup S=0)
   P(X_i=a,X_G=1\mid Z\cup S=0).
\]

For zero-mean, `hbar=2`, `xxpp` input covariance `V`, let
`Q=(V+I)/2`. Vacuum conditioning gives the Schur complement `Q^Z`.
The kernel returns the common-scaled events

\[
 \overline M_{a,G}=M_{a,G}/P(Z=0)
 =\sum_{S\subseteq R}(-1)^{|S|}\beta_S h_{a,G,S},\qquad
 \beta_S=\det(Q^Z_{SS})^{-1/2}.
\]

All branch, node, and edge events must use this *same* scale. In particular,
`a=0` is a local vacuum requirement on `i` inside `h`; it is not separate
conditioning on `Z union {i}` followed by an omitted normalization factor.
The forest score is

\[
 s_a=w_a^{1-f+e}\prod_{j\in F}u_{a,j}^{1-d_j}
                \prod_{(j,l)\in E}v_{a,jl}.
\]

The exponents sum to `1-f+e+f-2e+e=1`. Replacing every event by its
common-scaled version therefore divides both scores by `P(Z=0)` and leaves
`s_1/(s_0+s_1)` unchanged. It can remove underflow due solely to that common
scalar. It does not remove inclusion-exclusion cancellation, Schur rounding,
small-determinant error, or score-normalization error.

For no omitted nodes the score is `w_a`; for one it is the single node event;
for two with their connecting edge it is the edge event. Those boundaries
are exact in real arithmetic. More generally the tree approximation is exact
when the branch-conditional omitted law has the relevant tree factorization,
or when the branch-wise multiplicative approximation errors cancel in the
likelihood ratio. Static unconditional MI does not establish either condition.
The Gaussian support argument from iteration 6 applies to the exact events;
it guarantees neither probability accuracy nor numerical acceptance.

## 2. Reuse that can actually be counted

For each `S subset R`, factor `Q^Z_SS` once, and solve
`H=L^{-1}Q^Z_{S,D}` for `D={i} union F` once. The only covariance blocks
needed are the self blocks, `i-j` blocks, and tree-edge blocks:
`K_uv=Q^Z_uv-H_u^T H_v`. There are `1+2f+e` such blocks. Reusing local
vacuum probabilities needs `1+2f+2e` nonempty determinants, each of dimension
at most six. The larger retained set is handled by the shared outer loop,
not by recomputing a separate full determinant for each requested event.

For subset size `s`, a straightforward implementation costs

\[
 O\{s^3+(f+1)s^2+(f+e+1)s+(f+e+1)\}.
\]

For a tree this yields the prefix bound

\[
 O\!\left(2^r[(r+1)^3+(f+1)(r+1)^2]\right).
\]

This retains the exponential `2^r` dependence. Compared with independent
event evaluation it shares the retained-subset factorization, solves, and
selected covariance products. It does not make exact unrestricted GBS
polynomial, and the exponent cannot be silently treated as a constant when
`k` grows with `m`.

The Gaussian zero state is an additional cost. Rebuilding a prefix with
`z` zero modes from scratch can require
`O(z^3+z^2(m-z)+z(m-z)^2)`, bounded by `O(m^3)`. Repeating that bound at every
step can give `O(m^4)` per sample. Along a single canonical sample path,
rank-two append updates cost `O(sum_t (m-z_t)^2)=O(m^3)`. That improvement
must be earned by the implementation and counted; it cannot be inferred
from the local determinant bound alone.

Canonical reuse means that the previously conditioned zero *sequence* is
an initial segment of the requested zero sequence in the fixed measurement
order. Mere set inclusion is insufficient. A different prior history,
enumeration backtracking, or an incompatible cached state requires a rebuild
from the original stored covariance. The resulting finite-precision state
must match a cold evaluation of that same canonical prefix. Rebuilds, invalidations,
appended zero updates, and per-precision work are separate diagnostics.

With canonical per-path updates, the append-case sampling bound is

\[
 O\{m^3+2^k[m(k+1)^3+m^2(k+1)^2]
       +C_{\rm selector}+C_{\rm tree}+C_{\rm setup}+C_{\rm rebuild}\}.
\]

Here `C_rebuild=0` means no additional full-state reconstructions beyond each
precision's first canonical replay. It is **not** always zero under the
implemented auto policy. A failed long-double zero update invalidates that
state; quad can succeed, after which the next prefix still first attempts
long double. Repeated failures at the same earlier zero update can repeatedly
replay a large prefix from the original covariance. The conservative total
zero-state work can therefore be `O(m^4)` on such a path. A failure only in
event evaluation or the score leaves a successfully prepared state reusable
and does not by itself cause this extra replay. The implementation counts
failed attempts, invalidations, rebuilds, and both precisions; no unconditional
`O(m^3)` total-state worst-case claim is justified.

Chain construction is linear in `f` per prefix. A dense deterministic Prim
implementation costs `O(f^2)` per prefix, hence `O(m^3)` across a path.
Unconditional pairwise MI requires `O(m^2)` small marginal calculations and
storage once per input/instance. The existing influence selector performs
its own conditional-event work and may dominate. None of those costs is
included in the shared-event expression unless explicitly added.

The streaming kernel needs `O(m^2)` state plus
`O(k^2+kf+f)` workspace. Actual bytes must include the immutable binary64
input, both precision states if allocated, the double selector state, static
MI storage, and allocator-owned work buffers. Quad does not fit in the bytes
of one double state. Packed samples cost `O(N ceil(m/64))` machine words;
logQ costs `O(N)`; requested traces can be much larger. Full-law output costs
`O(2^m)` and recursive state copies are an enumeration cost, not evidence for
the streaming sampler's memory bound. No exponential subset cache is assumed.

## 3. Precision policy defines the implemented law

The input target is the stored binary64 covariance, with the documented
input validation/symmetrization. Each precision forms `(V+I)/2` itself.
Quad must not begin by casting an already rounded long-double Schur state.

At each prefix:

1. Select `R` and `E` once using the frozen selector/tree rules.
2. Evaluate the entire long-double batch, event checks, merged log score,
   normalization, and binary64 output conversion checks.
3. On a numerical guard failure only, recompute the whole same prefix in
   quad at most once, with the same `R` and `E`.
4. Return its accepted binary64 transition or an explicit numerical failure.
5. Draw one uniform only after an accepted transition exists.

Input errors, resource failures, and program faults are not numerical
refinement triggers. Refinement does not redraw a uniform, change the
prefix, reselect the retained set, choose a different tree, alter a threshold,
or refill a rejected sample. The old binary64 selector's own numerical
failure remains a failure; the batch kernel cannot silently replace its
selection policy. Auto refinement is a deterministic algorithmic rule,
not conditioning on obtaining a successful random sample.

The indicators are explicit floating-point error heuristics, not directed
interval certificates. A nonzero physical probability can round to zero;
numeric zero must not become a support certificate. The limited analytic
certificate currently allowed is a required click on a mode that is exactly
an isolated vacuum mode in the original zero-mean input. A retained certified
vacuum click makes the conditioning event impossible. TMSV-derived zeros
after conditioning need additional proof if they are to be certified; this
implementation must fail closed rather than infer a certificate from rounding.

The complete computed law is `Q8` on bit strings plus the failure symbol.
Real-arithmetic equivalence of event formulas does not imply bitwise equality
to iteration 7, because its rounding and guards differ. Equal-prefix tests
must include cold state, reuse after other queries, and LD/quad history.
They should also audit selector state determinism; unrelated covariance-cache
parents can otherwise produce different near-tie retained sets.

## 4. Failure, enumeration, and raw sample evidence

Full-Q traversal must sum successful leaves without renormalization and add
every reachable numerical failure prefix's probability to a separate failure
mass. A zero-state update failure following a zero outcome belongs to that
*child's* weight, not the entire parent's weight. No zero update is needed
after the final measured mode. Traversal continues along other branches.
An arithmetic failure in the enumerator's own weight accumulation is a
failed enumeration, not a newly invented sampler failure event.

For a requested sample batch, the first failure stops the batch and preserves
every completed sample and its logQ. The partial current sample is separately
identified. There is no retry, refill, or successful-only renormalization.
`requested_samples`, `completed_samples`, and consumed uniforms are distinct.
LogQ uses the actual returned binary64 transition, including a refined one,
with a stable log/log1p accumulation. Raw physical-mode bits, traces of `R/F/E`,
precision/fallback, and supplied uniforms must permit independent replay.
Finite-generator output is distinguished from the ideal-continuous-uniform
law used by full enumeration, as in previous rounds.

## 5. Proposed minimum informative experiment

These are recommendations; the root's frozen configuration is the authority
for the eventual panel. Do not describe this draft as a completed preregistration.

**Fresh full-TV panel.** Two fresh circuits at each of `m=8,12`, with the same
three predeclared lossy/bright/stress profiles, give 12 inputs. Compare retained
`k=4,6` and chain/static-MI-tree `k=2,4`, giving 72 full computed laws. Publish
all cells, total failure masses, numerical-work counters, and absolute
`TV(P,Q_extended)`. For all new `m=8` forest laws cross-check independent
table events with the same retained/tree choices. Frozen historical lossy
inputs that previously failed are useful named regressions, but are not new
quality evidence. Small deterministic fixtures precede this panel.

**Complete native cost panel.** Fix the first new `m=12` circuit's three
profiles before reading quality results, both `N=64` and `N=1024`, all six
methods, and three repetitions: 108 runs. Use a fixed randomized method order
per repetition and archive seeds/order. This is a proposed minimal size, not
a claim of a stable universal speed estimate. Every process begins with empty
algorithmic caches. Do not call operating-system page caches cold unless they
were actually controlled. Both the retained baseline and new sampler have
cross-prefix probability caching disabled.

Measure complete native `setup + sampling`, including physicality validation,
selector/static-MI preparation, state creation/copy/update, tree construction,
all refinement, RNG, sample storage, and requested logQ. Separately report
input parsing, output aggregation, output serialization/process overhead, and
wrapper wall time. Compiler time is archived as build provenance and kept
distinct from a reusable binary's sampling time. A tiny C++ event kernel
compared with the complete Python/native baseline is not a sampler speed win.
All runs use the same host/thread policy and same requested output products.

**Decision rule.** Continue to optimize the kernel if event sharing reduces
measured complete cost or opens a defensible quality/cost Pareto point. Report
failure if overhead/selector cost erases the kernel gain, if guards create
material failure mass, or if the same retained/tree transition cannot be
reproduced. State a speed advantage at equal quality only for the same fixed
target and predeclared TV acceptance criterion, or a clearly reported Pareto
comparison `TV_new <= TV_old + numerical_tolerance` and `time_new < time_old`.
Do not select cases or a timing statistic after seeing which ones win. Neither
a passed local gate nor this small exact panel certifies large-mode TV.

## 6. Required implementation gate and pending work

The gate should bind source/header/build/actual binary and cover: exact
small-event identities; `f=0,1,2`; both trees and all-zero MI ties; fixed `R/E`
traces against the prescribed Python rules; raw uniforms and actual logQ;
complete success-plus-failure mass; child-state failure placement; original
isolated-vacuum certificates; noncertified numeric zero; LD failure followed
by one quad attempt; quad failure without RNG/retry; cold/reused canonical
states; raw sample checkpoints on a stopped batch; and wrong ABI/input errors.

The new CLI is owned in `experiments/batched_forest_sampler.cpp` with a separate
build script and Python wrapper. It may reuse the unchanged legacy selector
implementation while linking the event kernel as a separate translation unit.
The shared C ABI owns event guards, score normalization, and precision policy;
the CLI must not reproduce a diverging version. Static MI values and resulting
deterministic edges require explicit agreement checks: a floating near-tie
cannot silently be described as the same tree if it chooses different edges.

## 7. Pre-formal implementation review and completed checks

The implemented CLI includes `click_sampler.cpp` unchanged under a renamed
legacy `main`, and links `batched_forest_kernel.cpp` as a separate translation
unit. It supports natural measurement order, the two fixed trees, retained
budgets 0--8, and at most 14 modes for full enumeration. Those CLI limits
are intentionally narrower than the stand-alone event ABI. It always uses
zero cross-prefix probability-cache entries. The legacy selector's bounded
within-step vacuum workspace is reset for each prefix, matching the retained
baseline's default behavior.

Read-only review of the shared kernel confirmed the following structural
properties: each subset gets one retained factorization and shared solve;
only node/current/edge blocks and two-/four-/six-dimensional local determinants
are requested; `a=0` remains a local event under the same zero-state scale;
the merged exponent calculation implements the forest formula; canonical
zero-sequence compatibility is checked before reuse; and each precision
rebuilds from the original binary64 covariance. The default analytic-zero
predicate checks the original exactly isolated vacuum rows and columns.
Numeric zero alone fails. This is an algebra/indexing review, not an assertion
that heuristic indicators rigorously enclose the real-arithmetic answer.

The CLI selects `R/F/E` before calling the shared scorer, uses the returned
binary64 probability for both sampling and logQ, and makes no random draw
inside refinement. A conditional failure in enumeration receives its current
prefix weight; a selector zero-update failure receives the zero child's
weight. Other branches continue and there is no renormalization. A final-mode
zero outcome performs no unnecessary state update. A stopped sample batch
preserves completed expanded samples when requested, always preserves a
lossless packed checkpoint, and identifies the separate partial prefix.

The wrapper validates completed evidence even on a failing process return:
complete/partial uniform consumption, raw/packed agreement, counts, sample
order, trace prefixes, injected-uniform thresholds, forbidden zero branches,
and complete-sample logQ versus the actual trace. Only then does it raise
`CppForestError` carrying the checkpoint. It does not convert an incomplete
batch into a successful fixed-size batch. Timeouts without a native JSON
return have no reconstructed checkpoint and are not retried.

The wrapper also exposes a checked timing-name adapter, `timings_seconds`,
alongside the original native `timings` fields. `kernel_diagnostics` groups
setup/sampling/enumeration records; phase counters and seconds are increments,
while maximum/payload fields retain their explicitly reported cumulative
process scope. The kernel's tracked major numerical-container bytes exclude
some query bookkeeping, stack/allocator overhead, caller output arrays, the
selector, samples, and Python objects. They are neither resident-set size nor
a complete allocator peak. This exclusion does not alter the stated
asymptotic streaming bound.

The CLI build uses the matched native flags `-O3 -std=c++20 -ffp-contract=off
-DEIGEN_DONT_PARALLELIZE -march=native`, with no fast-math option. The build
record binds the CLI, included unchanged retained source, shared kernel,
header, builder dependencies, compiler, Eigen headers, and the actual
`quadmath.h` and resolved `libquadmath.so`. Sources and dependency hashes are
rechecked before blessing the staged executable. The root's final gate must
bind this exact executable and the separate shared-library binary.

Six integration tests passed with zero skips (0.598 seconds in the focused
run). They include the root's four tests for analytic thermal uniforms,
raw/logQ/full-law consistency, isolated vacuum, stopped-batch preservation,
and explicit long-double enumeration failure. Two added tests cover:

* A fixed development input, `m=6`, Haar seed 881001, squeezing 0.6 and
  transmissivity 0.35, with chain/static-MI and `k=0,2`. Every one of the 63
  reachable prefixes agrees with the canonical Python interface on retained
  set, omitted sequence, ordered tree edges, and transition probability.
  Twelve injected-uniform samples per method also precede enumeration to
  exercise cold/reused history consistency and actual logQ. In the separate
  no-sampling development check, native/Python full-law TV differences were
  exactly zero; independent complete-target table-event differences were at
  most `3.22713e-15`. The test tolerances are `2e-12` for orchestration and
  `2e-11` for the independent table reference.
* An independent thermal product input with all static MI weights exactly
  zero. The tree retains its zero-weight spanning edges and uses the required
  physical-index tie rule: on omitted modes `(0,1,2)`, static MI selects
  `(0,1),(0,2)`, while the chain selects `(0,1),(1,2)`.

These small development fixtures are not part of the fresh formal matrix.
The full suite, not the focused test count, is the release gate.

Frozen CLI-side identifiers at this review:

| Artifact | SHA256 |
|---|---|
| `experiments/batched_forest_sampler.cpp` | `eb6773cf76738ce30296bfdef2c1c42d60d84a57cf101b701349ca429dbce948` |
| `experiments/build_batched_forest_sampler.py` | `adece4b5cbd5422f74df8c4fa19158014bd682bc4118f689e6547561831a0a8d` |
| `src/gbslab/cpp_forest.py` | `3988149b0f731e1027e56c2e1973c35f31164011950a2e0b5b4e08940a33f999` |
| `tests/test_cpp_forest.py` | `046932584a5d68dbe4e8e45ee8139bc289b1fa3586ebae7d258751e5deaf4ec4` |
| Native forest executable | `770aea52bd8a0e39e52b77a389b00a5bc7f0d6e5c8f5fe9476993b9bf2a76f00` |
| Native forest build record | `62f505216b7da0db3398ce48449b3c797b861e2c655617dcb01a48cf1e5f52e7` |

The included shared-kernel/source hashes in that executable match the current
source closure at this review. A later kernel/header change requires an
explicit rebuild and a new gate, not a claim that this binary contains it.

## 8. Frozen formal configuration and runner review

The root accepted and materialized the proposed experiment in
`configs/iteration8_batched_forest.json`, SHA256
`40dfe24ba2516244faa86784ee2d403609e9b5ffbbcc207700e34280ebc534a4`.
The reviewed configuration fixes:

* Modes 8 and 12, base seeds 440801 and 440802, actual circuit seed
  `base + 1000*m`, and three profiles sharing each circuit: lossy
  `(r,eta)=(0.6,0.35)`, bright `(0.9,0.65)`, stress `(1.2,0.85)`.
* Twelve inputs and six methods, hence 72 primary native full laws. All 72
  are compared with their Python orchestration/reference law; the 24 new
  forest laws at `m=8` additionally use independent target-table event sums
  with the actual native-traced retained set and tree edges. Sharing the C
  event kernel makes the forest Python comparison an orchestration check,
  not a second independent determinant implementation.
* Three fixed `m=12`, base-seed-440801 profiles, sizes 64/1024, six methods,
  and three repetitions: 108 native timing processes. The order is shuffled
  by fixed seed 440899. The reported matched cost is the median of the three
  full cold batches, with min/max retained; the six profile-by-size ratios
  are summarized by their median. Seeds are shared per repetition, without
  claiming paired sample trajectories across different transition laws.
* Cost output flags are fixed to raw output off, logQ off, and profiling off
  for both samplers. Both still materialize packed samples. Thus the native
  primary cost includes every requested product; it does not claim to include
  unrequested logQ or JSON serialization. Input/aggregation and wrapper wall
  remain separate reported costs.
* A fixed Pareto comparison requires complete new-law TV no greater than
  matched baseline TV plus `1e-10`, strictly smaller median native cost, and
  no failed/missing repetitions. All fixed comparisons are reported. The
  cross-backend TV check threshold `2e-8` is an arithmetic/orchestration
  agreement threshold, not a certificate of distance to physical target P.
* Twelve historical `m=10,12` chain/tree `k=4` regressions are stored apart
  from fresh quality and timing. Old numerical failures cannot become new
  input successes by relabeling their history.

Read-only runner review found that initial input linkage and snapshot
coverage could be stronger. The root added timing-input SHA checks against
the quality panel, equal quality/timing configuration binding, explicit
forest-test source snapshots, complete historical case/method key checks,
and builder/quadmath dependency checks in the final gate. Baseline full-law
process failure remains an unavailable primary law, with its failed payload
retained; a Python diagnostic is not substituted as a native success.

No mathematical or protocol blocker remains for running the root's frozen
gate and panel. This auditor has not run the primary matrix and has not
inspected its outcomes while approving this pre-formal protocol. The full
gate, measured counters, observed quality/cost results, and any limitations
must be reported after execution; none is inferred from the small fixture
successes above.

## 9. Full-gate closeout

The root's complete gate subsequently passed at
`results/iteration8_validation/validation.json`, SHA256
`aa4d55c84602ad3e3f54955b73d470085a67e491d96f5e9652feef253a3ca666`.
The recorded outcome is 268 tests, zero failures, zero errors, zero skips,
zero expected failures, and zero unexpected successes, in
`36.46241725499999` seconds. It explicitly verifies that the bound sources
and three actual candidate binaries stayed unchanged during the suite.
The shared library is
`8c4217c57559c741be9a65adcd390e92fa52ed8daacb68136452eab04b282263`;
the retained native executable is
`a8d5a830e7167aab20f829bee74bc404736f2551ff42e6e48491caf301d0fc3f`;
the full forest executable is the `770aea52...` binary listed above.

The gate exercises real LD-to-quad refinement, raw/table Gaussian events,
canonical history/order/cache invariance, numerical failure before the next
uniform, preserved failure mass, and propagation of nonnumerical errors.
It also explicitly exercises positive probabilities beyond the finite quad
budget and positive branches that round to binary64 endpoints: those remain
failures. Passing the refinement fixture does not mean quad resolves all
positive events. The final accepted transition still has binary64 output
precision, and the floating error indicators remain heuristic.

The theory claims are closed with the following limits: local determinants
are at most six-dimensional; the outer retained-subset count remains `2^r`;
selector, static MI, tree construction, both precision attempts, and every
state rebuild must remain in full native cost. The append-case state bound
does not suppress the repeated-invalidation `C_rebuild` cost or its conservative
`O(m^4)` worst case. Tracked scalar operations are not total hardware FLOPs,
and tracked numerical-buffer payloads are not full process memory. No
uniform-TV, general exact-GBS complexity, or speed conclusion follows from
the gate itself.

This closes the implementation/protocol approval. It does not review or
endorse the subsequently generated quality, regression, or timing results;
those require a separate read-only raw-artifact and report audit. No formal
matrix or timing run was executed by this auditor.
