# Iteration 81 methods and reproduction

The repository's frozen-source convention applies: never rerun into an existing
output directory or edit a previously executed source. Preserve partial failures.

## Factor IPF

`native/gaussian_pair_factor81_v1.cpp` implements P_l=diag(A_l) K diag(B_l).
Let G=A^T diag(w) B. A global update sets K=Q/G entrywise. Row and column
updates set A=p/(B K^T), B=p/(A K). Reciprocal label scaling balances A and B
without changing P. The physical pair table is (P+P^T)/2. Finite support zeros
are handled explicitly; nonconvergence and incompatible support return errors.
All compute-intensive work uses native Eigen/OpenMP matrix multiplication.
Storage is O(m²+Lm), arithmetic per iteration remains O(Lm²).

`iteration81_factor_pilot_v2.py` is the completed calibration driver. Small/wide
results are compared to iteration80 dense tables. The large case uses n=128,
m=384,L=65664,seed=810001,16 workers. The finite-field sign design plus axes
has independently checked fourth moments. Inputs, factors and moments are
chunked. Core coefficient bytes are not peak resident memory. The reported
large minimum_probability=0 is a nonnegative lower bound, not an enumerated
minimum over 9.68 billion pair entries. No factor sampling plan is yet supplied.

## Synthetic speed/accuracy pairing

`iteration81_prefix_build_v1.py` builds a thin recorder using preserved official
Frugal SIMD arithmetic. Audited samples match unmodified official samples with
identical RNG state, and recorded prefix thresholds match those samples.
`iteration81_accuracy_comparison_v1.py` enumerates all small or first-eight
prefixes on the exact iteration80 timing inputs. `gbslab.frugal_law` rebuilds
the law using the ideal iid uint64-to-GCC-double distribution, preserving
NaN/out-of-range comparison semantics; no threshold clipping is introduced.
Continuous-uniform controls are archived separately. CUDA prefix replay is
independently compared to the C++ direct-click law.

The PNR B8 generating function keeps total-pair counts J<=4 exact. For the
transport tail, s_l(S)=sum of the normalized pair table over unobserved ports,
so the no-click generating function is sum_l w_l[(1-z)/(1-z s_l)]^(n/2),
with its J<=4 contribution replaced by the exact pure-state coefficients.
The sphere control uses its complete-homogeneous-polynomial moments through
J=256 and an explicitly bounded positive negative-binomial remainder.
No Monte Carlo estimate is substituted for these numeric marginal laws.

## Jiuzhang raw inputs and explicit reference

Primary data: https://quantum.ustc.edu.cn/node/1121 . The inventory script reads
RAR5 entry headers through HTTP byte ranges. The extraction script copies each
original compressed entry into an independent non-solid RAR container and
checks the original CRC32 and uncompressed length after system-tar extraction.
No source from the archives is executed. `rarfile==4.5` is needed only for the
inventory; requests and system libarchive/tar handle the saved extraction path.

```
python experiments/iteration81_jiuzhang_extract_v1.py --output ../external_sources/jiuzhang3_iteration81_sample_v1 --sample
python experiments/iteration81_jiuzhang_accuracy_v3.py --output results/iteration81_jiuzhang_accuracy_replication
```

The first command requires a fresh output directory; the second uses the
declared external raw path and existing native libraries. Each binary record
contains 144 bytes; bit 0 of a detector byte is its most significant bit. Eight
adjacent bits correspond to one demultiplexed optical output. The native
histogram is independently checked against NumPy bit extraction on all records.

For each spectral sector, paired TMSV source covariance has diagonal cosh(2r),
x-pair entries +sinh(2r), and p-pair entries -sinh(2r). For passive lossy T,
O=[[Re T,-Im T],[Im T,Re T]], and C_out=I+O(C_in-I)O^T. No loss renormalization
or histogram fitting is applied. The published matrix has shape 1152×50; its
50-dimensional input axis uses adjacent H/V source pairs. Two independent
common spectral sectors are an explicit interpretation of the public README.
We also compute main-only and 25 source-distinct minor sectors. These declared
alternatives expose rather than hide physical-model sensitivity.

For independent sectors, a vacuum event has probability equal to the product
of sector vacuum probabilities. Inclusion-exclusion gives the final threshold
law. Both algorithms sample sectors independently and OR bits. Frugal's
official single-Gaussian sampler is used twice, not passed a sum covariance.

Pilot event A={patterns where pilot_frequency>reference_probability} is fixed
using 100000 initial records. On independent holdout of N=1392642, the one-sided
TV lower bound is max(0,hat_p(A)-q(A)-sqrt(log(1/delta)/(2N))). The upper bound
is min(1,TV(hat_p,q)+sqrt((256 log 2+log(1/delta))/(2N))). Use delta=.01/18
for nine lower and nine upper bounds, with a union bound. Validity assumes iid
records and a fixed reference; correlations or parameter fitting are not certified.

The v1 driver hits the native exact-symmetry guard. V2 symmetrizes only the
floating antisymmetric residue before every method but stops on the B8 numerical
failure. V3 retains that failure and completes all predeclared cases. Failed
complete-law configurations are not timed selectively on successful paths.

## Verification and archives

`iteration81_tests_v1.py` runs the complete suite with all configured historical
native libraries. `verify_iteration81_v1.py` loads no sampler DLL and checks
frozen hashes, factor identities, 80-digit vacuum determinants, exact Frugal
threshold laws, all raw records and confidence formulas. Timing is already
finished before either test job, and no heavy jobs overlap timing.

`iteration81_archive_v1.py` stores the oversized incomplete v1 input as lossless
16MiB pieces. Restore with `--restore NEW_FILE`; existing paths are never
overwritten. The external 214940448-byte experimental file stays outside Git;
its URL, byte range, CRC, SHA256 and extraction recipe are retained. Paper-source
provenance records arXiv 2304.12240v3; see the paper and its stated license for
attribution. An unavailable publisher supplement was not treated as evidence.
