#ifndef GBSLAB_BATCHED_FOREST_KERNEL_H
#define GBSLAB_BATCHED_FOREST_KERNEL_H

/* Shared native event kernel, ABI 1.  No Python or Eigen dependency.
 * Compile the companion .cpp with g++ -std=c++20 -ffp-contract=off and link
 * libquadmath.  Fast-math is unsupported.  Handles are single-threaded.
 * Every covariance is binary64, zero-mean, hbar=2, xxpp.  Each precision forms
 * (V+I)/2 itself.  The event indicators are heuristics, not interval bounds.
 */
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BF_ABI_VERSION 1
#define BF_MAX_MODES 512
#define BF_MAX_RETAINED 16
#define BF_ERROR_CAPACITY 512

enum BFStatus { BF_OK=0, BF_INVALID_ARGUMENT=1, BF_NUMERICAL_ERROR=2,
                BF_RESOURCE_ERROR=3, BF_INTERNAL_ERROR=4 };
enum BFPrecision { BF_LONG_DOUBLE=0, BF_QUAD=1, BF_AUTO=2 };
enum BFZeroReason { BF_SCORE_POSITIVE=0, BF_ZERO_BRANCH=1,
                    BF_ZERO_NODE=2, BF_ZERO_EDGE=3 };

enum BFCounter {
    BF_QUERIES=0, BF_QUERY_SUCCESSES, BF_NUMERIC_FAILURES,
    BF_STATE_REBUILDS, BF_STATE_REUSES, BF_STATE_APPENDS,
    BF_ZERO_UPDATES, BF_ZERO_FACTORIZATIONS, BF_ZERO_SOLVE_COLUMNS,
    BF_ZERO_SCHUR_BLOCKS, BF_SUBSETS, BF_SUBSET_FACTORIZATIONS,
    BF_SUBSET_DIMENSION_SUM, BF_SUBSET_CUBIC_WORK,
    BF_SHARED_SOLVE_COLUMNS, BF_SHARED_SOLVE_DOT_TERMS,
    BF_SPARSE_SCHUR_BLOCKS, BF_SPARSE_SCHUR_DOT_TERMS,
    BF_LOCAL_DETERMINANTS, BF_LOCAL_DIMENSION_SUM, BF_LOCAL_CUBIC_WORK,
    BF_EVENT_ACCUMULATIONS, BF_TRACKED_ADD_SUB, BF_TRACKED_MULTIPLY,
    BF_TRACKED_DIVIDE, BF_TRACKED_SQRT, BF_OUTPUT_CONVERSIONS,
    BF_ANALYTIC_ZERO_EVENTS, BF_MAXIMUM_RETAINED, BF_MAXIMUM_OMITTED,
    BF_MAXIMUM_ZEROS, BF_STATE_INVALIDATIONS, BF_COUNTER_COUNT
};
enum BFTime { BF_QUERY_SECONDS=0, BF_STATE_SECONDS, BF_EVENT_SECONDS,
              BF_OUTPUT_SECONDS, BF_TIME_COUNT };

typedef struct BFHandle BFHandle;

typedef struct BFQuery {
    const int *zeros;
    const int *retained;
    const int *omitted;
    const int *edges;       /* flat physical-mode pairs, length 2*edge_count */
    int mode;
    int zero_count;
    int retained_count;
    int omitted_count;
    int edge_count;
} BFQuery;

/* Output count = 1 + 2*g, g=1+f+e.
 * index 0: P(R=1 | Z=0).
 * index 1+a*g: P(R=1, X_mode=a | Z=0).
 * index 2+a*g+j: same joint event plus omitted[j]=1, 0<=j<f.
 * index 2+a*g+f+l: same joint event plus both endpoints of edges[l]=1.
 * All outputs have exactly the same scale: original joint mass / P(Z=0).
 * Numeric zero alone is NOT a certificate. exact_zero=1 requires a required
 * click on an original, exactly isolated vacuum mode; then value=indicator=0.
 */
typedef struct BFOutput {
    long double *values;
    long double *indicators;
    uint8_t *exact_zero;
    size_t capacity;
} BFOutput;

typedef struct BFConditional {
    double probability;
    int precision_used;
    int refinement_attempted;
    int zero_score_reason[2];
    long double log_scores[2]; /* ignored when zero_score_reason[a] != 0 */
    long double log_indicators[2];
    long double maximum_event_relative_indicator;
    long double maximum_log_score_indicator;
    char first_error[BF_ERROR_CAPACITY]; /* LD numeric failure when refined */
} BFConditional;

typedef struct BFDiagnostics {
    uint64_t counters[2][BF_COUNTER_COUNT];
    double seconds[2][BF_TIME_COUNT];
    uint64_t probability_calls;
    uint64_t auto_refinements;
    uint64_t auto_refinement_successes;
    uint64_t auto_refinement_failures;
    uint64_t reset_requests;
    uint64_t original_input_bytes;
    uint64_t state_payload_bytes[2];
    uint64_t peak_state_payload_bytes[2];
    uint64_t peak_workspace_payload_bytes[2];
    uint64_t peak_kernel_payload_bytes;
    double setup_seconds;
} BFDiagnostics;

typedef struct BFFormats {
    int abi_version;
    int long_double_bytes;
    int long_double_mantissa_bits;
    int quad_bytes;
    int quad_mantissa_bits;
    int maximum_modes;
    int maximum_retained;
    size_t query_bytes;
    size_t output_bytes;
    size_t conditional_bytes;
    size_t diagnostics_bytes;
} BFFormats;

int bf_formats(BFFormats *result);
BFHandle *bf_create(int modes, const double *covariance_xxpp,
                    const int *measurement_order, char *error, size_t error_capacity);
void bf_destroy(BFHandle *handle);
/* Mark both precision states invalid.  Counters remain cumulative.  The next
 * query starts from the original input.  Caller uses this at a new sample.
 */
void bf_reset(BFHandle *handle);
int bf_diagnostics(BFHandle *handle, BFDiagnostics *result);

/* One explicit-precision batch.  precision must be LD or QUAD.
 * Z must follow the handle's fixed measurement order.  Only canonical-prefix
 * Z states may be appended; all other histories rebuild from original V.
 * force_rebuild!=0 is available for independent cold-state checks.
 * The kernel accepts any acyclic edge forest on F; Python's chain/static-MI
 * rules produce a full spanning tree.  R/F/Z/mode must be disjoint.
 */
int bf_query(BFHandle *handle, const BFQuery *query, int precision,
             int force_rebuild, BFOutput *output, char *error, size_t error_capacity);

/* Shared forest score, guards and AUTO state machine for Python and CLI.
 * R/E are fixed in query before entering this call.  AUTO does at most one LD
 * batch and, only on a numerical failure, one whole-prefix QUAD batch.  Quad
 * state is independent, originating from binary64 V in quad throughout;
 * canonical-prefix reuse is allowed.  No epsilon, threshold tuning or retry.
 * Parameter/resource/program errors are NOT refinement triggers.
 * The output arrays contain the final attempted batch.  On success the
 * probability is binary64, with the frozen positive-branch endpoint guard.
 */
int bf_probability(BFHandle *handle, const BFQuery *query, int precision_policy,
                   long double max_relative_roundoff, BFOutput *output,
                   BFConditional *conditional, char *error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif
