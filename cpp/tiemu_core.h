#ifndef TIEMU_CORE_H
#define TIEMU_CORE_H

/*
 * Standalone Tiemu numerical core, C ABI version 1.
 *
 * This header is valid C and C++. It has no interpreter, array-library, or
 * binding-library dependency. Define TIEMU_CORE_BUILD while building the shared
 * library; consumers do not define it. All arrays use native-endian IEEE double
 * or int64_t elements in contiguous row-major order.
 *
 * Array lengths and output capacities count elements, never bytes. A null array
 * pointer is accepted only when its required length is zero. An optional
 * distribution is absent exactly when distribution == NULL and its length is 0.
 * Inputs are read-only and must remain alive and unchanged until the call ends.
 * Output buffers are caller-owned, must have at least the required capacity,
 * and must not overlap any input or another output buffer. The core never frees
 * caller buffers. Shape/count/overflow checks precede access; the caller remains
 * responsible for providing allocations of the stated sizes. Output contents
 * after a failed call are unspecified.
 *
 * Every int-returning entry point reports TIEMU_OK on success. No C++ exception
 * crosses this ABI. tiemu_last_error() returns the calling thread's last error;
 * its string remains valid until that thread's next status-returning core call.
 * A successful status-returning call clears that thread's error text. Version
 * and last-error queries do not change it.
 *
 * Handles belong to the specific create/destroy pair. Operations on each live
 * handle are internally serialized. Destruction must not race with any operation
 * or a second destruction of that handle: callers must synchronize its lifetime.
 * destroy(&handle) closes/frees a live object and sets *handle to NULL; repeating
 * it with that NULL slot succeeds. Passing a stale or wrong-type handle is not
 * supported. Create requires a non-NULL output slot whose current value is NULL;
 * on failure the slot remains NULL. Batch close is separately idempotent and
 * retains the handle so subsequent evaluations return TIEMU_RUNTIME_ERROR.
 *
 * Arithmetic preserves the existing scalar operation and reduction order.
 * Parallelism is limited to independent rows; workers includes the calling
 * thread. Compile without fast-math or floating-point contraction.
 */

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(TIEMU_CORE_BUILD)
#    define TIEMU_API __declspec(dllexport)
#  else
#    define TIEMU_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define TIEMU_API __attribute__((visibility("default")))
#else
#  define TIEMU_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum tiemu_status {
    TIEMU_OK = 0,
    TIEMU_VALUE_ERROR = 1,
    TIEMU_INDEX_ERROR = 2,
    TIEMU_RUNTIME_ERROR = 3
};

#define TIEMU_CORE_ABI_VERSION 1u

TIEMU_API uint32_t tiemu_abi_version(void);
TIEMU_API const char *tiemu_last_error(void);

/* values: rows x titans; matrix: titans x paths; output: rows x paths.
 * rows/titans may be zero; paths must be positive. A vector is represented as
 * rows == 1. The caller chooses the presentation shape of the output buffer. */
TIEMU_API int tiemu_path_affinities(
    const double *values, size_t values_count, size_t rows, size_t titans,
    const double *matrix, size_t matrix_count, size_t paths,
    double *output, size_t output_count);

/* output has 4 elements: activity, stability, purity, score. */
TIEMU_API int tiemu_concepts(
    const double *paths, size_t path_count, double multiplier,
    const double *distribution, size_t distribution_count, int is_boss,
    double *output, size_t output_count);

TIEMU_API int tiemu_purify(
    const double *values, size_t count, double factor,
    double *output, size_t output_count);

TIEMU_API int tiemu_normalize(
    const double *values, size_t count, double max_norm,
    double *output, size_t output_count);

TIEMU_API int tiemu_normalize_pipeline(
    const double *values, size_t count, double factor,
    const double *tide, size_t tide_count, double max_norm,
    double *output, size_t output_count);

TIEMU_API int tiemu_synergy(
    const double *first, size_t first_count,
    const double *second, size_t second_count,
    double *out_first, size_t out_first_count,
    double *out_second, size_t out_second_count);

TIEMU_API int tiemu_clash(
    const double *winner, size_t winner_count,
    const double *loser, size_t loser_count, double modifier,
    double *out_winner, size_t out_winner_count,
    double *out_loser, size_t out_loser_count);

TIEMU_API int tiemu_mentorship(
    const double *stronger, size_t stronger_count,
    const double *weaker, size_t weaker_count, double score_ratio,
    double *out_stronger, size_t out_stronger_count,
    double *out_weaker, size_t out_weaker_count);

TIEMU_API int tiemu_repulsion(
    const double *values, size_t count,
    const double *noise, size_t noise_count,
    double *output, size_t output_count);

/* logits: paths x paths; output: paths. Entity accumulation order is retained. */
TIEMU_API int tiemu_vote_totals(
    const int64_t *dominant, size_t dominant_count,
    const double *weights, size_t weights_count,
    const double *logits, size_t logits_count, size_t paths,
    double *output, size_t output_count);

/* One ordered grid generation. noise is the flattened (grid*grid, 2) array.
 * Neighbor reads intentionally observe earlier writes in the same generation. */
TIEMU_API int tiemu_inorganic_step(
    const double *activity, size_t activity_count,
    const double *stability, size_t stability_count,
    const double *noise, size_t noise_count, int64_t grid,
    double *out_activity, size_t out_activity_count,
    double *out_stability, size_t out_stability_count);

/* A complete ordered grid segment, including the CPython random() MT stream.
 * mt is the 624 uint32 words followed by its index (0..624). Gaussian cache is
 * deliberately outside this uniform-only operation and remains caller-owned. */
TIEMU_API int tiemu_inorganic_run(
    const double *activity, size_t activity_count,
    const double *stability, size_t stability_count,
    const uint32_t *mt, size_t mt_count, int64_t steps, int64_t grid,
    double *out_activity, size_t out_activity_count,
    double *out_stability, size_t out_stability_count,
    uint32_t *out_mt, size_t out_mt_count);

TIEMU_API int tiemu_batch_create(int64_t workers, int64_t min_rows, void **out_handle);
TIEMU_API int tiemu_batch_close(void *handle);
TIEMU_API int tiemu_batch_destroy(void **handle);

/* values: rows x titans; matrix: titans x paths; multipliers/bosses: rows.
 * outputs: (rows x paths), (rows x 4). A nonfinite titan in a row substitutes
 * ones for that entire input row before mapping, without mutating the input. */
TIEMU_API int tiemu_batch_concepts(
    void *handle,
    const double *values, size_t values_count, size_t rows, size_t titans,
    const double *matrix, size_t matrix_count, size_t paths,
    const double *multipliers, size_t multiplier_count,
    const double *distribution, size_t distribution_count,
    const int64_t *bosses, size_t boss_count,
    double *out_paths, size_t out_paths_count,
    double *out_stats, size_t out_stats_count);

TIEMU_API int tiemu_histogram_create(
    const int64_t *indices, size_t count, int64_t paths, void **out_handle);
TIEMU_API int tiemu_histogram_update(void *handle, int64_t index, int64_t path);
TIEMU_API int tiemu_histogram_distribution(void *handle, double *output, size_t output_count);
TIEMU_API int tiemu_histogram_destroy(void **handle);

/* Initial score buckets retain removed entities by design. Only the fallback
 * alive sequence changes on remove; all rank results preserve original order. */
TIEMU_API int tiemu_opponent_create(
    const double *scores, size_t count, double bin_size, void **out_handle);
TIEMU_API int tiemu_opponent_alive_count(void *handle, int64_t *output);
TIEMU_API int tiemu_opponent_entity_at(void *handle, int64_t rank, int64_t *output);
TIEMU_API int tiemu_opponent_count(
    void *handle, int64_t entity, double current_score, int64_t *output);
TIEMU_API int tiemu_opponent_at(
    void *handle, int64_t entity, double current_score, int64_t rank, int64_t *output);
TIEMU_API int tiemu_opponent_remove(void *handle, int64_t entity);
TIEMU_API int tiemu_opponent_destroy(void **handle);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TIEMU_CORE_H */
