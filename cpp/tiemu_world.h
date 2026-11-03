#ifndef TIEMU_WORLD_H
#define TIEMU_WORLD_H

#include "tiemu_encounters.h"

/* Resident, ordered world state without an interpreter dependency.
 *
 * Each world owns its continuous state buffers and monotonic entity IDs.
 * sync replaces the live population order: stable_id == 0 creates a new ID;
 * nonzero IDs must be unique members of the current live state. Omitted IDs
 * are removed and never reused. Thus a removed/reinserted entity is a new
 * population member. Source arrays are borrowed only until sync returns;
 * their pointers are not retained. Source scalar types are a caller concern.
 * Each titan/path vector has exactly 12 readable doubles, paths may be NULL
 * to preserve a missing lazy cache. Invalid input is rejected before changing
 * the live world. Allocation failure may invalidate the previous borrowed view.
 *
 * view, births, scoring and golden plans expose BORROWED, READ-ONLY native buffers. They are valid only
 * until the next mutating call or destruction. The caller must hold its own
 * lifetime lock from obtaining a view through the last read, and must not pass
 * resident output pointers back as inputs to any resident operation. No escaping array view
 * may outlive that lock. Per-call locking alone cannot protect a borrowed view.
 * A caller may copy values into independently owned public arrays while locked.
 *
 * Each entry point serializes use of its handle internally. Destruction must
 * not race with any operation or borrowed-view read. All status/error/handle
 * ownership rules from tiemu_core.h also apply; errors use tiemu_last_error().
 * run_encounters retains ordered RNG/arithmetic semantics of encounters_run.
 * Config reincarnator and result/trace entity indices remain current population
 * POSITIONS, not stable IDs. Config entities must match the synchronized size.
 * A membership-only reconciliation preserves previous native values for retained
 * members; those values may lag public mirrors. It marks numeric state dirty:
 * view/run reject access until a complete sync. Scoring preparation also marks
 * numeric state dirty without replacing live population state or IDs. Golden
 * identity plans also mark numeric state dirty. Births, scoring and golden plans
 * use independent inputs and remain available while numeric-dirty.
 * An encounter-failure state is stricter: sync is required before
 * view/run/births/reconcile/scoring/golden plans can be used again.
 * On a failed encounter the native scratch state/index is unspecified; do not
 * commit it to public state. A complete sync is required before reusing it.
 */

#ifdef __cplusplus
extern "C" {
#endif

#define TIEMU_WORLD_ABI_VERSION 1u

typedef struct tiemu_world_source {
    uint64_t stable_id;
    const double *titans;
    const double *paths;
    double score;
    double hit_points;
    int32_t trait;
    int32_t aspect;
    uint8_t boss;
    uint8_t unlocked;
} tiemu_world_source;

typedef struct tiemu_world_view {
    size_t entities;
    const uint64_t *stable_ids;
    tiemu_encounter_state state; /* borrowed pointers; consumers only read */
} tiemu_world_view;

typedef struct tiemu_scoring_source {
    const double *titans;       /* exactly 12 doubles; nonfinite values allowed */
    const double *cached_paths; /* NULL means an absent lazy cache */
    uint8_t boss;               /* 0 or 1 */
} tiemu_scoring_source;

typedef struct tiemu_scoring_view {
    size_t rows;
    uint64_t token;
    const double *values;       /* rows * 12 repaired current titan values */
    const double *old_paths;    /* rows * 12 old cache, before any repair */
    const int64_t *bosses;      /* rows boss flags, matching batch_concepts */
    const uint8_t *repaired;    /* rows flags: whole titan row replaced by ones */
    const double *paths;        /* rows * 12 fresh paths; NULL until finish */
    const double *stats;        /* rows * 4 batch stats; NULL until finish */
} tiemu_scoring_view;

typedef struct tiemu_golden_source {
    double score;
    int32_t trait; /* TIEMU_MORTAL, TIEMU_GOLDEN or TIEMU_REINCARNATOR */
    uint8_t boss;  /* 0 or 1 */
} tiemu_golden_source;

typedef struct tiemu_golden_view {
    size_t reset_count;
    const size_t *reset_positions;
    size_t selected_count;
    const size_t *selected_positions;
} tiemu_golden_view;

TIEMU_API uint32_t tiemu_world_abi_version(void);
TIEMU_API int tiemu_world_create(void **out_handle);
TIEMU_API int tiemu_world_sync(void *handle, const tiemu_world_source *sources, size_t count);
/* Reconcile only membership/order and new rows. order has count IDs: nonzero
 * entries are distinct current live IDs; each zero consumes one added source
 * in order. Every added source must have stable_id == 0, and the zero count
 * must equal added_count. Result IDs are copied to caller-owned out_ids with
 * out_capacity >= count. Source/result buffers may not overlap resident or
 * each other's storage. Invalid input leaves live membership, numeric access
 * state and output unchanged. A successful call requires sync before numerical
 * access, even if retained-row public values happen to remain unchanged.
 */
TIEMU_API int tiemu_world_reconcile(
    void *handle, const uint64_t *order, size_t count,
    const tiemu_world_source *added, size_t added_count,
    uint64_t *out_ids, size_t out_capacity);
TIEMU_API int tiemu_world_get_view(void *handle, tiemu_world_view *output);
TIEMU_API int tiemu_world_run_encounters(
    void *handle, const tiemu_encounter_config *config,
    const uint32_t *numpy_mt, const uint32_t *python_mt, size_t rng_words,
    tiemu_encounter_result *result);

/* Complete newborn numeric stage, in original row order:
 * blueprint + already-drawn noise -> constructor clip -> optional already
 * scaled legacy modifier -> purification/tide/clip/normalization.
 * Blueprint/tide contain 12 doubles, noise contains rows*12 doubles. Modifier
 * is absent exactly when NULL/0, otherwise it contains 12 doubles. No RNG is
 * consumed and no live entity is appended until the caller's following sync
 * or membership reconciliation.
 * The returned independent newborn buffer belongs to the world, not the caller.
 * births never changes existing population state, IDs, or encounter RNG state.
 * Constructor clipping uses +0 for values equal to either signed zero. Finite
 * inputs may still overflow in intermediate arithmetic: those IEEE results
 * follow the existing normalize_pipeline kernel, without Python FP warnings.
 * The integrating caller restricts magnitude/error-mode support as needed.
 */
TIEMU_API int tiemu_world_births(
    void *handle, const double *blueprint, size_t blueprint_count,
    const double *noise, size_t noise_count, size_t rows,
    const double *modifier, size_t modifier_count,
    const double *tide, size_t tide_count, double purity_factor, double max_norm,
    const double **out_values, size_t *out_count);

/* Two-stage scoring owns scratch separate from live ordered population state.
 * Each source has 12 titan doubles and either NULL or 12 cached path doubles.
 * matrix has exactly 12 * 12 doubles. Cached paths, including stale/nonfinite
 * values, are copied verbatim. A missing cache is mapped with the existing
 * path_affinities kernel from raw titans BEFORE replacing any nonfinite titan
 * row with ones. No caller input pointer is retained or written. Source order
 * is preserved; source rows need not be members or ordered like the live world.
 *
 * prepare copies/validates into independent scratch before publishing a
 * nonzero, monotonically increasing token and marking live numeric state dirty.
 * Invalid input leaves the previous preparation, live state and output intact.
 * It never changes live population rows, membership, IDs or RNG state. Only a
 * complete sync makes view/run available again. The caller computes original
 * multiplier arithmetic from old_paths, preserving its numerical microkernel.
 *
 * finish accepts exactly rows multipliers and an absent distribution (NULL/0)
 * or 12 distribution doubles. It uses the supplied batch evaluator and its
 * existing thread pool; all path/score arithmetic stays in batch_concepts.
 * The token is single-use. Successful sync/reconcile/run_encounters/births or
 * another prepare invalidates it. A failed finish may consume the token; a
 * fresh prepare recovers. No failed finish writes the output structure. On
 * success, output retains the prepare token and additionally exposes paths
 * and stats. Empty prepares/finishes are valid with NULL source/multiplier
 * buffers, but still require the complete mapping and a live evaluator.
 *
 * Both the world and evaluator require external lifetime locks for the whole
 * prepare -> multiplier calculation -> finish -> copy transaction. Per-call
 * native locking cannot protect borrowed pointers between calls or concurrent
 * destruction. No returned view may escape those locks; independently copy
 * public result arrays before unlocking. Never pass resident output storage
 * back as source/matrix/multiplier/distribution or output-structure storage.
 */
TIEMU_API int tiemu_world_scoring_prepare(
    void *handle, const tiemu_scoring_source *sources, size_t count,
    const double *matrix, size_t matrix_count, tiemu_scoring_view *output);
TIEMU_API int tiemu_world_scoring_finish(
    void *handle, uint64_t token, void *evaluator_handle,
    const double *multipliers, size_t multiplier_count,
    const double *distribution, size_t distribution_count, tiemu_scoring_view *output);

/* Plan golden identity changes from independent CURRENT metadata, without
 * importing titan/path vectors or relying on stale live rows. Sources need not
 * match live membership. Scores must be finite, trait uses the encounter enum,
 * and boss is 0/1. Reset positions contain every old GoldenOne in source order,
 * including bosses. Candidates exclude bosses and Reincarnators; selected
 * positions contain the first min(candidates, max(12, golden_cap)) candidates
 * in descending score order, preserving source order for equal scores.
 *
 * The caller still performs identity writes and its original random shuffle.
 * This operation consumes no RNG and retains no caller pointer. Its borrowed
 * result follows the World lifetime-lock rules above. Invalid input leaves the
 * previous plan, output, numeric-access state and scoring token unchanged.
 * Success marks live numeric state dirty and invalidates prepared scoring,
 * but never changes live membership/IDs. Membership-dirty worlds are allowed;
 * failed encounters require a full sync first. Empty plans are valid.
 */
TIEMU_API int tiemu_world_golden_plan(
    void *handle, const tiemu_golden_source *sources, size_t count,
    int64_t golden_cap, tiemu_golden_view *output);
TIEMU_API int tiemu_world_destroy(void **handle);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TIEMU_WORLD_H */
