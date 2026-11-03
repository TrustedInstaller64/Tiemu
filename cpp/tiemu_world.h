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
 * view and births expose BORROWED, READ-ONLY native buffers. They are valid only
 * until the next mutating call or destruction. The caller must hold its own
 * lifetime lock from obtaining a view through the last read, and must not pass
 * resident output pointers back as sync/birth inputs. No escaping array view
 * may outlive that lock. Per-call locking alone cannot protect a borrowed view.
 * A caller may copy values into independently owned public arrays while locked.
 *
 * Each entry point serializes use of its handle internally. Destruction must
 * not race with any operation or borrowed-view read. All status/error/handle
 * ownership rules from tiemu_core.h also apply; errors use tiemu_last_error().
 * run_encounters retains ordered RNG/arithmetic semantics of encounters_run.
 * Config reincarnator and result/trace entity indices remain current population
 * POSITIONS, not stable IDs. Config entities must match the synchronized size.
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

TIEMU_API uint32_t tiemu_world_abi_version(void);
TIEMU_API int tiemu_world_create(void **out_handle);
TIEMU_API int tiemu_world_sync(void *handle, const tiemu_world_source *sources, size_t count);
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
 * consumed and no live entity is appended until the caller's following sync.
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
TIEMU_API int tiemu_world_destroy(void **handle);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TIEMU_WORLD_H */
