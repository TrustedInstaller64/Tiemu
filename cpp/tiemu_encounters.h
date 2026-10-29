#ifndef TIEMU_ENCOUNTERS_H
#define TIEMU_ENCOUNTERS_H

#include "tiemu_core.h"

/* Ordered, single-world encounter engine. This header is valid C and contains
 * no interpreter/binding/array-library types. All IDs are original population
 * positions; removed entities deliberately remain in the initial score bins.
 * Only independent phase boundaries, not dependent encounters, may run in
 * parallel. Maximum encounters per call: 5000. Dimensions are fixed at 12.
 *
 * The caller owns every buffer, its capacities and the live OpponentIndex.
 * Arrays must be contiguous, non-overlapping, and live until return. The index
 * must be freshly built from this population's scores; its lifetime must be
 * synchronized for the entire call. Its alive state is updated in place.
 *
 * State buffers are in/out scratch copies, not user-visible Python arrays.
 * On error, scratch contents and index state are unspecified; do not commit
 * them to application state. No C++ exception crosses the ABI. Read the
 * shared core thread-local error before the next status-returning core or
 * encounter call.
 */

#ifdef __cplusplus
extern "C" {
#endif

#define TIEMU_ENCOUNTERS_ABI_VERSION 1u
#define TIEMU_ENCOUNTER_WIDTH 12u
#define TIEMU_MT_STATE_WORDS 625u

enum tiemu_encounter_trait { TIEMU_MORTAL = 0, TIEMU_GOLDEN = 1, TIEMU_REINCARNATOR = 2 };
enum tiemu_encounter_relation {
    TIEMU_NEUTRAL = 0, TIEMU_CLASH = 1, TIEMU_SYNERGY = 2,
    TIEMU_REPULSION = 3, TIEMU_MENTORSHIP = 4
};
enum tiemu_encounter_event_kind {
    TIEMU_REINCARNATOR_DESTROY = 1, TIEMU_BATTLE_START = 2,
    TIEMU_BATTLE_DAMAGE = 3, TIEMU_BATTLE_DEFEAT = 4, TIEMU_BATTLE_COUNTER = 5
};

typedef struct tiemu_encounter_config {
    size_t entities;
    size_t encounters;
    void *opponent_index;
    const double *mapping;             /* 12 x 12, row-major */
    size_t mapping_count;
    const int32_t *relations;           /* 12 x 12, enum above */
    size_t relations_count;
    const double *interaction_modifiers;/* 12 x 12 */
    size_t modifiers_count;
    const double *tide;                 /* 12 */
    size_t tide_count;
    double purity_factor;              /* strictly positive, finite */
    double max_norm;                   /* positive, finite */
    double culling_strength;           /* finite */
    int64_t reincarnator;              /* -1 absent, otherwise original ID */
    int64_t negative_world_titan;       /* index of the original 负世 constant */
} tiemu_encounter_config;

typedef struct tiemu_encounter_state {
    size_t entity_capacity;             /* all scalar/flag arrays below */
    size_t vector_capacity;             /* all entity x 12 arrays below */
    double *titans;
    double *paths;
    double *scores;
    double *hit_points;
    int32_t *traits;
    int32_t *aspects;                   /* -1 None, otherwise 0..11 */
    uint8_t *bosses;
    uint8_t *unlocked;
    uint8_t *path_valid;
    uint8_t *titan_replaced;            /* output: original ndarray replaced */
    uint8_t *path_changed;              /* output: original cache invalidated/replaced */
    uint8_t *original_titan_mutated;    /* output: initial ndarray had an in-place boost */
    double *original_titans;            /* input initial values; output old-alias contents */
    uint32_t *numpy_mt;                /* 624 state words, then position */
    uint32_t *python_mt;
    size_t rng_words;                  /* exactly 625 for each RNG buffer */
} tiemu_encounter_state;

typedef struct tiemu_encounter_event {
    int32_t kind;
    int32_t path_or_aspect;
    int64_t first;                     /* reincarnator or golden ID */
    int64_t second;                    /* victim or boss ID */
    double damage;                    /* value at event time */
    double remaining_hp;
} tiemu_encounter_event;

typedef struct tiemu_encounter_trace {
    int64_t attempt;
    int64_t first;
    int64_t second;
    int64_t culled;                    /* -1 if no cull */
    int32_t applied;                   /* zero if a selected old-bin entity was dead */
    uint32_t numpy_mt[TIEMU_MT_STATE_WORDS];
    uint32_t python_mt[TIEMU_MT_STATE_WORDS];
} tiemu_encounter_trace;

typedef struct tiemu_encounter_result {
    int64_t *culled_ids;               /* unique, first-cull order */
    size_t culled_capacity;
    size_t culled_count;
    int64_t *updated_ids;              /* unique, first-touch/add order */
    size_t updated_capacity;
    size_t updated_count;
    tiemu_encounter_event *events;     /* up to 3 per encounter */
    size_t event_capacity;
    size_t event_count;
    tiemu_encounter_trace *trace;      /* NULL with capacity 0 disables tracing */
    size_t trace_capacity;
    size_t trace_count;
    size_t applied_encounters;
    double first_multiplier_paths[12]; /* first old-cache read; for seat-cache side effects */
} tiemu_encounter_result;

TIEMU_API uint32_t tiemu_encounters_abi_version(void);
TIEMU_API const char *tiemu_encounters_last_error(void);
TIEMU_API int tiemu_encounters_run(
    const tiemu_encounter_config *config,
    tiemu_encounter_state *state,
    tiemu_encounter_result *result);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TIEMU_ENCOUNTERS_H */
