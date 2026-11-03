#ifndef TIEMU_AEONIC_H
#define TIEMU_AEONIC_H

#include "tiemu_core.h"

#ifdef __cplusplus
extern "C" {
#endif

enum tiemu_aeonic_pool {
    TIEMU_AEONIC_ATTACKERS = 0,
    TIEMU_AEONIC_ALL = 1,
    TIEMU_AEONIC_HOLDERS = 2, /* current titans OR current fire-seed holders */
    TIEMU_AEONIC_TITANS = 3
};

/* Ordered candidate indices for one fixed population, without an interpreter
 * or RNG dependency. Every returned value is a zero-based population position.
 * create copies three arrays of count flags (each exactly 0 or 1). reincarnator
 * is -1 or a valid population position. The preferred attacker list is the
 * optional reincarnator followed by initial golden positions in population
 * order; a golden reincarnator intentionally occurs twice. Golden membership
 * stays fixed for this handle; the caller rebuilds it whenever those identities
 * may change before the original selector's next population scan.
 * If that preferred list is empty, attackers are current titans, falling back
 * to all population positions when no titans remain. Other pools always retain
 * population order. update changes current titan/fire-seed flags atomically.
 * Construction is O(count); dynamic updates and filtered rank lookup are
 * O(log count), with O(1) pool sizes. No candidate array is rebuilt per query.
 *
 * excluded is -1 (none) or a population position, never an ordinal within a
 * filtered list. The attacker pool only accepts -1. pool_at selects the given
 * zero-based rank after exclusion. Empty populations/pools are valid; every
 * rank in an empty pool is invalid. Null handles/output pointers and invalid
 * flags, pool kinds, exclusions or ranks return TIEMU_VALUE_ERROR and do not
 * change flags, handles or output values. Allocation failure is reported by
 * the ordinary tiemu_core.h error contract; no partial handle is published.
 *
 * Each operation locks the handle internally. A caller must also hold its own
 * lifetime/transaction lock across pool_size -> RNG selection -> pool_at, so
 * another update cannot change membership between those calls. Destruction
 * must not race an operation. Inputs are borrowed only during create; scalar
 * results are copied to caller storage. destroy clears the handle slot and
 * is idempotent. All status/error rules from tiemu_core.h apply.
 */
TIEMU_API int tiemu_aeonic_create(
    const uint8_t *golden, const uint8_t *titans, const uint8_t *holders,
    size_t count, int64_t reincarnator, void **out_handle);
TIEMU_API int tiemu_aeonic_pool_size(
    void *handle, uint32_t pool, int64_t excluded, size_t *out_count);
TIEMU_API int tiemu_aeonic_pool_at(
    void *handle, uint32_t pool, int64_t excluded, size_t rank, size_t *out_position);
TIEMU_API int tiemu_aeonic_update(
    void *handle, size_t position, uint8_t titan, uint8_t holder);
TIEMU_API int tiemu_aeonic_destroy(void **handle);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TIEMU_AEONIC_H */
