#include "tiemu_core.h"
#include "tiemu_encounters.h"
#include "tiemu_world.h"
#include "tiemu_aeonic.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

/* A C-only client proves the core can load and execute outside Python. */
int main(void) {
    const double values[3] = {2.0, 3.0, 5.0};
    const double matrix[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    double output[3] = {0, 0, 0};
    void *index = NULL;
    int64_t count = -1;
    if (tiemu_abi_version() != TIEMU_CORE_ABI_VERSION) return 1;
    if (tiemu_path_affinities(values, 3, 1, 3, matrix, 9, 3, output, 3)) return 2;
    if (output[0] != 0.2 || output[1] != 0.3 || output[2] != 0.5) return 3;
    if (tiemu_opponent_create(values, 3, 10.0, &index)) return 4;
    if (tiemu_opponent_alive_count(index, &count) || count != 3) return 5;
    if (tiemu_opponent_remove(index, 1)) return 6;
    if (tiemu_opponent_alive_count(index, &count) || count != 2) return 7;
    if (tiemu_opponent_destroy(&index) || index != NULL) return 8;
    if (tiemu_opponent_destroy(&index)) return 9;
    if (tiemu_concepts(NULL, 0, 1.0, NULL, 0, 0, output, 3) != TIEMU_VALUE_ERROR) return 10;
    if (strlen(tiemu_last_error()) == 0) return 11;
    if (tiemu_encounters_abi_version() != TIEMU_ENCOUNTERS_ABI_VERSION) return 12;
    if (tiemu_encounters_run(NULL, NULL, NULL) != TIEMU_VALUE_ERROR) return 13;
    if (strlen(tiemu_encounters_last_error()) == 0) return 14;
    {
        void *world = NULL;
        tiemu_world_source sources[2] = {{0}};
        tiemu_world_view view;
        double titans[12], noise[24] = {0}, tide[12] = {0};
        const double *newborns = NULL;
        size_t newborn_count = 0;
        uint64_t first_id, second_id;
        size_t i;
        for (i = 0; i < 12; ++i) titans[i] = 1.0;
        for (i = 0; i < 2; ++i) {
            sources[i].titans = titans;
            sources[i].hit_points = 100.0;
            sources[i].aspect = -1;
        }
        if (tiemu_world_abi_version() != TIEMU_WORLD_ABI_VERSION) return 15;
        if (tiemu_world_create(&world)) return 16;
        if (tiemu_world_sync(world, sources, 2)) return 17;
        if (tiemu_world_get_view(world, &view) || view.entities != 2) return 18;
        first_id = view.stable_ids[0]; second_id = view.stable_ids[1];
        if (!first_id || second_id <= first_id || view.state.titans[0] != 1.0) return 19;
        titans[0] = 9.0;
        if (view.state.titans[0] != 1.0) return 20; /* owned source copy */
        sources[0].stable_id = second_id; sources[1].stable_id = first_id;
        if (tiemu_world_sync(world, sources, 2)) return 21;
        if (tiemu_world_get_view(world, &view) || view.stable_ids[0] != second_id
                || view.stable_ids[1] != first_id || view.state.titans[0] != 9.0) return 22;
        sources[1].stable_id = second_id;
        if (tiemu_world_sync(world, sources, 2) != TIEMU_VALUE_ERROR) return 23;
        if (tiemu_world_get_view(world, &view) || view.stable_ids[1] != first_id) return 24;
        titans[0] = 1.0;
        if (tiemu_world_births(world, titans, 12, noise, 24, 2, NULL, 0, tide, 12,
                               0.0, 100.0, &newborns, &newborn_count)) return 25;
        if (newborn_count != 24) return 26;
        for (i = 0; i < newborn_count; ++i) if (newborns[i] != 1.0) return 27;
        {
            uint64_t order[2], assigned[2];
            order[0] = first_id; order[1] = 0;
            sources[0].stable_id = 0;
            if (tiemu_world_reconcile(world, order, 2, sources, 1, assigned, 2)) return 33;
            if (assigned[0] != first_id || assigned[1] <= second_id) return 34;
            if (tiemu_world_get_view(world, &view) != TIEMU_RUNTIME_ERROR) return 35;
            if (tiemu_world_births(world, titans, 12, noise, 24, 2, NULL, 0, tide, 12,
                                   0.0, 100.0, &newborns, &newborn_count)) return 36;
            if (tiemu_world_get_view(world, &view) != TIEMU_RUNTIME_ERROR) return 37;
            sources[0].stable_id = assigned[0]; sources[1].stable_id = assigned[1];
            if (tiemu_world_sync(world, sources, 2)) return 38;
            if (tiemu_world_get_view(world, &view) || view.stable_ids[1] != assigned[1]) return 39;
        }
        {
            void *evaluator = NULL;
            double current[24], old_cache[12], mapping[144] = {0};
            double repaired[24], multipliers[2] = {0.91, 1.19};
            double expected_paths[24], expected_stats[8];
            int64_t bosses[2] = {0, 1};
            tiemu_scoring_source scoring_sources[2] = {{0}};
            tiemu_scoring_view prepared, finished, sentinel = {0};
            unsigned char before[sizeof(sentinel)];
            uint64_t token;
            for (i = 0; i < 24; ++i) current[i] = repaired[i] = 1.0;
            for (i = 0; i < 12; ++i) {
                old_cache[i] = 0.25;
                mapping[i * 12 + i] = 1.0;
            }
            current[12] = NAN;
            scoring_sources[0].titans = current;
            scoring_sources[0].cached_paths = old_cache;
            scoring_sources[1].titans = current + 12;
            scoring_sources[1].boss = 1;
            if (tiemu_batch_create(2, 1, &evaluator)) return 40;
            if (tiemu_batch_concepts(evaluator, repaired, 24, 2, 12, mapping, 144, 12,
                    multipliers, 2, NULL, 0, bosses, 2, expected_paths, 24, expected_stats, 8)) return 41;
            if (tiemu_world_scoring_prepare(world, scoring_sources, 2, mapping, 144, &prepared)) return 42;
            token = prepared.token;
            if (!token || prepared.rows != 2 || prepared.paths || prepared.stats) return 43;
            if (prepared.repaired[0] || !prepared.repaired[1] || prepared.bosses[1] != 1) return 44;
            for (i = 0; i < 24; ++i) if (prepared.values[i] != 1.0) return 45;
            for (i = 0; i < 12; ++i)
                if (prepared.old_paths[i] != 0.25 || !isnan(prepared.old_paths[12 + i])) return 46;
            if (!isnan(current[12]) || current[0] != 1.0 || old_cache[0] != 0.25) return 47;
            if (tiemu_world_get_view(world, &view) != TIEMU_RUNTIME_ERROR) return 48;
            sentinel.rows = 999; sentinel.token = 123;
            memcpy(before, &sentinel, sizeof(sentinel));
            if (tiemu_world_scoring_prepare(world, scoring_sources, 2, mapping, 143,
                    &sentinel) != TIEMU_VALUE_ERROR || memcmp(before, &sentinel, sizeof(sentinel))) return 49;
            /* The old preparation owns every input even after caller changes. */
            current[0] = 99.0; old_cache[0] = 99.0; mapping[0] = 99.0;
            if (tiemu_world_scoring_finish(world, token, evaluator, multipliers, 2, NULL, 0, &finished)) return 50;
            if (finished.token != token || finished.rows != 2) return 51;
            for (i = 0; i < 24; ++i) if (finished.paths[i] != expected_paths[i]) return 52;
            for (i = 0; i < 8; ++i) if (finished.stats[i] != expected_stats[i]) return 53;
            if (tiemu_world_scoring_finish(world, token, evaluator, multipliers, 2, NULL, 0,
                    &sentinel) != TIEMU_RUNTIME_ERROR || memcmp(before, &sentinel, sizeof(sentinel))) return 54;
            if (tiemu_world_sync(world, sources, 2)) return 55;
            if (tiemu_world_get_view(world, &view) || view.entities != 2
                    || view.stable_ids[0] != sources[0].stable_id || view.state.titans[0] != 1.0) return 56;
            if (tiemu_world_scoring_prepare(world, NULL, 0, mapping, 144, &prepared)) return 57;
            if (!prepared.token || prepared.token == token || prepared.rows) return 58;
            if (tiemu_world_scoring_finish(world, prepared.token, evaluator, NULL, 0, NULL, 0, &finished)) return 59;
            if (finished.rows) return 60;
            if (tiemu_world_scoring_prepare(world, scoring_sources, 2, mapping, 144, &prepared)) return 61;
            if (tiemu_world_births(world, titans, 12, noise, 24, 2, NULL, 0, tide, 12,
                                   0.0, 100.0, &newborns, &newborn_count)) return 62;
            if (tiemu_world_scoring_finish(world, prepared.token, evaluator, multipliers, 2,
                    NULL, 0, &sentinel) != TIEMU_RUNTIME_ERROR) return 63;
            if (tiemu_world_scoring_prepare(world, scoring_sources, 2, mapping, 144, &prepared)) return 64;
            if (tiemu_batch_close(evaluator)) return 65;
            if (tiemu_world_scoring_finish(world, prepared.token, evaluator, multipliers, 2, NULL, 0,
                    &sentinel) != TIEMU_RUNTIME_ERROR || memcmp(before, &sentinel, sizeof(sentinel))) return 66;
            if (tiemu_batch_destroy(&evaluator) || evaluator != NULL) return 67;
        }
        if (tiemu_world_sync(world, NULL, 0)) return 28;
        sources[0].stable_id = 0;
        if (tiemu_world_sync(world, sources, 1)) return 29;
        if (tiemu_world_get_view(world, &view) || view.stable_ids[0] <= second_id) return 30;
        if (tiemu_world_destroy(&world) || world != NULL) return 31;
        if (tiemu_world_destroy(&world)) return 32;
    }
    {
        void *aeonic = NULL;
        uint8_t golden[4] = {0, 1, 0, 1};
        uint8_t titans[4] = {1, 0, 1, 0};
        uint8_t holders[4] = {0, 0, 1, 1};
        size_t size = 999, position = 999;
        if (tiemu_aeonic_create(golden, titans, holders, 4, 1, &aeonic)) return 68;
        memset(golden, 0, sizeof(golden));
        memset(titans, 0, sizeof(titans));
        memset(holders, 0, sizeof(holders));
        if (tiemu_aeonic_pool_size(aeonic, TIEMU_AEONIC_ATTACKERS, -1, &size) || size != 3) return 69;
        if (tiemu_aeonic_pool_at(aeonic, TIEMU_AEONIC_ATTACKERS, -1, 0, &position) || position != 1) return 70;
        if (tiemu_aeonic_pool_at(aeonic, TIEMU_AEONIC_ATTACKERS, -1, 1, &position) || position != 1) return 71;
        if (tiemu_aeonic_pool_at(aeonic, TIEMU_AEONIC_ATTACKERS, -1, 2, &position) || position != 3) return 72;
        if (tiemu_aeonic_pool_size(aeonic, TIEMU_AEONIC_HOLDERS, 2, &size) || size != 2) return 73;
        if (tiemu_aeonic_pool_at(aeonic, TIEMU_AEONIC_HOLDERS, 2, 1, &position) || position != 3) return 74;
        if (tiemu_aeonic_pool_at(aeonic, TIEMU_AEONIC_TITANS, 0, 0, &position) || position != 2) return 75;
        if (tiemu_aeonic_update(aeonic, 3, 1, 0) || tiemu_aeonic_update(aeonic, 0, 0, 0)) return 76;
        if (tiemu_aeonic_pool_size(aeonic, TIEMU_AEONIC_TITANS, -1, &size) || size != 2) return 77;
        if (tiemu_aeonic_pool_at(aeonic, TIEMU_AEONIC_TITANS, -1, 1, &position) || position != 3) return 78;
        if (tiemu_aeonic_update(aeonic, 2, 2, 0) != TIEMU_VALUE_ERROR) return 79;
        if (tiemu_aeonic_pool_at(aeonic, TIEMU_AEONIC_HOLDERS, 3, 0, &position) || position != 2) return 80;
        position = 999;
        if (tiemu_aeonic_pool_at(aeonic, TIEMU_AEONIC_ALL, 0, 3, &position) != TIEMU_VALUE_ERROR
                || position != 999) return 81;
        size = 999;
        if (tiemu_aeonic_pool_size(aeonic, TIEMU_AEONIC_ATTACKERS, 0, &size) != TIEMU_VALUE_ERROR
                || size != 999) return 82;
        if (tiemu_aeonic_destroy(&aeonic) || aeonic != NULL) return 83;
        if (tiemu_aeonic_destroy(&aeonic)) return 84;
        titans[2] = 1;
        if (tiemu_aeonic_create(golden, titans, holders, 4, -1, &aeonic)) return 85;
        if (tiemu_aeonic_pool_at(aeonic, TIEMU_AEONIC_ATTACKERS, -1, 0, &position) || position != 2) return 86;
        if (tiemu_aeonic_update(aeonic, 2, 0, 0)) return 87;
        if (tiemu_aeonic_pool_size(aeonic, TIEMU_AEONIC_ATTACKERS, -1, &size) || size != 4) return 88;
        if (tiemu_aeonic_pool_at(aeonic, TIEMU_AEONIC_ATTACKERS, -1, 3, &position) || position != 3) return 89;
        if (tiemu_aeonic_update(aeonic, 1, 1, 0)) return 90;
        if (tiemu_aeonic_pool_size(aeonic, TIEMU_AEONIC_ATTACKERS, -1, &size) || size != 1) return 91;
        if (tiemu_aeonic_pool_at(aeonic, TIEMU_AEONIC_ATTACKERS, -1, 0, &position) || position != 1) return 92;
        if (tiemu_aeonic_destroy(&aeonic)) return 93;
        if (tiemu_aeonic_create(NULL, NULL, NULL, 0, -1, &aeonic)) return 94;
        if (tiemu_aeonic_pool_size(aeonic, TIEMU_AEONIC_ALL, -1, &size) || size != 0) return 95;
        position = 999;
        if (tiemu_aeonic_pool_at(aeonic, TIEMU_AEONIC_ATTACKERS, -1, 0, &position) != TIEMU_VALUE_ERROR
                || position != 999) return 96;
        if (tiemu_aeonic_destroy(&aeonic)) return 97;
    }
    puts("standalone C client: ABI, arithmetic, resident lifecycle, births, scoring, aeonic index and error contracts passed");
    return 0;
}
