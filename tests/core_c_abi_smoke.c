#include "tiemu_core.h"
#include "tiemu_encounters.h"
#include "tiemu_world.h"
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
        if (tiemu_world_sync(world, NULL, 0)) return 28;
        sources[0].stable_id = 0;
        if (tiemu_world_sync(world, sources, 1)) return 29;
        if (tiemu_world_get_view(world, &view) || view.stable_ids[0] <= second_id) return 30;
        if (tiemu_world_destroy(&world) || world != NULL) return 31;
        if (tiemu_world_destroy(&world)) return 32;
    }
    puts("standalone C client: ABI, arithmetic, resident lifecycle, births and error contracts passed");
    return 0;
}
