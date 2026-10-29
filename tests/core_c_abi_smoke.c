#include "tiemu_core.h"
#include "tiemu_encounters.h"
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
    puts("standalone C client: ABI, arithmetic, state handle and error contracts passed");
    return 0;
}
