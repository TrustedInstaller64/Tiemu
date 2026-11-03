// Resident state and ordered birth/encounter stages: no interpreter dependency.
#ifndef TIEMU_CORE_BUILD
#define TIEMU_CORE_BUILD
#endif
#include "tiemu_world.h"
#include "tiemu_internal.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {
constexpr std::size_t width = TIEMU_ENCOUNTER_WIDTH;

void checked(int status) {
    if (status == TIEMU_OK) return;
    const std::string message = tiemu_last_error();
    if (status == TIEMU_VALUE_ERROR) throw std::invalid_argument(message);
    if (status == TIEMU_INDEX_ERROR) throw std::out_of_range(message);
    throw std::runtime_error(message);
}

template<class T>
void required(const T *pointer, std::size_t count, const char *name) {
    if (count && !pointer) throw std::invalid_argument(std::string(name) + " is null");
}

std::size_t vector_size(std::size_t rows) {
    if (rows > std::numeric_limits<std::uint32_t>::max()
            || rows > std::numeric_limits<std::size_t>::max() / sizeof(tiemu_world_source)
            || rows > std::numeric_limits<std::size_t>::max() / width / sizeof(double))
        throw std::invalid_argument("resident population exceeds supported size");
    return rows * width;
}

void finite_array(const double *values, std::size_t count, const char *name) {
    required(values, count, name);
    for (std::size_t i = 0; i < count; ++i)
        if (!std::isfinite(values[i]))
            throw std::invalid_argument(std::string(name) + " must be finite");
}

struct World {
    std::mutex mutex;
    std::size_t entities = 0;
    std::uint64_t next_id = 1;
    bool needs_sync = false;
    std::vector<std::uint64_t> ids, next_ids;
    std::unordered_set<std::uint64_t> live_ids, next_live_ids;
    std::vector<double> titans, paths, scores, hp, original_titans;
    std::vector<std::int32_t> traits, aspects;
    std::vector<std::uint8_t> bosses, unlocked, valid, replaced, changed, old_mutated;
    std::array<std::uint32_t, TIEMU_MT_STATE_WORDS> numpy_mt{}, python_mt{};
    std::vector<double> newborns;

    void reserve(std::size_t count, std::size_t vectors) {
        // All potentially throwing allocations precede any live-value write.
        // Capacity remains allocated across subsequent generations.
        titans.reserve(vectors); paths.reserve(vectors); original_titans.reserve(vectors);
        scores.reserve(count); hp.reserve(count); traits.reserve(count); aspects.reserve(count);
        bosses.reserve(count); unlocked.reserve(count); valid.reserve(count);
        replaced.reserve(count); changed.reserve(count); old_mutated.reserve(count);
    }

    void resize(std::size_t count, std::size_t vectors) {
        titans.resize(vectors); paths.resize(vectors); original_titans.resize(vectors);
        scores.resize(count); hp.resize(count); traits.resize(count); aspects.resize(count);
        bosses.resize(count); unlocked.resize(count); valid.resize(count);
        replaced.resize(count); changed.resize(count); old_mutated.resize(count);
    }

    tiemu_encounter_state state() {
        return {entities, entities * width, titans.data(), paths.data(), scores.data(), hp.data(),
                traits.data(), aspects.data(), bosses.data(), unlocked.data(), valid.data(),
                replaced.data(), changed.data(), old_mutated.data(), original_titans.data(),
                numpy_mt.data(), python_mt.data(), TIEMU_MT_STATE_WORDS};
    }

    void ready() const {
        if (needs_sync)
            throw std::runtime_error("resident world requires a complete sync after failed encounters");
    }
};

World &live(void *handle) {
    if (!handle) throw std::invalid_argument("resident world handle is null");
    return *static_cast<World *>(handle);
}

void sync(World &world, const tiemu_world_source *sources, std::size_t count) {
    const auto vectors = vector_size(count);
    required(sources, count, "resident sources");
    world.next_ids.resize(count);
    world.next_live_ids.clear();
    world.next_live_ids.reserve(count);
    auto next_id = world.next_id;
    for (std::size_t i = 0; i < count; ++i) {
        const auto &row = sources[i];
        auto id = row.stable_id;
        if (id) {
            if (world.live_ids.find(id) == world.live_ids.end())
                throw std::invalid_argument("resident stable ID is not live");
        } else {
            if (!next_id) throw std::invalid_argument("resident stable IDs exhausted");
            id = next_id;
            next_id = next_id == std::numeric_limits<std::uint64_t>::max() ? 0 : next_id + 1;
        }
        if (!world.next_live_ids.insert(id).second)
            throw std::invalid_argument("resident stable IDs must be unique");
        world.next_ids[i] = id;
        if (!std::isfinite(row.score) || !std::isfinite(row.hit_points)
                || row.trait < TIEMU_MORTAL || row.trait > TIEMU_REINCARNATOR
                || row.aspect < -1 || row.aspect >= static_cast<std::int32_t>(width)
                || row.boss > 1 || row.unlocked > 1)
            throw std::invalid_argument("resident entity metadata is invalid");
        finite_array(row.titans, width, "resident titans");
        if (row.paths) finite_array(row.paths, width, "resident paths");
    }
    world.reserve(count, vectors);
    world.resize(count, vectors);
    // Validation/allocation are complete. No source pointer is retained.
    for (std::size_t i = 0; i < count; ++i) {
        const auto &row = sources[i];
        std::copy_n(row.titans, width, world.titans.data() + i * width);
        std::copy_n(row.titans, width, world.original_titans.data() + i * width);
        if (row.paths) std::copy_n(row.paths, width, world.paths.data() + i * width);
        else std::fill_n(world.paths.data() + i * width, width, 0.0);
        world.scores[i] = row.score;
        world.hp[i] = row.hit_points;
        world.traits[i] = row.trait;
        world.aspects[i] = row.aspect;
        world.bosses[i] = row.boss;
        world.unlocked[i] = row.unlocked;
        world.valid[i] = row.paths ? 1 : 0;
        world.replaced[i] = world.changed[i] = world.old_mutated[i] = 0;
    }
    world.ids.swap(world.next_ids);
    world.live_ids.swap(world.next_live_ids);
    world.entities = count;
    world.next_id = next_id;
    world.needs_sync = false;
}
} // namespace

extern "C" {
uint32_t tiemu_world_abi_version(void) { return TIEMU_WORLD_ABI_VERSION; }

int tiemu_world_create(void **out_handle) {
    return tiemu_detail::guarded([&] {
        if (!out_handle || *out_handle)
            throw std::invalid_argument("resident create requires an empty handle slot");
        *out_handle = new World();
    });
}

int tiemu_world_sync(void *handle, const tiemu_world_source *sources, size_t count) {
    return tiemu_detail::guarded([&] {
        auto &world = live(handle);
        std::lock_guard<std::mutex> lock(world.mutex);
        sync(world, sources, count);
    });
}

int tiemu_world_get_view(void *handle, tiemu_world_view *output) {
    return tiemu_detail::guarded([&] {
        auto &world = live(handle);
        std::lock_guard<std::mutex> lock(world.mutex);
        if (!output) throw std::invalid_argument("resident view output is null");
        world.ready();
        *output = {world.entities, world.ids.data(), world.state()};
    });
}

int tiemu_world_run_encounters(void *handle, const tiemu_encounter_config *config,
        const uint32_t *numpy_mt, const uint32_t *python_mt, size_t rng_words,
        tiemu_encounter_result *result) {
    return tiemu_detail::guarded([&] {
        auto &world = live(handle);
        std::lock_guard<std::mutex> lock(world.mutex);
        world.ready();
        if (!config || !result) throw std::invalid_argument("resident encounter config/result is null");
        if (config->entities != world.entities)
            throw std::invalid_argument("encounter population must match synchronized resident state");
        if (rng_words != TIEMU_MT_STATE_WORDS || !numpy_mt || !python_mt
                || numpy_mt[TIEMU_MT_STATE_WORDS-1] > 624 || python_mt[TIEMU_MT_STATE_WORDS-1] > 624)
            throw std::invalid_argument("resident encounters require complete valid MT states");
        std::copy_n(numpy_mt, TIEMU_MT_STATE_WORDS, world.numpy_mt.data());
        std::copy_n(python_mt, TIEMU_MT_STATE_WORDS, world.python_mt.data());
        // Each segment has its own initial public-array alias, including when
        // another resident segment follows without a Python synchronization.
        std::copy(world.titans.begin(), world.titans.end(), world.original_titans.begin());
        auto state = world.state();
        const auto status = tiemu_encounters_run(config, &state, result);
        if (status != TIEMU_OK) world.needs_sync = true;
        checked(status);
    });
}

int tiemu_world_births(void *handle, const double *blueprint, size_t blueprint_count,
        const double *noise, size_t noise_count, size_t rows,
        const double *modifier, size_t modifier_count, const double *tide, size_t tide_count,
        double purity_factor, double max_norm, const double **out_values, size_t *out_count) {
    return tiemu_detail::guarded([&] {
        auto &world = live(handle);
        std::lock_guard<std::mutex> lock(world.mutex);
        world.ready();
        const auto count = vector_size(rows);
        if (blueprint_count != width || tide_count != width || noise_count != count
                || (modifier ? modifier_count != width : modifier_count != 0)
                || !std::isfinite(purity_factor) || !std::isfinite(max_norm) || max_norm <= 0
                || !out_values || !out_count)
            throw std::invalid_argument("resident birth shape/scalar/output is invalid");
        finite_array(blueprint, width, "birth blueprint");
        finite_array(noise, noise_count, "birth noise");
        finite_array(tide, width, "birth tide");
        if (modifier) finite_array(modifier, width, "birth modifier");
        world.newborns.resize(count);
        for (std::size_t row = 0; row < rows; ++row) {
            std::array<double, width> values;
            // Separate stores retain the original ufunc operation boundaries.
            for (std::size_t j = 0; j < width; ++j) values[j] = blueprint[j] + noise[row*width+j];
            // The optimized NumPy maximum(value, +0) used by constructor clip
            // returns the positive bound for a signed-zero tie on x86/ARM.
            for (auto &value : values) if (value <= 0.0) value = 0.0;
            if (modifier) for (std::size_t j = 0; j < width; ++j) values[j] += modifier[j];
            checked(tiemu_normalize_pipeline(values.data(), width, purity_factor, tide, width,
                                             max_norm, world.newborns.data() + row*width, width));
        }
        *out_values = world.newborns.data();
        *out_count = count;
    });
}

int tiemu_world_destroy(void **handle) {
    return tiemu_detail::guarded([&] {
        if (!handle) throw std::invalid_argument("resident destroy handle slot is null");
        // The caller synchronizes destruction with every operation/view read.
        delete static_cast<World *>(*handle);
        *handle = nullptr;
    });
}
} // extern "C"
