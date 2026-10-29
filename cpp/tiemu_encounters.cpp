// Ordered whole-encounter engine: no Python, NumPy or binding-runtime API.
#ifndef TIEMU_CORE_BUILD
#define TIEMU_CORE_BUILD
#endif
#include "tiemu_encounters.h"
#include "tiemu_internal.h"
#include "tiemu_random.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr std::size_t width = TIEMU_ENCOUNTER_WIDTH;

void checked(int status) {
    if (status != TIEMU_OK) {
        const std::string message = tiemu_last_error();
        if (status == TIEMU_VALUE_ERROR) throw std::invalid_argument(message);
        if (status == TIEMU_INDEX_ERROR) throw std::out_of_range(message);
        throw std::runtime_error(message);
    }
}

template <class T>
void required(const T *pointer, std::size_t count, const char *name) {
    if (count && !pointer) throw std::invalid_argument(std::string(name) + " is null");
}

void validate(const tiemu_encounter_config &c, const tiemu_encounter_state &s,
              const tiemu_encounter_result &r) {
    if (c.entities > std::numeric_limits<std::uint32_t>::max()
            || c.entities > std::numeric_limits<std::size_t>::max()/width)
        throw std::invalid_argument("population exceeds supported encounter size");
    if (c.encounters > 5000) throw std::invalid_argument("encounters must not exceed 5000");
    if (s.entity_capacity < c.entities || s.vector_capacity < c.entities*width)
        throw std::invalid_argument("encounter state capacity is too small");
    if (c.mapping_count != width*width || c.relations_count != width*width
            || c.modifiers_count != width*width || c.tide_count != width)
        throw std::invalid_argument("encounter engine requires 12-dimensional inputs");
    if (!std::isfinite(c.purity_factor) || c.purity_factor <= 0
            || !std::isfinite(c.max_norm) || c.max_norm <= 0
            || !std::isfinite(c.culling_strength))
        throw std::invalid_argument("encounter scalar parameters are invalid");
    if (c.reincarnator < -1 || (c.reincarnator >= 0
            && static_cast<std::size_t>(c.reincarnator) >= c.entities)
            || c.negative_world_titan < 0 || c.negative_world_titan >= 12)
        throw std::invalid_argument("encounter entity/titan index is invalid");
    if (s.rng_words != TIEMU_MT_STATE_WORDS)
        throw std::invalid_argument("encounters require two complete MT states");
    required(c.mapping, c.mapping_count, "mapping");
    required(c.relations, c.relations_count, "relations");
    required(c.interaction_modifiers, c.modifiers_count, "interaction modifiers");
    required(c.tide, c.tide_count, "tide");
    required(s.titans, c.entities*width, "titans");
    required(s.paths, c.entities*width, "paths");
    required(s.original_titans, c.entities*width, "original titans");
    required(s.scores, c.entities, "scores");
    required(s.hit_points, c.entities, "hit points");
    required(s.traits, c.entities, "traits");
    required(s.aspects, c.entities, "aspects");
    required(s.bosses, c.entities, "bosses");
    required(s.unlocked, c.entities, "unlocked");
    required(s.path_valid, c.entities, "path validity");
    required(s.titan_replaced, c.entities, "titan replacement flags");
    required(s.path_changed, c.entities, "path change flags");
    required(s.original_titan_mutated, c.entities, "old-alias mutation flags");
    required(s.numpy_mt, s.rng_words, "NumPy MT state");
    required(s.python_mt, s.rng_words, "Python MT state");
    if (r.culled_capacity < c.entities || r.updated_capacity < c.entities
            || r.event_capacity < c.encounters*3)
        throw std::invalid_argument("encounter result capacity is too small");
    required(r.culled_ids, c.entities, "culled IDs");
    required(r.updated_ids, c.entities, "updated IDs");
    required(r.events, c.encounters*3, "events");
    if ((r.trace == nullptr && r.trace_capacity != 0)
            || (r.trace != nullptr && r.trace_capacity < c.encounters))
        throw std::invalid_argument("trace capacity must cover the complete encounter limit");
    for (std::size_t j = 0; j < width*width; ++j) {
        if (!std::isfinite(c.mapping[j]) || !std::isfinite(c.interaction_modifiers[j])
                || c.relations[j] < TIEMU_NEUTRAL || c.relations[j] > TIEMU_MENTORSHIP)
            throw std::invalid_argument("encounter matrix contains an invalid value");
    }
    for (std::size_t j = 0; j < width; ++j)
        if (!std::isfinite(c.tide[j])) throw std::invalid_argument("tide must be finite");
    for (std::size_t i = 0; i < c.entities; ++i) {
        if (!std::isfinite(s.scores[i]) || !std::isfinite(s.hit_points[i])
                || s.traits[i] < TIEMU_MORTAL || s.traits[i] > TIEMU_REINCARNATOR
                || s.aspects[i] < -1 || s.aspects[i] >= 12 || s.bosses[i] > 1
                || s.unlocked[i] > 1 || s.path_valid[i] > 1)
            throw std::invalid_argument("entity metadata contains an invalid value");
        for (std::size_t j = 0; j < width; ++j) {
            if (!std::isfinite(s.titans[i*width+j])
                    || !std::isfinite(s.original_titans[i*width+j])
                    || (s.path_valid[i] && !std::isfinite(s.paths[i*width+j])))
                throw std::invalid_argument("initial encounter arrays must be finite");
        }
    }
    if (!c.opponent_index && c.entities) throw std::invalid_argument("opponent index is null");
    if (c.opponent_index) {
        std::int64_t alive = 0;
        checked(tiemu_opponent_alive_count(c.opponent_index, &alive));
        if (alive != static_cast<std::int64_t>(c.entities))
            throw std::invalid_argument("opponent index must be freshly built for this population");
    }
}

class Engine {
    const tiemu_encounter_config &c;
    tiemu_encounter_state &s;
    tiemu_encounter_result &r;
    tiemu_detail::Mt19937 numpy;
    tiemu_detail::Mt19937 python;
    std::vector<std::uint8_t> culled;
    std::vector<std::uint8_t> updated;
    bool histogram_started = false;
    std::array<std::int64_t, 2> dirty{};
    std::size_t dirty_count = 0;

    double *titans(std::int64_t id) { return s.titans + static_cast<std::size_t>(id)*width; }
    double *paths(std::int64_t id) {
        const auto index = static_cast<std::size_t>(id);
        double *output = s.paths + index*width;
        if (!s.path_valid[index]) {
            checked(tiemu_path_affinities(titans(id), width, 1, width,
                                         c.mapping, c.mapping_count, width, output, width));
            s.path_valid[index] = 1;
            s.path_changed[index] = 1;
        }
        return output;
    }

    std::int64_t dominant(std::int64_t id) {
        const double *values = paths(id);
        std::int64_t result = 0;
        // NumPy argmax selects the first maximum (and first NaN if produced).
        for (std::int64_t j = 1; j < 12; ++j) {
            if (std::isnan(values[result])) break;
            if (std::isnan(values[j]) || values[j] > values[result]) result = j;
        }
        return result;
    }

    void refresh_distribution_caches() {
        // The frozen handlers never read the distribution's numeric value.
        // Preserve its observable lazy-cache reads, including dirty ID order.
        if (!histogram_started) {
            for (std::size_t i = 0; i < c.entities; ++i) paths(static_cast<std::int64_t>(i));
            histogram_started = true;
        } else {
            for (std::size_t i = 0; i < dirty_count; ++i) paths(dirty[i]);
        }
        dirty_count = 0;
    }

    void normalize(std::int64_t id) {
        std::array<double, width> output;
        checked(tiemu_normalize_pipeline(titans(id), width, c.purity_factor,
                                         c.tide, width, c.max_norm, output.data(), width));
        std::copy(output.begin(), output.end(), titans(id));
        s.titan_replaced[id] = 1;
        s.path_valid[id] = 0;
        s.path_changed[id] = 1;
    }

    void replace(std::int64_t id, const std::array<double, width> &values) {
        std::copy(values.begin(), values.end(), titans(id));
        s.titan_replaced[id] = 1;
    }

    void event(std::int32_t kind, std::int64_t first, std::int64_t second,
               std::int32_t path = -1, double damage = 0, double hp = 0) {
        if (r.event_count >= r.event_capacity) throw std::runtime_error("event capacity exhausted");
        r.events[r.event_count++] = {kind, path, first, second, damage, hp};
    }

    std::int64_t battle(std::int64_t golden, std::int64_t boss) {
        event(TIEMU_BATTLE_START, golden, boss, s.aspects[golden]);
        const double damage = s.scores[golden] * (0.2 + python.random53() * 0.3);
        s.hit_points[boss] -= damage;
        event(TIEMU_BATTLE_DAMAGE, golden, boss, -1, damage, s.hit_points[boss]);
        if (s.hit_points[boss] <= 0) {
            event(TIEMU_BATTLE_DEFEAT, golden, boss);
            s.unlocked[golden] = 1;
            s.scores[golden] *= 1.5;
            return boss;
        }
        if (python.random53() < 0.05) {
            event(TIEMU_BATTLE_COUNTER, golden, boss);
            return golden;
        }
        return -1;
    }

    std::int64_t interaction(std::int64_t first, std::int64_t second) {
        const auto golden = (s.traits[first] == TIEMU_GOLDEN && s.aspects[first] >= 0) ? first
            : ((s.traits[second] == TIEMU_GOLDEN && s.aspects[second] >= 0) ? second : -1);
        const auto boss = s.bosses[first] ? first : (s.bosses[second] ? second : -1);
        if (golden >= 0 && boss >= 0 && s.aspects[golden] == s.aspects[boss])
            return battle(golden, boss);

        const bool has_reincarnator = c.reincarnator >= 0
            && (first == c.reincarnator || second == c.reincarnator);
        if (has_reincarnator && dominant(first) == dominant(second)) {
            const auto other = first == c.reincarnator ? second : first;
            if (s.traits[other] != TIEMU_GOLDEN) {
                event(TIEMU_REINCARNATOR_DESTROY, c.reincarnator, other,
                      static_cast<std::int32_t>(dominant(c.reincarnator)));
                return other;
            }
        }
        if (has_reincarnator) {
            const auto other = first == c.reincarnator ? second : first;
            const double factor = s.traits[other] == TIEMU_GOLDEN ? 1.2 : 1.02;
            const auto offset = static_cast<std::size_t>(c.reincarnator)*width
                + static_cast<std::size_t>(c.negative_world_titan);
            s.titans[offset] *= factor;
            if (!s.titan_replaced[c.reincarnator]) {
                s.original_titans[offset] *= factor;
                s.original_titan_mutated[c.reincarnator] = 1;
            }
            normalize(c.reincarnator);
        }
        if (s.traits[first] == TIEMU_GOLDEN || s.traits[second] == TIEMU_GOLDEN) return -1;
        const auto stronger = s.scores[first] > s.scores[second] ? first : second;
        const auto weaker = stronger == first ? second : first;
        if (numpy.random53() < std::tanh((s.scores[stronger] - s.scores[weaker])*c.culling_strength))
            return weaker;
        const auto first_path = dominant(first);
        const auto second_path = dominant(second);
        const auto relation = c.relations[first_path*12 + second_path];
        std::array<double, width> out_first, out_second;
        if (relation == TIEMU_CLASH) {
            if (!std::isfinite(s.scores[first]) || !std::isfinite(s.scores[second])) return -1;
            const auto winner = s.scores[first]*(1 + numpy.random53()*0.2) > s.scores[second] ? first : second;
            const auto loser = winner == first ? second : first;
            const double modifier = c.interaction_modifiers[dominant(winner)*12 + dominant(loser)];
            checked(tiemu_clash(titans(winner), width, titans(loser), width, modifier,
                                out_first.data(), width, out_second.data(), width));
            replace(winner, out_first);
            replace(loser, out_second);
            normalize(winner);
            normalize(loser);
            if (numpy.random53() < 0.1*(s.scores[winner]/(s.scores[loser] + 1e-6) - 1)) return loser;
        } else if (relation == TIEMU_SYNERGY) {
            checked(tiemu_synergy(titans(first), width, titans(second), width,
                                  out_first.data(), width, out_second.data(), width));
            replace(first, out_first);
            replace(second, out_second);
            normalize(first);
            normalize(second);
        } else if (relation == TIEMU_REPULSION) {
            for (const auto id : {first, second}) {
                std::array<double, width> noise;
                for (auto &value : noise) value = numpy.random53();
                checked(tiemu_repulsion(titans(id), width, noise.data(), width, out_first.data(), width));
                replace(id, out_first);
                normalize(id);
            }
        } else if (relation == TIEMU_MENTORSHIP) {
            if (!std::isfinite(s.scores[stronger]) || !std::isfinite(s.scores[weaker])
                    || s.scores[weaker] <= 0) return -1;
            const double ratio = s.scores[stronger]/(s.scores[weaker] + 1e-6);
            checked(tiemu_mentorship(titans(stronger), width, titans(weaker), width, ratio,
                                     out_first.data(), width, out_second.data(), width));
            replace(stronger, out_first);
            replace(weaker, out_second);
            normalize(stronger);
            normalize(weaker);
        }
        return -1;
    }

public:
    Engine(const tiemu_encounter_config &config, tiemu_encounter_state &state,
           tiemu_encounter_result &result)
        : c(config), s(state), r(result), numpy(state.numpy_mt), python(state.python_mt),
          culled(config.entities, 0), updated(config.entities, 0) {}

    void run() {
        r.culled_count = r.updated_count = r.event_count = r.trace_count = r.applied_encounters = 0;
        std::fill_n(s.titan_replaced, c.entities, 0);
        std::fill_n(s.path_changed, c.entities, 0);
        std::fill_n(s.original_titan_mutated, c.entities, 0);
        for (std::size_t attempt = 0; attempt < c.encounters; ++attempt) {
            std::int64_t alive = 0, first = -1, second = -1, count = 0, removed = -1;
            bool applied = false;
            checked(tiemu_opponent_alive_count(c.opponent_index, &alive));
            if (alive < 2) break;
            checked(tiemu_opponent_entity_at(c.opponent_index,
                numpy.interval(static_cast<std::uint32_t>(alive - 1)), &first));
            checked(tiemu_opponent_count(c.opponent_index, first, s.scores[first], &count));
            if (count > 0) {
                checked(tiemu_opponent_at(c.opponent_index, first, s.scores[first],
                    numpy.interval(static_cast<std::uint32_t>(count - 1)), &second));
                // Both ranks have already consumed their original RNG draws.
                if (!culled[first] && !culled[second]) {
                    double *old_paths = paths(first);
                    if (!r.applied_encounters)
                        std::copy_n(old_paths, width, r.first_multiplier_paths);
                    refresh_distribution_caches();
                    removed = interaction(first, second);
                    if (removed >= 0) {
                        culled[removed] = 1;
                        r.culled_ids[r.culled_count++] = removed;
                        checked(tiemu_opponent_remove(c.opponent_index, removed));
                    }
                    for (const auto id : {first, second}) {
                        if (!updated[id]) {
                            updated[id] = 1;
                            r.updated_ids[r.updated_count++] = id;
                        }
                    }
                    dirty = {std::min(first, second), std::max(first, second)};
                    dirty_count = first == second ? 1 : 2;
                    ++r.applied_encounters;
                    applied = true;
                }
            }
            if (r.trace) {
                auto &entry = r.trace[r.trace_count++];
                entry.attempt = static_cast<std::int64_t>(attempt);
                entry.first = first;
                entry.second = second;
                entry.culled = removed;
                entry.applied = applied ? 1 : 0;
                numpy.write(entry.numpy_mt);
                python.write(entry.python_mt);
            }
        }
        numpy.write(s.numpy_mt);
        python.write(s.python_mt);
    }
};
} // namespace

extern "C" {
uint32_t tiemu_encounters_abi_version(void) { return TIEMU_ENCOUNTERS_ABI_VERSION; }
const char *tiemu_encounters_last_error(void) { return tiemu_detail::last_error(); }
int tiemu_encounters_run(const tiemu_encounter_config *config, tiemu_encounter_state *state,
                         tiemu_encounter_result *result) {
    return tiemu_detail::guarded([&] {
        if (!config || !state || !result) throw std::invalid_argument("encounter structs must not be null");
        validate(*config, *state, *result);
        Engine(*config, *state, *result).run();
    });
}
}
