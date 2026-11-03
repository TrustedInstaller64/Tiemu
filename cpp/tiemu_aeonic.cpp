// Ordered generation-local candidate pools. No interpreter or RNG dependency.
#ifndef TIEMU_CORE_BUILD
#define TIEMU_CORE_BUILD
#endif
#include "tiemu_aeonic.h"
#include "tiemu_internal.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace {
using Position = std::size_t;

Position low_bit(Position value) noexcept { return value & (~value + 1); }

class Fenwick {
    std::vector<Position> tree_;
    Position count_ = 0, total_ = 0, highest_bit_ = 0;

public:
    template<class Flag>
    void build(Position count, Flag flag) {
        count_ = count;
        tree_.assign(count + 1, 0);
        total_ = 0;
        highest_bit_ = count ? 1 : 0;
        while (highest_bit_ && highest_bit_ <= count / 2) highest_bit_ <<= 1;
        // Each node contributes once to its parent, yielding linear build.
        for (Position i = 1; i <= count; ++i) {
            const Position value = flag(i - 1) ? 1 : 0;
            tree_[i] += value;
            total_ += value;
            const Position parent = i + low_bit(i);
            if (parent <= count) tree_[parent] += tree_[i];
        }
    }

    Position total() const noexcept { return total_; }

    Position before(Position position) const noexcept {
        Position sum = 0;
        for (Position i = position; i; i -= low_bit(i)) sum += tree_[i];
        return sum;
    }

    void update(Position position, bool previous, bool next) noexcept {
        if (previous == next) return;
        if (next) ++total_; else --total_;
        for (Position i = position + 1; i <= count_; i += low_bit(i)) {
            if (next) ++tree_[i]; else --tree_[i];
        }
    }

    Position at(Position rank) const noexcept {
        Position position = 0;
        for (Position bit = highest_bit_; bit; bit >>= 1) {
            const Position next = position + bit;
            if (next <= count_ && tree_[next] <= rank) {
                rank -= tree_[next];
                position = next;
            }
        }
        return position;
    }
};

struct AeonicIndex {
    std::mutex mutex;
    Position count;
    std::vector<std::uint8_t> titans, holders;
    std::vector<Position> preferred_attackers;
    Fenwick titan_pool, holder_pool;

    AeonicIndex(const std::uint8_t *golden, const std::uint8_t *titan_flags,
                const std::uint8_t *holder_flags, Position size, std::int64_t reincarnator)
        : count(size), titans(size), holders(size) {
        preferred_attackers.reserve(size + (reincarnator >= 0 ? 1 : 0));
        if (reincarnator >= 0) preferred_attackers.push_back(static_cast<Position>(reincarnator));
        for (Position i = 0; i < count; ++i) {
            titans[i] = titan_flags[i];
            holders[i] = holder_flags[i];
            if (golden[i]) preferred_attackers.push_back(i);
        }
        titan_pool.build(count, [&](Position i) { return titans[i] != 0; });
        holder_pool.build(count, [&](Position i) { return titans[i] || holders[i]; });
    }

    void validate_query(std::uint32_t pool, std::int64_t excluded) const {
        if (pool > TIEMU_AEONIC_TITANS || excluded < -1
                || (excluded >= 0 && static_cast<std::uint64_t>(excluded) >= count)
                || (pool == TIEMU_AEONIC_ATTACKERS && excluded != -1))
            throw std::invalid_argument("aeonic pool or excluded position is invalid");
    }

    bool contains(std::uint32_t pool, Position position) const noexcept {
        return pool == TIEMU_AEONIC_TITANS ? titans[position] != 0
            : (titans[position] || holders[position]);
    }

    Position size(std::uint32_t pool, std::int64_t excluded) const noexcept {
        if (pool == TIEMU_AEONIC_ATTACKERS) {
            if (!preferred_attackers.empty()) return preferred_attackers.size();
            return titan_pool.total() ? titan_pool.total() : count;
        }
        if (pool == TIEMU_AEONIC_ALL) return count - (excluded >= 0 ? 1 : 0);
        const auto &tree = pool == TIEMU_AEONIC_TITANS ? titan_pool : holder_pool;
        return tree.total() - (excluded >= 0 && contains(pool, static_cast<Position>(excluded)) ? 1 : 0);
    }

    Position at(std::uint32_t pool, std::int64_t excluded, Position rank) const {
        if (rank >= size(pool, excluded))
            throw std::invalid_argument("aeonic candidate rank is out of range");
        if (pool == TIEMU_AEONIC_ATTACKERS) {
            if (!preferred_attackers.empty()) return preferred_attackers[rank];
            return titan_pool.total() ? titan_pool.at(rank) : rank;
        }
        if (pool == TIEMU_AEONIC_ALL)
            return rank + (excluded >= 0 && rank >= static_cast<Position>(excluded) ? 1 : 0);
        const auto &tree = pool == TIEMU_AEONIC_TITANS ? titan_pool : holder_pool;
        // Skip the excluded member's rank, not the excluded population index.
        if (excluded >= 0 && contains(pool, static_cast<Position>(excluded))
                && rank >= tree.before(static_cast<Position>(excluded))) ++rank;
        return tree.at(rank);
    }

    void update(Position position, std::uint8_t titan, std::uint8_t holder) {
        if (position >= count || titan > 1 || holder > 1)
            throw std::invalid_argument("aeonic update position or flags are invalid");
        // No allocation or throwing operation follows validation.
        titan_pool.update(position, titans[position] != 0, titan != 0);
        holder_pool.update(position, titans[position] || holders[position], titan || holder);
        titans[position] = titan;
        holders[position] = holder;
    }
};

AeonicIndex &live(void *handle) {
    if (!handle) throw std::invalid_argument("aeonic index handle is null");
    return *static_cast<AeonicIndex *>(handle);
}

void validate_sources(const std::uint8_t *golden, const std::uint8_t *titans,
                      const std::uint8_t *holders, Position count, std::int64_t reincarnator) {
    if (count > std::numeric_limits<std::uint32_t>::max()
            || count >= static_cast<Position>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(Position))
        throw std::invalid_argument("aeonic population exceeds supported size");
    if ((count && (!golden || !titans || !holders)) || reincarnator < -1
            || (reincarnator >= 0 && static_cast<std::uint64_t>(reincarnator) >= count))
        throw std::invalid_argument("aeonic flag buffers or reincarnator are invalid");
    for (Position i = 0; i < count; ++i)
        if (golden[i] > 1 || titans[i] > 1 || holders[i] > 1)
            throw std::invalid_argument("aeonic flags must be zero or one");
}
} // namespace

extern "C" {
int tiemu_aeonic_create(const uint8_t *golden, const uint8_t *titans, const uint8_t *holders,
        size_t count, int64_t reincarnator, void **out_handle) {
    return tiemu_detail::guarded([&] {
        if (!out_handle || *out_handle)
            throw std::invalid_argument("aeonic create requires an empty handle slot");
        validate_sources(golden, titans, holders, count, reincarnator);
        *out_handle = new AeonicIndex(golden, titans, holders, count, reincarnator);
    });
}

int tiemu_aeonic_pool_size(void *handle, uint32_t pool, int64_t excluded, size_t *out_count) {
    return tiemu_detail::guarded([&] {
        auto &index = live(handle);
        std::lock_guard<std::mutex> lock(index.mutex);
        if (!out_count) throw std::invalid_argument("aeonic size output is null");
        index.validate_query(pool, excluded);
        *out_count = index.size(pool, excluded);
    });
}

int tiemu_aeonic_pool_at(void *handle, uint32_t pool, int64_t excluded,
        size_t rank, size_t *out_position) {
    return tiemu_detail::guarded([&] {
        auto &index = live(handle);
        std::lock_guard<std::mutex> lock(index.mutex);
        if (!out_position) throw std::invalid_argument("aeonic position output is null");
        index.validate_query(pool, excluded);
        const auto position = index.at(pool, excluded, rank);
        *out_position = position;
    });
}

int tiemu_aeonic_update(void *handle, size_t position, uint8_t titan, uint8_t holder) {
    return tiemu_detail::guarded([&] {
        auto &index = live(handle);
        std::lock_guard<std::mutex> lock(index.mutex);
        index.update(position, titan, holder);
    });
}

int tiemu_aeonic_destroy(void **handle) {
    return tiemu_detail::guarded([&] {
        if (!handle) throw std::invalid_argument("aeonic destroy handle slot is null");
        // The caller excludes concurrent operations/destruction with its lock.
        delete static_cast<AeonicIndex *>(*handle);
        *handle = nullptr;
    });
}
} // extern "C"
