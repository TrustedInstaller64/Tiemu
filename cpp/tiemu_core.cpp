// Ordered standalone C++ numerical and state primitives. No interpreter API.
#ifndef TIEMU_CORE_BUILD
#define TIEMU_CORE_BUILD
#endif
#include "tiemu_internal.h"
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
using Size = std::int64_t;
using tiemu_detail::guarded;
static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559,
              "Tiemu core requires IEEE 754 binary64 doubles");

Size signed_size(std::size_t count) {
    if (count > static_cast<std::size_t>(std::numeric_limits<Size>::max()))
        throw std::invalid_argument("array size exceeds the supported integer range");
    return static_cast<Size>(count);
}

std::size_t product(std::size_t first, std::size_t second) {
    if (second && first > std::numeric_limits<std::size_t>::max() / second)
        throw std::invalid_argument("array shape is too large");
    return first * second;
}

template <class T>
void input_buffer(const T *pointer, std::size_t count, const char *message) {
    if (count > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(T))
        throw std::invalid_argument("array extent exceeds addressable memory");
    if (count && !pointer) throw std::invalid_argument(message);
}

template <class T>
void output_buffer(T *pointer, std::size_t capacity, std::size_t required, const char *message) {
    if (capacity < required) throw std::invalid_argument(message);
    input_buffer(pointer, required, message);
}

// Avoid even a zero offset from a null pointer for valid empty input arrays.
template <class T>
T *offset(T *pointer, Size index) noexcept {
    return pointer ? pointer + index : nullptr;
}

void distribution_buffer(const double *distribution, std::size_t count, std::size_t paths) {
    if (!distribution && count == 0) return;
    if (count != paths) throw std::invalid_argument("distribution must match paths");
    input_buffer(distribution, count, "distribution must be a numeric vector");
}

void pair_buffers(const double *first, std::size_t first_count,
                  const double *second, std::size_t second_count,
                  double *out_first, std::size_t out_first_count,
                  double *out_second, std::size_t out_second_count) {
    if (first_count != second_count)
        throw std::invalid_argument("interaction vectors must have the same length");
    input_buffer(first, first_count, "first vector is null");
    input_buffer(second, second_count, "second vector is null");
    output_buffer(out_first, out_first_count, first_count, "first output buffer is too small or null");
    output_buffer(out_second, out_second_count, first_count, "second output buffer is too small or null");
}

template <class T>
T &object(void *handle) {
    if (!handle) throw std::invalid_argument("native handle is null");
    return *static_cast<T *>(handle);
}

void empty_slot(void **slot) {
    if (!slot || *slot) throw std::invalid_argument("output handle slot must be nonnull and empty");
}

template <class T>
void destroy(void **slot) {
    if (!slot) throw std::invalid_argument("handle slot is null");
    delete static_cast<T *>(*slot);
    *slot = nullptr;
}

Size first_argmax(const double *values, Size count) {
    Size best = 0;
    for (Size i = 0; i < count; ++i) {
        if (std::isnan(values[i])) return i;
        if (values[i] > values[best]) best = i;
    }
    return best;
}

void map_path_row(const double *input, const double *weights, Size titans,
                  Size paths, double *result, bool replace_with_ones = false) {
    double total = 0.0;
    for (Size path = 0; path < paths; ++path) {
        double value = 0.0;
        for (Size titan = 0; titan < titans; ++titan)
            value += (replace_with_ones ? 1.0 : input[titan]) * weights[titan * paths + path];
        value = value < 0.0 ? 0.0 : value;
        result[path] = value;
        total += value;
    }
    if (total > 0.0)
        for (Size path = 0; path < paths; ++path) result[path] /= total;
}

void score_path_row(const double *values, Size count, double multiplier,
                    const double *penalties, bool is_boss, double *result) {
    double act_sq = 0.0, stab_sq = 0.0, total = 0.0;
    for (Size i = 0; i < count; ++i) {
        (i < 6 ? act_sq : stab_sq) += values[i] * values[i];
        total += values[i];
    }
    const Size dominant = first_argmax(values, count);
    const double activity = std::sqrt(act_sq) * 10.0;
    const double stability = std::sqrt(stab_sq) * 10.0;
    const double purity = total == 0.0 ? 0.0 : values[dominant] / total;
    const double saturation = penalties ? 1.0 / (1.0 + 2.0 * penalties[dominant]) : 1.0;
    double score = (activity + stability) * (1.0 + purity) * saturation * multiplier;
    if (is_boss) score *= 5.0;
    result[0] = activity;
    result[1] = stability;
    result[2] = purity;
    result[3] = score;
}

class BatchEvaluator {
    struct Job {
        const double *values;
        const double *matrix;
        const double *multipliers;
        const double *distribution;
        const std::int64_t *bosses;
        double *paths;
        double *stats;
        Size rows, titans, path_count;
    };
    const Size workers_, min_rows_;
    std::mutex request_mutex_, pool_mutex_;
    std::condition_variable ready_, completed_;
    std::vector<std::thread> threads_;
    Job job_{};
    std::uint64_t generation_ = 0;
    Size remaining_ = 0;
    bool stopping_ = false, closed_ = false;

    static void evaluate(const Job &job, Size begin, Size end) noexcept {
        for (Size row = begin; row < end; ++row) {
            const double *values = offset(job.values, row * job.titans);
            bool replace_with_ones = false;
            for (Size titan = 0; titan < job.titans; ++titan)
                if (!std::isfinite(values[titan])) { replace_with_ones = true; break; }
            double *paths = job.paths + row * job.path_count;
            map_path_row(values, job.matrix, job.titans, job.path_count, paths, replace_with_ones);
            score_path_row(paths, job.path_count, job.multipliers[row], job.distribution,
                           job.bosses[row] != 0, job.stats + row * 4);
        }
    }

    void evaluate_partition(const Job &job, Size index) const noexcept {
        const Size quotient = job.rows / workers_, remainder = job.rows % workers_;
        const Size begin = quotient * index + std::min(index, remainder);
        const Size end = begin + quotient + (index < remainder ? 1 : 0);
        evaluate(job, begin, end);
    }

    void worker(Size index) {
        std::uint64_t observed_generation = 0;
        std::unique_lock<std::mutex> lock(pool_mutex_);
        for (;;) {
            ready_.wait(lock, [&] { return stopping_ || generation_ != observed_generation; });
            if (stopping_) return;
            observed_generation = generation_;
            const Job job = job_;
            lock.unlock();
            evaluate_partition(job, index);
            lock.lock();
            if (--remaining_ == 0) completed_.notify_one();
        }
    }

    void stop_threads() noexcept {
        {
            std::lock_guard<std::mutex> lock(pool_mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        for (auto &thread : threads_)
            if (thread.joinable()) thread.join();
        threads_.clear();
    }

    void start_threads() {
        if (!threads_.empty() || workers_ == 1) return;
        try {
            threads_.reserve(static_cast<std::size_t>(workers_ - 1));
            for (Size i = 1; i < workers_; ++i) threads_.emplace_back([this, i] { worker(i); });
        } catch (...) {
            stop_threads();
            stopping_ = false;
            throw;
        }
    }

public:
    BatchEvaluator(Size workers, Size min_rows) : workers_(workers), min_rows_(min_rows) {
        if (workers < 1) throw std::invalid_argument("workers must be positive");
        if (min_rows < 1) throw std::invalid_argument("min_rows must be positive");
        if (static_cast<std::uint64_t>(workers) > std::numeric_limits<std::size_t>::max())
            throw std::invalid_argument("workers exceeds the addressable range");
    }
    BatchEvaluator(const BatchEvaluator &) = delete;
    BatchEvaluator &operator=(const BatchEvaluator &) = delete;
    ~BatchEvaluator() { close(); }

    void close() {
        std::lock_guard<std::mutex> request(request_mutex_);
        if (closed_) return;
        closed_ = true;
        stop_threads();
    }

    void concepts(const double *values, const double *matrix, const double *multipliers,
                  const double *distribution, const std::int64_t *bosses,
                  double *paths, double *stats, Size rows, Size titans, Size path_count) {
        const Job job{values, matrix, multipliers, distribution, bosses,
                      paths, stats, rows, titans, path_count};
        std::lock_guard<std::mutex> request(request_mutex_);
        if (closed_) throw std::runtime_error("batch evaluator is closed");
        if (workers_ == 1 || rows < min_rows_) {
            evaluate(job, 0, rows);
        } else {
            start_threads();
            {
                std::lock_guard<std::mutex> lock(pool_mutex_);
                job_ = job;
                remaining_ = static_cast<Size>(threads_.size());
                ++generation_;
            }
            ready_.notify_all();
            evaluate_partition(job, 0);
            std::unique_lock<std::mutex> lock(pool_mutex_);
            completed_.wait(lock, [&] { return remaining_ == 0; });
        }
    }
};

class PathHistogram {
    std::vector<std::int64_t> indices_, counts_;
    std::mutex mutex_;
public:
    PathHistogram(const std::int64_t *indices, std::size_t count, Size paths) {
        if (paths <= 0) throw std::invalid_argument("number of paths must be positive");
        if (static_cast<std::uint64_t>(paths)
            > static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(Size))
            throw std::invalid_argument("number of paths exceeds addressable memory");
        signed_size(count);
        counts_.assign(static_cast<std::size_t>(paths), 0);
        if (count) indices_.assign(indices, indices + count);
        for (auto path : indices_) {
            if (path < 0 || path >= paths) throw std::out_of_range("path index out of range");
            ++counts_[path];
        }
    }

    void update(Size index, Size path) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index < 0 || static_cast<std::size_t>(index) >= indices_.size())
            throw std::out_of_range("entity index out of range");
        if (path < 0 || static_cast<std::size_t>(path) >= counts_.size())
            throw std::out_of_range("path index out of range");
        --counts_[indices_[index]];
        ++counts_[path];
        indices_[index] = path;
    }

    void distribution(double *output, std::size_t capacity) {
        std::lock_guard<std::mutex> lock(mutex_);
        output_buffer(output, capacity, counts_.size(), "distribution output is too small or null");
        for (std::size_t path = 0; path < counts_.size(); ++path)
            output[path] = indices_.empty() ? 1.0 / counts_.size()
                : static_cast<double>(counts_[path]) / indices_.size();
    }
};

class OpponentIndex {
    double bin_size_;
    Size alive_count_;
    std::vector<bool> alive_;
    std::vector<Size> fenwick_, bin_positions_;
    std::vector<std::int64_t> initial_bins_;
    std::map<std::int64_t, std::vector<Size>> bins_;
    std::mutex mutex_;

    std::int64_t bin(double score) const {
        if (!std::isfinite(score)) throw std::invalid_argument("scores must be finite");
        double remainder = std::fmod(score, bin_size_);
        double quotient = (score - remainder) / bin_size_;
        if (remainder != 0.0 && remainder < 0.0) quotient -= 1.0;
        double floored = std::floor(quotient);
        if (quotient - floored > 0.5) floored += 1.0;
        if (!std::isfinite(floored)
            || floored <= static_cast<double>(std::numeric_limits<std::int64_t>::min())
            || floored >= static_cast<double>(std::numeric_limits<std::int64_t>::max()))
            throw std::invalid_argument("score bin is outside the supported integer range");
        return static_cast<std::int64_t>(floored);
    }

    void check_entity(Size entity) const {
        if (entity < 0 || static_cast<std::size_t>(entity) >= alive_.size())
            throw std::out_of_range("entity index out of range");
    }

    Size alive_before(Size index) const {
        Size count = 0;
        for (Size i = index; i > 0; i -= i & -i) count += fenwick_[i];
        return count;
    }

    Size nearby_count(Size entity, std::int64_t current_bin) const {
        Size count = 0;
        for (std::int64_t b = current_bin - 1; b <= current_bin + 1; ++b) {
            auto found = bins_.find(b);
            if (found != bins_.end())
                count += static_cast<Size>(found->second.size()) - (initial_bins_[entity] == b ? 1 : 0);
        }
        return count;
    }

    Size entity_at_unlocked(Size rank) const {
        if (rank < 0 || rank >= alive_count_) throw std::out_of_range("alive rank out of range");
        Size index = 0, bit = 1;
        while (bit <= static_cast<Size>(alive_.size()) / 2) bit <<= 1;
        for (; bit > 0; bit >>= 1) {
            const Size next = index + bit;
            if (next < static_cast<Size>(fenwick_.size()) && fenwick_[next] <= rank) {
                index = next;
                rank -= fenwick_[next];
            }
        }
        return index;
    }

public:
    OpponentIndex(const double *scores, std::size_t count, double bin_size)
        : bin_size_(bin_size), alive_count_(signed_size(count)) {
        if (!std::isfinite(bin_size) || bin_size <= 0.0)
            throw std::invalid_argument("bin size must be finite and positive");
        if (count >= static_cast<std::size_t>(std::numeric_limits<Size>::max()))
            throw std::invalid_argument("too many entities");
        alive_.assign(count, true);
        fenwick_.resize(count + 1, 0);
        initial_bins_.reserve(count);
        bin_positions_.reserve(count);
        for (Size i = 0; i < alive_count_; ++i) {
            const auto key = bin(scores[i]);
            initial_bins_.push_back(key);
            auto &bucket = bins_[key];
            bin_positions_.push_back(static_cast<Size>(bucket.size()));
            bucket.push_back(i);
            fenwick_[i + 1] = (i + 1) & -(i + 1);
        }
    }

    Size alive_count() {
        std::lock_guard<std::mutex> lock(mutex_);
        return alive_count_;
    }

    Size entity_at(Size rank) {
        std::lock_guard<std::mutex> lock(mutex_);
        return entity_at_unlocked(rank);
    }

    Size opponent_count(Size entity, double current_score) {
        std::lock_guard<std::mutex> lock(mutex_);
        check_entity(entity);
        const Size nearby = nearby_count(entity, bin(current_score));
        return nearby ? nearby : alive_count_ - (alive_[entity] ? 1 : 0);
    }

    Size opponent_at(Size entity, double current_score, Size rank) {
        std::lock_guard<std::mutex> lock(mutex_);
        check_entity(entity);
        const auto current_bin = bin(current_score);
        const Size nearby = nearby_count(entity, current_bin);
        const Size count = nearby ? nearby : alive_count_ - (alive_[entity] ? 1 : 0);
        if (rank < 0 || rank >= count) throw std::out_of_range("opponent rank out of range");
        if (!nearby) {
            if (alive_[entity] && rank >= alive_before(entity)) ++rank;
            return entity_at_unlocked(rank);
        }
        for (std::int64_t b = current_bin - 1; b <= current_bin + 1; ++b) {
            auto found = bins_.find(b);
            if (found == bins_.end()) continue;
            const bool contains_self = initial_bins_[entity] == b;
            const Size size = static_cast<Size>(found->second.size()) - (contains_self ? 1 : 0);
            if (rank < size) {
                if (contains_self && rank >= bin_positions_[entity]) ++rank;
                return found->second[rank];
            }
            rank -= size;
        }
        throw std::logic_error("opponent index invariant failed");
    }

    void remove(Size entity) {
        std::lock_guard<std::mutex> lock(mutex_);
        check_entity(entity);
        if (!alive_[entity]) return;
        alive_[entity] = false;
        --alive_count_;
        const Size limit = static_cast<Size>(fenwick_.size());
        for (Size i = entity + 1; i < limit;) {
            --fenwick_[i];
            const Size step = i & -i;
            if (step >= limit - i) break;
            i += step;
        }
        // Retain original buckets, including dead candidates, exactly as before.
    }
};

void mapping_buffers(const double *values, std::size_t values_count,
                     std::size_t rows, std::size_t titans,
                     const double *matrix, std::size_t matrix_count, std::size_t paths,
                     double *output, std::size_t output_count) {
    if (paths == 0 || values_count != product(rows, titans) || matrix_count != product(titans, paths))
        throw std::invalid_argument("mapping matrix shape must match affinities and contain paths");
    input_buffer(values, values_count, "values buffer is null");
    input_buffer(matrix, matrix_count, "mapping matrix is null");
    output_buffer(output, output_count, product(rows, paths), "path output is too small or null");
    signed_size(rows);
    signed_size(titans);
    signed_size(paths);
}
} // namespace

extern "C" {

uint32_t tiemu_abi_version(void) { return TIEMU_CORE_ABI_VERSION; }
const char *tiemu_last_error(void) { return tiemu_detail::last_error(); }

int tiemu_path_affinities(const double *values, size_t values_count, size_t rows, size_t titans,
                          const double *matrix, size_t matrix_count, size_t paths,
                          double *output, size_t output_count) {
    return guarded([&] {
        mapping_buffers(values, values_count, rows, titans, matrix, matrix_count, paths, output, output_count);
        const Size row_count = signed_size(rows), titan_count = signed_size(titans), path_count = signed_size(paths);
        for (Size row = 0; row < row_count; ++row)
            map_path_row(offset(values, row * titan_count), matrix, titan_count, path_count,
                         output + row * path_count);
    });
}

int tiemu_concepts(const double *paths, size_t path_count, double multiplier,
                   const double *distribution, size_t distribution_count, int is_boss,
                   double *output, size_t output_count) {
    return guarded([&] {
        if (path_count == 0) throw std::invalid_argument("paths must contain at least one path");
        input_buffer(paths, path_count, "paths buffer is null");
        distribution_buffer(distribution, distribution_count, path_count);
        output_buffer(output, output_count, 4, "concept output is too small or null");
        score_path_row(paths, signed_size(path_count), multiplier, distribution, is_boss != 0, output);
    });
}

int tiemu_purify(const double *values, size_t count, double factor, double *output, size_t output_count) {
    return guarded([&] {
        input_buffer(values, count, "values buffer is null");
        output_buffer(output, output_count, count, "purify output is too small or null");
        const Size n = signed_size(count);
        for (Size i = 0; i < n; ++i) output[i] = values[i];
        if ((factor > 0.0 || std::isnan(factor)) && n >= 2) {
            const Size dominant = first_argmax(values, n);
            const double reduction = values[dominant] * factor;
            for (Size i = 0; i < n; ++i) {
                const double value = i == dominant ? values[i] : values[i] - reduction;
                output[i] = value < 0.0 ? 0.0 : value;
            }
        }
    });
}

int tiemu_normalize(const double *values, size_t count, double max_norm,
                    double *output, size_t output_count) {
    return guarded([&] {
        input_buffer(values, count, "values buffer is null");
        output_buffer(output, output_count, count, "normalize output is too small or null");
        const Size n = signed_size(count);
        double squared = 0.0;
        for (Size i = 0; i < n; ++i) squared += values[i] * values[i];
        const double norm = std::sqrt(squared);
        const bool rescale = !std::isfinite(norm) || norm == 0.0 || norm > max_norm;
        for (Size i = 0; i < n; ++i)
            output[i] = rescale ? (values[i] / (norm + 1e-9)) * max_norm : values[i];
    });
}

int tiemu_normalize_pipeline(const double *values, size_t count, double factor,
                             const double *tide, size_t tide_count, double max_norm,
                             double *output, size_t output_count) {
    return guarded([&] {
        if (count != tide_count) throw std::invalid_argument("tide must match values");
        input_buffer(values, count, "values buffer is null");
        input_buffer(tide, tide_count, "tide buffer is null");
        output_buffer(output, output_count, count, "pipeline output is too small or null");
        const Size n = signed_size(count);
        for (Size i = 0; i < n; ++i) output[i] = values[i];
        if (!(factor <= 0.0) && n >= 2) {
            const Size dominant = first_argmax(values, n);
            const double reduction = values[dominant] * factor;
            for (Size i = 0; i < n; ++i) {
                const double value = i == dominant ? values[i] : values[i] - reduction;
                output[i] = value < 0.0 ? 0.0 : value;
            }
        }
        for (Size i = 0; i < n; ++i) {
            const double value = output[i] + tide[i];
            output[i] = value < 0.0 ? 0.0 : value;
        }
        double squared = 0.0;
        for (Size i = 0; i < n; ++i) squared += output[i] * output[i];
        const double norm = std::sqrt(squared);
        if (!std::isfinite(norm) || norm == 0.0 || norm > max_norm)
            for (Size i = 0; i < n; ++i) output[i] = (output[i] / (norm + 1e-9)) * max_norm;
    });
}

int tiemu_synergy(const double *first, size_t first_count, const double *second, size_t second_count,
                  double *out_first, size_t out_first_count, double *out_second, size_t out_second_count) {
    return guarded([&] {
        pair_buffers(first, first_count, second, second_count, out_first, out_first_count, out_second, out_second_count);
        const Size n = signed_size(first_count);
        for (Size i = 0; i < n; ++i) {
            out_first[i] = first[i] + second[i] * 0.02;
            out_second[i] = second[i] + out_first[i] * 0.02;
        }
    });
}

int tiemu_clash(const double *winner, size_t winner_count, const double *loser, size_t loser_count,
                double modifier, double *out_winner, size_t out_winner_count,
                double *out_loser, size_t out_loser_count) {
    return guarded([&] {
        pair_buffers(winner, winner_count, loser, loser_count, out_winner, out_winner_count, out_loser, out_loser_count);
        const Size n = signed_size(winner_count);
        for (Size i = 0; i < n; ++i) {
            const double influence = (loser[i] * 0.1) * modifier;
            out_winner[i] = winner[i] + influence;
            out_loser[i] = loser[i] - influence * 0.5;
        }
    });
}

int tiemu_mentorship(const double *stronger, size_t stronger_count, const double *weaker, size_t weaker_count,
                     double score_ratio, double *out_stronger, size_t out_stronger_count,
                     double *out_weaker, size_t out_weaker_count) {
    return guarded([&] {
        pair_buffers(stronger, stronger_count, weaker, weaker_count, out_stronger, out_stronger_count,
                     out_weaker, out_weaker_count);
        const Size n = signed_size(stronger_count);
        const double strength = 0.05 * std::tanh(score_ratio - 1.0);
        for (Size i = 0; i < n; ++i) {
            out_weaker[i] = weaker[i] + (stronger[i] - weaker[i]) * strength;
            out_stronger[i] = stronger[i] + (out_weaker[i] - stronger[i]) * 0.005;
        }
    });
}

int tiemu_repulsion(const double *values, size_t count, const double *noise, size_t noise_count,
                    double *output, size_t output_count) {
    return guarded([&] {
        if (count != noise_count) throw std::invalid_argument("interaction vectors must have the same length");
        if (count == 0) throw std::invalid_argument("repulsion requires a nonempty vector");
        input_buffer(values, count, "values buffer is null");
        input_buffer(noise, noise_count, "noise buffer is null");
        output_buffer(output, output_count, count, "repulsion output is too small or null");
        const Size n = signed_size(count), dominant = first_argmax(values, n);
        const double damage = values[dominant] * 0.05;
        for (Size i = 0; i < n; ++i) {
            const double damaged = i == dominant ? values[i] - damage : values[i];
            output[i] = damaged + (noise[i] * damage) * 0.1;
        }
    });
}

int tiemu_vote_totals(const int64_t *dominant, size_t dominant_count,
                      const double *weights, size_t weights_count,
                      const double *logits, size_t logits_count, size_t paths,
                      double *output, size_t output_count) {
    return guarded([&] {
        if (dominant_count != weights_count)
            throw std::invalid_argument("indices and weights must have the same length");
        if (paths == 0 || logits_count != product(paths, paths))
            throw std::invalid_argument("vote logits must be a nonempty square matrix");
        input_buffer(dominant, dominant_count, "dominant indices are null");
        input_buffer(weights, weights_count, "weighted scores are null");
        input_buffer(logits, logits_count, "vote logits are null");
        output_buffer(output, output_count, paths, "vote output is too small or null");
        const Size n = signed_size(dominant_count), path_count = signed_size(paths);
        for (Size i = 0; i < n; ++i)
            if (dominant[i] < 0 || dominant[i] >= path_count)
                throw std::out_of_range("dominant path index out of range");
        std::fill_n(output, paths, 0.0);
        std::vector<double> proposals(logits_count);
        for (Size index = 0; index < path_count; ++index) {
            const double *row = logits + index * path_count;
            double *proposal = proposals.data() + index * path_count;
            const double maximum = row[first_argmax(row, path_count)];
            double sum = 0.0;
            for (Size path = 0; path < path_count; ++path) {
                proposal[path] = std::exp(row[path] - maximum);
                sum += proposal[path];
            }
            for (Size path = 0; path < path_count; ++path) proposal[path] /= sum;
        }
        for (Size i = 0; i < n; ++i) {
            const double *proposal = proposals.data() + dominant[i] * path_count;
            for (Size path = 0; path < path_count; ++path) output[path] += proposal[path] * weights[i];
        }
    });
}

int tiemu_inorganic_step(const double *activity, size_t activity_count,
                         const double *stability, size_t stability_count,
                         const double *noise, size_t noise_count, int64_t grid,
                         double *out_activity, size_t out_activity_count,
                         double *out_stability, size_t out_stability_count) {
    return guarded([&] {
        if (grid <= 0 || grid > std::numeric_limits<Size>::max() / grid)
            throw std::invalid_argument("grid size must be positive and bounded");
        const Size cells = grid * grid;
        if (static_cast<std::uint64_t>(cells) > std::numeric_limits<std::size_t>::max())
            throw std::invalid_argument("grid size exceeds addressable memory");
        const std::size_t cell_count = static_cast<std::size_t>(cells);
        if (activity_count != cell_count || stability_count != cell_count
            || noise_count != product(cell_count, 2))
            throw std::invalid_argument("activity/stability must match grid squared and noise must have shape (cells, 2)");
        input_buffer(activity, activity_count, "activity is null");
        input_buffer(stability, stability_count, "stability is null");
        input_buffer(noise, noise_count, "noise is null");
        output_buffer(out_activity, out_activity_count, cell_count, "activity output is too small or null");
        output_buffer(out_stability, out_stability_count, cell_count, "stability output is too small or null");
        std::copy_n(activity, cell_count, out_activity);
        std::copy_n(stability, cell_count, out_stability);
        for (Size i = 0; i < grid; ++i) {
            for (Size j = 0; j < grid; ++j) {
                const Size cell = i * grid + j;
                double sum_a = 0.0, sum_s = 0.0;
                Size count = 0;
                for (int di = -1; di <= 1; ++di) {
                    for (int dj = -1; dj <= 1; ++dj) {
                        if (di == 0 && dj == 0) continue;
                        const Size ni = i + di, nj = j + dj;
                        if (ni >= 0 && ni < grid && nj >= 0 && nj < grid) {
                            const Size neighbor = ni * grid + nj;
                            sum_a += out_activity[neighbor];
                            sum_s += out_stability[neighbor];
                            ++count;
                        }
                    }
                }
                const double avg_a = count ? sum_a / count : 0.0;
                const double avg_s = count ? sum_s / count : 0.0;
                const double change_a = (avg_a - out_activity[cell]) * 0.1 + noise[cell * 2];
                const double change_s = (avg_s - out_stability[cell]) * 0.1 + noise[cell * 2 + 1];
                out_activity[cell] = std::max(0.0, out_activity[cell] + change_a);
                out_stability[cell] = std::max(0.0, out_stability[cell] + change_s);
            }
        }
    });
}

int tiemu_batch_create(int64_t workers, int64_t min_rows, void **out_handle) {
    return guarded([&] { empty_slot(out_handle); *out_handle = new BatchEvaluator(workers, min_rows); });
}
int tiemu_batch_close(void *handle) {
    return guarded([&] { object<BatchEvaluator>(handle).close(); });
}
int tiemu_batch_destroy(void **handle) {
    return guarded([&] { destroy<BatchEvaluator>(handle); });
}

int tiemu_batch_concepts(void *handle, const double *values, size_t values_count, size_t rows, size_t titans,
                         const double *matrix, size_t matrix_count, size_t paths,
                         const double *multipliers, size_t multiplier_count,
                         const double *distribution, size_t distribution_count,
                         const int64_t *bosses, size_t boss_count,
                         double *out_paths, size_t out_paths_count, double *out_stats, size_t out_stats_count) {
    return guarded([&] {
        auto &evaluator = object<BatchEvaluator>(handle);
        mapping_buffers(values, values_count, rows, titans, matrix, matrix_count, paths, out_paths, out_paths_count);
        if (multiplier_count != rows || boss_count != rows)
            throw std::invalid_argument("multipliers and bosses must match the row count");
        input_buffer(multipliers, multiplier_count, "multipliers are null");
        input_buffer(bosses, boss_count, "bosses are null");
        distribution_buffer(distribution, distribution_count, paths);
        output_buffer(out_stats, out_stats_count, product(rows, 4), "stats output is too small or null");
        evaluator.concepts(values, matrix, multipliers, distribution, bosses,
                           out_paths, out_stats, signed_size(rows), signed_size(titans), signed_size(paths));
    });
}

int tiemu_histogram_create(const int64_t *indices, size_t count, int64_t paths, void **out_handle) {
    return guarded([&] {
        empty_slot(out_handle);
        input_buffer(indices, count, "indices are null");
        *out_handle = new PathHistogram(indices, count, paths);
    });
}
int tiemu_histogram_update(void *handle, int64_t index, int64_t path) {
    return guarded([&] { object<PathHistogram>(handle).update(index, path); });
}
int tiemu_histogram_distribution(void *handle, double *output, size_t output_count) {
    return guarded([&] { object<PathHistogram>(handle).distribution(output, output_count); });
}
int tiemu_histogram_destroy(void **handle) {
    return guarded([&] { destroy<PathHistogram>(handle); });
}

int tiemu_opponent_create(const double *scores, size_t count, double bin_size, void **out_handle) {
    return guarded([&] {
        empty_slot(out_handle);
        input_buffer(scores, count, "scores are null");
        *out_handle = new OpponentIndex(scores, count, bin_size);
    });
}
int tiemu_opponent_alive_count(void *handle, int64_t *output) {
    return guarded([&] {
        output_buffer(output, 1, 1, "alive count output is null");
        *output = object<OpponentIndex>(handle).alive_count();
    });
}
int tiemu_opponent_entity_at(void *handle, int64_t rank, int64_t *output) {
    return guarded([&] {
        output_buffer(output, 1, 1, "entity output is null");
        *output = object<OpponentIndex>(handle).entity_at(rank);
    });
}
int tiemu_opponent_count(void *handle, int64_t entity, double current_score, int64_t *output) {
    return guarded([&] {
        output_buffer(output, 1, 1, "opponent count output is null");
        *output = object<OpponentIndex>(handle).opponent_count(entity, current_score);
    });
}
int tiemu_opponent_at(void *handle, int64_t entity, double current_score, int64_t rank, int64_t *output) {
    return guarded([&] {
        output_buffer(output, 1, 1, "opponent output is null");
        *output = object<OpponentIndex>(handle).opponent_at(entity, current_score, rank);
    });
}
int tiemu_opponent_remove(void *handle, int64_t entity) {
    return guarded([&] { object<OpponentIndex>(handle).remove(entity); });
}
int tiemu_opponent_destroy(void **handle) {
    return guarded([&] { destroy<OpponentIndex>(handle); });
}

} // extern "C"
