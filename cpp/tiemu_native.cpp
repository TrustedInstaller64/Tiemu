// The Python caller owns RNG, object lifetimes, and the ordering of world events.
// No fast-math or parallel reductions: preserve the original transition rules.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace py = pybind11;
using Doubles = py::array_t<double, py::array::c_style | py::array::forcecast>;
// Safe integer conversions only: a floating path id must not silently truncate.
using Indices = py::array_t<std::int64_t, py::array::c_style>;
using Size = py::ssize_t;

template <typename Array>
void require_vector(const Array &array, const char *name) {
    if (array.ndim() != 1) throw py::value_error(std::string(name) + " must be one-dimensional");
}

Size first_argmax(const double *values, Size size) {
    Size best = 0;
    for (Size i = 0; i < size; ++i) {
        // NumPy argmax returns the first NaN if one is present.
        if (std::isnan(values[i])) return i;
        if (values[i] > values[best]) best = i;
    }
    return best;
}

py::array_t<double> path_affinities(const Doubles &values, const Doubles &matrix) {
    if ((values.ndim() != 1 && values.ndim() != 2) || matrix.ndim() != 2)
        throw py::value_error("values must be a vector or matrix and matrix must be two-dimensional");
    const Size titans = values.shape(values.ndim() - 1);
    if (matrix.shape(0) != titans || matrix.shape(1) == 0)
        throw py::value_error("mapping matrix shape must match affinities and contain paths");
    const Size rows = values.ndim() == 1 ? 1 : values.shape(0);
    const Size paths = matrix.shape(1);
    std::vector<Size> shape = values.ndim() == 1 ? std::vector<Size>{paths} : std::vector<Size>{rows, paths};
    py::array_t<double> output(shape);
    const double *input = values.data(), *weights = matrix.data();
    double *result = output.mutable_data();
    {
        py::gil_scoped_release release;
        for (Size row = 0; row < rows; ++row) {
            double total = 0.0;
            for (Size path = 0; path < paths; ++path) {
                double value = 0.0;
                for (Size titan = 0; titan < titans; ++titan)
                    value += input[row * titans + titan] * weights[titan * paths + path];
                value = value < 0.0 ? 0.0 : value; // preserve NaN, like np.maximum
                result[row * paths + path] = value;
                total += value;
            }
            if (total > 0.0)
                for (Size path = 0; path < paths; ++path) result[row * paths + path] /= total;
        }
    }
    return output;
}

py::tuple concepts(const Doubles &paths, double multiplier, py::object distribution, bool is_boss) {
    require_vector(paths, "paths");
    const Size count = paths.size();
    if (count == 0) throw py::value_error("paths must contain at least one path");
    Doubles dist;
    const double *penalties = nullptr;
    if (!distribution.is_none()) {
        dist = Doubles::ensure(distribution);
        if (!dist) throw py::value_error("distribution must be a numeric vector");
        require_vector(dist, "distribution");
        if (dist.size() != count) throw py::value_error("distribution must match paths");
        penalties = dist.data();
    }
    const double *values = paths.data();
    double activity, stability, purity, score;
    {
        py::gil_scoped_release release;
        double act_sq = 0.0, stab_sq = 0.0, total = 0.0;
        for (Size i = 0; i < count; ++i) {
            (i < 6 ? act_sq : stab_sq) += values[i] * values[i];
            total += values[i];
        }
        const Size dominant = first_argmax(values, count);
        activity = std::sqrt(act_sq) * 10.0;
        stability = std::sqrt(stab_sq) * 10.0;
        purity = total == 0.0 ? 0.0 : values[dominant] / total;
        const double saturation = penalties ? 1.0 / (1.0 + 2.0 * penalties[dominant]) : 1.0;
        score = (activity + stability) * (1.0 + purity) * saturation * multiplier;
        if (is_boss) score *= 5.0;
    }
    return py::make_tuple(activity, stability, purity, score);
}

py::array_t<double> purify(const Doubles &values, double factor) {
    require_vector(values, "values");
    const Size count = values.size();
    py::array_t<double> output(count);
    const double *input = values.data();
    double *result = output.mutable_data();
    {
        py::gil_scoped_release release;
        std::copy_n(input, count, result);
        if (factor > 0.0 || std::isnan(factor)) {
            if (count >= 2) {
                const Size dominant = first_argmax(input, count);
                const double reduction = input[dominant] * factor;
                for (Size i = 0; i < count; ++i) {
                    const double value = i == dominant ? input[i] : input[i] - reduction;
                    result[i] = value < 0.0 ? 0.0 : value;
                }
            }
        }
    }
    return output;
}

py::array_t<double> normalize(const Doubles &values, double max_norm) {
    require_vector(values, "values");
    const Size count = values.size();
    py::array_t<double> output(count);
    const double *input = values.data();
    double *result = output.mutable_data();
    {
        py::gil_scoped_release release;
        double squared = 0.0;
        for (Size i = 0; i < count; ++i) squared += input[i] * input[i];
        const double norm = std::sqrt(squared);
        const bool rescale = !std::isfinite(norm) || norm == 0.0 || norm > max_norm;
        for (Size i = 0; i < count; ++i)
            result[i] = rescale ? (input[i] / (norm + 1e-9)) * max_norm : input[i];
    }
    return output;
}

Size require_pair(const Doubles &first, const Doubles &second) {
    require_vector(first, "first vector");
    require_vector(second, "second vector");
    if (first.size() != second.size()) throw py::value_error("interaction vectors must have the same length");
    return first.size();
}

py::tuple synergy(const Doubles &first, const Doubles &second) {
    const Size count = require_pair(first, second);
    py::array_t<double> out_first(count), out_second(count);
    const double *a = first.data(), *b = second.data();
    double *new_a = out_first.mutable_data(), *new_b = out_second.mutable_data();
    {
        py::gil_scoped_release release;
        for (Size i = 0; i < count; ++i) {
            new_a[i] = a[i] + b[i] * 0.02;
            new_b[i] = b[i] + new_a[i] * 0.02;
        }
    }
    return py::make_tuple(out_first, out_second);
}

py::tuple clash(const Doubles &winner, const Doubles &loser, double modifier) {
    const Size count = require_pair(winner, loser);
    py::array_t<double> out_winner(count), out_loser(count);
    const double *w = winner.data(), *l = loser.data();
    double *new_w = out_winner.mutable_data(), *new_l = out_loser.mutable_data();
    {
        py::gil_scoped_release release;
        for (Size i = 0; i < count; ++i) {
            const double influence = (l[i] * 0.1) * modifier;
            new_w[i] = w[i] + influence;
            new_l[i] = l[i] - influence * 0.5;
        }
    }
    return py::make_tuple(out_winner, out_loser);
}

py::tuple mentorship(const Doubles &stronger, const Doubles &weaker, double score_ratio) {
    const Size count = require_pair(stronger, weaker);
    py::array_t<double> out_stronger(count), out_weaker(count);
    const double *s = stronger.data(), *w = weaker.data();
    double *new_s = out_stronger.mutable_data(), *new_w = out_weaker.mutable_data();
    {
        py::gil_scoped_release release;
        const double strength = 0.05 * std::tanh(score_ratio - 1.0);
        for (Size i = 0; i < count; ++i) {
            new_w[i] = w[i] + (s[i] - w[i]) * strength;
            new_s[i] = s[i] + (new_w[i] - s[i]) * 0.005;
        }
    }
    return py::make_tuple(out_stronger, out_weaker);
}

py::array_t<double> repulsion(const Doubles &values, const Doubles &noise) {
    const Size count = require_pair(values, noise);
    if (count == 0) throw py::value_error("repulsion requires a nonempty vector");
    py::array_t<double> output(count);
    const double *input = values.data(), *random_values = noise.data();
    double *result = output.mutable_data();
    {
        py::gil_scoped_release release;
        const Size dominant = first_argmax(input, count);
        const double damage = input[dominant] * 0.05;
        for (Size i = 0; i < count; ++i) {
            const double damaged = i == dominant ? input[i] - damage : input[i];
            result[i] = damaged + (random_values[i] * damage) * 0.1;
        }
    }
    return output;
}

py::array_t<double> vote_totals(const Indices &dominant, const Doubles &weights, const Doubles &logits) {
    require_vector(dominant, "dominant indices");
    require_vector(weights, "weighted scores");
    if (dominant.size() != weights.size()) throw py::value_error("indices and weights must have the same length");
    if (logits.ndim() != 2 || logits.shape(0) != logits.shape(1) || logits.shape(0) == 0)
        throw py::value_error("vote logits must be a nonempty square matrix");
    const Size count = dominant.size(), paths = logits.shape(0);
    const auto *indices = dominant.data();
    for (Size i = 0; i < count; ++i)
        if (indices[i] < 0 || indices[i] >= paths) throw py::index_error("dominant path index out of range");
    py::array_t<double> output(paths);
    double *result = output.mutable_data();
    const double *scores = weights.data(), *matrix = logits.data();
    {
        py::gil_scoped_release release;
        std::fill_n(result, paths, 0.0);
        std::vector<double> proposal(static_cast<std::size_t>(paths));
        for (Size i = 0; i < count; ++i) {
            const double *row = matrix + indices[i] * paths;
            const double maximum = row[first_argmax(row, paths)];
            double sum = 0.0;
            for (Size path = 0; path < paths; ++path) {
                proposal[path] = std::exp(row[path] - maximum);
                sum += proposal[path];
            }
            // Preserve even signed/nonfinite weights and accumulation order.
            for (Size path = 0; path < paths; ++path) result[path] += (proposal[path] / sum) * scores[i];
        }
    }
    return output;
}

py::tuple inorganic_step(const Doubles &activity, const Doubles &stability, const Doubles &noise, Size grid) {
    require_vector(activity, "activity");
    require_vector(stability, "stability");
    if (grid <= 0 || grid > std::numeric_limits<Size>::max() / grid)
        throw py::value_error("grid size must be positive and bounded");
    const Size cells = grid * grid;
    if (activity.size() != cells || stability.size() != cells || noise.ndim() != 2 || noise.shape(0) != cells || noise.shape(1) != 2)
        throw py::value_error("activity/stability must match grid squared and noise must have shape (cells, 2)");
    py::array_t<double> out_activity(cells), out_stability(cells);
    double *a = out_activity.mutable_data(), *s = out_stability.mutable_data();
    const double *input_a = activity.data(), *input_s = stability.data(), *n = noise.data();
    {
        py::gil_scoped_release release;
        std::copy_n(input_a, cells, a);
        std::copy_n(input_s, cells, s);
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
                            sum_a += a[neighbor]; sum_s += s[neighbor]; ++count;
                        }
                    }
                }
                const double avg_a = count ? sum_a / count : 0.0;
                const double avg_s = count ? sum_s / count : 0.0;
                const double change_a = (avg_a - a[cell]) * 0.1 + n[cell * 2];
                const double change_s = (avg_s - s[cell]) * 0.1 + n[cell * 2 + 1];
                a[cell] = std::max(0.0, a[cell] + change_a);
                s[cell] = std::max(0.0, s[cell] + change_s);
            }
        }
    }
    return py::make_tuple(out_activity, out_stability);
}

class PathHistogram {
    std::vector<std::int64_t> indices_, counts_;
public:
    PathHistogram(const Indices &indices, Size paths) {
        require_vector(indices, "indices");
        if (paths <= 0) throw py::value_error("number of paths must be positive");
        counts_.assign(static_cast<std::size_t>(paths), 0);
        indices_.assign(indices.data(), indices.data() + indices.size());
        for (auto path : indices_) {
            if (path < 0 || path >= paths) throw py::index_error("path index out of range");
            ++counts_[path];
        }
    }
    void update(Size index, std::int64_t path) {
        if (index < 0 || static_cast<std::size_t>(index) >= indices_.size()) throw py::index_error("entity index out of range");
        if (path < 0 || static_cast<std::size_t>(path) >= counts_.size()) throw py::index_error("path index out of range");
        --counts_[indices_[index]]; ++counts_[path]; indices_[index] = path;
    }
    py::array_t<double> distribution() const {
        py::array_t<double> result(counts_.size());
        double *out = result.mutable_data();
        for (std::size_t path = 0; path < counts_.size(); ++path)
            out[path] = indices_.empty() ? 1.0 / counts_.size() : static_cast<double>(counts_[path]) / indices_.size();
        return result;
    }
};

class OpponentIndex {
    double bin_size_;
    Size alive_count_;
    std::vector<bool> alive_;
    std::vector<Size> fenwick_, bin_positions_;
    std::vector<std::int64_t> initial_bins_;
    std::map<std::int64_t, std::vector<Size>> bins_;

    std::int64_t bin(double score) const {
        if (!std::isfinite(score)) throw py::value_error("scores must be finite");
        // Match Python's float //, including e.g. 1.0 // 0.1 == 9.0.
        double remainder = std::fmod(score, bin_size_);
        double quotient = (score - remainder) / bin_size_;
        if (remainder != 0.0 && remainder < 0.0) quotient -= 1.0;
        double floored = std::floor(quotient);
        if (quotient - floored > 0.5) floored += 1.0;
        if (!std::isfinite(floored) || floored <= static_cast<double>(std::numeric_limits<std::int64_t>::min()) || floored >= static_cast<double>(std::numeric_limits<std::int64_t>::max()))
            throw py::value_error("score bin is outside the supported integer range");
        return static_cast<std::int64_t>(floored);
    }
    void check_entity(Size entity) const {
        if (entity < 0 || static_cast<std::size_t>(entity) >= alive_.size()) throw py::index_error("entity index out of range");
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
            if (found != bins_.end()) count += static_cast<Size>(found->second.size()) - (initial_bins_[entity] == b ? 1 : 0);
        }
        return count;
    }
public:
    OpponentIndex(const Doubles &scores, double bin_size) : bin_size_(bin_size), alive_count_(scores.size()) {
        require_vector(scores, "scores");
        if (!std::isfinite(bin_size) || bin_size <= 0.0) throw py::value_error("bin size must be finite and positive");
        const Size count = scores.size();
        alive_.assign(count, true); fenwick_.resize(count + 1, 0);
        initial_bins_.reserve(count); bin_positions_.reserve(count);
        for (Size i = 0; i < count; ++i) {
            const auto key = bin(scores.data()[i]);
            initial_bins_.push_back(key);
            auto &bucket = bins_[key]; bin_positions_.push_back(bucket.size()); bucket.push_back(i);
            fenwick_[i + 1] = (i + 1) & -(i + 1);
        }
    }
    Size alive_count() const { return alive_count_; }
    Size entity_at(Size rank) const {
        if (rank < 0 || rank >= alive_count_) throw py::index_error("alive rank out of range");
        Size index = 0, bit = 1;
        while (bit <= static_cast<Size>(alive_.size()) / 2) bit <<= 1;
        for (; bit > 0; bit >>= 1) {
            const Size next = index + bit;
            if (next < static_cast<Size>(fenwick_.size()) && fenwick_[next] <= rank) {
                index = next; rank -= fenwick_[next];
            }
        }
        return index;
    }
    Size opponent_count(Size entity, double current_score) const {
        check_entity(entity);
        const Size nearby = nearby_count(entity, bin(current_score));
        return nearby ? nearby : alive_count_ - (alive_[entity] ? 1 : 0);
    }
    Size opponent_at(Size entity, double current_score, Size rank) const {
        check_entity(entity);
        const auto current_bin = bin(current_score);
        const Size nearby = nearby_count(entity, current_bin);
        const Size count = nearby ? nearby : alive_count_ - (alive_[entity] ? 1 : 0);
        if (rank < 0 || rank >= count) throw py::index_error("opponent rank out of range");
        if (!nearby) {
            if (alive_[entity] && rank >= alive_before(entity)) ++rank;
            return entity_at(rank);
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
        check_entity(entity);
        if (!alive_[entity]) return;
        alive_[entity] = false; --alive_count_;
        for (Size i = entity + 1; i < static_cast<Size>(fenwick_.size()); i += i & -i) --fenwick_[i];
        // Deliberately retain the original buckets, including dead candidates.
    }
};

PYBIND11_MODULE(_tiemu_native, module) {
    module.doc() = "Tiemu ordered float64 simulation kernels; RNG remains in Python";
    module.def("path_affinities", &path_affinities, py::arg("values"), py::arg("matrix"));
    module.def("concepts", &concepts, py::arg("paths"), py::arg("multiplier"), py::arg("distribution") = py::none(), py::arg("is_boss") = false);
    module.def("purify", &purify, py::arg("values"), py::arg("factor"));
    module.def("normalize", &normalize, py::arg("values"), py::arg("max_norm"));
    module.def("synergy", &synergy, py::arg("a"), py::arg("b"));
    module.def("clash", &clash, py::arg("winner"), py::arg("loser"), py::arg("modifier"));
    module.def("mentorship", &mentorship, py::arg("stronger"), py::arg("weaker"), py::arg("score_ratio"));
    module.def("repulsion", &repulsion, py::arg("values"), py::arg("noise"));
    module.def("vote_totals", &vote_totals, py::arg("dom_indices"), py::arg("weighted_scores"), py::arg("vote_logits"));
    module.def("inorganic_step", &inorganic_step, py::arg("activity"), py::arg("stability"), py::arg("noise"), py::arg("grid_size"));
    py::class_<PathHistogram>(module, "PathHistogram")
        .def(py::init<const Indices &, Size>(), py::arg("indices"), py::arg("n_paths"))
        .def("update", &PathHistogram::update, py::arg("index"), py::arg("dom"))
        .def("distribution", &PathHistogram::distribution);
    py::class_<OpponentIndex>(module, "OpponentIndex")
        .def(py::init<const Doubles &, double>(), py::arg("scores"), py::arg("bin_size"))
        .def_property_readonly("alive_count", &OpponentIndex::alive_count)
        .def("entity_at", &OpponentIndex::entity_at, py::arg("rank"))
        .def("opponent_count", &OpponentIndex::opponent_count, py::arg("entity_index"), py::arg("current_score"))
        .def("opponent_at", &OpponentIndex::opponent_at, py::arg("entity_index"), py::arg("current_score"), py::arg("rank"))
        .def("remove", &OpponentIndex::remove, py::arg("entity_index"));
}
