#include "tiemu_core.h"
#include "tiemu_internal.h"
#include "tiemu_random.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

extern "C" int tiemu_inorganic_run(
    const double *activity, std::size_t activity_count,
    const double *stability, std::size_t stability_count,
    const std::uint32_t *mt, std::size_t mt_count, std::int64_t steps, std::int64_t grid,
    double *out_activity, std::size_t out_activity_count,
    double *out_stability, std::size_t out_stability_count,
    std::uint32_t *out_mt, std::size_t out_mt_count) {
    return tiemu_detail::guarded([&] {
        if (steps < 0 || grid <= 0)
            throw std::invalid_argument("steps must be nonnegative and grid must be positive");
        const auto side = static_cast<std::size_t>(grid);
        if (side > std::numeric_limits<std::size_t>::max() / side)
            throw std::invalid_argument("grid size overflow");
        const std::size_t count = side * side;
        if (count > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(double))
            throw std::invalid_argument("grid exceeds the addressable element range");
        if (!activity || !stability || activity_count != count || stability_count != count)
            throw std::invalid_argument("activity and stability must match the square grid");
        if (!out_activity || !out_stability || out_activity_count < count || out_stability_count < count)
            throw std::invalid_argument("grid output buffer too small");
        if (!mt || mt_count != 625 || !out_mt || out_mt_count < 625)
            throw std::invalid_argument("MT state must have 625 uint32 elements");
        tiemu_detail::Mt19937 rng(mt);
        std::copy_n(activity, count, out_activity);
        std::copy_n(stability, count, out_stability);
        for (std::int64_t step = 0; step < steps; ++step) {
            for (std::int64_t i = 0; i < grid; ++i) {
                for (std::int64_t j = 0; j < grid; ++j) {
                    const std::size_t cell = static_cast<std::size_t>(i) * side + j;
                    double sum_a = 0.0, sum_s = 0.0;
                    int neighbors = 0;
                    for (int di = -1; di <= 1; ++di) {
                        for (int dj = -1; dj <= 1; ++dj) {
                            if ((di == 0 && dj == 0) || i + di < 0 || i + di >= grid
                                || j + dj < 0 || j + dj >= grid) continue;
                            const std::size_t neighbor = static_cast<std::size_t>(i + di) * side + j + dj;
                            sum_a += out_activity[neighbor];
                            sum_s += out_stability[neighbor];
                            ++neighbors;
                        }
                    }
                    const double average_a = neighbors ? sum_a / neighbors : 0.0;
                    const double average_s = neighbors ? sum_s / neighbors : 0.0;
                    const double noise_a = -0.1 + (0.1 - (-0.1)) * rng.random53();
                    const double noise_s = -0.1 + (0.1 - (-0.1)) * rng.random53();
                    const double change_a = (average_a - out_activity[cell]) * 0.1 + noise_a;
                    const double change_s = (average_s - out_stability[cell]) * 0.1 + noise_s;
                    out_activity[cell] = std::max(0.0, out_activity[cell] + change_a);
                    out_stability[cell] = std::max(0.0, out_stability[cell] + change_s);
                }
            }
        }
        rng.write(out_mt);
    });
}
