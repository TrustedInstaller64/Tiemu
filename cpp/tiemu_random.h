#ifndef TIEMU_RANDOM_H
#define TIEMU_RANDOM_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace tiemu_detail {

// MT19937 state transition and 53-bit doubles used by CPython random and
// NumPy's legacy RandomState. Import/export the existing state rather than
// reseeding or substituting std::uniform_* distributions.
class Mt19937 {
    std::uint32_t words_[624];
    std::uint32_t position_;

public:
    explicit Mt19937(const std::uint32_t *state) {
        if (!state || state[624] > 624)
            throw std::invalid_argument("MT state must contain index 0..624");
        std::copy_n(state, 624, words_);
        position_ = state[624];
    }

    std::uint32_t next() {
        if (position_ == 624) {
            for (std::size_t i = 0; i < 624; ++i) {
                const std::uint32_t joined = (words_[i] & 0x80000000u)
                    | (words_[(i + 1) % 624] & 0x7fffffffu);
                words_[i] = words_[(i + 397) % 624] ^ (joined >> 1)
                    ^ ((joined & 1u) ? 0x9908b0dfu : 0u);
            }
            position_ = 0;
        }
        std::uint32_t value = words_[position_++];
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c5680u;
        value ^= (value << 15) & 0xefc60000u;
        value ^= value >> 18;
        return value;
    }

    double random53() {
        const std::uint32_t high = next() >> 5;
        const std::uint32_t low = next() >> 6;
        return (high * 67108864.0 + low) * (1.0 / 9007199254740992.0);
    }

    // NumPy RandomState's inclusive bounded-uint32 masked rejection. In
    // particular an interval of size one consumes no MT word.
    std::uint32_t interval(std::uint32_t maximum) {
        if (maximum == 0) return 0;
        std::uint32_t mask = maximum;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        std::uint32_t value;
        do { value = next() & mask; } while (value > maximum);
        return value;
    }

    void write(std::uint32_t *output) const {
        if (!output) throw std::invalid_argument("MT output is null");
        std::copy_n(words_, 624, output);
        output[624] = position_;
    }
};

} // namespace tiemu_detail
#endif
