#pragma once
/**
 * rolling_buffer.h — Fixed-capacity circular buffer of feature vectors.
 *
 * Stores up to BUFFER_CAPACITY rows of N_FEATURES doubles each in a flat
 * C array (no heap allocation after construction). Oldest entries are
 * silently overwritten when the buffer is full, matching Python's
 * collections.deque(maxlen=N) semantics.
 *
 * get_matrix() copies the contents into a caller-supplied buffer in
 * chronological order with a pair of memcpy calls, making it suitable
 * for zero-copy hand-off to a numpy ndarray.
 */
#include <cstddef>
#include <cstring>

namespace fdir {

constexpr std::size_t N_FEATURES       = 7;
constexpr std::size_t BUFFER_CAPACITY  = 5000;

class RollingFeatureBuffer {
public:
    RollingFeatureBuffer() noexcept : _head(0), _size(0) {}

    // Append one feature vector (must point to exactly N_FEATURES doubles).
    inline void push(const double* __restrict__ features) noexcept {
        std::memcpy(_data[_head], features, N_FEATURES * sizeof(double));
        _head = (_head + 1) % BUFFER_CAPACITY;
        if (_size < BUFFER_CAPACITY)
            ++_size;
    }

    /**
     * Copy buffer contents in chronological order into `output`.
     * `output` must hold at least size() * N_FEATURES doubles.
     * Returns the number of rows written.
     */
    std::size_t get_matrix(double* __restrict__ output) const noexcept {
        if (_size == 0) return 0;

        if (_size < BUFFER_CAPACITY) {
            // Buffer not yet full — data lives at [0, _size)
            std::memcpy(output, _data[0], _size * N_FEATURES * sizeof(double));
        } else {
            // Full circular buffer: oldest entry is at _head.
            std::size_t tail        = _head;
            std::size_t first_chunk = BUFFER_CAPACITY - tail;  // rows from tail→end

            std::memcpy(output,
                        _data[tail],
                        first_chunk * N_FEATURES * sizeof(double));
            std::memcpy(output + first_chunk * N_FEATURES,
                        _data[0],
                        tail * N_FEATURES * sizeof(double));
        }
        return _size;
    }

    std::size_t size()     const noexcept { return _size; }
    bool        is_empty() const noexcept { return _size == 0; }

    bool ready_for_fit(std::size_t min_samples) const noexcept {
        return _size >= min_samples;
    }

    static constexpr std::size_t capacity()   noexcept { return BUFFER_CAPACITY; }
    static constexpr std::size_t n_features() noexcept { return N_FEATURES; }

    void reset() noexcept { _head = 0; _size = 0; }

private:
    // Row-major storage: _data[row][feature]
    double      _data[BUFFER_CAPACITY][N_FEATURES];
    std::size_t _head;   // next write slot (oldest entry when full)
    std::size_t _size;   // number of valid rows (≤ BUFFER_CAPACITY)
};

} // namespace fdir
