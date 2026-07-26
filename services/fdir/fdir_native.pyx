# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Cython wrapper around the C++ RollingFeatureBuffer.

Exposes PyRollingBuffer — a drop-in replacement for the Python deque in
AnomalyDetector._window.  The key wins are:

  * push()      — stores 7 doubles with a single memcpy (no Python boxing)
  * get_matrix()— copies the circular buffer into a numpy array with at most
                  two memcpy calls, avoiding the O(N) Python-list iteration
                  that np.array(list(deque)) requires.

Memory layout: 5000 × 7 × 8 B = 280 KB — fits comfortably in L2 cache.
"""
import  numpy as np
cimport numpy as np
cdef extern from "<stddef.h>":
    ctypedef unsigned long size_t

# ── C++ class declaration ──────────────────────────────────────────────────────
cdef extern from "rolling_buffer.h" namespace "fdir":
    cdef cppclass RollingFeatureBuffer:
        RollingFeatureBuffer()                          except +
        void   push(const double* features)             nogil
        size_t get_matrix(double* output)               nogil
        size_t size()                                   nogil
        bint   is_empty()                               nogil
        bint   ready_for_fit(size_t min_samples)        nogil
        void   reset()                                  nogil
        size_t capacity()
        size_t n_features()


# ── Python-visible wrapper class ───────────────────────────────────────────────
cdef class PyRollingBuffer:
    """
    Circular buffer for FDIR feature vectors, backed by a C++ value object.

    Usage::

        buf = PyRollingBuffer()
        buf.push(battery_percent=85.0, cpu_temp_celsius=50.0, ...)
        X = buf.get_matrix()          # (n, 7) float64 ndarray, zero extra copy
        if buf.ready_for_fit(500):
            model.fit(X)
    """
    cdef RollingFeatureBuffer _buf   # embedded by value (no heap alloc)

    # ── mutation ───────────────────────────────────────────────────────────────
    def push(
        self,
        double battery_percent,
        double cpu_temp_celsius,
        double motor_temp_celsius,
        double motor_load_left,
        double motor_load_right,
        double velocity_ms,
        double obstacle_density,
    ):
        """Append one feature vector.  Overwrites oldest entry when full."""
        cdef double feat[7]
        feat[0] = battery_percent
        feat[1] = cpu_temp_celsius
        feat[2] = motor_temp_celsius
        feat[3] = motor_load_left
        feat[4] = motor_load_right
        feat[5] = velocity_ms
        feat[6] = obstacle_density
        with nogil:
            self._buf.push(feat)

    def reset(self):
        """Clear the buffer (useful for testing)."""
        self._buf.reset()

    # ── observation ────────────────────────────────────────────────────────────
    def get_matrix(self):
        """
        Return buffer contents as a C-contiguous (n, 7) float64 ndarray in
        chronological order.  Uses at most two memcpy calls regardless of
        wrap-around state.
        """
        cdef size_t n = self._buf.size()
        if n == 0:
            return np.empty((0, 7), dtype=np.float64)

        cdef np.ndarray[double, ndim=2, mode='c'] arr = \
            np.empty((n, 7), dtype=np.float64)

        cdef double[::1] flat = arr.ravel()   # contiguous view
        with nogil:
            self._buf.get_matrix(&flat[0])

        return arr

    # ── properties ─────────────────────────────────────────────────────────────
    @property
    def size(self):
        return self._buf.size()

    @property
    def capacity(self):
        return self._buf.capacity()

    @property
    def n_features(self):
        return self._buf.n_features()

    def ready_for_fit(self, size_t min_samples = 500):
        return bool(self._buf.ready_for_fit(min_samples))

    def __len__(self):
        return self._buf.size()

    def __repr__(self):
        return (f"PyRollingBuffer(size={self._buf.size()}, "
                f"capacity={self._buf.capacity()})")
