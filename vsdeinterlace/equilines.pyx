# cython: boundscheck=False, initializedcheck=False, language_level=3, nonecheck=False, overflowcheck=False, wraparound=False

cimport cython
from cython cimport view
from libc.math cimport fabs
from cython.parallel import prange


cdef int clamp(const int val, const int low, const int high) noexcept nogil:
    return min(max(val, low), high)


cpdef void equilines_filter(
    const unsigned short [:, ::view.contiguous] src, unsigned short [:, ::view.contiguous] dst,
    const int threshold
) noexcept nogil:
    cdef int height = src.shape[0]
    cdef int width = src.shape[1]

    cdef int top_acc, bott_acc, delta
    cdef int y, y1, x

    with nogil:
        for y in prange(0, height, 2):
            y1 = y + 1
            top_acc = 0
            bott_acc = 0

            for x in prange(width):
                top_acc += src[y, x]
                bott_acc += src[y1, x]

            delta = (bott_acc - top_acc) // (width * 2)

            if abs(delta) < threshold:
                for x in prange(width):
                    dst[y, x] = clamp(src[y, x] + delta, 0, 65535)
                    dst[y1, x] = clamp(src[y1, x] - delta, 0, 65535)
